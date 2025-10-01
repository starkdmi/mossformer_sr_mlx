import mlx.core as mx
import mlx.nn as nn

from conv_module import ConvModule
from fsmn import UniDeepFsmn
from layer_norm import CLayerNorm

# Helper functions
def exists(val):
    """Checks if a value exists (is not None)."""
    return val is not None

def default(val, d):
    """Returns a default value if the given value does not exist."""
    return val if exists(val) else d

def padding_to_multiple_of(n, mult):
    """Calculates the amount of padding needed to make a number a multiple of another."""
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder

# Scale Normalization class
class ScaleNorm(nn.Module):
    """ScaleNorm implements a scaled normalization technique for neural network layers."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5  # Calculate scale factor
        self.eps = eps  # Set epsilon
        self.g = mx.ones((1,))  # Initialize scaling parameter

    def __call__(self, x):
        """Forward pass for the ScaleNorm layer."""
        norm = mx.linalg.norm(x, axis=-1, keepdims=True) * self.scale  # Compute norm
        return x / mx.maximum(norm, self.eps) * self.g  # Normalize and scale

# Absolute positional encodings class
class ScaledSinuEmbedding(nn.Module):
    """ScaledSinuEmbedding provides sinusoidal positional encodings for inputs."""

    def __init__(self, dim):
        super().__init__()
        self.scale = mx.ones((1,))  # Initialize scale
        inv_freq = 1. / (10000 ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        self.inv_freq = inv_freq  # Store as attribute

    def __call__(self, x):
        """Forward pass for the ScaledSinuEmbedding layer."""
        n = x.shape[1]  # Extract sequence length
        t = mx.arange(n, dtype=mx.float32)  # Create time steps
        # Calculate sine and cosine embeddings
        sinu = mx.expand_dims(t, 1) * mx.expand_dims(self.inv_freq, 0)
        emb = mx.concatenate([mx.sin(sinu), mx.cos(sinu)], axis=-1)  # Concatenate sine and cosine
        return emb * self.scale  # Scale the embeddings

class OffsetScale(nn.Module):
    """OffsetScale applies learned offsets and scales to the input tensor."""

    def __init__(self, dim, heads=1):
        super().__init__()
        self.heads = heads
        self.dim = dim
        # Initialize with same pattern as PyTorch - will be overwritten by loaded weights
        self.gamma = mx.ones((heads, dim)) * 0.02
        self.beta = mx.zeros((heads, dim))  # Initialize offset parameters

    def __call__(self, x):
        """Forward pass for the OffsetScale layer."""
        
        # Apply scaling and offsets - x shape: (..., dim)
        # gamma/beta shape: (heads, dim)
        # Output shape: (..., heads, dim)
        out = mx.expand_dims(x, -2) * mx.expand_dims(self.gamma, 0) + mx.expand_dims(self.beta, 0)
        # Split into list of tensors for each head
        return [out[..., i, :] for i in range(out.shape[-2])]

# Feed-Forward Convolutional Module
class FFConvM(nn.Module):
    """FFConvM is a feed-forward convolutional module with normalization and dropout."""

    def __init__(
        self,
        dim_in,
        dim_out,
        norm_klass=nn.LayerNorm,
        dropout=0.1
    ):
        super().__init__()
        self.norm = norm_klass(dim_in)
        self.linear = nn.Linear(dim_in, dim_out)
        self.conv = ConvModule(dim_out)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x):
        """Forward pass for the FFConvM module."""
        
        x = self.norm(x)  # Normalize input
        
        x = self.linear(x)  # Linear transformation
        
        x = nn.silu(x)  # SiLU activation
        
        x = self.conv(x)  # Convolution module
        
        # x = self.dropout(x)  # Apply dropout
        
        return x

class FLASH_ShareA_FFConvM(nn.Module):
    """ 
    Fast Shared Dual Attention Mechanism with feed-forward convolutional blocks.
    Published in paper: "MossFormer: Pushing the Performance Limit of Monaural Speech Separation 
    using Gated Single-Head Transformer with Convolution-Augmented Joint Self-Attentions", ICASSP 2023.
    """
    
    def __init__(
        self,
        *,
        dim,
        group_size=256,
        query_key_dim=128,
        expansion_factor=1.,
        causal=False,
        dropout=0.1,
        rotary_pos_emb=None,
        norm_klass=nn.LayerNorm,
        shift_tokens=True
    ):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)        
        self.group_size = group_size
        self.causal = causal
        self.shift_tokens = shift_tokens

        # Initialize positional embeddings, dropout, and projections
        self.rotary_pos_emb = rotary_pos_emb
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward layers
        self.to_hidden = FFConvM(
            dim_in=dim,
            dim_out=hidden_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        self.to_qk = FFConvM(
            dim_in=dim,
            dim_out=query_key_dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )
        
        # Offset and scale for query and key
        self.qk_offset_scale = OffsetScale(query_key_dim, heads=4)

        self.to_out = FFConvM(
            dim_in=dim * 2,
            dim_out=dim,
            norm_klass=norm_klass,
            dropout=dropout,
        )

    def __call__(self, x, *, mask=None):
        """Forward pass for FLASH layer."""
        
        # Pre-normalization step
        normed_x = x 
        residual = x  # Save residual for skip connection

        # Token shifting if enabled
        if self.shift_tokens:
            x_shift, x_pass = mx.split(normed_x, 2, axis=-1)
            x_shift = mx.pad(x_shift, [(0, 0), (1, 0), (0, 0)], constant_values=0.)
            x_shift = x_shift[:, :-1, :]  # Remove last position
            normed_x = mx.concatenate([x_shift, x_pass], axis=-1)

        # Initial projections
        hidden = self.to_hidden(normed_x)
        v, u = mx.split(hidden, 2, axis=-1)
        qk = self.to_qk(normed_x)

        # Offset and scale
        quad_q, lin_q, quad_k, lin_k = self.qk_offset_scale(qk)
        
        att_v, att_u = self.cal_attention(x, quad_q, lin_q, quad_k, lin_k, v, u, mask)

        # Output calculation with gating
        out = (att_u * v) * mx.sigmoid(att_v * u)
        
        x = x + self.to_out(out)  # Residual connection
        
        return x

    def cal_attention(self, x, quad_q, lin_q, quad_k, lin_k, v, u, mask=None):
        """Calculate attention output using quadratic and linear attention mechanisms."""
        b, n, device, g = x.shape[0], x.shape[-2], None, self.group_size
        
        # Apply mask to linear keys if provided
        if exists(mask):
            lin_mask = mx.expand_dims(mask, -1)
            lin_k = mx.where(lin_mask, lin_k, 0.)

        # Rotate queries and keys with rotary positional embeddings
        if exists(self.rotary_pos_emb):
            
            quad_q = self.rotary_pos_emb.rotate_queries_or_keys(quad_q)
            lin_q = self.rotary_pos_emb.rotate_queries_or_keys(lin_q)
            quad_k = self.rotary_pos_emb.rotate_queries_or_keys(quad_k)
            lin_k = self.rotary_pos_emb.rotate_queries_or_keys(lin_k)
            
        # Padding for group processing
        padding = padding_to_multiple_of(n, g)
        if padding > 0:
            pad_width = [(0, 0), (0, padding), (0, 0)]
            quad_q = mx.pad(quad_q, pad_width, constant_values=0.)
            quad_k = mx.pad(quad_k, pad_width, constant_values=0.)
            lin_q = mx.pad(lin_q, pad_width, constant_values=0.)
            lin_k = mx.pad(lin_k, pad_width, constant_values=0.)
            v = mx.pad(v, pad_width, constant_values=0.)
            u = mx.pad(u, pad_width, constant_values=0.)
            if mask is None:
                mask = mx.ones((b, n), dtype=mx.bool_)
            mask = mx.pad(mask, [(0, 0), (0, padding)], constant_values=False)

        # Group along sequence for attention
        # Reshape from (b, g*n, d) to (b, g, n, d)
        def reshape_for_groups(t):
            b_size, seq_len, d = t.shape
            return mx.reshape(t, (b_size, seq_len // self.group_size, self.group_size, d))
        
        quad_q = reshape_for_groups(quad_q)
        quad_k = reshape_for_groups(quad_k)
        lin_q = reshape_for_groups(lin_q)
        lin_k = reshape_for_groups(lin_k)
        v = reshape_for_groups(v)
        u = reshape_for_groups(u)

        if exists(mask):
            mask = mx.reshape(mask, (b, -1, self.group_size))
            mask = mx.expand_dims(mask, 2)

        # Calculate quadratic attention output
        # quad_q/k shape: (b, g, n, d)
        sim = mx.matmul(quad_q, mx.swapaxes(quad_k, -2, -1)) / g
        
        attn = mx.square(nn.relu(sim))  # ReLU activation and square
        
        # attn = self.dropout(attn)
        
        # Apply mask to attention if provided
        if exists(mask):
            attn = mx.where(mask, attn, 0.)

        # Apply causal mask if needed
        if self.causal:
            causal_mask = mx.triu(mx.ones((g, g), dtype=mx.bool_), k=1)
            causal_mask = mx.expand_dims(mx.expand_dims(causal_mask, 0), 0)
            attn = mx.where(causal_mask, 0., attn)

        # Calculate output from attention
        quad_out_v = mx.matmul(attn, v)
        quad_out_u = mx.matmul(attn, u)
        
        # Calculate linear attention output
        if self.causal:
            # Causal linear attention with cumulative sum
            lin_kv = mx.matmul(mx.swapaxes(lin_k, -2, -1), v) / g
            lin_kv = mx.cumsum(lin_kv, axis=1)
            # Shift for causality
            lin_kv = mx.pad(lin_kv, [(0, 0), (1, 0), (0, 0), (0, 0)], constant_values=0.)[:, :-1]
            lin_out_v = mx.matmul(lin_q, lin_kv)

            lin_ku = mx.matmul(mx.swapaxes(lin_k, -2, -1), u) / g
            lin_ku = mx.cumsum(lin_ku, axis=1)
            lin_ku = mx.pad(lin_ku, [(0, 0), (1, 0), (0, 0), (0, 0)], constant_values=0.)[:, :-1]
            lin_out_u = mx.matmul(lin_q, lin_ku)
        else:
            # Non-causal linear attention
            # Compute attention per group then aggregate
            # lin_k, v shape: (b, num_groups, group_size, d)
            # We need to compute global key-value products across all positions
            lin_k_reshaped = mx.reshape(lin_k, (b, -1, lin_k.shape[-1]))  # (b, total_seq, d)
            v_reshaped = mx.reshape(v, (b, -1, v.shape[-1]))  # (b, total_seq, d)
            u_reshaped = mx.reshape(u, (b, -1, u.shape[-1]))  # (b, total_seq, d)
            
            # Global attention: all queries attend to all keys
            # PyTorch einsum 'b g n d, b g n e -> b d e' sums over g and n, but only divides by n
            # Info: n is the original sequence length before padding, not the padded length

            lin_kv = mx.matmul(mx.swapaxes(lin_k_reshaped, -2, -1), v_reshaped) / n  # (b, d, e)
            
            lin_out_v = mx.matmul(lin_q, lin_kv)  # (b, g, n, e)
            
            lin_ku = mx.matmul(mx.swapaxes(lin_k_reshaped, -2, -1), u_reshaped) / n  # (b, d, e)
            lin_out_u = mx.matmul(lin_q, lin_ku)  # (b, g, n, e)

        # Reshape and remove padding from outputs
        def reshape_from_groups(t):
            b_size, n_groups, group_size, d = t.shape
            return mx.reshape(t, (b_size, n_groups * group_size, d))[:, :n]
        
        final_v = reshape_from_groups(quad_out_v + lin_out_v)
        final_u = reshape_from_groups(quad_out_u + lin_out_u)
        
        return (final_v, final_u)


class Gated_FSMN(nn.Module):
    """Gated Frequency Selective Memory Network (FSMN) class."""
    
    def __init__(self, in_channels, out_channels, lorder, hidden_size):
        super().__init__()
        # Feedforward network for the first branch (u)
        self.to_u = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        # Feedforward network for the second branch (v)
        self.to_v = FFConvM(
            dim_in=in_channels,
            dim_out=hidden_size,
            norm_klass=nn.LayerNorm,
            dropout=0.1,
        )
        # Frequency selective memory network
        self.fsmn = UniDeepFsmn(in_channels, out_channels, lorder, hidden_size)

    def __call__(self, x):
        """Forward pass for the Gated FSMN."""
        input = x
        x_u = self.to_u(x)  # Process input through the first branch
        x_v = self.to_v(x)  # Process input through the second branch
        x_u = self.fsmn(x_u)  # Apply FSMN to the output of the first branch
        x = x_v * x_u + input  # Combine outputs with the original input
        return x

class Gated_FSMN_Block(nn.Module):
    """A 1-D convolutional block that incorporates a gated FSMN."""
    
    def __init__(self, dim, inner_channels=256, group_size=256, norm_type='scalenorm'):
        super().__init__()
        # Choose normalization class based on the provided type
        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # First convolutional layer with PReLU activation
        self.conv1 = nn.Conv1d(dim, inner_channels, kernel_size=1)
        self.prelu1 = nn.PReLU()
        self.norm1 = CLayerNorm(inner_channels)  # Normalization after first convolution
        self.gated_fsmn = Gated_FSMN(inner_channels, inner_channels, lorder=20, hidden_size=inner_channels)
        self.norm2 = CLayerNorm(inner_channels)  # Normalization after FSMN
        self.conv2 = nn.Conv1d(inner_channels, dim, kernel_size=1)  # Final convolutional layer

    def __call__(self, input):
        """Forward pass for the Gated FSMN Block."""
        
        # input shape: (batch, time, dim)
        # MLX Conv1d expects (batch, time, channels)
        x = self.conv1(input)
        
        x = self.prelu1(x)
        
        # CLayerNorm expects (batch, channels, time) format
        x = mx.transpose(x, (0, 2, 1))  # to (batch, channels, time)
        conv1 = self.norm1(x)
        
        # Back to (batch, time, channels) for FSMN
        x = mx.transpose(conv1, (0, 2, 1))
        seq_out = self.gated_fsmn(x)
        # Transpose for norm2
        x = mx.transpose(seq_out, (0, 2, 1))  # to (batch, channels, time)
        norm2 = self.norm2(x)
        # Back to (batch, time, channels) for conv2
        x = mx.transpose(norm2, (0, 2, 1))
        conv2 = self.conv2(x)
        # Add residual
        output = conv2 + input
        
        return output

# Rotary Embedding implementation for MLX
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        # Use MLX's built-in RoPE
        self.rope = nn.RoPE(dims=dim, traditional=True, base=base)
        self.dim = dim

    def rotate_queries_or_keys(self, x):
        """Apply rotary embeddings to queries or keys."""
        # Simply use the built-in RoPE
        return self.rope(x)

    def apply_rotary_emb(self, x, cos, sin):
        """Apply the rotary embedding to x."""
        # Split the last dimension into two halves
        d = x.shape[-1]
        x1 = x[..., :d//2]
        x2 = x[..., d//2:]
        
        # Apply rotation
        cos = cos[..., :d//2]
        sin = sin[..., :d//2]
        
        rotated = mx.concatenate([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], axis=-1)
        
        return rotated

class MossformerBlock_GFSMN(nn.Module):
    """Mossformer Block with Gated FSMN."""
    
    def __init__(self, *, dim, depth, group_size=256, query_key_dim=128, expansion_factor=4., 
                 causal=False, attn_dropout=0.1, norm_type='scalenorm', shift_tokens=True):
        super().__init__()
        assert norm_type in ('scalenorm', 'layernorm'), 'norm_type must be one of scalenorm or layernorm'

        if norm_type == 'scalenorm':
            norm_klass = ScaleNorm
        elif norm_type == 'layernorm':
            norm_klass = nn.LayerNorm

        self.group_size = group_size

        # Rotary positional embedding for attention
        rotary_pos_emb = RotaryEmbedding(dim=min(32, query_key_dim))

        # Create a list of Gated FSMN blocks
        self.fsmn = [Gated_FSMN_Block(dim) for _ in range(depth)]

        # Create a list of attention layers using FLASH_ShareA_FFConvM
        self.layers = [
            FLASH_ShareA_FFConvM(
                dim=dim,
                group_size=group_size,
                query_key_dim=query_key_dim,
                expansion_factor=expansion_factor,
                causal=causal,
                dropout=attn_dropout,
                rotary_pos_emb=rotary_pos_emb,
                norm_klass=norm_klass,
                shift_tokens=shift_tokens
            ) for _ in range(depth)
        ]

    def __call__(self, x, *, mask=None):
        """Forward pass for the Mossformer Block with Gated FSMN."""
        
        for idx, flash in enumerate(self.layers):
            x = flash(x, mask=mask)
            
            x = self.fsmn[idx](x)
            
        return x