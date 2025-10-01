import mlx.core as mx
import mlx.nn as nn

from mossformer2_block import ScaledSinuEmbedding, MossformerBlock_GFSMN
from conv_module import select_norm

EPS = 1e-8

class Encoder(nn.Module):
    """Convolutional Encoder Layer."""

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super().__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def __call__(self, x):
        """Return the encoded output."""
        # Input: B x L (if 2D) or B x C x L (if 3D)
        if x.ndim == 2:
            # B x L -> B x L x 1
            x = mx.expand_dims(x, axis=-1)
        else:
            # B x C x L -> B x L x C
            x = mx.transpose(x, (0, 2, 1))
        
        # B x L x C -> B x L x N
        x = self.conv1d(x)
        x = nn.relu(x)
        
        # Convert back to B x N x L for compatibility
        x = mx.transpose(x, (0, 2, 1))
        return x

class Decoder(nn.Module):
    """A decoder layer that consists of ConvTranspose1d."""

    def __init__(self, in_channels, out_channels, kernel_size, stride, bias=False):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias
        )

    def __call__(self, x):
        """Return the decoded output."""
        if x.ndim not in [2, 3]:
            raise RuntimeError(f"{self.__class__.__name__} accept 2/3D tensor as input")
        
        # Input: B x N x L -> B x L x N for MLX
        if x.ndim == 3:
            x = mx.transpose(x, (0, 2, 1))
        elif x.ndim == 2:
            x = mx.expand_dims(x, axis=-1)
        
        # B x L x N -> B x L_out x C_out
        x = self.conv_transpose(x)
        
        # If output channels is 1, squeeze and return B x L_out
        if x.shape[-1] == 1:
            x = mx.squeeze(x, axis=-1)
        else:
            # Otherwise convert back to B x C x L format
            x = mx.transpose(x, (0, 2, 1))
        
        return x

class MossFormerM(nn.Module):
    """This class implements the transformer encoder."""
    
    def __init__(
        self,
        num_blocks,
        d_model=None,
        causal=False,
        group_size=256,
        query_key_dim=128,
        expansion_factor=4.,
        attn_dropout=0.1
    ):
        super().__init__()

        self.mossformerM = MossformerBlock_GFSMN(
            dim=d_model,
            depth=num_blocks,
            group_size=group_size,
            query_key_dim=query_key_dim,
            expansion_factor=expansion_factor,
            causal=causal,
            attn_dropout=attn_dropout
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def __call__(self, src):
        """
        Arguments
        ----------
        src : mx.array
            Tensor shape [B, L, N],
            where, B = Batchsize,
                   L = time points
                   N = number of filters
        """
        
        output = self.mossformerM(src)
        
        output = self.norm(output)
        
        return output

class Computation_Block(nn.Module):
    """Computation block for dual-path processing."""

    def __init__(
        self,
        num_blocks,
        out_channels,
        norm="ln",
        skip_around_intra=True,
    ):
        super().__init__()

        # MossFormer+: MossFormer with recurrence
        self.intra_mdl = MossFormerM(num_blocks=num_blocks, d_model=out_channels)
        self.skip_around_intra = skip_around_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = select_norm(norm, out_channels, 3)

    def __call__(self, x):
        """Returns the output tensor."""
        B, N, S = x.shape
        
        # intra RNN
        # [B, S, N]
        intra = mx.transpose(x, (0, 2, 1))

        intra = self.intra_mdl(intra)
        
        # [B, N, S]
        intra = mx.transpose(intra, (0, 2, 1))
        if self.norm is not None:
            intra = self.intra_norm(intra)
            
        # [B, N, S]
        if self.skip_around_intra:
            intra = intra + x
            
        out = intra
        return out

class MossFormer_MaskNet(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet."""

    def __init__(
        self,
        in_channels,
        out_channels,
        out_channels_final,
        num_blocks=24,
        norm="ln",
        num_spks=1,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super().__init__()
        self.num_spks = num_spks
        self.num_blocks = num_blocks
        self.norm = select_norm(norm, in_channels, 3)
        self.conv1d_encoder = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = ScaledSinuEmbedding(out_channels)

        self.mdl = Computation_Block(
            num_blocks,
            out_channels,
            norm,
            skip_around_intra=skip_around_intra,
        )

        self.conv1d_out = nn.Conv1d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.conv1_decoder = nn.Conv1d(out_channels, out_channels_final, 1, bias=False)
        self.prelu = nn.PReLU()
        
        # gated output layer
        self.output_conv = nn.Conv1d(out_channels, out_channels, 1)
        self.output_gate_conv = nn.Conv1d(out_channels, out_channels, 1)

    def __call__(self, x):
        """Returns the output tensor."""
        # Input: [B, N, L] (PyTorch format: batch, channels, length)
        # GroupNorm with pytorch_compatible=True expects [B, C, ...spatial...] format
        # So we keep it as [B, N, L] for the norm
        
        x = self.norm(x)  # Apply norm on [B, N, L] format
        
        # Now transpose for Conv1d which expects [B, L, N]
        x = mx.transpose(x, (0, 2, 1))  # [B, L, N]
        x = self.conv1d_encoder(x)  # Output: [B, L, out_channels]
        
        if self.use_global_pos_enc:
            # x is [B, L, N]
            emb = self.pos_enc(x)  # [B, L, N]
            x = x + emb
            
        # We need to go back to [B, N, S] format for the mdl processing
        x = mx.transpose(x, (0, 2, 1))  # [B, N, S]
        
        # [B, N, S]
        x = self.mdl(x)
        
        x = self.prelu(x)
        
        # Convert back to [B, S, N] for Conv1d
        x = mx.transpose(x, (0, 2, 1))  # [B, S, N]
        
        # [B, S, N*spks]
        x = self.conv1d_out(x)
        
        B, S, _ = x.shape

        # [B*spks, S, N]
        x = mx.reshape(x, (B * self.num_spks, S, -1))
        
        # [B*spks, S, N]
        output_val = mx.tanh(self.output_conv(x))
        gate_val = mx.sigmoid(self.output_gate_conv(x))
        
        x = output_val * gate_val
        
        # [B*spks, S, N]
        x = self.conv1_decoder(x)
        
        # [B, spks, S, N]
        _, S, N = x.shape
        x = mx.reshape(x, (B, self.num_spks, S, N))
        x = nn.relu(x)
        
        # Convert back to [B, spks, N, S] then [spks, B, N, S]
        x = mx.transpose(x, (0, 1, 3, 2))  # [B, spks, N, S]
        x = mx.transpose(x, (1, 0, 2, 3))  # [spks, B, N, S]

        return x[0]

class MossFormer(nn.Module):
    def __init__(
        self,
        in_channels=512,
        out_channels=512,
        num_blocks=24,
        kernel_size=16,
        norm="ln",
        num_spks=2,
        skip_around_intra=True,
        use_global_pos_enc=True,
        max_length=20000,
    ):
        super().__init__()
        self.num_spks = num_spks
        self.enc = Encoder(kernel_size=kernel_size, out_channels=in_channels, in_channels=180)
        self.mask_net = MossFormer_MaskNet(
            in_channels=in_channels,
            out_channels=out_channels,
            out_channels_final=in_channels,
            num_blocks=num_blocks,
            norm=norm,
            num_spks=num_spks,
            skip_around_intra=skip_around_intra,
            use_global_pos_enc=use_global_pos_enc,
            max_length=max_length,
        )
        self.dec = Decoder(
           in_channels=out_channels,
           out_channels=1,
           kernel_size=kernel_size,
           stride=kernel_size//2,
           bias=False
        )
    
    def __call__(self, input):
        x = self.enc(input)
        mask = self.mask_net(x)
        x = mx.stack([x] * self.num_spks)
        sep_x = x * mask

        # Decoding - use vmap for parallel processing of speakers
        # sep_x has shape (num_spks, B, N, L)
        # Apply decoder to all speakers in parallel
        decoded = mx.vmap(self.dec, in_axes=0, out_axes=0)(sep_x)  # Shape: (num_spks, B, L)
        
        # Transpose to match expected output shape (B, L, num_spks)
        est_source = mx.transpose(decoded, (1, 2, 0))
        T_origin = input.shape[1]
        T_est = est_source.shape[1]
        if T_origin > T_est:
            est_source = mx.pad(est_source, [(0, 0), (0, T_origin - T_est), (0, 0)])
        else:
            est_source = est_source[:, :T_origin, :]

        # Extract speakers using list slicing
        out = [est_source[:,:,spk] for spk in range(self.num_spks)]
        return out