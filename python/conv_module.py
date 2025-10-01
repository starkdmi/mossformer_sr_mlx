import mlx.core as mx
import mlx.nn as nn

EPS = 1e-8

class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization."""

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.shape = shape

        if self.elementwise_affine:
            if shape == 3:
                self.weight = mx.ones((self.dim, 1))
                self.bias = mx.zeros((self.dim, 1))
            if shape == 4:
                self.weight = mx.ones((self.dim, 1, 1))
                self.bias = mx.zeros((self.dim, 1, 1))

    def __call__(self, x):
        """Returns the normalized tensor."""
        if x.ndim == 3:
            mean = mx.mean(x, axis=(1, 2), keepdims=True)
            var = mx.mean((x - mean) ** 2, axis=(1, 2), keepdims=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) * mx.rsqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) * mx.rsqrt(var + self.eps)

        if x.ndim == 4:
            mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
            var = mx.mean((x - mean) ** 2, axis=(1, 2, 3), keepdims=True)
            if self.elementwise_affine:
                x = self.weight * (x - mean) * mx.rsqrt(var + self.eps) + self.bias
            else:
                x = (x - mean) * mx.rsqrt(var + self.eps)
        return x

class CumulativeLayerNorm(nn.LayerNorm):
    """Calculate Cumulative Layer Normalization."""

    def __init__(self, dim, elementwise_affine=True):
        super().__init__(dim, affine=elementwise_affine, eps=1e-8)

    def __call__(self, x):
        """Returns the normalized tensor."""
        # x: N x C x K x S or N x C x L
        if x.ndim == 4:
            # N x K x S x C
            x = mx.transpose(x, (0, 2, 3, 1))
            # N x K x S x C == only channel norm
            x = super().__call__(x)
            # N x C x K x S
            x = mx.transpose(x, (0, 3, 1, 2))
        if x.ndim == 3:
            # N x L x C
            x = mx.transpose(x, (0, 2, 1))
            # N x L x C == only channel norm
            x = super().__call__(x)
            # N x C x L
            x = mx.transpose(x, (0, 2, 1))
        return x

class GroupNormWrapper(nn.Module):
    """Wrapper for MLX GroupNorm to handle PyTorch-style input format."""
    def __init__(self, num_groups, dims, eps=1e-8):
        super().__init__()
        self.num_groups = num_groups
        self.dims = dims
        self.eps = eps
        # Initialize parameters to match PyTorch
        self.weight = mx.ones((dims,))
        self.bias = mx.zeros((dims,))
    
    def __call__(self, x):
        # Input: [B, C, L] (PyTorch format)
        # For GroupNorm with 1 group, we normalize across all channels and spatial dimensions
        B, C, L = x.shape
        
        # Reshape to combine all dimensions except batch
        x_reshaped = x.reshape(B, -1)  # [B, C*L]
        
        # Compute mean and variance across all features
        mean = mx.mean(x_reshaped, axis=1, keepdims=True)  # [B, 1]
        var = mx.var(x_reshaped, axis=1, keepdims=True)  # [B, 1]
        
        # Normalize
        x_norm = (x_reshaped - mean) * mx.rsqrt(var + self.eps)
        
        # Reshape back
        x_norm = x_norm.reshape(B, C, L)
        
        # Apply affine transformation with broadcasting
        # weight and bias are [C], need to expand for [B, C, L]
        weight = self.weight.reshape(1, C, 1)  # [1, C, 1]
        bias = self.bias.reshape(1, C, 1)      # [1, C, 1]
        
        x_out = x_norm * weight + bias
        
        return x_out

def select_norm(norm, dim, shape):
    """Just a wrapper to select the normalization type."""
    if norm == "gln":
        return GlobalLayerNorm(dim, shape, elementwise_affine=True)
    if norm == "cln":
        return CumulativeLayerNorm(dim, elementwise_affine=True)
    if norm == "ln":
        # PyTorch uses GroupNorm(1, dim) which is 1 group across all channels
        # Use our wrapper to handle the format difference
        return GroupNormWrapper(num_groups=1, dims=dim, eps=1e-8)
    else:
        return nn.BatchNorm(dim)

class DepthwiseConv1d(nn.Module):
    """Depthwise 1D convolution."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def __call__(self, inputs):
        return self.conv(inputs)

class ConvModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer.
    """
    
    def __init__(
        self,
        in_channels: int,
        kernel_size: int = 17,
        expansion_factor: int = 2,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.depthwise_conv = DepthwiseConv1d(
            in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2
        )

    def __call__(self, inputs):
        # inputs shape: (batch, time, dim)
        # For depthwise conv keep it in (batch, time, dim) format since MLX Conv1d expects NLC format

        x = self.depthwise_conv(inputs)
        
        output = inputs + x
        
        return output