import mlx.core as mx
import mlx.nn as nn

class CLayerNorm(nn.LayerNorm):
    """Channel-wise layer normalization."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, sample):
        """Forward function.

        Args:
            sample: [batch_size, channels, length]
        """
        if sample.ndim != 3:
            raise RuntimeError(f'{self.__class__.__name__} only accept 3-D tensor as input')
        # [N, C, T] -> [N, T, C]
        sample = mx.transpose(sample, (0, 2, 1))
        # LayerNorm
        sample = super().__call__(sample)
        # [N, T, C] -> [N, C, T]
        sample = mx.transpose(sample, (0, 2, 1))
        return sample