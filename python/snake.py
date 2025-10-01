import mlx.core as mx
import mlx.nn as nn

def snake(x, alpha):
    """Snake activation function"""
    return x + mx.reciprocal(alpha + 1e-9) * mx.power(mx.sin(alpha * x), 2)

class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = mx.ones((1, channels, 1))
    
    def __call__(self, x):
        return snake(x, self.alpha)