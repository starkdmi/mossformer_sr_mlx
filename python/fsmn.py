import mlx.core as mx
import mlx.nn as nn

class UniDeepFsmn(nn.Module):
    """
    UniDeepFsmn is a neural network module that implements a single-deep feedforward sequence memory network (FSMN).
    """

    def __init__(self, input_dim, output_dim, lorder=None, hidden_size=None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        if lorder is None:
            return
        self.lorder = lorder
        self.hidden_size = hidden_size
        
        # Initialize the layers
        self.linear = nn.Linear(input_dim, hidden_size)
        self.project = nn.Linear(hidden_size, output_dim, bias=False)
        # MLX Conv2d for grouped convolution
        self.conv1 = nn.Conv2d(
            output_dim, 
            output_dim, 
            kernel_size=(lorder + lorder - 1, 1), 
            stride=(1, 1), 
            groups=output_dim, 
            bias=False
        )

    def __call__(self, input):
        """
        Forward pass for the UniDeepFsmn model.
        """
        # input shape: (batch, time, channels)
        f1 = nn.relu(self.linear(input))  # Apply linear layer followed by ReLU activation
        p1 = self.project(f1)  # Project to output dimension
        
        # For Conv2d, we need shape (batch, height, width, channels) in MLX
        # Current: (batch, time, output_dim)
        # Target: (batch, time, 1, output_dim) for NHWC
        x = mx.expand_dims(p1, 2)  # Add width dimension: (B, T, 1, C)
        
        # Pad for causal convolution - pad the time dimension
        y = mx.pad(x, [(0, 0), (self.lorder - 1, self.lorder - 1), (0, 0), (0, 0)])
        
        # Apply convolution
        out = x + self.conv1(y)  # Add original input to convolution output
        
        # Remove the width dimension and return
        out = mx.squeeze(out, axis=2)  # Back to (B, T, C)
        return input + out  # Return enhanced input
