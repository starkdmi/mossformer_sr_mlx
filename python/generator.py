import mlx.core as mx
import mlx.nn as nn

from mossformer2 import MossFormer_MaskNet
from snake import Snake1d

def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

class ResBlock1(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = [
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                      padding=get_padding(kernel_size, dilation[0]), bias=True),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                      padding=get_padding(kernel_size, dilation[1]), bias=True),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                      padding=get_padding(kernel_size, dilation[2]), bias=True)
        ]
        self.convs1_activates = [
            Snake1d(channels),
            Snake1d(channels),
            Snake1d(channels)
        ]
        self.convs2 = [
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                      padding=get_padding(kernel_size, 1), bias=True),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                      padding=get_padding(kernel_size, 1), bias=True),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                      padding=get_padding(kernel_size, 1), bias=True)
        ]
        self.convs2_activates = [
            Snake1d(channels),
            Snake1d(channels),
            Snake1d(channels)
        ]

    def __call__(self, x):
        # Input x is B x N x L
        for c1, c2, act1, act2 in zip(self.convs1, self.convs2, 
                                       self.convs1_activates, self.convs2_activates):
            xt = act1(x)
            # Convert to B x L x N for Conv1d
            xt = mx.transpose(xt, (0, 2, 1))
            xt = c1(xt)
            # Back to B x N x L
            xt = mx.transpose(xt, (0, 2, 1))
            xt = act2(xt)
            # Convert to B x L x N for Conv1d
            xt = mx.transpose(xt, (0, 2, 1))
            xt = c2(xt)
            # Back to B x N x L
            xt = mx.transpose(xt, (0, 2, 1))
            x = xt + x
        return x

class ResBlock2(nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = [
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                      padding=get_padding(kernel_size, dilation[0]), bias=True),
            nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                      padding=get_padding(kernel_size, dilation[1]), bias=True)
        ]
        self.convs_activates = [
            Snake1d(channels),
            Snake1d(channels)
        ]

    def __call__(self, x):
        # Input x is B x N x L
        for c, act in zip(self.convs, self.convs_activates):
            xt = act(x)
            # Convert to B x L x N for Conv1d
            xt = mx.transpose(xt, (0, 2, 1))
            xt = c(xt)
            # Back to B x N x L
            xt = mx.transpose(xt, (0, 2, 1))
            x = xt + x
        return x

class Generator(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = nn.Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3, bias=True)
        resblock = ResBlock1 if h.resblock == '1' else ResBlock2

        self.ups = []
        self.snakes = []
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.snakes.append(Snake1d(h.upsample_initial_channel // (2**i)))
            self.ups.append(
                nn.ConvTranspose1d(h.upsample_initial_channel // (2**i), 
                                   h.upsample_initial_channel // (2**(i+1)),
                                   k, u, padding=(k-u)//2, bias=True))

        self.resblocks = []
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.snake_post = Snake1d(ch)
        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=True)
    
    def __call__(self, x):
        # Input: B x N x L -> B x L x N for MLX Conv1d
        
        x = mx.transpose(x, (0, 2, 1))
        
        x = self.conv_pre(x)
        
        
        for i in range(self.num_upsamples):
            # Convert back to B x N x L for snake activation
            x = mx.transpose(x, (0, 2, 1))
            x = self.snakes[i](x)
            # Back to B x L x N for ConvTranspose
            x = mx.transpose(x, (0, 2, 1))
            
            x = self.ups[i](x)
            
            
            # Convert to B x N x L for ResBlocks
            x = mx.transpose(x, (0, 2, 1))
            
            # Collect ResBlock outputs
            resblock_outputs = []
            for j in range(self.num_kernels):
                resblock_outputs.append(self.resblocks[i * self.num_kernels + j](x))
            
            # Stack and average
            xs = mx.stack(resblock_outputs, axis=0)  # Shape: (num_kernels, B, N, L)
            x = mx.mean(xs, axis=0)  # Average across kernels
            
            # Back to B x L x N
            x = mx.transpose(x, (0, 2, 1))
            
        # Final layers
        x = mx.transpose(x, (0, 2, 1))  # B x N x L for snake
        x = self.snake_post(x)
        x = mx.transpose(x, (0, 2, 1))  # B x L x N for conv
        x = self.conv_post(x)
        x = mx.tanh(x)
        
        
        # Output should be B x L (single channel)
        x = mx.squeeze(x, axis=-1)
        return x

class Mossformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mossformer = MossFormer_MaskNet(in_channels=80, out_channels=512, out_channels_final=80)

    def __call__(self, input):
        out = self.mossformer(input)
        return out
