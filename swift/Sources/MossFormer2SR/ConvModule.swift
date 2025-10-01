import MLX
import MLXNN

/// Depthwise 1D convolution
public class DepthwiseConv1d: Module {
    @ModuleInfo var conv: Conv1d
    
    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        bias: Bool = false
    ) {
        precondition(outChannels % inChannels == 0, "out_channels should be constant multiple of in_channels")
        
        self.conv = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            padding: padding,
            groups: inChannels,
            bias: bias
        )
        
        super.init()
    }
    
    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        return conv(inputs)
    }
}

/// Conformer convolution module
/// Starts with a pointwise convolution and a gated linear unit (GLU).
/// This is followed by a single 1-D depthwise convolution layer.
public class ConvModule: Module {
    @ModuleInfo var depthwise_conv: DepthwiseConv1d
    
    public init(
        inChannels: Int,
        kernelSize: Int = 17,
        expansionFactor: Int = 2,
        dropoutP: Float = 0.1
    ) {
        precondition((kernelSize - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding")
        precondition(expansionFactor == 2, "Currently, Only Supports expansion_factor 2")
        
        self.depthwise_conv = DepthwiseConv1d(
            inChannels: inChannels,
            outChannels: inChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: (kernelSize - 1) / 2
        )
        
        super.init()
    }
    
    public func callAsFunction(_ inputs: MLXArray) -> MLXArray {
        // inputs shape: (batch, time, dim)
        // MLX Conv1d expects NLC format, so no transpose needed
        let x = depthwise_conv(inputs)
        
        // Residual connection
        return inputs + x
    }
}