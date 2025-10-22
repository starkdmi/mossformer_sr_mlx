import MLX
import MLXNN

/// Drop-in replacement for MLXNN.Conv1d that reuses the optimized depthwise kernel.
public final class Conv1dFast: Module {
    public let inputChannels: Int
    public let outputChannels: Int
    public let kernelSize: Int
    public let stride: Int
    public let padding: Int
    public let groups: Int
    public let hasBias: Bool

    @ModuleInfo public var weight: MLXArray
    @ModuleInfo public var bias: MLXArray?

    public init(
        inputChannels: Int,
        outputChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        bias: Bool = true
    ) {
        self.inputChannels = inputChannels
        self.outputChannels = outputChannels
        self.kernelSize = kernelSize
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.hasBias = bias

        // Same layout MLX.Conv1d expects: (O, K, I_per_group)
        self.weight = MLXArray.zeros([outputChannels, kernelSize, inputChannels / groups])
        if bias {
            self.bias = MLXArray.zeros([outputChannels])
        }

        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = DepthwiseConv1dKernel.apply(
            x,
            weight: weight,
            stride: stride,
            padding: padding,
            groups: groups
        )
        if let bias {
            out = out + bias
        }
        return out
    }
}

/// Depthwise 1D convolution
public class DepthwiseConv1d: Module {
    @ModuleInfo var conv: Conv1dFast // Conv1d
    
    public init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        bias: Bool = false
    ) {
        precondition(outChannels % inChannels == 0, "out_channels should be constant multiple of in_channels")
        
        self.conv = Conv1dFast( // Conv1d
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
