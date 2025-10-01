import MLX
import MLXNN

// MARK: - Helper Function

func getPadding(kernelSize: Int, dilation: Int = 1) -> Int {
    return (kernelSize * dilation - dilation) / 2
}

// MARK: - ResBlock1

public class ResBlock1: Module {
    let h: AttrDict
    let convs1: [Conv1d]
    let convs1_activates: [Snake1d]
    let convs2: [Conv1d]
    let convs2_activates: [Snake1d]
    
    public init(h: AttrDict, channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3, 5]) {
        self.h = h
        
        self.convs1 = [
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                   stride: 1, padding: getPadding(kernelSize: kernelSize, dilation: dilation[0]), dilation: dilation[0], bias: true),
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                   stride: 1, padding: getPadding(kernelSize: kernelSize, dilation: dilation[1]), dilation: dilation[1], bias: true),
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                   stride: 1, padding: getPadding(kernelSize: kernelSize, dilation: dilation[2]), dilation: dilation[2], bias: true)
        ]
        
        self.convs1_activates = [
            Snake1d(channels: channels),
            Snake1d(channels: channels),
            Snake1d(channels: channels)
        ]
        
        self.convs2 = [
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                   stride: 1, padding: getPadding(kernelSize: kernelSize, dilation: 1), dilation: 1, bias: true),
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                   stride: 1, padding: getPadding(kernelSize: kernelSize, dilation: 1), dilation: 1, bias: true),
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                   stride: 1, padding: getPadding(kernelSize: kernelSize, dilation: 1), dilation: 1, bias: true)
        ]
        
        self.convs2_activates = [
            Snake1d(channels: channels),
            Snake1d(channels: channels),
            Snake1d(channels: channels)
        ]
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        
        // Input x is B x N x L
        for i in 0..<convs1.count {
            var xt = convs1_activates[i](result)
            // Convert to B x L x N for Conv1d
            xt = xt.transposed(0, 2, 1)
            xt = convs1[i](xt)
            // Back to B x N x L
            xt = xt.transposed(0, 2, 1)
            xt = convs2_activates[i](xt)
            // Convert to B x L x N for Conv1d
            xt = xt.transposed(0, 2, 1)
            xt = convs2[i](xt)
            // Back to B x N x L
            xt = xt.transposed(0, 2, 1)
            result = xt + result
        }
        
        return result
    }
}

// MARK: - ResBlock2

public class ResBlock2: Module {
    let h: AttrDict
    let convs: [Conv1d]
    let convs_activates: [Snake1d]
    
    public init(h: AttrDict, channels: Int, kernelSize: Int = 3, dilation: [Int] = [1, 3]) {
        self.h = h
        
        self.convs = [
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                   stride: 1, padding: getPadding(kernelSize: kernelSize, dilation: dilation[0]), dilation: dilation[0], bias: true),
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
                   stride: 1, padding: getPadding(kernelSize: kernelSize, dilation: dilation[1]), dilation: dilation[1], bias: true)
        ]
        
        self.convs_activates = [
            Snake1d(channels: channels),
            Snake1d(channels: channels)
        ]
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        
        // Input x is B x N x L
        for i in 0..<convs.count {
            var xt = convs_activates[i](result)
            // Convert to B x L x N for Conv1d
            xt = xt.transposed(0, 2, 1)
            xt = convs[i](xt)
            // Back to B x N x L
            xt = xt.transposed(0, 2, 1)
            result = xt + result
        }
        
        return result
    }
}

// MARK: - AttrDict

/// A dictionary that allows attribute-style access.
public class AttrDict {
    private var dict: [String: Any]
    
    public init(_ dict: [String: Any] = [:]) {
        self.dict = dict
    }
    
    public subscript(key: String) -> Any? {
        get { return dict[key] }
        set { dict[key] = newValue }
    }
    
    // Computed properties for common HiFi-GAN parameters
    public var resblock: String {
        return dict["resblock"] as? String ?? "1"
    }
    
    public var upsample_rates: [Int] {
        return dict["upsample_rates"] as? [Int] ?? []
    }
    
    public var upsample_kernel_sizes: [Int] {
        return dict["upsample_kernel_sizes"] as? [Int] ?? []
    }
    
    public var upsample_initial_channel: Int {
        return dict["upsample_initial_channel"] as? Int ?? 512
    }
    
    public var resblock_kernel_sizes: [Int] {
        return dict["resblock_kernel_sizes"] as? [Int] ?? []
    }
    
    public var resblock_dilation_sizes: [[Int]] {
        return dict["resblock_dilation_sizes"] as? [[Int]] ?? []
    }
}

// MARK: - Generator

public class Generator: Module {
    let h: AttrDict
    let numKernels: Int
    let numUpsamples: Int
    
    @ModuleInfo var conv_pre: Conv1d
    var ups: [ConvTransposed1d] = []
    var snakes: [Snake1d] = []
    var resblocks: [Module] = []
    @ModuleInfo var snake_post: Snake1d
    @ModuleInfo var conv_post: Conv1d
    
    public init(h: AttrDict) {
        self.h = h
        self.numKernels = h.resblock_kernel_sizes.count
        self.numUpsamples = h.upsample_rates.count
        
        self.conv_pre = Conv1d(inputChannels: 80, outputChannels: h.upsample_initial_channel, kernelSize: 7, stride: 1, padding: 3, bias: true)
        
        let resblockClass: (AttrDict, Int, Int, [Int]) -> Module = h.resblock == "1" ? ResBlock1.init : ResBlock2.init
        
        // Create upsampling layers and snake activations
        for i in 0..<numUpsamples {
            let u = h.upsample_rates[i]
            let k = h.upsample_kernel_sizes[i]
            
            self.snakes.append(Snake1d(channels: h.upsample_initial_channel / (1 << i)))
            self.ups.append(
                ConvTransposed1d(
                    inputChannels: h.upsample_initial_channel / (1 << i),
                    outputChannels: h.upsample_initial_channel / (1 << (i + 1)),
                    kernelSize: k,
                    stride: u,
                    padding: (k - u) / 2,
                    bias: true
                )
            )
        }
        
        // Create ResBlocks
        for i in 0..<numUpsamples {
            let ch = h.upsample_initial_channel / (1 << (i + 1))
            for j in 0..<numKernels {
                let k = h.resblock_kernel_sizes[j]
                let d = h.resblock_dilation_sizes[j]
                self.resblocks.append(resblockClass(h, ch, k, d))
            }
        }
        
        let finalChannels = h.upsample_initial_channel / (1 << numUpsamples)
        self.snake_post = Snake1d(channels: finalChannels)
        self.conv_post = Conv1d(inputChannels: finalChannels, outputChannels: 1, kernelSize: 7, stride: 1, padding: 3, bias: true)
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: B x N x L -> B x L x N for MLX Conv1d
        var result = x.transposed(0, 2, 1)
        
        result = conv_pre(result)
        
        for i in 0..<numUpsamples {
            // Convert back to B x N x L for snake activation
            result = result.transposed(0, 2, 1)
            result = snakes[i](result)
            // Back to B x L x N for ConvTranspose
            result = result.transposed(0, 2, 1)
            
            result = ups[i](result)
            
            // Convert to B x N x L for ResBlocks
            result = result.transposed(0, 2, 1)
            
            // Collect ResBlock outputs
            var resblockOutputs: [MLXArray] = []
            for j in 0..<numKernels {
                let resblockIdx = i * numKernels + j
                let resblock = resblocks[resblockIdx]
                if let resblock1 = resblock as? ResBlock1 {
                    resblockOutputs.append(resblock1(result))
                } else if let resblock2 = resblock as? ResBlock2 {
                    resblockOutputs.append(resblock2(result))
                } else {
                    fatalError("Unknown resblock type")
                }
            }
            
            // Stack and average
            let xs = MLX.stacked(resblockOutputs, axis: 0)  // Shape: (num_kernels, B, N, L)
            result = MLX.mean(xs, axis: 0)  // Average across kernels
            
            // Back to B x L x N
            result = result.transposed(0, 2, 1)
        }
        
        // Final layers
        result = result.transposed(0, 2, 1)  // B x N x L for snake
        result = snake_post(result)
        result = result.transposed(0, 2, 1)  // B x L x N for conv
        result = conv_post(result)
        result = MLX.tanh(result)
        
        // Output should be B x L (single channel)
        result = result.squeezed(axis: -1)
        return result
    }
}