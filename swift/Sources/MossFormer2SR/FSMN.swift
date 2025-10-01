import MLX
import MLXNN

/// UniDeepFsmn is a neural network module that implements a single-deep feedforward sequence memory network (FSMN).
public class UniDeepFsmn: Module {
    let inputDim: Int
    let outputDim: Int
    let lorder: Int?
    let hiddenSize: Int?
    
    @ModuleInfo var linear: Linear
    @ModuleInfo var project: Linear
    @ModuleInfo var conv1: Conv2d
    
    public init(inputDim: Int, outputDim: Int, lorder: Int? = nil, hiddenSize: Int? = nil) {
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.lorder = lorder
        self.hiddenSize = hiddenSize
        
        if let lorder = lorder, let hiddenSize = hiddenSize {
            // Initialize the layers
            self.linear = Linear(inputDim, hiddenSize)
            self.project = Linear(hiddenSize, outputDim, bias: false)
            // MLX Conv2d for grouped convolution
            self.conv1 = Conv2d(
                inputChannels: outputDim,
                outputChannels: outputDim,
                kernelSize: [lorder + lorder - 1, 1],
                stride: [1, 1],
                groups: outputDim,
                bias: false
            )
        } else {
            // Dummy initialization - should not be used
            self.linear = Linear(1, 1)
            self.project = Linear(1, 1)
            self.conv1 = Conv2d(inputChannels: 1, outputChannels: 1, kernelSize: [1, 1])
        }
        
        super.init()
    }
    
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        guard let lorder = lorder else {
            return input
        }
        
        // input shape: (batch, time, channels)
        let f1 = relu(linear(input))  // Apply linear layer followed by ReLU activation
        let p1 = project(f1)  // Project to output dimension
        
        // For Conv2d, we need shape (batch, height, width, channels) in MLX
        // Current: (batch, time, output_dim)
        // Target: (batch, time, 1, output_dim) for NHWC
        let x = p1.expandedDimensions(axis: 2)  // Add width dimension: (B, T, 1, C)
        
        // Pad for causal convolution - pad the time dimension
        let y = MLX.padded(
            x,
            widths: [IntOrPair(0), IntOrPair([lorder - 1, lorder - 1]), IntOrPair(0), IntOrPair(0)]
        )
        
        // Apply convolution
        let out = x + conv1(y)  // Add original input to convolution output
        
        // Remove the width dimension and return
        let squeezed = out.squeezed(axis: 2)  // Back to (B, T, C)
        return input + squeezed  // Return enhanced input
    }
}