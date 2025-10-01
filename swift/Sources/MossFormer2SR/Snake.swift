import MLX
import MLXNN

/// Snake activation function
public func snake(_ x: MLXArray, alpha: MLXArray) -> MLXArray {
    return x + MLX.reciprocal(alpha + 1e-9) * MLX.pow(MLX.sin(alpha * x), 2)
}

/// Snake1d activation module
public class Snake1d: Module {
    @ModuleInfo var alpha: MLXArray
    
    public init(channels: Int) {
        self.alpha = MLXArray.ones([1, channels, 1])
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        return snake(x, alpha: alpha)
    }
}