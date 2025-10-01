import MLX
import MLXNN

/// Global Layer Normalization
public class GlobalLayerNorm: Module {
    let dim: Int
    let eps: Float
    let elementwiseAffine: Bool
    let shape: Int
    
    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray
    
    public init(dim: Int, shape: Int, eps: Float = 1e-8, elementwiseAffine: Bool = true) {
        self.dim = dim
        self.eps = eps
        self.elementwiseAffine = elementwiseAffine
        self.shape = shape
        
        if elementwiseAffine {
            if shape == 3 {
                self.weight = MLXArray.ones([dim, 1])
                self.bias = MLXArray.zeros([dim, 1])
            } else if shape == 4 {
                self.weight = MLXArray.ones([dim, 1, 1])
                self.bias = MLXArray.zeros([dim, 1, 1])
            } else {
                self.weight = MLXArray.ones([dim])
                self.bias = MLXArray.zeros([dim])
            }
        } else {
            self.weight = MLXArray.ones([1])
            self.bias = MLXArray.zeros([1])
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        if x.ndim == 3 {
            let mean = MLX.mean(x, axes: [1, 2], keepDims: true)
            let variance = MLX.mean(MLX.pow(x - mean, 2), axes: [1, 2], keepDims: true)
            
            if elementwiseAffine {
                return weight * (x - mean) * MLX.rsqrt(variance + eps) + bias
            } else {
                return (x - mean) * MLX.rsqrt(variance + eps)
            }
        } else if x.ndim == 4 {
            let mean = MLX.mean(x, axes: [1, 2, 3], keepDims: true)
            let variance = MLX.mean(MLX.pow(x - mean, 2), axes: [1, 2, 3], keepDims: true)
            
            if elementwiseAffine {
                return weight * (x - mean) * MLX.rsqrt(variance + eps) + bias
            } else {
                return (x - mean) * MLX.rsqrt(variance + eps)
            }
        }
        return x
    }
}

/// Cumulative Layer Normalization
public class CumulativeLayerNorm: LayerNorm {
    public init(dim: Int, elementwiseAffine: Bool = true) {
        super.init(dimensions: dim, eps: 1e-8, affine: elementwiseAffine)
    }
    
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result = x
        
        if x.ndim == 4 {
            // N x C x K x S -> N x K x S x C
            result = result.transposed(0, 2, 3, 1)
            // Apply layer norm
            result = super.callAsFunction(result)
            // N x K x S x C -> N x C x K x S
            result = result.transposed(0, 3, 1, 2)
        } else if x.ndim == 3 {
            // N x C x L -> N x L x C
            result = result.transposed(0, 2, 1)
            // Apply layer norm
            result = super.callAsFunction(result)
            // N x L x C -> N x C x L
            result = result.transposed(0, 2, 1)
        }
        
        return result
    }
}

/// Channel-wise Layer Normalization
public class CLayerNorm: LayerNorm {
    public override func callAsFunction(_ sample: MLXArray) -> MLXArray {
        guard sample.ndim == 3 else {
            fatalError("\(type(of: self)) only accepts 3-D tensor as input")
        }
        
        // [N, C, T] -> [N, T, C]
        var result = sample.transposed(0, 2, 1)
        // Apply LayerNorm
        result = super.callAsFunction(result)
        // [N, T, C] -> [N, C, T]
        result = result.transposed(0, 2, 1)
        
        return result
    }
}

/// Group Norm Wrapper for PyTorch-style input
public class GroupNormWrapper: Module {
    let numGroups: Int
    let dims: Int
    let eps: Float
    
    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray
    
    public init(numGroups: Int, dims: Int, eps: Float = 1e-8) {
        self.numGroups = numGroups
        self.dims = dims
        self.eps = eps
        
        // Initialize parameters to match PyTorch
        self.weight = MLXArray.ones([dims])
        self.bias = MLXArray.zeros([dims])
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: [B, C, L] (PyTorch format)
        let B = x.shape[0]
        let C = x.shape[1]
        let L = x.shape[2]
        
        // Reshape to combine all dimensions except batch
        let xReshaped = x.reshaped([B, -1])  // [B, C*L]
        
        // Compute mean and variance across all features
        let mean = MLX.mean(xReshaped, axis: 1, keepDims: true)  // [B, 1]
        let variance = MLX.variance(xReshaped, axis: 1, keepDims: true)  // [B, 1]
        
        // Normalize
        let xNorm = (xReshaped - mean) * MLX.rsqrt(variance + eps)
        
        // Reshape back
        let xNormReshaped = xNorm.reshaped([B, C, L])
        
        // Apply affine transformation with broadcasting
        let weightExpanded = weight.reshaped([1, C, 1])  // [1, C, 1]
        let biasExpanded = bias.reshaped([1, C, 1])      // [1, C, 1]
        
        return xNormReshaped * weightExpanded + biasExpanded
    }
}

/// Select normalization type
public func selectNorm(norm: String, dim: Int, shape: Int) -> Module {
    switch norm {
    case "gln":
        return GlobalLayerNorm(dim: dim, shape: shape, elementwiseAffine: true)
    case "cln":
        return CumulativeLayerNorm(dim: dim, elementwiseAffine: true)
    case "ln":
        // PyTorch uses GroupNorm(1, dim) which is 1 group across all channels
        return GroupNormWrapper(numGroups: 1, dims: dim, eps: 1e-8)
    default:
        return BatchNorm(featureCount: dim)
    }
}