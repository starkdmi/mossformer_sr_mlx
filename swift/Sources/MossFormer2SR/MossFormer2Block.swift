import MLX
import MLXNN
import Foundation

// MARK: - Helper Functions

func exists<T>(_ val: T?) -> Bool {
    return val != nil
}

func `default`<T>(_ val: T?, _ d: T) -> T {
    return val ?? d
}

func paddingToMultipleOf(_ n: Int, _ mult: Int) -> Int {
    let remainder = n % mult
    return remainder == 0 ? 0 : mult - remainder
}

// MARK: - Scale Normalization

/// ScaleNorm implements a scaled normalization technique for neural network layers.
public class ScaleNorm: Module {
    let scale: Float
    let eps: Float
    @ModuleInfo var g: MLXArray
    
    public init(dim: Int, eps: Float = 1e-5) {
        self.scale = Float(1.0 / sqrt(Double(dim)))
        self.eps = eps
        self.g = MLXArray.ones([1])
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let norm = MLX.norm(x, axis: -1, keepDims: true) * scale
        return x / MLX.maximum(norm, MLXArray(eps)) * g
    }
}

// MARK: - Scaled Sinusoidal Embedding

/// ScaledSinuEmbedding provides sinusoidal positional encodings for inputs.
public class ScaledSinuEmbedding: Module {
    @ModuleInfo var scale: MLXArray
    let invFreq: MLXArray
    
    public init(_ dim: Int) {
        self.scale = MLXArray.ones([1])
        self.invFreq = 1.0 / MLX.pow(10000, MLXArray(0..<dim).asType(.float32)[.stride(by: 2)] / Float(dim))
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = x.shape[1]
        let t = MLXArray(0..<n).asType(.float32)
        
        // Calculate sine and cosine embeddings
        let sinu = t.expandedDimensions(axis: 1) * invFreq.expandedDimensions(axis: 0)
        let emb = MLX.concatenated([MLX.sin(sinu), MLX.cos(sinu)], axis: -1)
        return emb * scale
    }
}

// MARK: - Offset Scale

/// OffsetScale applies learned offsets and scales to the input tensor.
public class OffsetScale: Module {
    let heads: Int
    let dim: Int
    @ModuleInfo var gamma: MLXArray
    @ModuleInfo var beta: MLXArray
    
    public init(dim: Int, heads: Int = 1) {
        self.heads = heads
        self.dim = dim
        // Initialize with same pattern as PyTorch
        self.gamma = MLXArray.ones([heads, dim]) * 0.02
        self.beta = MLXArray.zeros([heads, dim])
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        // Apply scaling and offsets
        // x shape: (..., dim)
        // gamma/beta shape: (heads, dim)
        
        // Expand x to include head dimension: (..., dim) -> (..., 1, dim)
        let xExpanded = x.expandedDimensions(axis: -2)
        
        // Broadcast multiplication with gamma: (..., 1, dim) * (heads, dim) -> (..., heads, dim)
        let scaled = xExpanded * gamma
        
        // Add beta offset: (..., heads, dim) + (heads, dim) -> (..., heads, dim)
        let out = scaled + beta
        
        // Split into list of tensors for each head
        var results: [MLXArray] = []
        for i in 0..<heads {
            // Use .ellipsis to preserve all dimensions except the head dimension
            let headOutput = out[.ellipsis, i, 0...]
            results.append(headOutput)
        }
        return results
    }
}

// MARK: - Feed-Forward Convolutional Module

/// FFConvM is a feed-forward convolutional module with normalization and dropout.
public class FFConvM: Module {
    @ModuleInfo var norm: Module
    @ModuleInfo var linear: Linear
    @ModuleInfo var conv: ConvModule
    let dropout: Float
    
    public init(
        dimIn: Int,
        dimOut: Int,
        normKlass: (Int) -> Module = { LayerNorm(dimensions: $0) },
        dropout: Float = 0.1
    ) {
        self.norm = normKlass(dimIn)
        self.linear = Linear(dimIn, dimOut)
        self.conv = ConvModule(inChannels: dimOut)
        self.dropout = dropout
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var result: MLXArray
        if let ln = norm as? LayerNorm {
            result = ln(x)
        } else if let sn = norm as? ScaleNorm {
            result = sn(x)
        } else {
            result = x
        }
        result = linear(result)
        result = silu(result)
        result = conv(result)
        // Note: Dropout is commented out in Python to avoid training mode issues
        return result
    }
}

// MARK: - FLASH Share A FFConvM

/// Fast Shared Dual Attention Mechanism with feed-forward convolutional blocks.
public class FLASH_ShareA_FFConvM: Module {
    let groupSize: Int
    let causal: Bool
    let shiftTokens: Bool
    let rotaryPosEmb: RotaryEmbedding?
    let dropout: Float
    
    @ModuleInfo var to_hidden: FFConvM
    @ModuleInfo var to_qk: FFConvM
    @ModuleInfo var qk_offset_scale: OffsetScale
    @ModuleInfo var to_out: FFConvM
    
    public init(
        dim: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 1.0,
        causal: Bool = false,
        dropout: Float = 0.1,
        rotaryPosEmb: RotaryEmbedding? = nil,
        normKlass: @escaping (Int) -> Module = { LayerNorm(dimensions: $0) },
        shiftTokens: Bool = true
    ) {
        let hiddenDim = Int(Float(dim) * expansionFactor)
        self.groupSize = groupSize
        self.causal = causal
        self.shiftTokens = shiftTokens
        self.rotaryPosEmb = rotaryPosEmb
        self.dropout = dropout
        
        // Feed-forward layers
        self.to_hidden = FFConvM(
            dimIn: dim,
            dimOut: hiddenDim,
            normKlass: normKlass,
            dropout: dropout
        )
        self.to_qk = FFConvM(
            dimIn: dim,
            dimOut: queryKeyDim,
            normKlass: normKlass,
            dropout: dropout
        )
        
        // Offset and scale for query and key
        self.qk_offset_scale = OffsetScale(dim: queryKeyDim, heads: 4)
        
        self.to_out = FFConvM(
            dimIn: dim * 2,
            dimOut: dim,
            normKlass: normKlass,
            dropout: dropout
        )
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let normedX = x
        
        // Token shifting if enabled
        var processedX = normedX
        if shiftTokens {
            let split = MLX.split(normedX, indices: [normedX.shape[2] / 2], axis: -1)
            var xShift = split[0]
            let xPass = split[1]
            
            // Pad and shift
            xShift = MLX.padded(xShift, widths: [IntOrPair(0), IntOrPair([1, 0]), IntOrPair(0)])
            xShift = xShift[0..., 0..<(xShift.shape[1] - 1), 0...]
            processedX = MLX.concatenated([xShift, xPass], axis: -1)
        }
        
        // Initial projections
        let hidden = to_hidden(processedX)
        let hiddenSplit = MLX.split(hidden, indices: [hidden.shape[2] / 2], axis: -1)
        let v = hiddenSplit[0]
        let u = hiddenSplit[1]
        
        let qk = to_qk(processedX)
        // Offset and scale
        let qkOffsetScaled = qk_offset_scale(qk)
        let quadQ = qkOffsetScaled[0]
        let linQ = qkOffsetScaled[1]
        let quadK = qkOffsetScaled[2]
        let linK = qkOffsetScaled[3]
        
        let (attV, attU) = calAttention(
            x: x,
            quadQ: quadQ,
            linQ: linQ,
            quadK: quadK,
            linK: linK,
            v: v,
            u: u,
            mask: mask
        )
        
        // Output calculation with gating
        let out = (attU * v) * MLX.sigmoid(attV * u)
        
        // Residual connection
        return x + to_out(out)
    }
    
    private func calAttention(
        x: MLXArray,
        quadQ: MLXArray,
        linQ: MLXArray,
        quadK: MLXArray,
        linK: MLXArray,
        v: MLXArray,
        u: MLXArray,
        mask: MLXArray?
    ) -> (MLXArray, MLXArray) {
        let b = x.shape[0]
        let n = x.shape[1]
        let g = groupSize
        
        // Debug prints removed for cleaner output
        
        var quadQProc = quadQ
        var linQProc = linQ
        var quadKProc = quadK
        var linKProc = linK
        var vProc = v
        var uProc = u
        var maskProc = mask
        
        // Apply mask to linear keys if provided
        if let mask = mask {
            let linMask = mask.expandedDimensions(axis: -1)
            linKProc = MLX.where(linMask, linKProc, MLXArray(0))
        }
        
        // Rotate queries and keys with rotary positional embeddings
        if let rotaryPosEmb = rotaryPosEmb {
            quadQProc = rotaryPosEmb.rotateQueriesOrKeys(quadQProc)
            linQProc = rotaryPosEmb.rotateQueriesOrKeys(linQProc)
            quadKProc = rotaryPosEmb.rotateQueriesOrKeys(quadKProc)
            linKProc = rotaryPosEmb.rotateQueriesOrKeys(linKProc)
        }
        
        // Padding for group processing
        // Use the sequence length from v/u tensors which have the full sequence dimension
        let actualSeqLen = vProc.shape[1]
        let padding = paddingToMultipleOf(actualSeqLen, g)
        if padding > 0 {
            let padWidth = [IntOrPair(0), IntOrPair([0, padding]), IntOrPair(0)]
            quadQProc = MLX.padded(quadQProc, widths: padWidth)
            quadKProc = MLX.padded(quadKProc, widths: padWidth)
            linQProc = MLX.padded(linQProc, widths: padWidth)
            linKProc = MLX.padded(linKProc, widths: padWidth)
            vProc = MLX.padded(vProc, widths: padWidth)
            uProc = MLX.padded(uProc, widths: padWidth)
            
            if maskProc == nil {
                maskProc = MLXArray.ones([b, actualSeqLen]).asType(.bool)
            }
            maskProc = MLX.padded(maskProc!, widths: [IntOrPair(0), IntOrPair([0, padding])], value: MLXArray(false))
        }
        
        
        // Reshape for groups
        func reshapeForGroups(_ t: MLXArray) -> MLXArray {
            let bSize = t.shape[0]
            let seqLen = t.shape[1]
            let d = t.shape[2]
            
            
            return t.reshaped([bSize, seqLen / groupSize, groupSize, d])
        }
        
        quadQProc = reshapeForGroups(quadQProc)
        quadKProc = reshapeForGroups(quadKProc)
        linQProc = reshapeForGroups(linQProc)
        linKProc = reshapeForGroups(linKProc)
        vProc = reshapeForGroups(vProc)
        uProc = reshapeForGroups(uProc)
        
        if let mask = maskProc {
            maskProc = mask.reshaped([b, -1, groupSize]).expandedDimensions(axis: 2)
        }
        
        // Calculate quadratic attention
        let sim = MLX.matmul(quadQProc, quadKProc.swappedAxes(-2, -1)) / Float(g)
        var attn = MLX.square(relu(sim))
        
        // Apply mask to attention if provided
        if let mask = maskProc {
            attn = MLX.where(mask, attn, MLXArray(0))
        }
        
        // Apply causal mask if needed
        if causal {
            let causalMask = MLX.triu(MLXArray.ones([g, g]).asType(.bool), k: 1)
                .expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            attn = MLX.where(causalMask, MLXArray(0), attn)
        }
        
        // Calculate output from attention
        let quadOutV = MLX.matmul(attn, vProc)
        let quadOutU = MLX.matmul(attn, uProc)
        
        // Calculate linear attention output
        let linOutV: MLXArray
        let linOutU: MLXArray
        
        if causal {
            // Causal linear attention
            var linKV = MLX.matmul(linKProc.swappedAxes(-2, -1), vProc) / Float(g)
            linKV = MLX.cumsum(linKV, axis: 1)
            linKV = MLX.padded(linKV, widths: [IntOrPair(0), IntOrPair([1, 0]), IntOrPair(0), IntOrPair(0)])[0..., 0..<(linKV.shape[1] - 1), 0..., 0...]
            linOutV = MLX.matmul(linQProc, linKV)
            
            var linKU = MLX.matmul(linKProc.swappedAxes(-2, -1), uProc) / Float(g)
            linKU = MLX.cumsum(linKU, axis: 1)
            linKU = MLX.padded(linKU, widths: [IntOrPair(0), IntOrPair([1, 0]), IntOrPair(0), IntOrPair(0)])[0..., 0..<(linKU.shape[1] - 1), 0..., 0...]
            linOutU = MLX.matmul(linQProc, linKU)
        } else {
            // Non-causal linear attention
            let linKReshaped = linKProc.reshaped([b, -1, linKProc.shape[3]])
            let vReshaped = vProc.reshaped([b, -1, vProc.shape[3]])
            let uReshaped = uProc.reshaped([b, -1, uProc.shape[3]])
            
            // IMPORTANT: n is the ORIGINAL sequence length before padding
            let linKV = MLX.matmul(linKReshaped.swappedAxes(-2, -1), vReshaped) / Float(n)
            linOutV = MLX.matmul(linQProc, linKV)
            
            let linKU = MLX.matmul(linKReshaped.swappedAxes(-2, -1), uReshaped) / Float(n)
            linOutU = MLX.matmul(linQProc, linKU)
        }
        
        // Reshape and remove padding from outputs
        func reshapeFromGroups(_ t: MLXArray) -> MLXArray {
            let bSize = t.shape[0]
            let nGroups = t.shape[1]
            let groupSize = t.shape[2]
            let d = t.shape[3]
            return t.reshaped([bSize, nGroups * groupSize, d])[0..., 0..<n, 0...]
        }
        
        let finalV = reshapeFromGroups(quadOutV + linOutV)
        let finalU = reshapeFromGroups(quadOutU + linOutU)
        
        return (finalV, finalU)
    }
}

// MARK: - Gated FSMN

/// Gated Frequency Selective Memory Network (FSMN) class.
public class Gated_FSMN: Module {
    @ModuleInfo var to_u: FFConvM
    @ModuleInfo var to_v: FFConvM
    @ModuleInfo var fsmn: UniDeepFsmn
    
    public init(inChannels: Int, outChannels: Int, lorder: Int, hiddenSize: Int) {
        // Feedforward networks
        self.to_u = FFConvM(
            dimIn: inChannels,
            dimOut: hiddenSize,
            normKlass: { LayerNorm(dimensions: $0) },
            dropout: 0.1
        )
        self.to_v = FFConvM(
            dimIn: inChannels,
            dimOut: hiddenSize,
            normKlass: { LayerNorm(dimensions: $0) },
            dropout: 0.1
        )
        // Frequency selective memory network
        self.fsmn = UniDeepFsmn(inputDim: inChannels, outputDim: outChannels, lorder: lorder, hiddenSize: hiddenSize)
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input = x
        let xU = to_u(x)
        let xV = to_v(x)
        let xUProc = fsmn(xU)
        return xV * xUProc + input
    }
}

// MARK: - Gated FSMN Block

/// A 1-D convolutional block that incorporates a gated FSMN.
public class Gated_FSMN_Block: Module {
    let groupSize: Int
    
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var prelu1: PReLU
    @ModuleInfo var norm1: CLayerNorm
    @ModuleInfo var gated_fsmn: Gated_FSMN
    @ModuleInfo var norm2: CLayerNorm
    @ModuleInfo var conv2: Conv1d
    
    public init(dim: Int, innerChannels: Int = 256, groupSize: Int = 256, normType: String = "scalenorm") {
        self.groupSize = groupSize
        
        // First convolutional layer with PReLU activation
        self.conv1 = Conv1d(inputChannels: dim, outputChannels: innerChannels, kernelSize: 1)
        self.prelu1 = PReLU(count: 1)
        self.norm1 = CLayerNorm(dimensions: innerChannels)
        self.gated_fsmn = Gated_FSMN(inChannels: innerChannels, outChannels: innerChannels, lorder: 20, hiddenSize: innerChannels)
        self.norm2 = CLayerNorm(dimensions: innerChannels)
        self.conv2 = Conv1d(inputChannels: innerChannels, outputChannels: dim, kernelSize: 1)
        
        super.init()
    }
    
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        // MLX Conv1d expects (batch, time, channels)
        var x = conv1(input)
        x = prelu1(x)
        
        // CLayerNorm expects (batch, channels, time) format
        x = x.transposed(0, 2, 1)
        let conv1Out = norm1(x)
        
        // Back to (batch, time, channels) for FSMN
        x = conv1Out.transposed(0, 2, 1)
        let seqOut = gated_fsmn(x)
        
        // Transpose for norm2
        x = seqOut.transposed(0, 2, 1)
        let norm2Out = norm2(x)
        
        // Back to (batch, time, channels) for conv2
        x = norm2Out.transposed(0, 2, 1)
        let conv2Out = conv2(x)
        
        // Add residual
        return conv2Out + input
    }
}

// MARK: - Rotary Embedding

/// Rotary Embedding implementation for MLX
public class RotaryEmbedding: Module {
    let rope: RoPE
    let dim: Int
    
    public init(dim: Int, base: Int = 10000) {
        self.rope = RoPE(dimensions: dim, traditional: true, base: Float(base))
        self.dim = dim
        super.init()
    }
    
    public func rotateQueriesOrKeys(_ x: MLXArray) -> MLXArray {
        return rope(x)
    }
}

// MARK: - MossformerBlock with Gated FSMN

/// Mossformer Block with Gated FSMN.
public class MossformerBlock_GFSMN: Module {
    let groupSize: Int
    let fsmn: [Gated_FSMN_Block]
    let layers: [FLASH_ShareA_FFConvM]
    
    public init(
        dim: Int,
        depth: Int,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        causal: Bool = false,
        attnDropout: Float = 0.1,
        normType: String = "scalenorm",
        shiftTokens: Bool = true
    ) {
        precondition(normType == "scalenorm" || normType == "layernorm", "norm_type must be one of scalenorm or layernorm")
        
        let normKlass: (Int) -> Module = { dim in
            if normType == "scalenorm" {
                return ScaleNorm(dim: dim)
            } else {
                return LayerNorm(dimensions: dim)
            }
        }
        
        self.groupSize = groupSize
        
        // Rotary positional embedding for attention
        let rotaryPosEmb = RotaryEmbedding(dim: min(32, queryKeyDim))
        
        // Create a list of Gated FSMN blocks
        self.fsmn = (0..<depth).map { _ in
            Gated_FSMN_Block(dim: dim)
        }
        
        // Create a list of attention layers using FLASH_ShareA_FFConvM
        self.layers = (0..<depth).map { _ in
            FLASH_ShareA_FFConvM(
                dim: dim,
                groupSize: groupSize,
                queryKeyDim: queryKeyDim,
                expansionFactor: expansionFactor,
                causal: causal,
                dropout: attnDropout,
                rotaryPosEmb: rotaryPosEmb,
                normKlass: normKlass,
                shiftTokens: shiftTokens
            )
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var output = x
        
        for (idx, flash) in layers.enumerated() {
            output = flash(output, mask: mask)
            output = fsmn[idx](output)
        }
        
        return output
    }
}
