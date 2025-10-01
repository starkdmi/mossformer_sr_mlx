import MLX
import MLXNN

// MARK: - Encoder

/// Convolutional Encoder Layer.
public class Encoder: Module {
    @ModuleInfo var conv1d: Conv1d
    let inChannels: Int
    
    public init(kernelSize: Int = 2, outChannels: Int = 64, inChannels: Int = 1) {
        self.inChannels = inChannels
        self.conv1d = Conv1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: kernelSize / 2,
            groups: 1,
            bias: false
        )
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        
        // Input: B x L (if 2D) or B x C x L (if 3D)
        if x.ndim == 2 {
            // B x L -> B x L x 1
            input = input.expandedDimensions(axis: -1)
        } else {
            // B x C x L -> B x L x C
            input = input.transposed(0, 2, 1)
        }
        
        // B x L x C -> B x L x N
        var result = conv1d(input)
        result = relu(result)
        
        // Convert back to B x N x L for compatibility
        result = result.transposed(0, 2, 1)
        return result
    }
}

// MARK: - Decoder

/// A decoder layer that consists of ConvTransposed1d.
public class Decoder: Module {
    @ModuleInfo var conv_transpose: ConvTransposed1d
    
    public init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, bias: Bool = false) {
        self.conv_transpose = ConvTransposed1d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: kernelSize,
            stride: stride,
            bias: bias
        )
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard x.ndim == 2 || x.ndim == 3 else {
            fatalError("\(type(of: self)) accepts 2/3D tensor as input")
        }
        
        var input = x
        
        // Input: B x N x L -> B x L x N for MLX
        if x.ndim == 3 {
            input = input.transposed(0, 2, 1)
        } else if x.ndim == 2 {
            input = input.expandedDimensions(axis: -1)
        }
        
        // B x L x N -> B x L_out x C_out
        let result = conv_transpose(input)
        
        // If output channels is 1, squeeze and return B x L_out
        if result.shape[2] == 1 {
            return result.squeezed(axis: -1)
        } else {
            // Otherwise convert back to B x C x L format
            return result.transposed(0, 2, 1)
        }
    }
}

// MARK: - MossFormerM

/// This class implements the transformer encoder.
public class MossFormerM: Module {
    @ModuleInfo var mossformerM: MossformerBlock_GFSMN
    @ModuleInfo var norm: LayerNorm
    
    public init(
        numBlocks: Int,
        dModel: Int,
        causal: Bool = false,
        groupSize: Int = 256,
        queryKeyDim: Int = 128,
        expansionFactor: Float = 4.0,
        attnDropout: Float = 0.1
    ) {
        self.mossformerM = MossformerBlock_GFSMN(
            dim: dModel,
            depth: numBlocks,
            groupSize: groupSize,
            queryKeyDim: queryKeyDim,
            expansionFactor: expansionFactor,
            causal: causal,
            attnDropout: attnDropout
        )
        self.norm = LayerNorm(dimensions: dModel, eps: 1e-6)
        
        super.init()
    }
    
    public func callAsFunction(_ src: MLXArray) -> MLXArray {
        // src shape: [B, L, N]
        var output = mossformerM(src)
        output = norm(output)
        return output
    }
}

// MARK: - Computation Block

/// Computation block for dual-path processing.
public class Computation_Block: Module {
    @ModuleInfo var intra_mdl: MossFormerM
    let skipAroundIntra: Bool
    let norm: String?
    @ModuleInfo var intra_norm: Module?
    
    public init(
        numBlocks: Int,
        outChannels: Int,
        norm: String? = "ln",
        skipAroundIntra: Bool = true
    ) {
        self.skipAroundIntra = skipAroundIntra
        self.norm = norm
        
        // MossFormer+: MossFormer with recurrence
        self.intra_mdl = MossFormerM(numBlocks: numBlocks, dModel: outChannels)
        
        // Norm
        if let norm = norm {
            self.intra_norm = selectNorm(norm: norm, dim: outChannels, shape: 3)
        }
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // let B = x.shape[0]
        // let N = x.shape[1]
        // let S = x.shape[2]
        
        // intra RNN
        // [B, S, N]
        var intra = x.transposed(0, 2, 1)
        
        intra = intra_mdl(intra)
        
        // [B, N, S]
        intra = intra.transposed(0, 2, 1)
        
        if let intraNorm = intra_norm {
            if let gln = intraNorm as? GlobalLayerNorm {
                intra = gln(intra)
            } else if let cln = intraNorm as? CumulativeLayerNorm {
                intra = cln(intra)
            } else if let grp = intraNorm as? GroupNormWrapper {
                intra = grp(intra)
            } else if let ln = intraNorm as? LayerNorm {
                intra = ln(intra)
            } else {
                fatalError("Unknown norm type")
            }
        }
        
        // [B, N, S]
        if skipAroundIntra {
            intra = intra + x
        }
        
        return intra
    }
}

// MARK: - MossFormer MaskNet

/// The dual path model which is the basis for dualpathrnn, sepformer, dptnet.
public class MossFormer_MaskNet: Module {
    let numSpks: Int
    let numBlocks: Int
    let useGlobalPosEnc: Bool
    
    @ModuleInfo var norm: Module
    @ModuleInfo var conv1d_encoder: Conv1d
    @ModuleInfo var pos_enc: ScaledSinuEmbedding?
    @ModuleInfo var mdl: Computation_Block
    @ModuleInfo var conv1d_out: Conv1d
    @ModuleInfo var conv1_decoder: Conv1d
    @ModuleInfo var prelu: PReLU
    @ModuleInfo var output_conv: Conv1d
    @ModuleInfo var output_gate_conv: Conv1d
    
    public init(
        inChannels: Int,
        outChannels: Int,
        outChannelsFinal: Int,
        numBlocks: Int = 24,
        norm: String = "ln",
        numSpks: Int = 1,
        skipAroundIntra: Bool = true,
        useGlobalPosEnc: Bool = true,
        maxLength: Int = 20000
    ) {
        self.numSpks = numSpks
        self.numBlocks = numBlocks
        self.useGlobalPosEnc = useGlobalPosEnc
        
        self.norm = selectNorm(norm: norm, dim: inChannels, shape: 3)
        self.conv1d_encoder = Conv1d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
        
        if useGlobalPosEnc {
            self.pos_enc = ScaledSinuEmbedding(outChannels)
        }
        
        self.mdl = Computation_Block(
            numBlocks: numBlocks,
            outChannels: outChannels,
            norm: norm,
            skipAroundIntra: skipAroundIntra
        )
        
        self.conv1d_out = Conv1d(
            inputChannels: outChannels,
            outputChannels: outChannels * numSpks,
            kernelSize: 1
        )
        self.conv1_decoder = Conv1d(inputChannels: outChannels, outputChannels: outChannelsFinal, kernelSize: 1, bias: false)
        self.prelu = PReLU(count: 1)
        
        // gated output layer
        self.output_conv = Conv1d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 1)
        self.output_gate_conv = Conv1d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 1)
        
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Input: [B, N, L] (PyTorch format: batch, channels, length)
        var result: MLXArray
        if let gln = norm as? GlobalLayerNorm {
            result = gln(x)
        } else if let cln = norm as? CumulativeLayerNorm {
            result = cln(x)
        } else if let grp = norm as? GroupNormWrapper {
            result = grp(x)
        } else if let ln = norm as? LayerNorm {
            result = ln(x)
        } else {
            result = x
        }
        
        // Now transpose for Conv1d which expects [B, L, N]
        result = result.transposed(0, 2, 1)  // [B, L, N]
        result = conv1d_encoder(result)  // Output: [B, L, out_channels]
        
        if let posEnc = pos_enc {
            // x is [B, L, N]
            let emb = posEnc(result)  // [B, L, N]
            result = result + emb
        }
        
        // We need to go back to [B, N, S] format for the mdl processing
        result = result.transposed(0, 2, 1)  // [B, N, S]
        
        // [B, N, S]
        result = mdl(result)
        
        result = prelu(result)
        
        // Convert back to [B, S, N] for Conv1d
        result = result.transposed(0, 2, 1)  // [B, S, N]
        
        // [B, S, N*spks]
        result = conv1d_out(result)
        
        let B = result.shape[0]
        let S = result.shape[1]
        
        // [B*spks, S, N]
        result = result.reshaped([B * numSpks, S, -1])
        
        // [B*spks, S, N]
        let outputVal = MLX.tanh(output_conv(result))
        let gateVal = MLX.sigmoid(output_gate_conv(result))
        
        result = outputVal * gateVal
        
        // [B*spks, S, N]
        result = conv1_decoder(result)
        
        // [B, spks, S, N]
        let N = result.shape[2]
        result = result.reshaped([B, numSpks, S, N])
        result = relu(result)
        
        // Convert back to [B, spks, N, S] then [spks, B, N, S]
        result = result.transposed(0, 1, 3, 2)  // [B, spks, N, S]
        result = result.transposed(1, 0, 2, 3)  // [spks, B, N, S]
        
        return result[0]
    }
}

// MARK: - MossFormer Main Class

/// Main MossFormer model
public class MossFormer: Module {
    let numSpks: Int
    @ModuleInfo var enc: Encoder
    @ModuleInfo var mask_net: MossFormer_MaskNet
    @ModuleInfo var dec: Decoder
    
    public init(
        inChannels: Int = 512,
        outChannels: Int = 512,
        numBlocks: Int = 24,
        kernelSize: Int = 16,
        norm: String = "ln",
        numSpks: Int = 2,
        skipAroundIntra: Bool = true,
        useGlobalPosEnc: Bool = true,
        maxLength: Int = 20000
    ) {
        self.numSpks = numSpks
        self.enc = Encoder(kernelSize: kernelSize, outChannels: inChannels, inChannels: 180)
        self.mask_net = MossFormer_MaskNet(
            inChannels: inChannels,
            outChannels: outChannels,
            outChannelsFinal: inChannels,
            numBlocks: numBlocks,
            norm: norm,
            numSpks: numSpks,
            skipAroundIntra: skipAroundIntra,
            useGlobalPosEnc: useGlobalPosEnc,
            maxLength: maxLength
        )
        self.dec = Decoder(
            inChannels: outChannels,
            outChannels: 1,
            kernelSize: kernelSize,
            stride: kernelSize / 2,
            bias: false
        )
        
        super.init()
    }
    
    public func callAsFunction(_ input: MLXArray) -> [MLXArray] {
        let x = enc(input)
        let mask = mask_net(x)
        
        // Stack x for each speaker
        let xStacked = MLX.stacked(Array(repeating: x, count: numSpks))
        let sepX = xStacked * mask
        
        // Decoding - use vmap for parallel processing of speakers
        // sepX has shape (num_spks, B, N, L)
        // Decoding - process each speaker separately
        var decodedSpeakers: [MLXArray] = []
        for spk in 0..<numSpks {
            decodedSpeakers.append(dec(sepX[spk]))
        }
        let decoded = MLX.stacked(decodedSpeakers, axis: 0)  // Shape: (num_spks, B, L)
        
        // Transpose to match expected output shape (B, L, num_spks)
        let estSource = decoded.transposed(1, 2, 0)
        let tOrigin = input.shape[1]
        let tEst = estSource.shape[1]
        
        let finalEstSource: MLXArray
        if tOrigin > tEst {
            finalEstSource = MLX.padded(estSource, widths: [IntOrPair(0), IntOrPair([0, tOrigin - tEst]), IntOrPair(0)])
        } else {
            finalEstSource = estSource[0..., 0..<tOrigin, 0...]
        }
        
        // Extract speakers using list slicing
        return (0..<numSpks).map { spk in
            finalEstSource[0..., 0..., spk]
        }
    }
}

// MARK: - Simplified Mossformer for HiFi-GAN

/// Simplified Mossformer wrapper for HiFi-GAN integration
public class Mossformer: Module {
    @ModuleInfo var mossformer: MossFormer_MaskNet
    
    public override init() {
        self.mossformer = MossFormer_MaskNet(
            inChannels: 80,
            outChannels: 512,
            outChannelsFinal: 80
        )
        super.init()
    }
    
    public func callAsFunction(_ input: MLXArray) -> MLXArray {
        return mossformer(input)
    }
}
