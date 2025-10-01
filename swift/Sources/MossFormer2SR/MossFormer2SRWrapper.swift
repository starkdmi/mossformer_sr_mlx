import MLX
import MLXNN

/// The MossFormer2_SR_48K model for speech super-resolution.
///
/// This class encapsulates the functionality of the MossFormer2 and HiFi-Gan
/// Generator within a higher-level model. It processes input audio data to produce
/// higher-resolution outputs.
public class MossFormer2_SR_48K: Module {
    @ModuleInfo var model_m: Mossformer
    @ModuleInfo var model_g: Generator
    
    public init(args: AttrDict) {
        // Initialize the Mossformer model
        self.model_m = Mossformer()
        // Initialize the Generator with args
        self.model_g = Generator(h: args)
        super.init()
    }
    
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        /// Forward pass through the model.
        ///
        /// - Parameter x: Input tensor of dimension [B, N, S], where B is the batch size,
        ///                N is the number of mel bins (80 in this case), and S is the
        ///                sequence length (e.g., time frames).
        /// - Returns: Bandwidth expanded audio output tensor from the model.
        let mossformerOutput = model_m(x)  // Get outputs from Mossformer
        let outputs = model_g(mossformerOutput)  // Generate audio with HiFi-GAN
        return outputs
    }
}