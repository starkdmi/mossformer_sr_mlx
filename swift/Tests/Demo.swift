import XCTest
import AVFoundation
import AudioUtils
import MLX
import MLXNN
@testable import MossFormer2SR

final class MossFormer2_SRTests: XCTestCase {
    
    /// Calculate standard deviation
    func std(_ array: MLXArray) -> MLXArray {
        let mean = MLX.mean(array)
        let variance = MLX.mean(MLX.pow(array - mean, 2))
        return MLX.sqrt(variance)
    }
    
    /// Process audio file through the model
    /// Info: Could be improved with prewarming all deps - stft, mel spec, atd.
    func processAudio(
        model: MossFormer2_SR_48K,
        audioPath: String,
        outputPath: String,
        args: AttrDict
    ) throws -> MLXArray {
        // Load audio at any sample rate and resample to 48kHz if needed
        let audioLoadStart = Date()
        let config = AudioLoader.Configuration(
            targetSampleRate: 48000,
            normalizationMode: .none,
            resamplingMethod: .avAudioConverter(
                algorithm: AVSampleRateConverterAlgorithm_Mastering,
                quality: .max
            ),
        )
        let audioLoader = AudioLoader(config: config)
        let audio48k = try audioLoader.loadMono(from: URL(fileURLWithPath: audioPath)).reshaped([1, -1])
        MLX.eval(audio48k)
        print("⏱️ Audio loading: \(String(format: "%.3f", Date().timeIntervalSince(audioLoadStart)))s")
        
        // Get audio as 1D array
        let inputs = audio48k[0]
        let inputLen = inputs.shape[0]
        
        // Config
        let samplingRate = args["sampling_rate"] as? Int ?? 48000
        let oneTimeDecodeLength = args["one_time_decode_length"] as? Float ?? 20.0
        let decodeWindow = args["decode_window"] as? Float ?? 4.0
        let hopSize = args["hop_size"] as? Int ?? 256
        let nFFT = args["n_fft"] as? Int ?? 1024
        let numMels = args["num_mels"] as? Int ?? 80
        let winSize = args["win_size"] as? Int ?? 1024
        let fmin = args["fmin"] as? Float ?? 0
        let fmax = args["fmax"] as? Float ?? 8000
        
        var outputs: MLXArray
        
        // Check if sliding window is needed
        if Float(inputLen) > Float(samplingRate) * oneTimeDecodeLength {
            print("Using sliding window for long audio...")
            // Long audio processing with sliding window
            let window = Int(Float(samplingRate) * decodeWindow)
            let stride = Int(Float(window) * 0.75)
            var t = inputs.shape[0]
            
            // Pad if necessary
            var paddedInputs = inputs
            if t < window {
                paddedInputs = MLX.concatenated([inputs, MLXArray.zeros([window - t])])
            } else if t < window + stride {
                let padding = window + stride - t
                paddedInputs = MLX.concatenated([inputs, MLXArray.zeros([padding])])
            } else {
                if (t - window) % stride != 0 {
                    let padding = stride - ((t - window) % stride)
                    paddedInputs = MLX.concatenated([inputs, MLXArray.zeros([padding])])
                }
            }
            
            t = paddedInputs.shape[0]
            outputs = MLXArray.zeros([t])
            let giveUpLength = (window - stride) / 2
            var currentIdx = 0
            
            while currentIdx + window <= t {
                let audioSegment = paddedInputs[currentIdx..<(currentIdx + window)]
                
                // Compute mel spectrogram for segment
                let audioSegment2D = audioSegment.expandedDimensions(axis: 0)
                let melStart = Date()
                let melSegment = try! melSpectrogram(
                    audioSegment2D,
                    nFFT: nFFT,
                    numMels: numMels,
                    samplingRate: samplingRate,
                    hopSize: hopSize,
                    winSize: winSize,
                    fmin: fmin,
                    fmax: fmax
                )
                MLX.eval(melSegment)
                print("⏱️ Mel spectrogram (segment): \(String(format: "%.3f", Date().timeIntervalSince(melStart)))s")
                
                // Run inference
                let inferenceStart = Date()
                let generatorOutputSegment = model(melSegment)
                MLX.eval(generatorOutputSegment)
                print("⏱️ Model inference (segment): \(String(format: "%.3f", Date().timeIntervalSince(inferenceStart)))s")
                let squeezedOutput = generatorOutputSegment.squeezed()
                
                let offset = audioSegment.shape[0] - squeezedOutput.shape[0]
                
                if currentIdx == 0 {
                    let endIdx = window - giveUpLength
                    if offset != 0 {
                        outputs[currentIdx..<(currentIdx + endIdx)] = squeezedOutput[0..<(endIdx - offset)]
                    } else {
                        outputs[currentIdx..<(currentIdx + endIdx)] = squeezedOutput[0..<endIdx]
                    }
                } else {
                    let trimmedOutput = squeezedOutput[(squeezedOutput.shape[0] - window)...]
                    let startIdx = currentIdx + giveUpLength
                    let endIdx = currentIdx + window - giveUpLength
                    if offset != 0 {
                        outputs[startIdx..<endIdx] = trimmedOutput[giveUpLength..<(window - giveUpLength - offset)]
                    } else {
                        outputs[startIdx..<endIdx] = trimmedOutput[giveUpLength..<(window - giveUpLength)]
                    }
                }
                
                currentIdx += stride
            }
        } else {
            // Short audio - process at once
            
            let melStart = Date()
            let inputs2D = inputs.expandedDimensions(axis: 0)
            let melSpec = try! melSpectrogram(
                inputs2D,
                nFFT: nFFT,
                numMels: numMels,
                samplingRate: samplingRate,
                hopSize: hopSize,
                winSize: winSize,
                fmin: fmin,
                fmax: fmax
            )
            MLX.eval(melSpec)
            print("⏱️ Mel spectrogram: \(String(format: "%.3f", Date().timeIntervalSince(melStart)))s")
            
            let melMin = MLX.min(melSpec).item(Float.self)
            let melMax = MLX.max(melSpec).item(Float.self)
            let melMean = MLX.mean(melSpec).item(Float.self)
            let melStd = std(melSpec).item(Float.self)
            // print("Mel spectrogram stats - Min: \(String(format: "%.6f", melMin)), Max: \(String(format: "%.6f", melMax)), Mean: \(String(format: "%.6f", melMean)), Std: \(String(format: "%.6f", melStd))")

            // Run inference
            let startTime = Date()
            let output = model(melSpec)
            MLX.eval(output)  // Force evaluation for timing
            let inferenceTime = Date().timeIntervalSince(startTime)
            
            outputs = output.squeezed()
            // print("Output shape: \(outputs.shape)")
            print("⏱️ Inference time: \(String(format: "%.3f", inferenceTime))s")
            let outMin = MLX.min(outputs).item(Float.self)
            let outMax = MLX.max(outputs).item(Float.self)
            let outMean = MLX.mean(MLX.abs(outputs)).item(Float.self)
            let outStd = std(outputs).item(Float.self)
            // print("Raw model output stats - Min: \(String(format: "%.6f", outMin)), Max: \(String(format: "%.6f", outMax)), Mean: \(String(format: "%.6f", outMean)), Std: \(String(format: "%.6f", outStd))")
        }

        // Apply bandwidth substitution
        let bwStart = Date()
        outputs = try! bandwidthSub(inputs, outputs, fs: 48000)
        MLX.eval(outputs)
        print("⏱️ Bandwidth substitution: \(String(format: "%.3f", Date().timeIntervalSince(bwStart)))s")

        // Trim to original length
        outputs = outputs[0..<inputLen]
        
        // Print output statistics before saving
        let maxVal = MLX.max(MLX.abs(outputs)).item(Float.self)
        let meanVal = MLX.mean(MLX.abs(outputs)).item(Float.self)
        let stdVal = std(outputs).item(Float.self)
        print("Final output stats - Max: \(String(format: "%.6f", maxVal)), Mean: \(String(format: "%.6f", meanVal)), Std: \(String(format: "%.6f", stdVal))")

        let audioSaver = AudioSaver(config: .init(sampleRate: 48000))
        do {
            let saveStart = Date()
            try audioSaver.save(outputs, to: outputPath)
            print("⏱️ Audio saving: \(String(format: "%.3f", Date().timeIntervalSince(saveStart)))s")
            print("Successfully saved output to: \(outputPath)")
            print("Audio duration: \(Float(outputs.shape[0])/48000) seconds at 48kHz")
        } catch {
            print("Failed to save audio: \(error), path \(outputPath)")
            // Still continue with the test
        }
        
        return outputs
    }

    func testMossFormer2SR() throws {
        // Download model from HuggingFace
        print("Downloading model from HuggingFace...")
        let downloader = ModelDownloader()
        let (configURL, weightsURL) = try downloader.downloadModelSync()
        print("Model downloaded to:")
        print("  Config: \(configURL.path)")
        print("  Weights: \(weightsURL.path)")

        // Output paths
        let testDir = URL(fileURLWithPath: #filePath).deletingLastPathComponent()
        let inputPath = testDir.appendingPathComponent("test_16k.wav").path
        let outputPath = testDir.appendingPathComponent("superres_48k.wav").path

        // Load model configuration
        let configData = try Data(contentsOf: configURL)
        let modelConfig = try JSONSerialization.jsonObject(with: configData) as! [String: Any]
        
        // Create model
        let args = AttrDict(modelConfig)
        
        // Set decode parameters (matching PyTorch)
        args["one_time_decode_length"] = 20.0
        args["decode_window"] = 4.0
        
        let model = MossFormer2_SR_48K(args: args)

        // Load weights
        print("Loading weights from \(weightsURL.path)...")
        print("Weights URL: \(weightsURL), exists: \(FileManager.default.fileExists(atPath: weightsURL.path))")
        
        // Try loading the weights
        let loadStart = Date()
        let weights = try MLX.loadArrays(url: weightsURL)
        print("⏱️ Weight loading: \(String(format: "%.3f", Date().timeIntervalSince(loadStart)))s")
        
        // Filter out any incompatible keys
        let filteredWeights = weights.filter { key, _ in
            !key.contains("num_batches_tracked")
        }
        
        // Update model with loaded weights
        let parameters = ModuleParameters.unflattened(filteredWeights)
        
        model.update(parameters: parameters)
        print("Weight loading complete, total weights: \(filteredWeights.count)")
        
        // Set model to eval mode
        eval(model)
        
        // Process audio
        let totalStart = Date()
        _ = try processAudio(model: model, audioPath: inputPath, outputPath: outputPath, args: args)
        print("⏱️ Total processing time: \(String(format: "%.3f", Date().timeIntervalSince(totalStart)))s")
    }
}
