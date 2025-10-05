import Foundation
import Hub

/// Downloads and caches MossFormer2 model files from HuggingFace
class ModelDownloader {
    private let modelId = "starkdmi/MossFormer2_SR_48K_MLX"

    /// Downloads model configuration and weights from HuggingFace
    /// - Returns: Tuple of (configPath, weightsPath)
    func downloadModel() async throws -> (config: URL, weights: URL) {
        let repo = Hub.Repo(id: modelId)

        print("Starting download from HuggingFace: \(modelId)")

        // Download both config.json and model_fp32.safetensors
        let modelDirectory = try await HubApi.shared.snapshot(
            from: repo,
            matching: ["config.json", "model_fp32.safetensors"],
            progressHandler: { progress in
                print("Download progress: \(Int(progress.fractionCompleted * 100))%")
            }
        )

        print("Download completed to: \(modelDirectory.path)")

        let configURL = modelDirectory.appendingPathComponent("config.json")
        let weightsURL = modelDirectory.appendingPathComponent("model_fp32.safetensors")

        return (configURL, weightsURL)
    }

    /// Synchronous wrapper for downloading model
    /// - Returns: Tuple of (configPath, weightsPath)
    func downloadModelSync() throws -> (config: URL, weights: URL) {
        var result: (config: URL, weights: URL)?
        var error: Error?

        let group = DispatchGroup()
        group.enter()

        Task {
            do {
                result = try await downloadModel()
            } catch let err {
                error = err
            }
            group.leave()
        }

        group.wait()

        if let error = error {
            throw error
        }

        guard let result = result else {
            throw NSError(
                domain: "ModelDownloader",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Failed to download model"]
            )
        }

        return result
    }
}
