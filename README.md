# MossFormer2 Super Resolution (48K)

Audio super-resolution model for upsampling from 16kHz to 48kHz using MLX. Python and Swift implementations.

## Usage

### Python

```bash
cd python
pip install -r requirements.txt
python generate.py --input input_16k.wav --output output_48k.wav
```

### Swift

```bash
cd swift
xcodebuild build -scheme generate -configuration Release -destination 'platform=macOS' -derivedDataPath .build/DerivedData -quiet
.build/DerivedData/Build/Products/Release/generate -i input_16k.wav -o output_48k.wav
```

## Performance

MLX outputs match the original PyTorch implementation with ~1e-5 precision.

| Framework  | Speed (× faster than input) |
| ---------- | --------------------------- |
| Swift MLX  | **1.61×**                   |
| Python MLX | **4.28×**                   |

## Model

HuggingFace: [starkdmi/MossFormer2_SR_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SR_48K_MLX) (439 MB)

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
