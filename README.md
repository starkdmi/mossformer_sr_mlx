# MossFormer2 Super Resolution (48K)

Audio super-resolution model for upsampling from 16kHz to 48kHz using MLX. Python and Swift implementations.

## Usage

### Python

```bash
cd python
pip install -r requirements.txt
python demo.py --input input.wav --output output.wav
```

See [`python/demo.py`](python/demo.py) for implementation details.

### Swift

See [`swift/Tests/Demo.swift`](swift/Tests/Demo.swift) for implementation details.

## Quality

MLX Python and Swift outputs match the original PyTorch implementation with ~1e-5 precision.

## Model

Original model: [alibabasglab/MossFormer2_SR_48K](https://huggingface.co/alibabasglab/MossFormer2_SR_48K)

MLX weights: [starkdmi/MossFormer2_SR_48K_MLX](https://huggingface.co/starkdmi/MossFormer2_SR_48K_MLX)

Source: [ClearerVoice-Studio](https://github.com/modelscope/ClearerVoice-Studio)

## License

See [LICENSE](LICENSE).
