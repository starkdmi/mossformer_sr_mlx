import json
import time
import argparse
import librosa # audio load/resample
import numpy as np # single line convert for soundfile
import soundfile as sf # save audio

import mlx.core as mx
from mlx.utils import tree_unflatten
from huggingface_hub import hf_hub_download

from mossformer2_sr_wrapper import MossFormer2_SR_48K, AttrDict
from bandwidth_sub import bandwidth_sub
from mel_spec import mel_spectrogram

def process_audio(model, audio_path, output_path, args):
    """Process audio file through the model."""
    # Load audio at original sample rate
    audio_original, sr_original = librosa.load(audio_path, sr=None, mono=True)
    print(f"Original file sample rate: {sr_original} Hz")
    print(f"Loaded audio: {len(audio_original)} samples ({len(audio_original)/sr_original:.6f} seconds)")
    print(f"Original audio stats - Min: {np.min(audio_original):.6f}, Max: {np.max(audio_original):.6f}, Mean: {np.mean(np.abs(audio_original)):.6f}, Std: {np.std(audio_original):.6f}")

    # Resample to 48kHz if needed
    if sr_original != 48000:
        print(f"\nResampling from {sr_original}Hz to 48000Hz...")
        audio_48k = librosa.resample(audio_original, orig_sr=sr_original, target_sr=48000)
        print(f"Resampled audio: {len(audio_48k)} samples ({len(audio_48k)/48000:.6f} seconds)")
    else:
        print(f"\nNo resampling needed - already at 48kHz")
        audio_48k = audio_original
    print(f"Input audio stats - Min: {np.min(audio_48k):.6f}, Max: {np.max(audio_48k):.6f}, Mean: {np.mean(np.abs(audio_48k)):.6f}, Std: {np.std(audio_48k):.6f}")

    # Convert to MLX array
    input_len = len(audio_48k)
    inputs = mx.array(audio_48k)

    # Check if sliding window is needed
    if input_len > args.sampling_rate * args.one_time_decode_length:
        print("Using sliding window for long audio...")
        # Long audio processing with sliding window
        window = int(args.sampling_rate * args.decode_window)
        stride = int(window * 0.75)
        t = inputs.shape[0]
        
        # Pad if necessary
        if t < window:
            inputs = mx.concatenate([inputs, mx.zeros(window - t)])
        elif t < window + stride:
            padding = window + stride - t
            inputs = mx.concatenate([inputs, mx.zeros(padding)])
        else:
            if (t - window) % stride != 0:
                padding = stride - ((t - window) % stride)
                inputs = mx.concatenate([inputs, mx.zeros(padding)])

        t = inputs.shape[0]
        outputs = mx.zeros(t)
        give_up_length = (window - stride) // 2
        current_idx = 0

        while current_idx + window <= t:
            audio_segment = inputs[current_idx:current_idx + window]

            # Compute mel spectrogram for segment
            # mel_spectrogram_mlx expects (batch, time) format
            audio_segment_2d = mx.expand_dims(audio_segment, axis=0)
            mel_segment = mel_spectrogram(
                audio_segment_2d, 
                n_fft=args.n_fft,
                num_mels=args.num_mels,
                sampling_rate=args.sampling_rate,
                hop_size=args.hop_size,
                win_size=args.win_size,
                fmin=args.fmin,
                fmax=args.fmax
            )

            # mel_segment is already in batch format from mel_spectrogram_mlx
            mel_input = mel_segment

            # Run inference
            generator_output_segment = model(mel_input)
            generator_output_segment = mx.squeeze(generator_output_segment)

            offset = audio_segment.shape[0] - generator_output_segment.shape[0]

            if current_idx == 0:
                outputs[current_idx:current_idx + window - give_up_length] = \
                    generator_output_segment[:-give_up_length+offset] if offset != 0 else \
                    generator_output_segment[:-give_up_length]
            else:
                generator_output_segment = generator_output_segment[-window:]
                outputs[current_idx + give_up_length:current_idx + window - give_up_length] = \
                    generator_output_segment[give_up_length:-give_up_length+offset] if offset != 0 else \
                    generator_output_segment[give_up_length:-give_up_length]

            current_idx += stride
    else:
        # Short audio - process at once
        # mel_spectrogram_mlx expects (batch, time) format
        inputs_2d = mx.expand_dims(inputs, axis=0)
        mel_spec = mel_spectrogram(
            inputs_2d,
            n_fft=args.n_fft,
            num_mels=args.num_mels,
            sampling_rate=args.sampling_rate,
            hop_size=args.hop_size,
            win_size=args.win_size,
            fmin=args.fmin,
            fmax=args.fmax
        )

        # mel_spec is already in batch format from mel_spectrogram_mlx
        mel_input = mel_spec
        # print(f"Mel spectrogram stats - Min: {mx.min(mel_input):.6f}, Max: {mx.max(mel_input):.6f}, Mean: {mx.mean(mel_input):.6f}, Std: {mx.std(mel_input):.6f}")

        # Run inference
        start_time = time.time()
        output = model(mel_input)
        mx.eval(output) # Force evaluation
        inference_time = time.time() - start_time

        outputs = mx.squeeze(output)
        # print(f"Output shape: {outputs.shape}")
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Raw model output stats - Min: {mx.min(outputs):.6f}, Max: {mx.max(outputs):.6f}, Mean: {mx.mean(mx.abs(outputs)):.6f}, Std: {mx.std(outputs):.6f}")

    # Apply bandwidth substitution
    print("Applying bandwidth substitution...")
    # print(f"Before bandwidth_sub - Min: {mx.min(outputs):.6f}, Max: {mx.max(outputs):.6f}, Mean: {mx.mean(mx.abs(outputs)):.6f}, Std: {mx.std(outputs):.6f}")
    outputs = bandwidth_sub(inputs, outputs, fs=48000)
    # print(f"After bandwidth_sub - Min: {mx.min(outputs):.6f}, Max: {mx.max(outputs):.6f}, Mean: {mx.mean(mx.abs(outputs)):.6f}, Std: {mx.std(outputs):.6f}")

    # Trim to original length
    outputs = outputs[:input_len]
    # Trim/pad to original length
    # if outputs.shape[0] < input_len:
    #     padding = input_len - outputs.shape[0]
    #     outputs = mx.concatenate([outputs, mx.zeros(padding)])
    # elif outputs.shape[0] > input_len:
    #     outputs = outputs[:input_len]

    # Convert to numpy only for saving
    outputs_np = np.array(outputs)

    # Save at 48kHz
    sf.write(output_path, outputs_np, 48000)
    print(f"Saved output to {output_path} ({len(outputs_np)/48000:.2f} seconds at 48kHz)")
    print(f"Final output stats - Max: {mx.max(mx.abs(outputs)):.6f}, Mean: {mx.mean(mx.abs(outputs)):.6f}, Std: {mx.std(outputs):.6f}")

    return outputs

def main():
    parser = argparse.ArgumentParser(description="MossFormer2 Super Resolution")
    parser.add_argument("--input", required=True, help="Input audio file")
    parser.add_argument("--output", required=True, help="Output audio file")
    parser.add_argument("--model", help="Local model weights path (optional)")
    parser.add_argument("--config", help="Local config path (optional)")

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    # Check if custom weights path provided, otherwise download from HuggingFace
    if args.model:
        weights_path = args.model
        config_path = args.config if args.config else "MossFormer2_SR_48K.json"
    else:
        # Download from HuggingFace
        print("Downloading model from HuggingFace...")
        weights_path = hf_hub_download(
            repo_id="starkdmi/MossFormer2_SR_48K_MLX",
            filename="model_fp32.safetensors"
        )
        config_path = hf_hub_download(
            repo_id="starkdmi/MossFormer2_SR_48K_MLX",
            filename="config.json"
        )

    # Load model configuration
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    # Create model
    model_args = AttrDict(model_config)

    # Set decode parameters (matching PyTorch)
    model_args.one_time_decode_length = 20.0
    model_args.decode_window = 4.0

    model = MossFormer2_SR_48K(model_args)

    # Load weights from a native mlx formatted file
    print(f"Loading weights from {weights_path}...")
    weights = mx.load(weights_path)
    model.update(tree_unflatten(list(weights.items())))
    print("Weight loading complete")

    # Set model to eval mode
    model.eval()
    print("Model set to eval mode")

    # Process audio
    process_audio(model, input_path, output_path, model_args)

if __name__ == "__main__":
    main()