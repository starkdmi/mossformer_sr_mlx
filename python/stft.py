import mlx.core as mx

def create_window(win_type: str, win_len: int, periodic: bool = False) -> mx.array:
    """Create optimized window function"""
    if win_type == 'hamming':
        n = mx.arange(win_len, dtype=mx.float32)
        if periodic:
            return 0.54 - 0.46 * mx.cos(2 * mx.pi * n / win_len)
        else:
            return 0.54 - 0.46 * mx.cos(2 * mx.pi * n / (win_len - 1))
    elif win_type == 'hann':
        n = mx.arange(win_len, dtype=mx.float32)
        if periodic:
            return 0.5 * (1 - mx.cos(2 * mx.pi * n / win_len))
        else:
            return 0.5 * (1 - mx.cos(2 * mx.pi * n / (win_len - 1)))
    else:
        raise ValueError(f"Unsupported window type: {win_type}")

@mx.compile
def stft(x: mx.array, n_fft: int, hop_length: int, win_length: int, 
         window: mx.array, center: bool = True):
    batch_size, signal_len = x.shape

    # Efficient padding using slice operations
    if center:
        pad_amount = n_fft // 2
        # Direct slice-based reflection (fastest approach)
        x_padded = mx.concatenate([
            x[:, 1:pad_amount + 1][:, ::-1],   # Left reflection
            x,                                 # Original signal
            x[:, -pad_amount - 1:-1][:, ::-1]  # Right reflection
        ], axis=-1)
        padded_len = signal_len + 2 * pad_amount
    else:
        x_padded = x
        padded_len = signal_len

    # Single-shot framing and windowing
    num_frames = (padded_len - win_length) // hop_length + 1
    
    # Create frames with optimized strides
    frames = mx.as_strided(
        x_padded, 
        shape=(batch_size, num_frames, win_length), 
        strides=(padded_len, hop_length, 1)
    )
    
    # Apply window and handle FFT size in one operation
    if win_length == n_fft:
        # Perfect case - no padding needed
        windowed_frames = frames * window
    else:
        # Apply original window to frames
        windowed_frames = frames * window[:win_length]
        # Pad frames to n_fft
        windowed_frames = mx.concatenate([
            windowed_frames, 
            mx.zeros((batch_size, num_frames, n_fft - win_length))
        ], axis=-1)

    # FFT with immediate transpose for memory efficiency
    stft_complex = mx.fft.rfft(windowed_frames, n=n_fft, axis=-1).transpose(0, 2, 1)

    return mx.real(stft_complex), mx.imag(stft_complex)