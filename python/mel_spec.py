import mlx.core as mx
from stft import stft, create_window

# Global cache for mel basis and windows
mel_basis = {}
hann_window = {}

def hz_to_mel(frequencies, htk=False):
    """Convert Hz to Mels"""
    if htk:
        return 2595.0 * mx.log10(1.0 + frequencies / 700.0)
    else:
        # Slaney's formula
        f_min = 0.0
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = mx.log(6.4) / 27.0
        
        # For frequencies < 1000 Hz
        mels = (frequencies - f_min) / f_sp
        
        # For frequencies >= 1000 Hz
        log_region = frequencies >= min_log_hz
        mels = mx.where(
            log_region,
            min_log_mel + mx.log(frequencies / min_log_hz) / logstep,
            mels
        )
        return mels

def mel_to_hz(mels, htk=False):
    """Convert Mels to Hz"""
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    else:
        # Slaney's formula  
        f_min = 0.0
        f_sp = 200.0 / 3
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp
        logstep = mx.log(6.4) / 27.0
        
        # For mels < min_log_mel
        freqs = f_min + f_sp * mels
        
        # For mels >= min_log_mel
        log_region = mels >= min_log_mel
        freqs = mx.where(
            log_region,
            min_log_hz * mx.exp(logstep * (mels - min_log_mel)),
            freqs
        )
        return freqs

def fft_frequencies(sr, n_fft):
    """Compute FFT bin center frequencies"""
    return mx.linspace(0, float(sr) / 2, int(1 + n_fft // 2))

def mel_frequencies(n_mels, fmin=0.0, fmax=11025.0, htk=False):
    """Compute mel band center frequencies"""
    # Convert frequency limits to mel scale
    min_mel = float(hz_to_mel(fmin, htk=htk))
    max_mel = float(hz_to_mel(fmax, htk=htk))
    
    # Equally spaced mel values - need n_mels points
    # Use numpy-style linspace with endpoint=True
    mels = mx.linspace(min_mel, max_mel, n_mels)
    
    # Convert back to Hz
    return mel_to_hz(mels, htk=htk)

def create_mel_filter_bank(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm='slaney'):
    """Create mel filter bank matrix"""
    if fmax is None:
        fmax = float(sr) / 2
        
    # Initialize the weights matrix
    weights = mx.zeros((n_mels, int(1 + n_fft // 2)), dtype=mx.float32)
    
    # Center frequencies of each FFT bin
    fftfreqs = fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Center frequencies of mel bands - uniformly spaced between limits
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    
    # Calculate differences between adjacent mel frequencies
    # Implement diff manually as MLX doesn't have it
    fdiff = mel_f[1:] - mel_f[:-1]
    
    # Create triangular filters
    # Use broadcasting to compute all filters at once
    ramps = mx.expand_dims(mel_f, axis=-1) - mx.expand_dims(fftfreqs, axis=0)
    
    # Vectorized computation using broadcasting - replaces the loop
    # lower and upper slopes for all bins
    lower = -ramps[:-2] / fdiff[:-1][:, None]  # Shape: (n_mels, n_freq)
    upper = ramps[2:] / fdiff[1:][:, None]     # Shape: (n_mels, n_freq)
    
    # Intersect them with each other and zero
    weights = mx.maximum(0, mx.minimum(lower, upper))
    
    # Apply normalization
    if isinstance(norm, str):
        if norm == "slaney":
            # Slaney-style mel is scaled to be approx constant energy per channel
            enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
            weights *= enorm[:, None]
        else:
            raise ValueError(f"Unsupported norm={norm}")
    elif norm is not None:
        # Apply Lp normalization
        weights = normalize(weights, norm=norm, axis=-1)
        
    return weights

def normalize(x, norm=2, axis=-1):
    """Normalize array along axis"""
    if norm == 1:
        return x / mx.sum(mx.abs(x), axis=axis, keepdims=True)
    elif norm == 2:
        return x / mx.sqrt(mx.sum(x * x, axis=axis, keepdims=True))
    elif norm == mx.inf:
        return x / mx.max(mx.abs(x), axis=axis, keepdims=True)
    else:
        return x / mx.power(mx.sum(mx.power(mx.abs(x), norm), axis=axis, keepdims=True), 1.0/norm)

def hann_window_mlx(window_length):
    """Create a Hann window using stft_mlx"""
    # torch.hann_window uses periodic=True by default
    return create_window('hann', window_length, periodic=True)


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """Dynamic range compression"""
    return mx.log(mx.clip(x, clip_val, None) * C)

def spectral_normalize(magnitudes):
    """Spectral normalization"""
    return dynamic_range_compression(magnitudes)

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """Mel spectrogram in MLX"""
    global mel_basis, hann_window
    
    # Create cache keys
    device_key = "mlx"  # MLX uses unified memory
    fmax_key = str(fmax) + '_' + device_key
    
    # Create mel basis if not cached
    if fmax_key not in mel_basis:
        mel = create_mel_filter_bank(
            sr=sampling_rate, 
            n_fft=n_fft, 
            n_mels=num_mels, 
            fmin=fmin, 
            fmax=fmax
        )
        mel_basis[fmax_key] = mel
        hann_window[device_key] = hann_window_mlx(win_size)
    
    # Pad signal - torch implementation always pads regardless of center
    y = mx.expand_dims(y, axis=1)
    pad_amount = int((n_fft - hop_size) / 2)
    # Use edge padding to match torch's reflect padding behavior
    # First, handle the case where signal is shorter than pad amount
    if y.shape[2] > pad_amount:
        # Reflect padding approximation using edge values
        left_pad = y[:, :, 1:pad_amount+1][:, :, ::-1]
        right_pad = y[:, :, -pad_amount-1:-1][:, :, ::-1]
        y = mx.concatenate([left_pad, y, right_pad], axis=2)
    else:
        # Fall back to constant padding for very short signals
        y = mx.pad(y, [(0, 0), (0, 0), (pad_amount, pad_amount)], mode='constant')
    y = mx.squeeze(y, axis=1)
    
    # Compute STFT using imported function
    spec_real, spec_imag = stft(
        y, 
        n_fft, 
        hop_length=hop_size, 
        win_length=win_size, 
        window=hann_window[device_key],
        center=center
    )
    
    # Convert to magnitude
    spec = mx.sqrt(spec_real * spec_real + spec_imag * spec_imag + 1e-9)
    
    # Apply mel filter bank
    spec = mx.matmul(mel_basis[fmax_key], spec)
    
    # Apply spectral normalization
    spec = spectral_normalize(spec)
    
    return spec