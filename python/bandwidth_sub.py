import mlx.core as mx
from mlx_filters import detect_bandwidth, lowpass_filter, highpass_filter

def bandwidth_sub(low_bandwidth_audio, high_bandwidth_audio, fs=48000):
    """
    MLX implementation of bandwidth substitution.

    Parameters:
    -----------
    low_bandwidth_audio : mx.array
        Audio signal with limited bandwidth (e.g., upsampled from lower sample rate)
    high_bandwidth_audio : mx.array
        Audio signal with full bandwidth
    fs : int
        Sampling frequency (default: 48000 Hz)
        
    Returns:
    --------
    mx.array
        Audio with substituted bandwidth
    """
    # Detect effective bandwidth of the first signal
    f_low, f_high = detect_bandwidth(low_bandwidth_audio, fs)
    
    # Replace the lower frequency content of the second audio
    substituted_audio = replace_bandwidth(low_bandwidth_audio, high_bandwidth_audio, fs, f_low, f_high)
    
    # Optional: Smooth the transition
    smoothed_audio = smooth_transition(substituted_audio, low_bandwidth_audio, fs)
    
    return smoothed_audio

def replace_bandwidth(signal1, signal2, fs, f_low, f_high):
    """
    Replace frequency content between f_low and f_high from signal1 into signal2.
    
    Parameters:
    -----------
    signal1 : mx.array
        Source signal for frequency content
    signal2 : mx.array
        Target signal to be modified
    fs : int
        Sampling frequency
    f_low : float
        Low frequency boundary
    f_high : float
        High frequency boundary
        
    Returns:
    --------
    mx.array
        Combined signal
    """
    # Extract effective band from signal1 (frequencies below f_high)
    effective_band = lowpass_filter(signal1, fs, f_high)
    
    # Extract high frequency content from signal2 (frequencies above f_high)
    signal2_highpass = highpass_filter(signal2, fs, f_high)
    
    # Match lengths
    min_length = min(effective_band.shape[0], signal2_highpass.shape[0])
    effective_band = effective_band[:min_length]
    signal2_highpass = signal2_highpass[:min_length]
    
    # Combine signals
    return signal2_highpass + effective_band

def smooth_transition(signal1, signal2, fs, transition_band=100):
    """
    Apply smooth transition between two signals.
    
    Parameters:
    -----------
    signal1 : mx.array
        Primary signal
    signal2 : mx.array
        Secondary signal to transition from
    fs : int
        Sampling frequency
    transition_band : int
        Transition band width in Hz (default: 100)
        
    Returns:
    --------
    mx.array
        Smoothly transitioned signal
    """
    # Calculate fade length
    fade_length = int(transition_band * fs / 1000)
    
    # Create fade curve using MLX
    fade = mx.linspace(0, 1, fade_length)
    
    # Get minimum length
    min_length = min(signal1.shape[0], signal2.shape[0])
    
    # Create full crossfade array
    if fade_length < min_length:
        ones_length = min_length - fade_length
        crossfade = mx.concatenate([fade, mx.ones(ones_length)])
    else:
        crossfade = fade[:min_length]
    
    # Apply crossfade
    smoothed_signal = (1 - crossfade) * signal2[:min_length] + crossfade * signal1[:min_length]
    
    return smoothed_signal
