from pydub import AudioSegment
from pydub.effects import low_pass_filter, high_pass_filter
import numpy as np
import os

def encode_decode(signal, codec="pcm_alaw", temp_file_path="temp_audio.wav"):
    """
    Encode and decode the audio signal using specified codec.
    
    Args:
        signal (AudioSegment): Input audio segment.
        codec (str): Codec name for encoding (default: "pcm_alaw").
        temp_file_path (str): Path to save the temporary file (default: "temp_audio.wav").
    
    Returns:
        AudioSegment: Audio segment after encoding and decoding.
    """
    # Export the signal to a temporary file in the specified codec format
    signal.export(temp_file_path, format="wav", codec=codec)

    # Read the encoded file back into an AudioSegment
    decoded_signal = AudioSegment.from_file(temp_file_path)

    # Clean up the temporary file
    os.remove(temp_file_path)

    return decoded_signal

def normalize_signal(signal, target_dbfs=-26):
    """
    Normalize audio to target dBFS level using apply_gain
    
    Args:
        audio_segment (AudioSegment): Input audio segment
        target_dbfs (float): Target dBFS level (default: -26)
    
    Returns:
        AudioSegment: Normalized audio segment
    """
    gain_needed = target_dbfs - signal.dBFS
    return signal.apply_gain(gain_needed)

def adjust_noise_volume_snr(noise, signal, target_snr_db):
    """
    Add noise to audio maintaining a specific Signal-to-Noise Ratio (SNR)
    
    Args:
        noise (AudioSegment): The noise audio segment to be adjusted.
        audio_segment (AudioSegment): Input audio segment to which noise will be added.
        target_snr_db (float): Desired SNR in decibels.
    
    Returns:
        AudioSegment: Audio with added noise at specified SNR.
    """
    # Get the signal and noise powers
    signal_rms = signal.rms
    noise_rms = noise.rms

    # print(f"Signal RMS: {signal_rms}")
    # print(f"Signal dBFS: {signal.dBFS}")
    # print(f"Signal Max dBFS: {signal.max_dBFS}")
    # print(f"Noise RMS: {noise_rms}")
    # print(f"Noise dBFS: {noise.dBFS}")
    # print(f"Noise Max dBFS: {noise.max_dBFS}")
    
    # Calculate the gain needed for desired SNR
    # SNR = 20 * log10(signal_rms / noise_rms_desired)
    # Therefore: noise_rms_desired = signal_rms / (10^(SNR/20))
    desired_noise_rms = signal_rms / (10 ** (target_snr_db / 20))

    # print(f"Desired Noise RMS: {desired_noise_rms}")

    # Calculate required gain
    gain_db = 20 * np.log10(desired_noise_rms / noise_rms)

    # print(f"Gain (dB): {gain_db}")

    # Apply gain to noise
    adjusted_noise = noise.apply_gain(gain_db)

    # print(f"Adjusted Noise RMS: {adjusted_noise.rms}")
    # print(f"Adjusted Noise dBFS: {adjusted_noise.dBFS}")
    # print(f"Adjusted Noise Max dBFS: {adjusted_noise.max_dBFS}")

    return adjusted_noise

def resample_signal(signal_path, sampling_rate=8000):
    signal = AudioSegment.from_file(signal_path)
    signal.set_frame_rate(sampling_rate)

    return signal

def simulate_narrowband(signal, sampling_rate=8000):
    """
    Simulate narrowband telephony by resampling to 8 kHz and applying a 300-3400 Hz bandpass filter.
    
    Args:
        signal (AudioSegment): Input audio segment.
    
    Returns:
        AudioSegment: Audio segment processed to simulate narrowband telephony.
    """
    # Step 1: Downsample to 8 kHz
    signal = signal.set_frame_rate(sampling_rate)
    # Step 2: Apply a high-pass filter to remove frequencies below 300 Hz
    signal = high_pass_filter(signal, 300)
    # Step 3: Apply a low-pass filter to remove frequencies above 3400 Hz
    signal = low_pass_filter(signal, 3400)

    return signal

def overlay_signal(signal, snr, noise_file_path):
    noise = AudioSegment.from_file(noise_file_path)
    # Loop noise if needed to match signal length
    noise = (noise * (len(signal) // len(noise) + 1))[:len(signal)]
    noise = noise.split_to_mono()[0]
    noise = simulate_narrowband(noise)
    noise = normalize_signal(noise)
    noise = adjust_noise_volume_snr(noise, signal, snr)
    
    # Overlay the noise onto the original audio
    return signal.overlay(noise)

def export_file(signal, path):
    signal.export(
        path, 
        format="wav",
        parameters=[
            "-ar", str(signal.frame_rate),   # Preserve sample rate
            "-ac", "1", # Channels: mono
            "-sample_fmt", "s16",
        ]
    )

    return path

def generate_degraded_signal(input_path, output_path, deg_type, snr=20):
    """
    Read audio file, normalize it to -26 dBFS, degrade it, and save to new location
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save degraded audio
        snr (float): Signal-to-noise ratio (dB)
    """
    # Read input audio file using pydub
    signal = AudioSegment.from_file(input_path)
    
    # Pre-process:

    # Convert signal to mono
    signal = signal.split_to_mono()[0]

    # Simulate narrowband
    signal = simulate_narrowband(signal)
    
    # Normalize to -26 dBFS
    signal = normalize_signal(signal)

    # Save temp file with new reference file
    temp_ref_path = os.path.join(os.path.dirname(input_path), 'temp_ref', f"{os.path.splitext(os.path.basename(input_path))[0]}_temp_ref.wav")  # Updated export path
    export_file(encode_decode(signal), temp_ref_path)

    # Perform degradation
    signal = deg_type.apply_degradation(signal, snr)

    # Normalize to -26 dBFS
    signal = normalize_signal(signal)

    # Encode and decode using G.711 codec
    signal = encode_decode(signal)

    # Save degraded audio with original parameters
    export_file(signal, output_path)

    return temp_ref_path
