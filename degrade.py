from pydub import AudioSegment
from pydub.generators import WhiteNoise
from pydub.effects import low_pass_filter, high_pass_filter
import numpy as np
import os

def encode_decode_g711(signal):
    """
    Encode and decode the audio signal using G.711 codec.
    
    Args:
        signal (AudioSegment): Input audio segment.
    
    Returns:
        AudioSegment: Audio segment after G.711 encoding and decoding.
    """
    # Export the signal to a temporary file in G.711 format
    temp_g711_path = "temp_g711.wav"
    signal.export(temp_g711_path, format="wav", codec="pcm_alaw")

    # Read the G.711 encoded file back into an AudioSegment
    decoded_signal = AudioSegment.from_file(temp_g711_path)

    # Clean up the temporary file
    os.remove(temp_g711_path)

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

    print(f"Signal RMS: {signal_rms}")
    print(f"Signal dBFS: {signal.dBFS}")
    print(f"Signal Max dBFS: {signal.max_dBFS}")
    print(f"Noise RMS: {noise_rms}")
    print(f"Noise dBFS: {noise.dBFS}")
    print(f"Noise Max dBFS: {noise.max_dBFS}")
    
    # Calculate the gain needed for desired SNR
    # SNR = 20 * log10(signal_rms / noise_rms_desired)
    # Therefore: noise_rms_desired = signal_rms / (10^(SNR/20))
    desired_noise_rms = signal_rms / (10 ** (target_snr_db / 20))

    print(f"Desired Noise RMS: {desired_noise_rms}")

    # Calculate required gain
    gain_db = 20 * np.log10(desired_noise_rms / noise_rms)

    print(f"Gain (dB): {gain_db}")

    # Apply gain to noise
    adjusted_noise = noise.apply_gain(gain_db)

    print(f"Adjusted Noise RMS: {adjusted_noise.rms}")
    print(f"Adjusted Noise dBFS: {adjusted_noise.dBFS}")
    print(f"Adjusted Noise Max dBFS: {adjusted_noise.max_dBFS}")
    print("----------------------------------------------------")

    return adjusted_noise

def simulate_narrowband(signal):
    """
    Simulate narrowband telephony by resampling to 8 kHz and applying a 300-3400 Hz bandpass filter.
    
    Args:
        signal (AudioSegment): Input audio segment.
    
    Returns:
        AudioSegment: Audio segment processed to simulate narrowband telephony.
    """
    # Step 1: Downsample to 8 kHz
    signal = signal.set_frame_rate(8000)

    # Step 2: Apply a high-pass filter to remove frequencies below 300 Hz
    signal = high_pass_filter(signal, 300)

    # Step 3: Apply a low-pass filter to remove frequencies above 3400 Hz
    signal = low_pass_filter(signal, 3400)

    return signal

def add_white_noise(signal, snr, noise_file_path=None):
    if noise_file_path:
        noise = AudioSegment.from_file(noise_file_path)
        # Loop noise if needed to match signal length
        noise = (noise * (len(signal) // len(noise) + 1))[:len(signal)]
    else:
        # Generate white noise audio segment with the same duration
        noise = WhiteNoise().to_audio_segment(duration=len(signal))

    noise = simulate_narrowband(signal)
    noise = normalize_signal(noise)
    noise = adjust_noise_volume_snr(noise, signal, snr)
    
    # Overlay the noise onto the original audio
    return signal.overlay(noise)

def create_degraded_audio(input_path, output_path, snr=20):
    """
    Read audio file, normalize it to -26 dBFS, degrade it, and save to new location
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save degraded audio
        noise_level (float): Level of random noise to add (0 to 1)
    """
    # Read input audio file using pydub
    signal = AudioSegment.from_file(input_path)

    # Simulate narrowband
    signal = simulate_narrowband(signal)
    
    # Normalize to -26 dBFS
    normalized_signal = normalize_signal(signal)

    # Save temp file with new reference file
    temp_ref_path = input_path.replace('.wav', '_temp_ref.wav')  # Updated export path
    normalized_signal.export(
        temp_ref_path, 
        format="wav",
        parameters=["-ar", str(signal.frame_rate)]  # Preserve sample rate
    )

    # Add noise
    noise_file_path = './audio/noise/LTASmatched_noise.wav'
    degraded_signal = add_white_noise(normalized_signal, snr)

    # Normalize to -26 dBFS
    normalized_signal = normalize_signal(degraded_signal)

    # Encode and decode using G.711 codec
    normalized_signal = encode_decode_g711(normalized_signal)

    # Save degraded audio with original parameters
    normalized_signal.export(
        output_path, 
        format="wav",
        parameters=["-ar", str(signal.frame_rate)]  # Preserve sample rate
    )

    return temp_ref_path
