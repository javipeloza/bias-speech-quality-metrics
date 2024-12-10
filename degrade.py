from pydub import AudioSegment
from pydub.generators import WhiteNoise
import numpy as np

def normalize_audio(audio_segment, target_dbfs=-26):
    """
    Normalize audio to target dBFS level using apply_gain
    
    Args:
        audio_segment (AudioSegment): Input audio segment
        target_dbfs (float): Target dBFS level (default: -26)
    
    Returns:
        AudioSegment: Normalized audio segment
    """
    gain_needed = target_dbfs - audio_segment.max_dBFS
    return audio_segment.apply_gain(gain_needed)

def adjust_noise_volume_snr(noise, audio_segment, target_snr_db):
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
    signal_rms = audio_segment.rms
    noise_rms = noise.rms

    print(f"Signal RMS: {signal_rms}")
    print(f"Signal dBFS: {audio_segment.dBFS}")
    print(f"Signal Max dBFS: {audio_segment.max_dBFS}")
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

def add_white_noise(audio_segment, snr, noise_file_path=None):
    if noise_file_path:
        noise = AudioSegment.from_file(noise_file_path)
        # Loop noise if needed to match signal length
        noise = (noise * (len(audio_segment) // len(noise) + 1))[:len(audio_segment)]
    else:
        # Generate white noise audio segment with the same duration
        noise = WhiteNoise().to_audio_segment(duration=len(audio_segment))

    noise = normalize_audio(noise)
    noise = adjust_noise_volume_snr(noise, audio_segment, snr)
    
    # Overlay the noise onto the original audio
    return audio_segment.overlay(noise)

def create_degraded_audio(input_path, output_path, snr=20):
    """
    Read audio file, normalize it to -26 dBFS, degrade it, and save to new location
    
    Args:
        input_path (str): Path to input audio file
        output_path (str): Path to save degraded audio
        noise_level (float): Level of random noise to add (0 to 1)
    """
    # Read input audio file using pydub
    audio_segment = AudioSegment.from_file(input_path)
    
    # Normalize to -26 dBFS
    normalized_audio = normalize_audio(audio_segment)
    # noise_file_path = './audio/noise/LTASmatched_noise.wav'

    # Add noise
    degraded_audio_segment = add_white_noise(normalized_audio, snr)

    # Normalize to -26 dBFS
    normalized_audio = normalize_audio(degraded_audio_segment)

    # Save degraded audio with original parameters
    degraded_audio_segment.export(
        output_path, 
        format="wav",
        parameters=["-ar", str(audio_segment.frame_rate)]  # Preserve sample rate
    )
