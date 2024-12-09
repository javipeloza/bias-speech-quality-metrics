from pydub import AudioSegment
from pydub.generators import WhiteNoise

def normalize_audio(audio_segment, target_dbfs=-26):
    """
    Normalize audio to target dBFS level using apply_gain
    
    Args:
        audio_segment (AudioSegment): Input audio segment
        target_dbfs (float): Target dBFS level (default: -26)
    
    Returns:
        AudioSegment: Normalized audio segment
    """
    gain_needed = target_dbfs - audio_segment.dBFS
    return audio_segment.apply_gain(gain_needed)

def adjust_noise_volume(noise, noise_level):
    """
    Adjust the volume of the noise based on noise_level.
    
    Args:
        noise (AudioSegment): The noise audio segment.
        noise_level (float): The level of noise to add (0 to 1).
    
    Returns:
        AudioSegment: The adjusted noise audio segment.
    """
    # Convert noise_level to decibels (negative values to reduce volume)
    noise_reduction_db = -20 * (1/noise_level)  # This creates a logarithmic relationship
    return noise - abs(noise_reduction_db)

def overlay_noise(audio_segment, noise_file, noise_level):
    # Load the noise file
    noise = AudioSegment.from_file(noise_file)
    
    # Repeat the noise to match the length of the audio segment
    noise = (noise * (len(audio_segment) // len(noise) + 1))[:len(audio_segment)]
    
    noise = normalize_audio(noise)

    # Adjust the noise level
    noise = adjust_noise_volume(noise, noise_level * 1.5)
    
    # Overlay the noise on the audio segment
    audio_segment = audio_segment.overlay(noise)
    
    return audio_segment

def add_white_noise(audio_segment, noise_level=0.005, noise_file=None):
    # If noise_level is 0, return the original audio unchanged
    if noise_level == 0:
        return audio_segment
    
    if noise_file:
        return overlay_noise(audio_segment, noise_file, noise_level)
        
    # Generate white noise audio segment with the same duration
    noise = WhiteNoise().to_audio_segment(duration=len(audio_segment))
    noise = adjust_noise_volume(noise, noise_level)
    
    # Overlay the noise onto the original audio
    return audio_segment.overlay(noise)

def create_degraded_audio(input_path, output_path, noise_level=0.005):
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

    # Add white noise using overlay
    degraded_audio_segment = add_white_noise(normalized_audio, noise_level)

    # Normalize to -26 dBFS
    normalized_audio = normalize_audio(degraded_audio_segment)

    # Save degraded audio with original parameters
    degraded_audio_segment.export(
        output_path, 
        format="wav",
        parameters=["-ar", str(audio_segment.frame_rate)]  # Preserve sample rate
    )
