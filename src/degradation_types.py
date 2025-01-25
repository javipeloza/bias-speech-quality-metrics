from degrade import overlay_signal
import os

class DegradationType:
    """
    Base class for different types of audio degradation.

    This class serves as a base for applying various types of noise to an audio signal.

    Attributes:
        audio_dir (str): Absolute path to the directory containing audio files.
    """
    audio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio'))

    @property
    def name(self):
        """
        Name of the degradation type.

        Returns:
            str: The name of the degradation type.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("Must implement name property")

    def apply_degradation(self, signal, snr):
        """
        Applies the degradation effect to an audio signal.

        Args:
            signal (AudioSegment): The input audio signal.
            snr (float): Signal-to-noise ratio in dB.

        Returns:
            AudioSegment: The degraded audio signal.

        Raises:
            NotImplementedError: If not implemented in a subclass.
        """
        raise NotImplementedError("Must implement apply_degradation method")

class BlueNoise(DegradationType):
    """
    Applies blue noise degradation to an audio signal.
    """
    @property
    def name(self):
        return "blue_noise"

    def apply_degradation(self, signal, snr):
        noise_file_path = f'{self.audio_dir}/noise/blue_noise.wav'
        signal = overlay_signal(signal, snr, noise_file_path)
        return signal
    
# Noisy crowd is the equivalent to babble noise
class NoisyCrowd(DegradationType):
    """
    Applies babble noise (noisy crowd) degradation to an audio signal.
    """
    @property
    def name(self):
        return "noisy_crowd"

    def apply_degradation(self, signal, snr):
        noise_file_path = f'{self.audio_dir}/noise/babble_noise.wav'
        signal = overlay_signal(signal, snr, noise_file_path)
        return signal
    
class PinkNoise(DegradationType):
    """
    Applies pink noise degradation to an audio signal.
    """
    @property
    def name(self):
        return "pink_noise"

    def apply_degradation(self, signal, snr):
        noise_file_path = f'{self.audio_dir}/noise/pink_noise.wav'
        signal =    (signal, snr, noise_file_path)
        return signal
