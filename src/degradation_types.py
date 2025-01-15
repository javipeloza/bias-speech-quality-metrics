from degrade import overlay_signal
import os

class DegradationType:
    audio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio'))

    @property
    def name(self):
        raise NotImplementedError("Must implement name property")

    """Base class for different types of degradation"""
    def apply_degradation(self, signal, snr):
        raise NotImplementedError("Must implement apply_degradation method")

class BlueNoise(DegradationType):
    @property
    def name(self):
        return "blue_noise"

    def apply_degradation(self, signal, snr):
        noise_file_path = f'{self.audio_dir}/noise/blue_noise.wav'
        signal = overlay_signal(signal, snr, noise_file_path)
        return signal
    
class NoisyCrowd(DegradationType):
    @property
    def name(self):
        return "noisy_crowd"

    def apply_degradation(self, signal, snr):
        noise_file_path = f'{self.audio_dir}/noise/noisy_crowd_2.wav'
        signal = overlay_signal(signal, snr, noise_file_path)
        return signal
    
class PinkNoise(DegradationType):
    @property
    def name(self):
        return "pink_noise"

    def apply_degradation(self, signal, snr):
        noise_file_path = f'{self.audio_dir}/noise/pink_noise.wav'
        signal =    (signal, snr, noise_file_path)
        return signal
