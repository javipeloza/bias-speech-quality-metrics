from degrade import overlay_signal

class DegradationType:
    @property
    def name(self):
        raise NotImplementedError("Must implement name property")

    """Base class for different types of degradation"""
    def apply_degradation(self, signal, snr):
        raise NotImplementedError("Must implement apply_degradation method")


class NoiseType(DegradationType):
    @property
    def name(self):
        return "noise" 

    def apply_degradation(self, signal, snr):
        noise_file_path = './audio/noise/LTASmatched_noise.wav'
        noise_file_path = './audio/noise/pink_noise.wav'
        signal = overlay_signal(signal, snr, noise_file_path)
        return signal
