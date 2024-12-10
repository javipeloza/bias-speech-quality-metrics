from degrade import create_degraded_audio

class DegradationType:
    """Base class for different types of degradation"""
    def apply_degradation(self, ref_path, deg_path, level):
        raise NotImplementedError("Must implement apply_degradation method")
    
    @property
    def name(self):
        raise NotImplementedError("Must implement name property")

class NoiseType(DegradationType):
    def apply_degradation(self, ref_path, deg_path, level):
        return create_degraded_audio(ref_path, deg_path, snr=level)
    
    @property
    def name(self):
        return "noise" 
