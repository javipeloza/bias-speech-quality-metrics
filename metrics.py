from pesq import pesq

class MetricStrategy:
    """Base class for quality metrics"""
    def calculate_score(self, ref, deg):
        raise NotImplementedError("Must implement calculate_score method")
    
    @property
    def name(self):
        raise NotImplementedError("Must implement name property")

class PESQStrategy(MetricStrategy):
    def calculate_score(self, ref, deg):
        return pesq(8000, ref, deg, 'nb')
    
    @property
    def name(self):
        return "PESQ"
    
class ViSQOLStrategy(MetricStrategy):
    def calculate_score(self, ref, deg):
        return pesq(8000, ref, deg, 'nb')  # Replace with actual ViSQOL implementation
    
    @property
    def name(self):
        return "ViSQOL" 