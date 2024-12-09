import os
import numpy as np
from scipy.io import wavfile
from pesq import pesq
from scipy.signal import resample
from converter import AudioConverter
from degrade import create_degraded_audio

class MetricStrategy:
    def calculate_score(self, ref, deg):
        raise NotImplementedError("Must implement calculate_score method")

class PESQStrategy(MetricStrategy):
    def calculate_score(self, ref, deg):
        return pesq(16000, ref, deg, 'wb')
    
class ViSQOLStrategy(MetricStrategy):
    def calculate_score(self, ref, deg):
        return pesq(16000, ref, deg, 'wb')


class AudioQualityAnalyzer:
    def __init__(self, language, ref_dir, deg_dir, metric_strategy):
        self.language = language
        self.ref_dir = ref_dir
        self.deg_dir = deg_dir
        self.metric_strategy = metric_strategy
        self.noise_levels = np.linspace(0, 1, 5)  # More granular: 0 to 1 in ~0.05 steps
        self.results = {}  # Dictionary to store results per file and noise level
        self.skipped_files = []

    def analyze(self):
        for file_name in os.listdir(self.ref_dir):
            if not file_name.endswith('.wav'):
                continue
                
            print(f'Analyzing file: {file_name}')
            ref_path = os.path.join(self.ref_dir, file_name)
            self.results[file_name] = {}

            try:
                # Read reference file once
                _, ref = wavfile.read(ref_path)
                
                # Test each noise level
                for noise_level in self.noise_levels:
                    print(f'  Testing noise level: {noise_level:.2f}')
                    deg_path = os.path.join(self.deg_dir, f'temp_{noise_level}_{file_name}')
                    
                    try:
                        # Create degraded version for this noise level
                        create_degraded_audio(ref_path, deg_path, noise_level=noise_level)
                        _, deg = wavfile.read(deg_path)
                        
                        # Calculate score
                        score = self.metric_strategy.calculate_score(ref, deg)
                        self.results[file_name][noise_level] = score
                        print(f'    Score: {score:.3f}')
                        
                        # Clean up temporary degraded file
                        # os.remove(deg_path)
                        
                    except Exception as e:
                        print(f"Error at noise level {noise_level}: {str(e)}")
                        self.results[file_name][noise_level] = None

            except Exception as e:
                print(f"Skipping file {file_name}: {str(e)}")
                self.skipped_files.append(file_name)
                continue

    def get_results_by_file(self, file_name):
        """Get all noise levels and scores for a specific file"""
        if file_name in self.results:
            noise_levels = sorted(self.results[file_name].keys())
            scores = [self.results[file_name][level] for level in noise_levels]
            return noise_levels, scores
        return None, None

    def get_average_scores_by_noise_level(self):
        """Get average scores across all files for each noise level"""
        avg_scores = {}
        for noise_level in self.noise_levels:
            scores = [
                results[noise_level]
                for results in self.results.values()
                if noise_level in results and results[noise_level] is not None
            ]
            if scores:
                avg_scores[noise_level] = np.mean(scores)
        return sorted(avg_scores.keys()), [avg_scores[k] for k in sorted(avg_scores.keys())]

    def get_processed_count(self):
        return len(self.results)

    def get_skipped_count(self):
        return len(self.skipped_files)
