import os
import numpy as np
from scipy.io import wavfile
from degrade import generate_degraded_signal

class AudioQualityAnalyzer:
    def __init__(self, language, ref_dir, deg_dir):
        self.language = language
        self.ref_dir = ref_dir
        self.deg_dir = deg_dir
        self.degradation_levels = list(np.arange(-20, 41, 5))
        # self.degradation_levels = [-20,20,80]

        """
            results: {
                file_name_1: {
                    noise: {
                        snr_1: {
                            'PESQ': 1.75,
                            'ViSQOL': 1.88
                        },
                        snr2: {...}
                    }
                }
            }
        """
        
        self.results = {}
        self.skipped_files = []
        
        # Lists to store available metrics and degradation types
        self.metrics = []
        self.degradation_types = []

    def add_metric(self, metric_strategy):
        """Add a new metric strategy"""
        self.metrics.append(metric_strategy)

    def add_degradation_type(self, degradation_type):
        """Add a new degradation type"""
        self.degradation_types.append(degradation_type)

    def analyze(self):
        for file_name in os.listdir(self.ref_dir):
            if not file_name.endswith('.wav'):
                continue
                
            print(f'Analyzing file: {file_name}')
            ref_path = os.path.join(self.ref_dir, file_name)
            self.results[file_name] = {}

            try:
                # Test each degradation type
                for deg_type in self.degradation_types:
                    print(f'Testing degradation type: {deg_type.name}')
                    self.results[file_name][deg_type.name] = {}
                    
                    # Test each degradation level
                    for level in self.degradation_levels:
                        print(f'Degradation intensity: {level:.2f}')
                        deg_path = os.path.join(
                            self.deg_dir, 
                            f'deg_{deg_type.name}_{level}_{file_name}'
                        )
                        
                        try:
                            # Apply degradation
                            temp_ref_path = generate_degraded_signal(ref_path, deg_path, deg_type, level)
                            
                            # Calculate scores for all metrics
                            self.results[file_name][deg_type.name][level] = {}
                            for metric in self.metrics:
                                score = metric.calculate_score(temp_ref_path, deg_path)
                                self.results[file_name][deg_type.name][level][metric.name] = score
                                print(f'{metric.name} Score: {score:.3f}')
                            
                        except Exception as e:
                            print(f"Error at {deg_type.name} level {level}: {str(e)}")
                            self.results[file_name][deg_type.name][level] = None

                        print("----------------------------------------------------")

            except Exception as e:
                print(f"Skipping file {file_name}: {str(e)}")
                self.skipped_files.append(file_name)
                continue

    def get_results_by_file(self, file_name, degradation_type=None, metric_name=None):
        """
        Get results for a specific file, optionally filtered by degradation type and metric
        Returns: (levels, scores)
        """
        if file_name not in self.results:
            return None, None
            
        if degradation_type is None:
            return self.results[file_name]
            
        if degradation_type not in self.results[file_name]:
            return None, None
            
        if metric_name is None:
            return self.results[file_name][degradation_type]
            
        levels = sorted(self.results[file_name][degradation_type].keys())
        scores = [
            self.results[file_name][degradation_type][level][metric_name] 
            for level in levels
            if level in self.results[file_name][degradation_type] 
            and self.results[file_name][degradation_type][level] is not None
            and metric_name in self.results[file_name][degradation_type][level]
        ]
        return levels, scores

    def get_average_scores(self, degradation_type, metric_name):
        """Get average scores across all files for a specific degradation type and metric"""
        avg_scores = {}
        for level in self.degradation_levels:
            scores = []
            for file_results in self.results.values():
                if (degradation_type in file_results and 
                    level in file_results[degradation_type] and 
                    file_results[degradation_type][level] is not None and
                    metric_name in file_results[degradation_type][level]):
                    scores.append(file_results[degradation_type][level][metric_name])
            
            if scores:
                avg_scores[level] = np.mean(scores)
                
        return sorted(avg_scores.keys()), [avg_scores[k] for k in sorted(avg_scores.keys())]

    def get_processed_count(self):
        return len(self.results)

    def get_skipped_count(self):
        return len(self.skipped_files)
    
    def get_results(self):
        return self.results
