import matplotlib.pyplot as plt
import numpy as np

class ResultsLogger:
    def __init__(self, file_path):
        self.file_path = file_path

    def log_results(self, analyzer, metric_name):
        """Log results from the AudioQualityAnalyzer"""
        noise_levels, avg_scores = analyzer.get_average_scores_by_noise_level()
        
        with open(self.file_path, 'a') as file:
            file.write(f"Metric: {metric_name}\n")
            file.write(f"Processed {analyzer.get_processed_count()} files\n")
            file.write(f"Skipped {analyzer.get_skipped_count()} files\n")
            file.write("\nNoise Level Results:\n")
            for noise, score in zip(noise_levels, avg_scores):
                file.write(f"Noise Level {noise:.2f}: {score:.3f}\n")
            file.write("\n")

    def display_results(self):
        with open(self.file_path, 'r') as file:
            print(file.read())

    def plot_results(self, analyzers, save_dir='./results'):
        """
        Plot results from multiple analyzers for comparison
        
        Args:
            analyzers (list): List of AudioQualityAnalyzer objects
            save_dir (str): Directory to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        # Plot each analyzer's results
        for analyzer in analyzers:
            noise_levels, scores = analyzer.get_average_scores_by_noise_level()
            plt.plot(noise_levels, scores, marker='o', label=f'{analyzer.language.capitalize()}')
        
        plt.xlabel('Noise Level')
        plt.ylabel('PESQ Score')
        
        # Create title with all languages
        languages = [analyzer.language.capitalize() for analyzer in analyzers]
        languages_str = ' vs '.join(languages)
        plt.title(f'Audio Quality vs Noise Level Comparison: {languages_str}')
        
        plt.legend()
        plt.grid(True)
        
        # Create filename with all languages
        languages_filename = '_'.join(analyzer.language.lower() for analyzer in analyzers)
        plt.savefig(f'{save_dir}/quality_vs_noise_comparison_{languages_filename}.png')
        plt.close()

