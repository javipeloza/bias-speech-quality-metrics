import matplotlib.pyplot as plt
import numpy as np

class ResultsLogger:
    def __init__(self, file_path):
        self.file_path = file_path

    def log_results(self, analyzer):
        """Log results from the AudioQualityAnalyzer with the new data structure"""
        with open(self.file_path, 'a') as file:
            file.write(f"\nAnalysis Results for {analyzer.language.capitalize()}\n")
            file.write(f"{'='*50}\n")
            file.write(f"Processed {analyzer.get_processed_count()} files\n")
            file.write(f"Skipped {analyzer.get_skipped_count()} files\n\n")

            # For each degradation type
            for deg_type in analyzer.degradation_types:
                file.write(f"\nDegradation Type: {deg_type.name}\n")
                file.write(f"{'-'*30}\n")

                # For each metric
                for metric in analyzer.metrics:
                    file.write(f"\n{metric.name} Results:\n")
                    
                    # Get average scores for this degradation type and metric
                    levels, scores = analyzer.get_average_scores(deg_type.name, metric.name)
                    
                    # Log average scores
                    for level, score in zip(levels, scores):
                        file.write(f"Level {level:.2f}: {score:.3f}\n")
                    
                    file.write("\n")

    def plot_results(self, analyzers, save_dir='./results'):
        """
        Create plots for each degradation type and metric combination
        
        Args:
            analyzers (list): List of AudioQualityAnalyzer objects
            save_dir (str): Directory to save the plots
        """
        if not analyzers:
            return

        # Get all degradation types and metrics from the first analyzer
        degradation_types = analyzers[0].degradation_types
        metrics = analyzers[0].metrics

        # For each degradation type
        for deg_type in degradation_types:
            # For each metric
            for metric in metrics:
                plt.figure(figsize=(12, 6))
                
                # Plot each language's results
                for analyzer in analyzers:
                    levels, scores = analyzer.get_average_scores(deg_type.name, metric.name)
                    if levels and scores:  # Only plot if we have data
                        plt.plot(levels, scores, marker='o', label=analyzer.language.capitalize())
                
                plt.xlabel(f'{deg_type.name.capitalize()} Level')
                plt.ylabel(f'{metric.name} Score')
                
                # Create title
                languages = [analyzer.language.capitalize() for analyzer in analyzers]
                languages_str = ' vs '.join(languages)
                plt.title(f'{metric.name} Scores vs {deg_type.name.capitalize()} Level: {languages_str}')
                
                plt.legend()
                plt.grid(True)
                
                # Create filename
                languages_filename = '_'.join(analyzer.language.lower() for analyzer in analyzers)
                plt.savefig(
                    f'{save_dir}/{deg_type.name}_{metric.name.lower()}_comparison_{languages_filename}.png'
                )
                plt.close()

    def display_results(self):
        """Display the contents of the results file"""
        try:
            with open(self.file_path, 'r') as file:
                print(file.read())
        except FileNotFoundError:
            print("No results file found.")

