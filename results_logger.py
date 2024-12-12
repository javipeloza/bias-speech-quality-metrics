import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class ResultsLogger:
    def __init__(self, file_path):
        self.file_path = file_path

    def log_results(self, analyzer):
        """Log results from AudioQualityAnalyzer"""
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

def plot_analysis_results(analyzers, save_dir='./results'):
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

def plot_statistical_results_table(statistical_results, languages):
    """
    Plot results in a table format where rows are metrics and columns are F-statistic and p-value.
    
    Args:
        statistical_results (dict): Dictionary containing statistical results with metrics as keys.
        languages (list): List of languages to include in the title and filename.
    """
    # Generate save path based on languages
    languages_filename = '_'.join(lang.lower() for lang in languages)
    save_path = f'./results/statistical_results_table_{languages_filename}.png'

    metrics = list(statistical_results.keys())
    f_statistics = [statistical_results[metric]['F-statistic'] for metric in metrics]
    p_values = [statistical_results[metric]['p-value'] for metric in metrics]

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(8, len(metrics) * 0.5 + 1))  # Adjust height based on number of metrics
    ax.axis('tight')
    ax.axis('off')

    # Create table data
    table_data = [["Metric", "F-statistic", "p-value"]] + [[metric, f_stat, p_val] for metric, f_stat, p_val in zip(metrics, f_statistics, p_values)]

    # Create the table
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')

    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Create title
    languages_str = ', '.join(languages)
    plt.title(f'Statistical Results for: {languages_str}')

    # Save the table as an image
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_post_hoc_results(post_hoc_results, output_dir='results'):
    """
    Save the post-hoc test results to a text file.

    Parameters:
    - post_hoc_results: Dictionary containing Tukey HSD test results for different metrics
    - output_dir: The directory where the text file will be saved
    """
    # Generate the filename
    filename = f'{output_dir}/comprehensive_post_hoc_tests.txt'

    # Save the results to a text file
    with open(filename, 'w') as file:
        file.write("Comprehensive Post-Hoc Test Results (Tukey HSD)\n")
        file.write("=" * 50 + "\n\n")
        
        # Iterate through metrics
        for metric, result in post_hoc_results.items():
            file.write(f"Metric: {metric}\n")
            file.write("-" * 20 + "\n")
            file.write(str(result) + "\n\n")
        
        # Add interpretation guide
        file.write("Interpretation Guide:\n")
        file.write("- Reject null hypothesis (significant difference) if p-adj < 0.05\n")
        file.write("- Positive mean diff indicates first group is higher\n")
        file.write("- Negative mean diff indicates second group is higher\n")
