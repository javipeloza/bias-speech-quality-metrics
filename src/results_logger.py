import matplotlib.pyplot as plt
import os
import json
from analyzer import AudioQualityAnalyzer

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

def plot_analysis_results(analyzers, save_dir=None):
    """
    Create plots for each degradation type and metric combination
    
    Args:
        analyzers (list): List of AudioQualityAnalyzer objects
        save_dir (str): Directory to save the plots (default: results directory)
    """
    if save_dir is None:
        results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
        save_dir = results_dir

    if not analyzers:
        return

    # Get all degradation types and metrics from the first analyzer
    degradation_types = analyzers[0].degradation_types
    metrics = analyzers[0].metrics

    # For each degradation type
    for deg_type_name in degradation_types:
        # For each metric
        for metric_name in metrics:
            plt.figure(figsize=(12, 6))
            
            # Plot each language's results
            for analyzer in analyzers:
                levels, scores = analyzer.get_average_scores(deg_type_name, metric_name)
                if levels and scores:  # Only plot if we have data
                    plt.plot(levels, scores, marker='o', label=analyzer.language.capitalize())
            
            plt.xlabel(f'{deg_type_name.capitalize()} Level')
            plt.ylabel(f'{metric_name} Score')
            
            # Create title
            languages = [analyzer.language.capitalize() for analyzer in analyzers]
            languages_str = ' vs '.join(languages)
            plt.title(f'{metric_name} Scores vs {deg_type_name.capitalize()} Level: {languages_str}')
            
            plt.legend()
            plt.grid(True)
            
            # Create filename
            languages_filename = '_'.join(analyzer.language.lower() for analyzer in analyzers)
            plt.savefig(
                f'{save_dir}/{deg_type_name}_{metric_name.lower()}_comparison_{languages_filename}.png'
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
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    save_path = f'{results_dir}/statistical_results_table_{languages_filename}.png'

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

def plot_post_hoc_results(post_hoc_results, output_dir=None):
    """
    Save the post-hoc test results to a text file.

    Parameters:
    - post_hoc_results: Dictionary containing Tukey HSD test results for different metrics
    - output_dir: The directory where the text file will be saved (default: results directory)
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

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

def plot_pairwise_language_comparison(pairwise_comparison_results, output_dir=None):
    """
    Plot pairwise language comparison results in a table format.

    Parameters:
    - pairwise_comparison_results: Dictionary containing pairwise comparison results
    - output_dir: Directory to save the plot (default: results directory)
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure and axis
    total_rows = sum(len(pairs) for pairs in pairwise_comparison_results.values()) + len(pairwise_comparison_results) + 1
    fig, ax = plt.subplots(figsize=(12, total_rows * 0.3))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = [
        ["Metric", "Language Pair", "F-statistic", "p-value", "Significant Difference"]
    ]

    for metric, comparisons in pairwise_comparison_results.items():
        # Add a row for the metric
        table_data.append([metric, "", "", "", ""])
        
        for (lang1, lang2), results in comparisons.items():
            table_data.append([
                "",
                f"{lang1} vs {lang2}",
                f"{results['f_statistic']:.4f}",
                f"{results['p_value']:.4f}",
                "Yes" if results['significant_difference'] else "No"
            ])

    # Create the table
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')

    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    # Color-code significant differences
    for i in range(1, len(table_data)):
        if table_data[i][4] == "Yes":
            for j in range(5):
                table[(i, j)].set_facecolor('lightcoral')

    # Create title
    plt.title('Pairwise Language Comparison Results')

    # Save the table as an image
    save_path = f'{output_dir}/pairwise_language_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Pairwise language comparison results plot saved to {save_path}")

def plot_comprehensive_language_bias_analysis(comprehensive_results, output_dir=None):
    """
    Plot comprehensive language bias analysis results in a table format.

    Parameters:
    - comprehensive_results: Dictionary containing comprehensive language bias analysis results
    - output_dir: Directory to save the plot (default: results directory)
    """
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, len(comprehensive_results) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = [
        ["Metric", "Languages", "F-statistic", "p-value", "Eta-squared", "Significant Bias"]
    ]

    for metric, results in comprehensive_results.items():
        table_data.append([
            metric,
            ', '.join(results['languages']),
            f"{results['f_statistic']:.4f}",
            f"{results['p_value']:.4f}",
            f"{results['eta_squared']:.4f}",
            "Yes" if results['significant_bias'] else "No"
        ])

    # Create the table
    table = ax.table(cellText=table_data, colLabels=None, cellLoc='center', loc='center')

    # Adjust table style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Color-code significant bias
    for i in range(1, len(table_data)):
        if table_data[i][5] == "Yes":
            for j in range(6):
                table[(i, j)].set_facecolor('lightcoral')

    # Create title
    plt.title('Comprehensive Language Bias Analysis Results')

    # Save the table as an image
    save_path = f'{output_dir}/comprehensive_language_bias_analysis.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    print(f"Comprehensive language bias analysis results plot saved to {save_path}")

def save_analyzers_to_json(analyzers, file_path):
    """Save analyzers to a JSON file."""
    with open(file_path, 'w') as json_file:
        json.dump([{
            'language': analyzer.language,
            'ref_dir': analyzer.ref_dir,
            'deg_dir': analyzer.deg_dir,
            'degradation_types': [degradation_type.name for degradation_type in analyzer.degradation_types],
            'metrics': [metric.name for metric in analyzer.metrics],
            'skipped_files': analyzer.skipped_files,
            'results': analyzer.results,
        } for analyzer in analyzers], json_file)  # Save the analyzers array

def save_analyzer_to_json(analyzer, file_path):
    """Save analyzer to a JSON file as a dictionary."""
    # Convert the analyzer object to a dictionary
    json_data = {
        'language': analyzer.language,
        'ref_dir': analyzer.ref_dir,
        'deg_dir': analyzer.deg_dir,
        'degradation_types': [degradation_type.name for degradation_type in analyzer.degradation_types],
        'metrics': [metric.name for metric in analyzer.metrics],
        'skipped_files': analyzer.skipped_files,
        'results': analyzer.results,
    }  # Ensure the analyzer is saved as a dictionary

    with open(file_path, 'w') as json_file:
        json.dump(json_data, json_file)  # Save the dictionary to JSON

def json_to_analyzers(json_file):
    """Load results from a JSON file into the analyzer."""
    with open(json_file, 'r') as file:
        data = json.load(file)

    analyzers = []
    
    for entry in data:
        # Create a new instance of AudioQualityAnalyzer using the loaded data
        analyzer = AudioQualityAnalyzer(
            language=entry['language'],
            ref_dir=entry['ref_dir'],
            deg_dir=entry['deg_dir']
        )
        
        # Restore degradation types
        for degradation_type in entry['degradation_types']:
            # Assuming you have a way to reconstruct the degradation type instances
            analyzer.add_degradation_type(degradation_type)  # Adjust as necessary

        # Restore metrics
        for metric in entry['metrics']:
            # Assuming you have a way to reconstruct the metric instances
            analyzer.add_metric(metric)  # Adjust as necessary

        # Restore other attributes
        analyzer.skipped_files = entry['skipped_files']
        analyzer.results = entry['results']
        
        analyzers.append(analyzer)

    return analyzers

def save_analyzer_to_txt(analyzer, file_path):
    """Save analyzer to a text file as a formatted string."""
    with open(file_path, 'w') as file:
        file.write(f"Language: {analyzer.language}\n")
        file.write(f"Reference Directory: {analyzer.ref_dir}\n")
        file.write(f"Degraded Directory: {analyzer.deg_dir}\n")
        file.write(f"Degradation Types: {', '.join(degradation_type.name for degradation_type in analyzer.degradation_types)}\n")
        file.write(f"Metrics: {', '.join(metric.name for metric in analyzer.metrics)}\n")
        file.write(f"Skipped Files: {analyzer.skipped_files}\n")
        file.write(f"Results: {analyzer.results}\n")
        file.write("=" * 50 + "\n")
