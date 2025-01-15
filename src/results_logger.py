import matplotlib.pyplot as plt
import os
import json
from analyzer import AudioQualityAnalyzer
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import f_oneway, kruskal

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


def plot_metrics_by_language(data, output_dir='plots'):
    """
    Creates plots for metric scores for each language and noise type
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
    """
    # Set style for better visualization
    plt.style.use('default')
    
    # Define colors for different noise types
    colors = {'blue_noise': 'blue', 'pink_noise': 'pink', 'noisy_crowd': 'red'}
    
    # Create plots for each language and metric
    metrics = ['PESQ', 'ViSQOL']
    
    for language in data.keys():
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Get the first (and only) audio file for this language
            audio_file = list(data[language].keys())[0]
            
            for noise_type in data[language][audio_file].keys():
                # Extract SNR levels and scores
                snr_levels = []
                scores = []
                
                for snr in data[language][audio_file][noise_type].keys():
                    snr_levels.append(int(snr))
                    scores.append(data[language][audio_file][noise_type][snr][metric])
                
                # Sort by SNR levels
                snr_scores = sorted(zip(snr_levels, scores))
                snr_levels, scores = zip(*snr_scores)
                
                # Plot the line
                ax.plot(snr_levels, scores, marker='o', label=noise_type, 
                       color=colors[noise_type], linewidth=2, markersize=6)
            
            # Customize the plot
            ax.set_xlabel('SNR (dB)', fontsize=12)
            ax.set_ylabel(f'{metric} Score', fontsize=12)
            ax.set_title(f'{metric} Scores vs SNR for {language.capitalize()}', 
                        fontsize=14, pad=20)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Save the plot
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, f'{language}_{metric.lower()}.png'), 
                       bbox_inches='tight', dpi=300)
            plt.close()

def plot_metric_correlation(data, output_dir='plots'):
    """
    Creates line plots comparing PESQ and ViSQOL scores for each language
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    # Set style for better visualization
    plt.style.use('default')
    
    # Define colors for different noise types
    colors = {'blue_noise': 'blue', 'pink_noise': 'pink', 'noisy_crowd': 'red'}
    
    # Create a plot for each language
    for language in data.keys():
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the first (and only) audio file for this language
        audio_file = list(data[language].keys())[0]
        
        for noise_type in data[language][audio_file].keys():
            # Extract PESQ and ViSQOL scores
            pesq_scores = []
            visqol_scores = []
            
            for snr in data[language][audio_file][noise_type].keys():
                pesq_scores.append(data[language][audio_file][noise_type][snr]['PESQ'])
                visqol_scores.append(data[language][audio_file][noise_type][snr]['ViSQOL'])
            
            # Create line plot
            ax.plot(visqol_scores, pesq_scores, 
                    label=noise_type, 
                    color=colors[noise_type],
                    marker='o',  # Add markers to the line
                    alpha=0.7,
                    linewidth=2)  # Line width
            
            # Set fixed axis limits from 1 to 5
            ax.set_xlim(1, 5)
            ax.set_ylim(1, 5)
        
        # Add a thin dotted black line from bottom-left to top-right
        ax.plot([1, 5], [1, 5], linestyle=':', color='black', linewidth=0.5)
        
        # Customize the plot
        ax.set_xlabel('ViSQOL Score', fontsize=12)
        ax.set_ylabel('PESQ Score', fontsize=12)
        ax.set_title(f'PESQ vs ViSQOL Scores for {language.capitalize()}', 
                    fontsize=14, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{language}_metric_correlation.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()


def plot_score_distribution_boxplots(data, output_dir='plots'):
    """
    Creates boxplots of PESQ and ViSQOL scores for each language across all noise types and SNR levels.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Prepare data for plotting
    pesq_scores = {lang: [] for lang in data.keys()}
    visqol_scores = {lang: [] for lang in data.keys()}
    
    # Collect scores for each language
    for language in data.keys():
        audio_file = list(data[language].keys())[0]  # Get the first audio file
        for noise_type in data[language][audio_file].keys():
            for snr in data[language][audio_file][noise_type].keys():
                pesq_scores[language].append(
                    data[language][audio_file][noise_type][snr]['PESQ']
                )
                visqol_scores[language].append(
                    data[language][audio_file][noise_type][snr]['ViSQOL']
                )
    
    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot PESQ scores
    box_pesq = ax1.boxplot([pesq_scores[lang] for lang in pesq_scores.keys()],
                labels=[lang.capitalize() for lang in pesq_scores.keys()])
    ax1.set_ylabel('PESQ Score')
    ax1.set_title('PESQ Score Distribution by Language')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylim(1, 5)
    
    # Set the median line color to blue for PESQ
    for median in box_pesq['medians']:
        median.set_color('blue')
    
    # Mark the mean with a triangle for PESQ (black)
    for i, lang in enumerate(pesq_scores.keys(), 1):
        mean_val = np.mean(pesq_scores[lang])
        ax1.plot(i, mean_val, marker='^', color='black', markersize=8)  # Triangle for mean
    
    # Plot ViSQOL scores
    box_visqol = ax2.boxplot([visqol_scores[lang] for lang in visqol_scores.keys()],
                labels=[lang.capitalize() for lang in visqol_scores.keys()])
    ax2.set_ylabel('ViSQOL Score')
    ax2.set_title('ViSQOL Score Distribution by Language')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_ylim(1, 5)
    
    # Set the median line color to blue for ViSQOL
    for median in box_visqol['medians']:
        median.set_color('blue')
    
    # Mark the mean with a triangle for ViSQOL (black)
    for i, lang in enumerate(visqol_scores.keys(), 1):
        mean_val = np.mean(visqol_scores[lang])
        ax2.plot(i, mean_val, marker='^', color='black', markersize=8)  # Triangle for mean
        
        # Draw a vertical line for the minimum value of ViSQOL
        min_val = np.min(visqol_scores[lang])
        ax2.plot([i, i], [1, min_val], color='blue', linestyle='-', linewidth=1.5)  # Min value line
    
    # Add a main title
    fig.suptitle('Score Distributions Across Languages', fontsize=14, y=1.05)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_score_distributions_by_noise(data, output_dir='plots'):
    """
    Creates boxplots of PESQ and ViSQOL scores for each language, separated by noise type.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Define noise types and prepare data structure
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    pesq_scores = {noise: {lang: [] for lang in data.keys()} for noise in noise_types}
    visqol_scores = {noise: {lang: [] for lang in data.keys()} for noise in noise_types}
    
    # Collect scores for each language and noise type
    for language in data.keys():
        audio_file = list(data[language].keys())[0]
        for noise_type in noise_types:
            for snr in data[language][audio_file][noise_type].keys():
                pesq_scores[noise_type][language].append(
                    data[language][audio_file][noise_type][snr]['PESQ']
                )
                visqol_scores[noise_type][language].append(
                    data[language][audio_file][noise_type][snr]['ViSQOL']
                )
    
    # Create subplots for each noise type
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # Plot PESQ scores (top row)
    for idx, noise_type in enumerate(noise_types):
        ax = axes[0, idx]
        box_pesq = ax.boxplot([pesq_scores[noise_type][lang] for lang in data.keys()],
                              labels=[lang.capitalize() for lang in data.keys()])
        ax.set_ylabel('PESQ Score' if idx == 0 else '')
        ax.set_title(f'{noise_type.replace("_", " ").title()}')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(1, 5)
        
        # Set the median line color to blue for PESQ
        for median in box_pesq['medians']:
            median.set_color('blue')
        
        # Mark the mean with a triangle (black) for PESQ
        for i, lang in enumerate(data.keys(), 1):
            mean_val = np.mean(pesq_scores[noise_type][lang])
            ax.plot(i, mean_val, marker='^', color='black', markersize=8)  # Triangle for mean
    
    # Plot ViSQOL scores (bottom row)
    for idx, noise_type in enumerate(noise_types):
        ax = axes[1, idx]
        box_visqol = ax.boxplot([visqol_scores[noise_type][lang] for lang in data.keys()],
                               labels=[lang.capitalize() for lang in data.keys()])
        ax.set_ylabel('ViSQOL Score' if idx == 0 else '')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_ylim(1, 5)
        
        # Set the median line color to blue for ViSQOL
        for median in box_visqol['medians']:
            median.set_color('blue')
        
        # Mark the mean with a triangle (black) for ViSQOL
        for i, lang in enumerate(data.keys(), 1):
            mean_val = np.mean(visqol_scores[noise_type][lang])
            ax.plot(i, mean_val, marker='^', color='black', markersize=8)  # Triangle for mean
            
            # Draw a vertical line for the minimum value of ViSQOL
            min_val = np.min(visqol_scores[noise_type][lang])
            ax.plot([i, i], [1, min_val], color='blue', linestyle='-', linewidth=1.5)  # Min value line
    
    # Add row labels
    fig.text(0.08, 0.75, 'PESQ Scores', rotation=90, fontsize=12)
    fig.text(0.08, 0.25, 'ViSQOL Scores', rotation=90, fontsize=12)
    
    # Add a main title
    fig.suptitle('Score Distributions by Noise Type', fontsize=14, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'score_distributions_by_noise.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_score_density_distributions(data, output_dir='plots'):
    """
    Creates KDE plots of PESQ and ViSQOL scores for each language.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Prepare data for plotting
    pesq_scores = {lang: [] for lang in data.keys()}
    visqol_scores = {lang: [] for lang in data.keys()}
    
    # Collect scores for each language
    for language in data.keys():
        audio_file = list(data[language].keys())[0]
        for noise_type in data[language][audio_file].keys():
            for snr in data[language][audio_file][noise_type].keys():
                pesq_scores[language].append(
                    data[language][audio_file][noise_type][snr]['PESQ']
                )
                visqol_scores[language].append(
                    data[language][audio_file][noise_type][snr]['ViSQOL']
                )
    
    # Create two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Define colors for each language
    colors = plt.cm.Set3(np.linspace(0, 1, len(data.keys())))
    
    # Plot PESQ score distributions
    for (language, scores), color in zip(pesq_scores.items(), colors):
        sns.kdeplot(data=scores, 
                   ax=ax1, 
                   label=language.capitalize(),
                   color=color,
                   fill=True,
                   alpha=0.3)
    
    ax1.set_xlabel('PESQ Score')
    ax1.set_ylabel('Density')
    ax1.set_title('PESQ Score Distribution by Language')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(1, 5)
    ax1.legend()
    
    # Plot ViSQOL score distributions
    for (language, scores), color in zip(visqol_scores.items(), colors):
        sns.kdeplot(data=scores, 
                   ax=ax2, 
                   label=language.capitalize(),
                   color=color,
                   fill=True,
                   alpha=0.3)
    
    ax2.set_xlabel('ViSQOL Score')
    ax2.set_ylabel('Density')
    ax2.set_title('ViSQOL Score Distribution by Language')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xlim(1, 5)
    ax2.legend()
    
    # Add a main title
    fig.suptitle('Score Density Distributions Across Languages', 
                fontsize=14, y=1.05)
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'score_density_distributions.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def plot_score_heatmaps(data, output_dir='plots'):
    """
    Creates heatmaps showing average PESQ and ViSQOL scores for each language
    across different noise types and SNR levels.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Get all unique SNR levels and noise types
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = sorted(list(data[list(data.keys())[0]][list(data[list(data.keys())[0]].keys())[0]]['blue_noise'].keys()))
    
    # Create figure with subplots for each language
    n_languages = len(data.keys())
    fig, axes = plt.subplots(n_languages, 2, figsize=(15, 5 * n_languages))
    
    # Create a heatmap for each language
    for lang_idx, (language, lang_data) in enumerate(data.items()):
        # Initialize data matrices for PESQ and ViSQOL
        pesq_matrix = np.zeros((len(noise_types), len(snr_levels)))
        visqol_matrix = np.zeros((len(noise_types), len(snr_levels)))
        
        # Get the first (and only) audio file
        audio_file = list(lang_data.keys())[0]
        
        # Fill matrices with scores
        for noise_idx, noise_type in enumerate(noise_types):
            for snr_idx, snr in enumerate(snr_levels):
                pesq_matrix[noise_idx, snr_idx] = data[language][audio_file][noise_type][snr]['PESQ']
                visqol_matrix[noise_idx, snr_idx] = data[language][audio_file][noise_type][snr]['ViSQOL']
        
        # Plot PESQ heatmap
        sns.heatmap(pesq_matrix,
                   ax=axes[lang_idx, 0],
                   cmap='YlOrRd',
                   vmin=1, vmax=5,
                   xticklabels=snr_levels,
                   yticklabels=[nt.replace('_', ' ').title() for nt in noise_types],
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'PESQ Score'})
        
        axes[lang_idx, 0].set_title(f'{language.capitalize()} - PESQ Scores')
        axes[lang_idx, 0].set_xlabel('SNR (dB)')
        axes[lang_idx, 0].set_ylabel('Noise Type')
        
        # Plot ViSQOL heatmap
        sns.heatmap(visqol_matrix,
                   ax=axes[lang_idx, 1],
                   cmap='YlOrRd',
                   vmin=1, vmax=5,
                   xticklabels=snr_levels,
                   yticklabels=[nt.replace('_', ' ').title() for nt in noise_types],
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'ViSQOL Score'})
        
        axes[lang_idx, 1].set_title(f'{language.capitalize()} - ViSQOL Scores')
        axes[lang_idx, 1].set_xlabel('SNR (dB)')
        axes[lang_idx, 1].set_ylabel('Noise Type')
    
    # Adjust layout
    plt.suptitle('Score Heatmaps by Language, Noise Type, and SNR Level', 
                fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'score_heatmaps.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_average_scores_by_snr(data, output_dir='plots'):
    """
    Creates grouped bar charts comparing average PESQ and ViSQOL scores for each language
    by noise type across different SNR levels.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Define SNR levels and noise types
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    languages = list(data.keys())
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Set width of bars and positions
    bar_width = 0.15
    x = np.arange(len(languages))
    
    # Define colors for noise types
    colors = {'blue_noise': 'blue', 
             'pink_noise': 'pink', 
             'noisy_crowd': 'red'}
    
    # Plot bars for each SNR level and noise type
    for snr_idx, snr in enumerate(snr_levels):
        # Calculate average scores
        pesq_averages = {noise_type: [] for noise_type in noise_types}
        visqol_averages = {noise_type: [] for noise_type in noise_types}
        
        for language in languages:
            audio_file = list(data[language].keys())[0]
            for noise_type in noise_types:
                try:
                    pesq_averages[noise_type].append(
                        data[language][audio_file][noise_type][snr]['PESQ']
                    )
                    visqol_averages[noise_type].append(
                        data[language][audio_file][noise_type][snr]['ViSQOL']
                    )
                except KeyError:
                    pesq_averages[noise_type].append(np.nan)
                    visqol_averages[noise_type].append(np.nan)
        
        # Plot bars for each noise type
        for noise_idx, noise_type in enumerate(noise_types):
            # Only add label to legend for the first SNR level
            label = noise_type.replace('_', ' ').title() if snr_idx == 0 else None
            
            # Calculate x positions for this group of bars
            pos = x + (snr_idx - len(snr_levels)/2 + noise_idx/3) * bar_width
            
            # Plot PESQ scores
            ax1.bar(pos, pesq_averages[noise_type], 
                   bar_width/3, label=label, 
                   color=colors[noise_type],
                   alpha=0.7 + 0.3*(snr_idx/len(snr_levels)))
            
            # Plot ViSQOL scores
            ax2.bar(pos, visqol_averages[noise_type], 
                   bar_width/3, label=label, 
                   color=colors[noise_type],
                   alpha=0.7 + 0.3*(snr_idx/len(snr_levels)))
    
    # Customize PESQ plot
    ax1.set_ylabel('PESQ Score')
    ax1.set_title('Average PESQ Scores by Language, Noise Type, and SNR Level')
    ax1.set_xticks(x)
    ax1.set_xticklabels([lang.capitalize() for lang in languages])
    ax1.set_ylim(1, 5)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create noise type legend handles
    noise_legend_elements = [plt.Rectangle((0,0),1,1, color=colors[noise_type], 
                           label=noise_type.replace('_', ' ').title())
                           for noise_type in noise_types]
    
    # Create SNR level legend elements
    snr_legend_elements = [plt.Line2D([0], [0], color='gray', 
                         label=f'{snr}dB', 
                         marker='s', 
                         linestyle='None',
                         alpha=0.7 + 0.3*(i/len(snr_levels))) 
                         for i, snr in enumerate(snr_levels)]
    
    # Add combined legend for PESQ plot
    ax1.legend(handles=[*noise_legend_elements, *snr_legend_elements],
              title='Noise Type                SNR Level',
              ncol=2,
              bbox_to_anchor=(1.15, 1))
    
    # Customize ViSQOL plot
    ax2.set_ylabel('ViSQOL Score')
    ax2.set_title('Average ViSQOL Scores by Language, Noise Type, and SNR Level')
    ax2.set_xticks(x)
    ax2.set_xticklabels([lang.capitalize() for lang in languages])
    ax2.set_ylim(1, 5)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add combined legend for ViSQOL plot
    ax2.legend(handles=[*noise_legend_elements, *snr_legend_elements],
              title='Noise Type                SNR Level',
              ncol=2,
              bbox_to_anchor=(1.15, 1))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'average_scores_by_snr.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_radar_charts(data, output_dir='plots'):
    """
    Creates radar charts comparing average PESQ and ViSQOL scores across languages
    for different noise types and SNR levels.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Define categories and prepare data
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    languages = list(data.keys())
    
    # Calculate average scores for each language, noise type, and SNR level
    def calculate_averages(metric):
        averages = {lang: {'by_noise': {}, 'by_snr': {}} for lang in languages}
        
        for lang in languages:
            audio_file = list(data[lang].keys())[0]
            
            # Calculate averages by noise type (across all SNR levels)
            for noise in noise_types:
                scores = []
                for snr in snr_levels:
                    try:
                        scores.append(data[lang][audio_file][noise][snr][metric])
                    except KeyError:
                        continue
                averages[lang]['by_noise'][noise] = np.mean(scores)
            
            # Calculate averages by SNR level (across all noise types)
            for snr in snr_levels:
                scores = []
                for noise in noise_types:
                    try:
                        scores.append(data[lang][audio_file][noise][snr][metric])
                    except KeyError:
                        continue
                averages[lang]['by_snr'][snr] = np.mean(scores)
        
        return averages
    
    pesq_averages = calculate_averages('PESQ')
    visqol_averages = calculate_averages('ViSQOL')
    
    # Create figure with 4 subplots (2x2)
    fig = plt.figure(figsize=(20, 20))
    
    # Helper function to create radar chart
    def create_radar_chart(ax, categories, values_dict, title, original_keys=None):
        """
        Helper function to create radar chart
        
        Args:
            original_keys: List of original dictionary keys if categories are formatted
        """
        # Number of variables
        num_vars = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        # Initialize the plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(0)
        
        # Plot data
        for lang_idx, (lang, values) in enumerate(values_dict.items()):
            # Use original keys for accessing values if provided
            if original_keys:
                values_list = [values[orig_key] for orig_key in original_keys]
            else:
                values_list = [values[cat] for cat in categories]
                
            values_list += values_list[:1]  # Complete the circle
            
            color = plt.cm.Set2(lang_idx / len(languages))
            ax.plot(angles, values_list, 'o-', linewidth=2, label=lang.capitalize(), color=color)
            ax.fill(angles, values_list, alpha=0.25, color=color)
        
        # Set chart properties
        ax.set_title(title, pad=20, fontsize=12)
        ax.set_ylim(1, 5)
        plt.xticks(angles[:-1], categories, fontsize=10)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(True)
    
    # Create radar charts
    # PESQ by noise type
    ax1 = fig.add_subplot(221, projection='polar')
    create_radar_chart(ax1, 
                      [nt.replace('_', ' ').title() for nt in noise_types],
                      {lang: pesq_averages[lang]['by_noise'] for lang in languages},
                      'Average PESQ Scores by Noise Type',
                      original_keys=noise_types)  # Pass original keys
    
    # PESQ by SNR
    ax2 = fig.add_subplot(222, projection='polar')
    create_radar_chart(ax2,
                      [f'{snr}dB' for snr in snr_levels],
                      {lang: pesq_averages[lang]['by_snr'] for lang in languages},
                      'Average PESQ Scores by SNR Level',
                      original_keys=snr_levels)  # Pass original keys
    
    # ViSQOL by noise type
    ax3 = fig.add_subplot(223, projection='polar')
    create_radar_chart(ax3,
                      [nt.replace('_', ' ').title() for nt in noise_types],
                      {lang: visqol_averages[lang]['by_noise'] for lang in languages},
                      'Average ViSQOL Scores by Noise Type',
                      original_keys=noise_types)  # Pass original keys
    
    # ViSQOL by SNR
    ax4 = fig.add_subplot(224, projection='polar')
    create_radar_chart(ax4,
                      [f'{snr}dB' for snr in snr_levels],
                      {lang: visqol_averages[lang]['by_snr'] for lang in languages},
                      'Average ViSQOL Scores by SNR Level',
                      original_keys=snr_levels)  # Pass original keys
    
    # Add a single legend for all subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, 
              loc='center',
              bbox_to_anchor=(0.5, 0.02),
              ncol=len(languages))
    
    # Add main title
    fig.suptitle('Radar Charts of Average Scores Across Languages', 
                fontsize=16, y=0.95)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'radar_charts.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    

def plot_pca_clusters(data, output_dir='plots'):
    """
    Performs PCA on the score profiles, visualizes clusters of languages, 
    and extracts insights to detect potential biases in metrics.

    Args:
        data (dict): Parsed JSON data containing the analysis results.
        output_dir (str): Directory where plots will be saved (default: 'plots').
    """
    plt.style.use('default')

    # Define noise types and SNR levels
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    languages = list(data.keys())

    # Prepare the data matrix
    score_matrix = []
    for language in languages:
        audio_file = list(data[language].keys())[0]
        scores = []
        for noise_type in noise_types:
            for snr in snr_levels:
                try:
                    scores.append(data[language][audio_file][noise_type][snr]['PESQ'])
                    scores.append(data[language][audio_file][noise_type][snr]['ViSQOL'])
                except KeyError:
                    scores.append(np.nan)
                    scores.append(np.nan)
        score_matrix.append(scores)

    # Convert to numpy array and handle missing values
    score_matrix = np.array(score_matrix)
    score_matrix = np.nan_to_num(score_matrix, nan=np.nanmean(score_matrix, axis=0))

    # Standardize the data
    scaler = StandardScaler()
    score_matrix_scaled = scaler.fit_transform(score_matrix)

    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(score_matrix_scaled)

    # Extract explained variance
    explained_variance = pca.explained_variance_ratio_
    print("Explained Variance Ratio (PC1, PC2):", explained_variance)

    # Extract PCA loadings
    pca_loadings = pca.components_
    print("PCA Loadings:")
    for i, component in enumerate(pca_loadings):
        print(f"Component {i+1}:", component)

    # Calculate mean and variance of scores per language
    metrics_summary = {}
    for i, language in enumerate(languages):
        metrics_summary[language] = {
            'PESQ_mean': np.nanmean(score_matrix[i, ::2]),  # Every other column is PESQ
            'PESQ_std': np.nanstd(score_matrix[i, ::2]),
            'ViSQOL_mean': np.nanmean(score_matrix[i, 1::2]),  # Every other column is ViSQOL
            'ViSQOL_std': np.nanstd(score_matrix[i, 1::2]),
        }

    print("Metrics Summary per Language:")
    for lang, stats in metrics_summary.items():
        print(lang.capitalize(), stats)

    # Plot the PCA results
    fig, ax = plt.subplots(figsize=(10, 8))

    # Define colors for each language
    colors = plt.cm.Set3(np.linspace(0, 1, len(languages)))

    for i, language in enumerate(languages):
        ax.scatter(pca_result[i, 0], pca_result[i, 1], 
                   color=colors[i], label=language.capitalize(), s=100)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA of Language Score Profiles')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()

    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'pca_clusters_bias_analysis.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_score_differences(data, baseline_language='english', output_dir='plots'):
    """
    Computes and plots score differences between a baseline language and other languages
    at each SNR level or noise type.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        baseline_language (str): The language to use as the baseline for comparison
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Define noise types and SNR levels
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    languages = list(data.keys())
    
    # Ensure the baseline language is in the data
    if baseline_language not in languages:
        raise ValueError(f"Baseline language '{baseline_language}' not found in data.")
    
    # Prepare data for plotting
    def compute_differences(metric):
        differences_by_noise = {lang: {noise: [] for noise in noise_types} for lang in languages if lang != baseline_language}
        differences_by_snr = {lang: {snr: [] for snr in snr_levels} for lang in languages if lang != baseline_language}
        
        baseline_scores_by_noise = {noise: [] for noise in noise_types}
        baseline_scores_by_snr = {snr: [] for snr in snr_levels}
        
        # Collect baseline scores
        audio_file = list(data[baseline_language].keys())[0]
        for noise in noise_types:
            for snr in snr_levels:
                try:
                    baseline_scores_by_noise[noise].append(data[baseline_language][audio_file][noise][snr][metric])
                    baseline_scores_by_snr[snr].append(data[baseline_language][audio_file][noise][snr][metric])
                except KeyError:
                    continue
        
        # Compute differences
        for lang in languages:
            if lang == baseline_language:
                continue
            audio_file = list(data[lang].keys())[0]
            for noise in noise_types:
                for snr in snr_levels:
                    try:
                        score = data[lang][audio_file][noise][snr][metric]
                        differences_by_noise[lang][noise].append(score - np.mean(baseline_scores_by_noise[noise]))
                        differences_by_snr[lang][snr].append(score - np.mean(baseline_scores_by_snr[snr]))
                    except KeyError:
                        continue
        
        return differences_by_noise, differences_by_snr
    
    pesq_differences_by_noise, pesq_differences_by_snr = compute_differences('PESQ')
    visqol_differences_by_noise, visqol_differences_by_snr = compute_differences('ViSQOL')
    
    # Plot differences by noise type
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # PESQ differences by noise type
    for lang, differences in pesq_differences_by_noise.items():
        avg_differences = [np.mean(differences[noise]) for noise in noise_types]
        axes[0].plot(noise_types, avg_differences, marker='o', label=lang.capitalize())
    
    axes[0].set_title('PESQ Score Differences by Noise Type')
    axes[0].set_ylabel('Difference from English')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # ViSQOL differences by noise type
    for lang, differences in visqol_differences_by_noise.items():
        avg_differences = [np.mean(differences[noise]) for noise in noise_types]
        axes[1].plot(noise_types, avg_differences, marker='o', label=lang.capitalize())
    
    axes[1].set_title('ViSQOL Score Differences by Noise Type')
    axes[1].set_ylabel('Difference from English')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'score_differences_by_noise.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot differences by SNR level
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # PESQ differences by SNR level
    for lang, differences in pesq_differences_by_snr.items():
        avg_differences = [np.mean(differences[snr]) for snr in snr_levels]
        axes[0].plot(snr_levels, avg_differences, marker='o', label=lang.capitalize())
    
    axes[0].set_title('PESQ Score Differences by SNR Level')
    axes[0].set_ylabel('Difference from English')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # ViSQOL differences by SNR level
    for lang, differences in visqol_differences_by_snr.items():
        avg_differences = [np.mean(differences[snr]) for snr in snr_levels]
        axes[1].plot(snr_levels, avg_differences, marker='o', label=lang.capitalize())
    
    axes[1].set_title('ViSQOL Score Differences by SNR Level')
    axes[1].set_ylabel('Difference from English')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'score_differences_by_snr.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def plot_relative_performance_ratios(data, target_languages, baseline_language='english', output_dir='plots'):
    """
    Computes and plots the relative performance ratios of PESQ/ViSQOL scores between
    target languages and a baseline language (English).
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        target_languages (list): List of target languages to compare against the baseline
        baseline_language (str): The language to use as the baseline for comparison
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Define noise types and SNR levels
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    
    # Ensure the baseline language is in the data
    if baseline_language not in data:
        raise ValueError(f"Baseline language '{baseline_language}' not found in data.")
    
    # Prepare data for plotting
    def compute_ratios(metric):
        ratios_by_noise = {lang: {noise: [] for noise in noise_types} for lang in target_languages}
        ratios_by_snr = {lang: {snr: [] for snr in snr_levels} for lang in target_languages}
        
        baseline_scores_by_noise = {noise: [] for noise in noise_types}
        baseline_scores_by_snr = {snr: [] for snr in snr_levels}
        
        # Collect baseline scores
        audio_file = list(data[baseline_language].keys())[0]
        for noise in noise_types:
            for snr in snr_levels:
                try:
                    baseline_scores_by_noise[noise].append(data[baseline_language][audio_file][noise][snr][metric])
                    baseline_scores_by_snr[snr].append(data[baseline_language][audio_file][noise][snr][metric])
                except KeyError:
                    continue
        
        # Compute ratios
        for lang in target_languages:
            if lang not in data:
                continue
            audio_file = list(data[lang].keys())[0]
            for noise in noise_types:
                for snr in snr_levels:
                    try:
                        score = data[lang][audio_file][noise][snr][metric]
                        ratios_by_noise[lang][noise].append(score / np.mean(baseline_scores_by_noise[noise]))
                        ratios_by_snr[lang][snr].append(score / np.mean(baseline_scores_by_snr[snr]))
                    except KeyError:
                        continue
        
        return ratios_by_noise, ratios_by_snr
    
    pesq_ratios_by_noise, pesq_ratios_by_snr = compute_ratios('PESQ')
    visqol_ratios_by_noise, visqol_ratios_by_snr = compute_ratios('ViSQOL')
    
    # Plot ratios by noise type
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # PESQ ratios by noise type
    for lang, ratios in pesq_ratios_by_noise.items():
        avg_ratios = [np.mean(ratios[noise]) for noise in noise_types]
        axes[0].bar(noise_types, avg_ratios, label=lang.capitalize())
    
    axes[0].set_title('PESQ Score Ratios by Noise Type')
    axes[0].set_ylabel('Ratio to English')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # ViSQOL ratios by noise type
    for lang, ratios in visqol_ratios_by_noise.items():
        avg_ratios = [np.mean(ratios[noise]) for noise in noise_types]
        axes[1].bar(noise_types, avg_ratios, label=lang.capitalize())
    
    axes[1].set_title('ViSQOL Score Ratios by Noise Type')
    axes[1].set_ylabel('Ratio to English')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'relative_performance_ratios_by_noise.png'),
                bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot ratios by SNR level
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # PESQ ratios by SNR level
    for lang, ratios in pesq_ratios_by_snr.items():
        avg_ratios = [np.mean(ratios[snr]) for snr in snr_levels]
        axes[0].bar(snr_levels, avg_ratios, label=lang.capitalize())
    
    axes[0].set_title('PESQ Score Ratios by SNR Level')
    axes[0].set_ylabel('Ratio to English')
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend()
    
    # ViSQOL ratios by SNR level
    for lang, ratios in visqol_ratios_by_snr.items():
        avg_ratios = [np.mean(ratios[snr]) for snr in snr_levels]
        axes[1].bar(snr_levels, avg_ratios, label=lang.capitalize())
    
    axes[1].set_title('ViSQOL Score Ratios by SNR Level')
    axes[1].set_ylabel('Ratio to English')
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'relative_performance_ratios_by_snr.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


def conduct_statistical_tests(data, output_dir='results'):
    """
    Conducts statistical tests to determine if score distributions differ significantly between languages.
    Saves results to a text file and a PNG file.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where results will be saved (default: 'results')
    """
    plt.style.use('default')
    
    # Define noise types and SNR levels
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    languages = list(data.keys())
    
    # Prepare data for statistical tests
    def collect_scores(metric):
        scores_by_language = {lang: [] for lang in languages}
        
        for language in languages:
            audio_file = list(data[language].keys())[0]
            for noise_type in noise_types:
                for snr in snr_levels:
                    try:
                        scores_by_language[language].append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
        
        return scores_by_language
    
    pesq_scores_by_language = collect_scores('PESQ')
    visqol_scores_by_language = collect_scores('ViSQOL')
    
    # Conduct ANOVA and Kruskal-Wallis tests
    def perform_tests(scores_by_language, metric_name):
        anova_result = f_oneway(*scores_by_language.values())
        kruskal_result = kruskal(*scores_by_language.values())
        
        # Save results to a text file
        with open(os.path.join(output_dir, f'{metric_name}_statistical_tests.txt'), 'w') as f:
            f.write(f"Statistical Tests for {metric_name} Scores\n")
            f.write("="*40 + "\n")
            f.write("ANOVA Test:\n")
            f.write(f"F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4e}\n\n")
            f.write("Kruskal-Wallis Test:\n")
            f.write(f"H-statistic: {kruskal_result.statistic:.4f}, p-value: {kruskal_result.pvalue:.4e}\n")
        
        return anova_result, kruskal_result
    
    pesq_anova, pesq_kruskal = perform_tests(pesq_scores_by_language, 'PESQ')
    visqol_anova, visqol_kruskal = perform_tests(visqol_scores_by_language, 'ViSQOL')
    
    # Plot results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot p-values
    metrics = ['PESQ', 'ViSQOL']
    anova_pvalues = [pesq_anova.pvalue, visqol_anova.pvalue]
    kruskal_pvalues = [pesq_kruskal.pvalue, visqol_kruskal.pvalue]
    
    bar_width = 0.35
    index = np.arange(len(metrics))
    
    ax.bar(index, anova_pvalues, bar_width, label='ANOVA', color='skyblue')
    ax.bar(index + bar_width, kruskal_pvalues, bar_width, label='Kruskal-Wallis', color='lightcoral')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('p-value')
    ax.set_title('Statistical Test p-values for Score Distributions')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(metrics)
    ax.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
    ax.legend()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'statistical_tests_pvalues.png'),
                bbox_inches='tight', dpi=300)
    plt.close()


from scipy.stats import ttest_ind, mannwhitneyu

def perform_pairwise_tests(data, languages_to_compare, test_type='t-test', output_dir='results'):
    """
    Performs pairwise t-tests or Mann-Whitney U tests to compare specific languages.
    Saves results to a text file and plots the results.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        languages_to_compare (list): List of language pairs to compare
        test_type (str): Type of test to perform ('t-test' or 'mann-whitney')
        output_dir (str): Directory where results will be saved (default: 'results')
    """
    plt.style.use('default')
    
    # Define noise types and SNR levels
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    
    # Prepare data for pairwise tests
    def collect_scores(metric):
        scores_by_language = {lang: [] for lang in data.keys()}
        
        for language in data.keys():
            audio_file = list(data[language].keys())[0]
            for noise_type in noise_types:
                for snr in snr_levels:
                    try:
                        scores_by_language[language].append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
        
        return scores_by_language
    
    pesq_scores_by_language = collect_scores('PESQ')
    visqol_scores_by_language = collect_scores('ViSQOL')
    
    # Perform pairwise tests
    def perform_tests(scores_by_language, metric_name):
        results = []
        
        for lang1, lang2 in languages_to_compare:
            if lang1 not in scores_by_language or lang2 not in scores_by_language:
                continue
            
            scores1 = scores_by_language[lang1]
            scores2 = scores_by_language[lang2]
            
            if test_type == 't-test':
                test_result = ttest_ind(scores1, scores2, equal_var=False)
                test_name = 'T-test'
            elif test_type == 'mann-whitney':
                test_result = mannwhitneyu(scores1, scores2, alternative='two-sided')
                test_name = 'Mann-Whitney U test'
            else:
                raise ValueError("Invalid test type. Choose 't-test' or 'mann-whitney'.")
            
            results.append((lang1, lang2, test_name, test_result.statistic, test_result.pvalue))
        
        # Save results to a text file
        with open(os.path.join(output_dir, f'{metric_name}_pairwise_tests.txt'), 'w') as f:
            f.write(f"Pairwise {test_name} for {metric_name} Scores\n")
            f.write("="*50 + "\n")
            for lang1, lang2, test_name, statistic, pvalue in results:
                f.write(f"{lang1.capitalize()} vs {lang2.capitalize()}:\n")
                f.write(f"Statistic: {statistic:.4f}, p-value: {pvalue:.4e}\n\n")
        
        return results
    
    pesq_results = perform_tests(pesq_scores_by_language, 'PESQ')
    visqol_results = perform_tests(visqol_scores_by_language, 'ViSQOL')

    test_name = "Pairwise Test" 
    
    # Plot results
    def plot_results(results, metric_name):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        labels = [f"{lang1.capitalize()} vs {lang2.capitalize()}" for lang1, lang2, _, _, _ in results]
        statistics = [statistic for _, _, _, statistic, _ in results]
        pvalues = [pvalue for _, _, _, _, pvalue in results]
        
        # Create bar plot for statistics
        ax.bar(labels, statistics, color='skyblue', label='Statistic')
        
        # Plot p-values as a secondary y-axis
        ax2 = ax.twinx()
        ax2.plot(labels, pvalues, color='red', marker='o', linestyle='None', label='p-value')
        
        ax.set_ylabel('Statistic')
        ax2.set_ylabel('p-value')
        ax.set_title(f"Pairwise {test_name} Results for {metric_name} Scores")
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        plt.xticks(rotation=45, ha='right')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric_name}_pairwise_tests.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    plot_results(pesq_results, 'PESQ')
    plot_results(visqol_results, 'ViSQOL')
    
    # Print results to console
    print("Pairwise Test Results:")
    for metric_name, results in zip(['PESQ', 'ViSQOL'], [pesq_results, visqol_results]):
        print(f"\n{metric_name} Scores:")
        for lang1, lang2, test_name, statistic, pvalue in results:
            print(f"{lang1.capitalize()} vs {lang2.capitalize()}: {test_name} - Statistic: {statistic:.4f}, p-value: {pvalue:.4e}")


def compute_effect_sizes(data, languages_to_compare, output_dir='results'):
    """
    Computes effect sizes (Cohen's d) to quantify the magnitude of differences in scores between languages.
    Saves results to a text file and creates a plot.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        languages_to_compare (list): List of language pairs to compare
        output_dir (str): Directory where results will be saved (default: 'results')
    """
    # Define noise types and SNR levels
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    
    # Prepare data for effect size computation
    def collect_scores(metric):
        scores_by_language = {lang: [] for lang in data.keys()}
        
        for language in data.keys():
            audio_file = list(data[language].keys())[0]
            for noise_type in noise_types:
                for snr in snr_levels:
                    try:
                        scores_by_language[language].append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
        
        return scores_by_language
    
    pesq_scores_by_language = collect_scores('PESQ')
    visqol_scores_by_language = collect_scores('ViSQOL')
    
    # Compute effect sizes
    def compute_cohens_d(scores1, scores2):
        # Calculate the means and standard deviations
        mean1, mean2 = np.mean(scores1), np.mean(scores2)
        std1, std2 = np.std(scores1, ddof=1), np.std(scores2, ddof=1)
        
        # Calculate pooled standard deviation
        pooled_std = np.sqrt(((len(scores1) - 1) * std1**2 + (len(scores2) - 1) * std2**2) / (len(scores1) + len(scores2) - 2))
        
        # Calculate Cohen's d
        d = (mean1 - mean2) / pooled_std
        return d
    
    def perform_effect_size_computation(scores_by_language, metric_name):
        results = []
        
        for lang1, lang2 in languages_to_compare:
            if lang1 not in scores_by_language or lang2 not in scores_by_language:
                continue
            
            scores1 = scores_by_language[lang1]
            scores2 = scores_by_language[lang2]
            
            d = compute_cohens_d(scores1, scores2)
            results.append((lang1, lang2, d))
        
        # Save results to a text file
        with open(os.path.join(output_dir, f'{metric_name}_effect_sizes.txt'), 'w') as f:
            f.write(f"Effect Sizes (Cohen's d) for {metric_name} Scores\n")
            f.write("="*50 + "\n")
            for lang1, lang2, d in results:
                f.write(f"{lang1.capitalize()} vs {lang2.capitalize()}:\n")
                f.write(f"Cohen's d: {d:.4f}\n\n")
        
        return results
    
    pesq_effect_sizes = perform_effect_size_computation(pesq_scores_by_language, 'PESQ')
    visqol_effect_sizes = perform_effect_size_computation(visqol_scores_by_language, 'ViSQOL')
    
    # Plot effect sizes
    def plot_effect_sizes(effect_sizes, metric_name):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data for plotting
        labels = [f"{lang1.capitalize()} vs {lang2.capitalize()}" for lang1, lang2, _ in effect_sizes]
        values = [d for _, _, d in effect_sizes]
        
        # Create bar plot
        ax.bar(labels, values, color='skyblue')
        
        ax.set_ylabel("Cohen's d")
        ax.set_title(f"Effect Sizes (Cohen's d) for {metric_name} Scores")
        ax.axhline(y=0, color='black', linewidth=0.8)
        ax.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric_name}_effect_sizes.png'), bbox_inches='tight', dpi=300)
        plt.close()
    
    plot_effect_sizes(pesq_effect_sizes, 'PESQ')
    plot_effect_sizes(visqol_effect_sizes, 'ViSQOL')
    
    # Print results to console
    print("Effect Size Results:")
    for metric_name, results in zip(['PESQ', 'ViSQOL'], [pesq_effect_sizes, visqol_effect_sizes]):
        print(f"\n{metric_name} Scores:")
        for lang1, lang2, d in results:
            print(f"{lang1.capitalize()} vs {lang2.capitalize()}: Cohen's d = {d:.4f}")

def plot_average_scores_by_snr_level(data, output_dir='plots'):
    """
    Plots the average PESQ and ViSQOL scores for each language across different SNR levels.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Define noise types and SNR levels
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    languages = list(data.keys())
    
    # Prepare data for plotting
    def collect_average_scores(metric):
        averages_by_language = {lang: {snr: [] for snr in snr_levels} for lang in languages}
        
        for language in languages:
            audio_file = list(data[language].keys())[0]
            for snr in snr_levels:
                scores = []
                for noise_type in noise_types:
                    try:
                        scores.append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
                averages_by_language[language][snr] = np.mean(scores) if scores else np.nan
        
        return averages_by_language
    
    pesq_averages_by_language = collect_average_scores('PESQ')
    visqol_averages_by_language = collect_average_scores('ViSQOL')
    
    # Plot PESQ scores
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for language, averages in pesq_averages_by_language.items():
        snr_values = list(averages.keys())
        avg_scores = list(averages.values())
        ax.plot(snr_values, avg_scores, marker='o', label=language.capitalize())
    
    ax.set_xlabel('SNR Level (dB)')
    ax.set_ylabel('Average PESQ Score')
    ax.set_title('Average PESQ Scores by SNR Level')
    ax.set_ylim(1, 5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save the PESQ plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'average_pesq_scores_by_snr.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot ViSQOL scores
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for language, averages in visqol_averages_by_language.items():
        snr_values = list(averages.keys())
        avg_scores = list(averages.values())
        ax.plot(snr_values, avg_scores, marker='o', label=language.capitalize())
    
    ax.set_xlabel('SNR Level (dB)')
    ax.set_ylabel('Average ViSQOL Score')
    ax.set_title('Average ViSQOL Scores by SNR Level')
    ax.set_ylim(1, 5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save the ViSQOL plot
    plt.savefig(os.path.join(output_dir, 'average_visqol_scores_by_snr.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_median_scores_by_snr_level(data, output_dir='plots'):
    """
    Plots the median PESQ and ViSQOL scores for each language across different SNR levels.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Define noise types and SNR levels
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    languages = list(data.keys())
    
    # Prepare data for plotting
    def collect_median_scores(metric):
        medians_by_language = {lang: {snr: [] for snr in snr_levels} for lang in languages}
        
        for language in languages:
            audio_file = list(data[language].keys())[0]
            for snr in snr_levels:
                scores = []
                for noise_type in noise_types:
                    try:
                        scores.append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
                medians_by_language[language][snr] = np.median(scores) if scores else np.nan
        
        return medians_by_language
    
    pesq_medians_by_language = collect_median_scores('PESQ')
    visqol_medians_by_language = collect_median_scores('ViSQOL')
    
    # Plot PESQ scores
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for language, medians in pesq_medians_by_language.items():
        snr_values = list(medians.keys())
        median_scores = list(medians.values())
        ax.plot(snr_values, median_scores, marker='o', label=language.capitalize())
    
    ax.set_xlabel('SNR Level (dB)')
    ax.set_ylabel('Median PESQ Score')
    ax.set_title('Median PESQ Scores by SNR Level')
    ax.set_ylim(1, 5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save the PESQ plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'median_pesq_scores_by_snr.png'), bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot ViSQOL scores
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for language, medians in visqol_medians_by_language.items():
        snr_values = list(medians.keys())
        median_scores = list(medians.values())
        ax.plot(snr_values, median_scores, marker='o', label=language.capitalize())
    
    ax.set_xlabel('SNR Level (dB)')
    ax.set_ylabel('Median ViSQOL Score')
    ax.set_title('Median ViSQOL Scores by SNR Level')
    ax.set_ylim(1, 5)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Save the ViSQOL plot
    plt.savefig(os.path.join(output_dir, 'median_visqol_scores_by_snr.png'), bbox_inches='tight', dpi=300)
    plt.close()

def plot_gender_based_scatter(data, output_dir='plots'):
    """
    Creates scatter plots for PESQ and ViSQOL scores for each language, coloring points by gender.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Define SNR levels
    snr_levels = ['-25', '-20', '-15', '-10', '-5', '0', '5', '10', '15', '20', '25', '30', '35', '40']
    languages = list(data.keys())
    
    # Define colors for gender
    gender_colors = {'_F_': 'red', '_M_': 'blue'}
    
    # Create plots for each language
    for language in languages:
        # Prepare data for plotting
        pesq_scores = {snr: {'_F_': [], '_M_': []} for snr in snr_levels}
        visqol_scores = {snr: {'_F_': [], '_M_': []} for snr in snr_levels}
        
        for audio_file, noise_data in data[language].items():
            gender = '_F_' if '_F_' in audio_file else '_M_' if '_M_' in audio_file else None
            if not gender:
                continue
            
            for noise_type, snr_data in noise_data.items():
                for snr, metrics in snr_data.items():
                    if snr in snr_levels:
                        pesq_scores[snr][gender].append(metrics['PESQ'])
                        visqol_scores[snr][gender].append(metrics['ViSQOL'])
        
        # Plot PESQ scores
        fig, ax = plt.subplots(figsize=(10, 6))
        for gender, color in gender_colors.items():
            snr_values = []
            scores = []
            for snr in snr_levels:
                snr_values.extend([float(snr)] * len(pesq_scores[snr][gender]))  # Convert to float
                scores.extend(pesq_scores[snr][gender])
            ax.scatter(snr_values, scores, color=color, label=f'{"Female" if gender == "_F_" else "Male"}', alpha=0.6)
            
            # Fit a cubic curve
            if len(snr_values) > 3:  # Ensure there are enough points to fit a curve
                coeffs = np.polyfit(snr_values, scores, 3)
                snr_sorted = np.sort(snr_values)
                cubic_trend_line = np.polyval(coeffs, snr_sorted)
                ax.plot(snr_sorted, cubic_trend_line, color=color, linestyle='-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('SNR Level (dB)')
        ax.set_ylabel('PESQ Score')
        ax.set_title(f'PESQ Scores by Gender for {language.capitalize()}')
        ax.set_ylim(1, 5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save the PESQ plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{language}_pesq_gender_scatter.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        # Plot ViSQOL scores
        fig, ax = plt.subplots(figsize=(10, 6))
        for gender, color in gender_colors.items():
            snr_values = []
            scores = []
            for snr in snr_levels:
                snr_values.extend([float(snr)] * len(visqol_scores[snr][gender]))  # Convert to float
                scores.extend(visqol_scores[snr][gender])
            ax.scatter(snr_values, scores, color=color, label=f'{"Female" if gender == "_F_" else "Male"}', alpha=0.6)
            
            # Fit a cubic curve
            if len(snr_values) > 3:  # Ensure there are enough points to fit a curve
                coeffs = np.polyfit(snr_values, scores, 3)
                snr_sorted = np.sort(snr_values)
                cubic_trend_line = np.polyval(coeffs, snr_sorted)
                ax.plot(snr_sorted, cubic_trend_line, color=color, linestyle='-', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('SNR Level (dB)')
        ax.set_ylabel('ViSQOL Score')
        ax.set_title(f'ViSQOL Scores by Gender for {language.capitalize()}')
        ax.set_ylim(1, 5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save the ViSQOL plot
        plt.savefig(os.path.join(output_dir, f'{language}_visqol_gender_scatter.png'), bbox_inches='tight', dpi=300)
        plt.close()

from sklearn.linear_model import LinearRegression

def plot_metric_correlation_by_gender(data, output_dir='plots'):
    """
    Creates scatter plots for PESQ vs ViSQOL scores for each language, coloring points by gender,
    and includes a cubic polynomial trend line for each gender.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    languages = list(data.keys())
    
    # Define colors for gender
    gender_colors = {'_F_': 'red', '_M_': 'blue'}
    
    # Create plots for each language
    for language in languages:
        # Prepare data for plotting
        pesq_scores = {'_F_': [], '_M_': []}
        visqol_scores = {'_F_': [], '_M_': []}
        
        for audio_file, noise_data in data[language].items():
            gender = '_F_' if '_F_' in audio_file else '_M_' if '_M_' in audio_file else None
            if not gender:
                continue
            
            for noise_type, snr_data in noise_data.items():
                for snr, metrics in snr_data.items():
                    pesq_scores[gender].append(metrics['PESQ'])
                    visqol_scores[gender].append(metrics['ViSQOL'])
        
        # Plot PESQ vs ViSQOL scores
        fig, ax = plt.subplots(figsize=(10, 6))
        for gender, color in gender_colors.items():
            x = np.array(visqol_scores[gender])
            y = np.array(pesq_scores[gender])
            ax.scatter(x, y, color=color, label=f'{"Female" if gender == "_F_" else "Male"}', alpha=0.6)
            
            # Fit a cubic polynomial to the data for each gender
            if len(x) > 1:  # Ensure there are enough points to fit a curve
                coeffs = np.polyfit(x, y, 3)
                cubic_trend_line = np.polyval(coeffs, x)
                # Sort the x values for a smooth line
                sorted_indices = np.argsort(x)
                ax.plot(x[sorted_indices], cubic_trend_line[sorted_indices], color=color, linestyle='-', linewidth=2)
        
        # Add a thin dotted black line from bottom-left to top-right
        ax.plot([1, 5], [1, 5], linestyle=':', color='black', linewidth=0.5)
        
        ax.set_xlabel('ViSQOL Score')
        ax.set_ylabel('PESQ Score')
        ax.set_title(f'PESQ vs ViSQOL Scores by Gender for {language.capitalize()}')
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save the plot
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{language}_metric_correlation_gender_scatter.png'), bbox_inches='tight', dpi=300)
        plt.close()


def plot_pesq_visqol_overlay(data, output_dir='plots'):
    """
    Creates overlay plots of PESQ and ViSQOL scores across different SNR levels for each language.
    
    Args:
        data (dict): Dictionary of languages containing analysis results
        output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for better visualization
    plt.style.use('default')
    
    # Define noise types and their colors
    noise_colors = [
        ('blue_noise', 'blue'),
        ('pink_noise', 'pink'),
        ('noisy_crowd', 'red')
    ]
    
    # Process each language
    for language in data.keys():
        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()  # Create second y-axis for ViSQOL
        
        # Get the first (and only) audio file for this language
        audio_file = list(data[language].keys())[0]
        
        # Process data for each noise type
        for noise_type, color in noise_colors:
            if noise_type not in data[language][audio_file]:
                continue
                
            # Extract SNR levels and scores
            snr_levels = []
            pesq_scores = []
            visqol_scores = []
            
            for snr in data[language][audio_file][noise_type].keys():
                snr_levels.append(float(snr))
                pesq_scores.append(data[language][audio_file][noise_type][snr]['PESQ'])
                visqol_scores.append(data[language][audio_file][noise_type][snr]['ViSQOL'])
            
            # Sort by SNR levels
            snr_scores = sorted(zip(snr_levels, pesq_scores, visqol_scores))
            snr_levels, pesq_scores, visqol_scores = zip(*snr_scores)
            
            # Plot PESQ scores on first y-axis
            line1 = ax1.plot(snr_levels, pesq_scores, marker='o', 
                            label=f'PESQ ({noise_type})',
                            color=color, linestyle='-', linewidth=2, markersize=6)
            
            # Plot ViSQOL scores on second y-axis
            line2 = ax2.plot(snr_levels, visqol_scores, marker='s',
                            label=f'ViSQOL ({noise_type})',
                            color=color, linestyle='--', linewidth=2, markersize=6)
            
        ax1.set_ylim(1, 5)
        ax2.set_ylim(1, 5)
        
        # Customize the plot
        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('PESQ Score', fontsize=12, color='darkblue')
        ax2.set_ylabel('ViSQOL Score', fontsize=12, color='darkred')
        
        # Set title with language
        plt.title(f'PESQ and ViSQOL Scores vs SNR - {language.upper()}', 
                 fontsize=14, pad=20)
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f'metrics_overlay_{language}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


def metrics_overlay_average(data, output_dir='plots'):
    """
    Creates overlay plots of averaged PESQ and ViSQOL scores across different SNR levels for each language.
    The scores are averaged across all noise types for each SNR level.
    
    Args:
        data (dict): Dictionary of languages containing analysis results
        output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for better visualization
    plt.style.use('default')
    
    # Define noise types
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    
    # Process each language
    for language in data.keys():
        # Create figure and axis
        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax2 = ax1.twinx()  # Create second y-axis for ViSQOL
        
        # Get the first (and only) audio file for this language
        audio_file = list(data[language].keys())[0]
        
        # Initialize dictionaries to store values for averaging
        snr_pesq = {}
        snr_visqol = {}
        
        # Collect all values for each SNR level
        for noise_type in noise_types:
            if noise_type not in data[language][audio_file]:
                continue
                
            for snr in data[language][audio_file][noise_type].keys():
                if snr not in snr_pesq:
                    snr_pesq[snr] = []
                    snr_visqol[snr] = []
                
                snr_pesq[snr].append(data[language][audio_file][noise_type][snr]['PESQ'])
                snr_visqol[snr].append(data[language][audio_file][noise_type][snr]['ViSQOL'])
        
        # Calculate averages and prepare for plotting
        snr_levels = []
        pesq_averages = []
        visqol_averages = []
        
        for snr in sorted(snr_pesq.keys(), key=float):
            snr_levels.append(float(snr))
            pesq_averages.append(np.mean(snr_pesq[snr]))
            visqol_averages.append(np.mean(snr_visqol[snr]))
        
        # Plot average PESQ scores on first y-axis
        line1 = ax1.plot(snr_levels, pesq_averages, marker='o', 
                        label='PESQ (average)',
                        color='darkblue', linestyle='-', linewidth=2, markersize=6)
        
        # Plot average ViSQOL scores on second y-axis
        line2 = ax2.plot(snr_levels, visqol_averages, marker='s',
                        label='ViSQOL (average)',
                        color='darkred', linestyle='--', linewidth=2, markersize=6)
        
        # Set y-axis limits from 1 to 5
        ax1.set_ylim(1, 5)
        ax2.set_ylim(1, 5)
        
        # Customize the plot
        ax1.set_xlabel('SNR (dB)', fontsize=12)
        ax1.set_ylabel('PESQ Score', fontsize=12, color='darkblue')
        ax2.set_ylabel('ViSQOL Score', fontsize=12, color='darkred')
        
        # Set title with language
        plt.title(f'Average PESQ and ViSQOL Scores vs SNR - {language.upper()}', 
                 fontsize=14, pad=20)
        
        # Add grid
        ax1.grid(True, linestyle='--', alpha=0.3)
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f'metrics_overlay_average_{language}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

import os
import numpy as np
import matplotlib.pyplot as plt

def plot_scores_by_noise_type(data, output_dir='plots'):
    """
    Creates one plot per noise type, showing two lines for each language
    (PESQ and ViSQOL) aggregated at each SNR level.
    
    Args:
        data (dict): Dictionary of languages containing analysis results
        output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define noise types and their colors for better visualization
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    
    # Process each noise type separately
    for noise_type in noise_types:
        # Create figure and axis
        plt.figure(figsize=(12, 7))
        
        # Iterate over languages to plot PESQ and ViSQOL lines
        for language, language_data in data.items():
            snr_pesq = {}
            snr_visqol = {}
            
            # Aggregate data by SNR for the current noise type
            for audio_file, audio_data in language_data.items():
                if noise_type in audio_data:
                    for snr, scores in audio_data[noise_type].items():
                        if snr not in snr_pesq:
                            snr_pesq[snr] = []
                            snr_visqol[snr] = []
                        
                        snr_pesq[snr].append(scores['PESQ'])
                        snr_visqol[snr].append(scores['ViSQOL'])
            
            # Prepare data for plotting
            snr_levels = []
            pesq_averages = []
            visqol_averages = []
            
            for snr in sorted(snr_pesq.keys(), key=float):
                snr_levels.append(float(snr))
                pesq_averages.append(np.mean(snr_pesq[snr]))
                visqol_averages.append(np.mean(snr_visqol[snr]))
            
            # Plot PESQ and ViSQOL lines for the current language
            plt.plot(snr_levels, pesq_averages, label=f'{language} - PESQ', marker='o', linestyle='-')
            plt.plot(snr_levels, visqol_averages, label=f'{language} - ViSQOL', marker='s', linestyle='--')
        
        # Customize the plot
        plt.title(f'PESQ and ViSQOL Scores by Language vs SNR ({noise_type.replace("_", " ").title()})', fontsize=16)
        plt.xlabel('SNR (dB)', fontsize=14)
        plt.ylabel('Score', fontsize=14)
        plt.ylim(1, 5)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='upper left', fontsize=10, title='Languages')
        
        # Save the plot
        output_path = os.path.join(output_dir, f'metrics_by_language_{noise_type}.png')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


def compute_score_differences_for_languages(data, lang1, lang2, snr_range=list(range(0,31))):
    """
    Computes the differences between PESQ and ViSQOL scores for two specified languages
    across a given SNR range.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results.
        lang1 (str): The first language.
        lang2 (str): The second language.
        snr_range (list): A list of SNR levels to consider (e.g., ['-20', '0', '5']).
    
    Returns:
        list: A list of differences in scores for each SNR level for the two languages.
              Each element is a tuple of (snr, difference).
    """
    # Define noise types
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    
    # Helper function to collect average scores for a given metric (PESQ or ViSQOL)
    def collect_average_scores(metric):
        averages_by_language = {lang: {snr: [] for snr in snr_range} for lang in [lang1, lang2]}
        
        for language in [lang1, lang2]:
            audio_file = list(data[language].keys())[0]  # Assuming one audio file per language
            for snr in snr_range:
                scores = []
                for noise_type in noise_types:
                    try:
                        scores.append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
                averages_by_language[language][snr] = np.mean(scores) if scores else np.nan
        
        return averages_by_language
    
    # Collect average PESQ and ViSQOL scores for the specified languages and SNR range
    pesq_averages_by_language = collect_average_scores('PESQ')
    visqol_averages_by_language = collect_average_scores('ViSQOL')
    
    # Compute the differences between PESQ and ViSQOL for the two languages and given SNR range
    differences = []
    
    for snr in snr_range:
        pesq_lang1 = pesq_averages_by_language[lang1][snr]
        pesq_lang2 = pesq_averages_by_language[lang2][snr]
        visqol_lang1 = visqol_averages_by_language[lang1][snr]
        visqol_lang2 = visqol_averages_by_language[lang2][snr]
        
        # Compute the difference (PESQ - ViSQOL) for both languages
        if not np.isnan(pesq_lang1) and not np.isnan(visqol_lang1) and not np.isnan(pesq_lang2) and not np.isnan(visqol_lang2):
            diff_lang1 = pesq_lang1 - visqol_lang1
            diff_lang2 = pesq_lang2 - visqol_lang2
            difference = (snr, diff_lang1, diff_lang2)
            differences.append(difference)
        else:
            differences.append((snr, np.nan, np.nan))
    
    return differences