import matplotlib.pyplot as plt
import os
import json
from analyzer import AudioQualityAnalyzer
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from scipy import interpolate
import matplotlib.lines as mlines
import pandas as pd

noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
snr_levels = [str(x) for x in np.arange(-25, 41, 5)]

# ----------------------- Saving results -----------------------------------------------------

def log_analyzer_results(analyzer, file_path):
    """Log results from AudioQualityAnalyzer"""
    with open(file_path, 'a') as file:
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

# ----------------------- Plotting results -----------------------------------------------------

def plot_average_scores_by_snr_level(data, output_dir='plots', txt_filename='average_pesq_visqol_by_snr.txt'):
    """
    Plots the average PESQ and ViSQOL scores for each language across different SNR levels,
    and saves the average values for each language and each SNR to a text file.
    Additionally, creates an overlay plot comparing PESQ and ViSQOL scores.
    """
    plt.style.use('default')
    languages = list(data.keys())
    
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
    
    pesq_averages_by_language = collect_average_scores( 'PESQ')
    visqol_averages_by_language = collect_average_scores('ViSQOL')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the average values to a text file
    with open(os.path.join(output_dir, txt_filename), 'w') as f:
        f.write("Average Scores by Language and SNR Level\n")
        f.write(f"{'Language':<12} {'SNR Level':<12} {'PESQ Average':<15} {'ViSQOL Average':<15}\n")
        f.write("-" * 54 + "\n")
        
        for language in languages:
            for snr in snr_levels:
                pesq_score = pesq_averages_by_language[language].get(snr, np.nan)
                visqol_score = visqol_averages_by_language[language].get(snr, np.nan)
                f.write(f"{language:<12} {snr:<12} {pesq_score:<15.3f} {visqol_score:<15.3f}\n")
    
    # Plot PESQ and ViSQOL scores separately
    for metric, averages_by_language, ylabel, filename in [
        ('PESQ', pesq_averages_by_language, 'Average PESQ Score', 'average_pesq_scores_by_snr.png'),
        ('ViSQOL', visqol_averages_by_language, 'Average ViSQOL Score', 'average_visqol_scores_by_snr.png')
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for language, averages in averages_by_language.items():
            snr_values = list(averages.keys())
            avg_scores = list(averages.values())
            ax.plot(snr_values, avg_scores, marker='o', label=language.capitalize())
        ax.set_xlabel('SNR Level (dB)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Average {metric} Scores by SNR Level')
        ax.set_ylim(1, 5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()
    
    # Define color mapping for languages
    language_colors = {
        'english': 'blue',
        'turkish': 'red',
        'korean': 'green'
    }

    # Create overlay plot comparing PESQ and ViSQOL
    fig, ax = plt.subplots(figsize=(10, 6))

    for language in languages:
        snr_values = list(pesq_averages_by_language[language].keys())
        pesq_scores = list(pesq_averages_by_language[language].values())
        visqol_scores = list(visqol_averages_by_language[language].values())

        color = language_colors.get(language.lower(), 'black')  # Default to black for unknown languages

        ax.plot(snr_values, pesq_scores, marker='o', linestyle='-', color=color, label=f'{language.capitalize()} - PESQ')
        ax.plot(snr_values, visqol_scores, marker='s', linestyle='--', color=color, label=f'{language.capitalize()} - ViSQOL')

    # Apply font styles to match LaTeX document
    # plt.rc('font', family='Times New Roman')  # Match LaTeX's times package

    ax.set_xlabel('Signal-to-Noise Ratio (dB)', fontsize=18)
    ax.set_ylabel('Score', fontsize=18)
    ax.set_title('Average PESQ and ViSQOL Scores by Language', fontsize=20, pad=10)

    ax.tick_params(axis='both', which='major', labelsize=12)  # Tick labels
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_ylim(1, 5)

    # Update legend styling
    ax.legend(loc='upper left', fontsize=12, title='Language - Metric', title_fontsize=12, frameon=True)

    # Save figure
    plt.savefig(os.path.join(output_dir, 'average_pesq_visqol_by_snr.svg'), bbox_inches='tight', format='svg')
    plt.savefig(os.path.join(output_dir, 'average_pesq_visqol_by_snr.png'), bbox_inches='tight', dpi=300)

def plot_metrics_by_language_and_degradation_type(data, output_dir='plots'):
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
        plt.savefig(os.path.join(output_dir, f'{language}_by_degradation_type.svg'), 
                       bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'{language}_by_degradation_type.png'), 
                       bbox_inches='tight', dpi=300)
        plt.close()

def plot_metrics_by_degradation_type(data, output_dir='plots'):
    """
    Creates one plot per noise type, showing two lines for each language
    (PESQ and ViSQOL) aggregated at each SNR level.
    
    Args:
        data (dict): Dictionary of languages containing analysis results
        output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
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
        output_path_png = os.path.join(output_dir, f'metrics_by_language_{noise_type}.png')
        output_path_svg = os.path.join(output_dir, f'metrics_by_language_{noise_type}.svg')
        plt.tight_layout()
        plt.savefig(output_path_png, bbox_inches='tight', dpi=300)
        plt.savefig(output_path_svg, bbox_inches='tight')
        plt.close()

def plot_metric_correlation_by_language_and_degradation_type(data, output_dir='plots'):
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
        plt.savefig(os.path.join(output_dir, f'{language}_metric_correlation_by_degradation_type.png'), 
                   bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, f'{language}_metric_correlation_by_degradation_type.svg'), 
                   bbox_inches='tight')
        plt.close()

def plot_average_scores_by_degradation_type(data, output_dir='plots'):
    """
    Creates a single plot overlaying results for different noise types,
    with two lines per noise type (one for PESQ and one for ViSQOL),
    aggregated at each SNR level across languages.
    Also saves the y-values (scores) for each line in a text file.

    Args:
        data (dict): Dictionary of languages containing analysis results
        output_dir (str): Directory to save the plots and text files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define noise types and their colors for better visualization
    noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
    colors = {'blue_noise': 'b', 'pink_noise': 'm', 'noisy_crowd': 'g'}
    
    # Rename "noisy_crowd" to "babble noise" for legend
    noise_labels = {'blue_noise': 'Blue Noise', 
                    'pink_noise': 'Pink Noise', 
                    'noisy_crowd': 'Babble Noise'}

    plt.figure(figsize=(12, 7))
    
    txt_output_path = os.path.join(output_dir, 'metrics_by_noise_type.txt')
    with open(txt_output_path, 'w') as txt_file:
        for noise_type in noise_types:
            snr_pesq = {}
            snr_visqol = {}
            
            # Aggregate data by SNR across all languages for the current noise type
            for language, language_data in data.items():
                for audio_file, audio_data in language_data.items():
                    if noise_type in audio_data:
                        for snr, scores in audio_data[noise_type].items():
                            if snr not in snr_pesq:
                                snr_pesq[snr] = []
                                snr_visqol[snr] = []
                            
                            snr_pesq[snr].append(scores['PESQ'])
                            snr_visqol[snr].append(scores['ViSQOL'])
            
            # Prepare data for plotting
            snr_levels = sorted(snr_pesq.keys(), key=float)
            pesq_averages = [np.mean(snr_pesq[snr]) for snr in snr_levels]
            visqol_averages = [np.mean(snr_visqol[snr]) for snr in snr_levels]
            
            # Write y-values to text file
            txt_file.write(f'{noise_labels[noise_type]} - PESQ: {list(map(float, pesq_averages))}\n')
            txt_file.write(f'{noise_labels[noise_type]} - ViSQOL: {list(map(float, visqol_averages))}\n\n')
            
            # Plot PESQ and ViSQOL lines for the current noise type
            plt.plot(snr_levels, pesq_averages, label=f'{noise_labels[noise_type]} - PESQ', 
                     marker='o', linestyle='-', color=colors[noise_type])
            plt.plot(snr_levels, visqol_averages, label=f'{noise_labels[noise_type]} - ViSQOL', 
                     marker='s', linestyle='--', color=colors[noise_type])
    
    # Customize the plot
    plt.title('Average PESQ and ViSQOL Scores by Degradation Type', fontsize=20, pad=15)
    plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    plt.ylim(1, 5)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=14, title='Degradation Type - Metric')

    # Save the plot
    output_path = os.path.join(output_dir, 'metrics_by_noise_type.svg')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def plot_score_distribution_boxplots(data, output_dir='plots', txt_filename='boxplot_statistics.txt'):
    """
    Creates boxplots of PESQ and ViSQOL scores for each language across all noise types and SNR levels,
    and saves the summary statistics including Q1 and Q3 in a text file.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
        txt_filename (str): Name of the text file to save the statistics (default: 'boxplot_statistics.txt')
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
                pesq_scores[language].append(data[language][audio_file][noise_type][snr]['PESQ'])
                visqol_scores[language].append(data[language][audio_file][noise_type][snr]['ViSQOL'])

    # Prepare a file to write the statistics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, txt_filename), 'w') as f:
        f.write("Language\tMetric\tMin\tQ1\tMedian\tQ3\tMax\tMean\n")

        # Create two subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot PESQ scores
        box_pesq = ax1.boxplot([pesq_scores[lang] for lang in pesq_scores.keys()],
                               labels=[lang.capitalize() for lang in pesq_scores.keys()],
                               whis=1.0, 
                               flierprops=dict(marker='', color='none'),
                               boxprops=dict(color='black', linewidth=1),
                               whiskerprops=dict(color='black', linewidth=1),  # Change whisker color and width
                               capprops=dict(color='black', linewidth=2))  # Change cap color and width
        ax1.set_ylabel('PESQ Scores', fontsize=18)
        ax1.set_title('PESQ Score Distribution by Language', fontsize=20, pad=15)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.set_ylim(1, 5)
        ax1.tick_params(axis='x', labelsize=18)  # Set language labels to 18pt

        # Keep the x-axis line, but remove the vertical lines at the x-tick positions
        ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax1.xaxis.set_tick_params(width=0)  # Removes the vertical tick marks

        # Prepare to create custom lines for the legend
        mean_line = mlines.Line2D([], [], color='blue', linestyle='--', linewidth=2, label='Mean')
        median_line = mlines.Line2D([], [], color='blue', linestyle='-', linewidth=2, label='Median')

        for i, lang in enumerate(pesq_scores.keys(), 1):
            # Collect statistics for PESQ
            scores = np.array(pesq_scores[lang])
            min_val = np.min(scores)
            q1_val = np.percentile(scores, 25)
            median_val = np.median(scores)
            q3_val = np.percentile(scores, 75)
            max_val = np.max(scores)
            mean_val = np.mean(scores)
            f.write(f"{lang.capitalize()}\tPESQ\t{min_val:.2f}\t{q1_val:.2f}\t{median_val:.2f}\t{q3_val:.2f}\t{max_val:.2f}\t{mean_val:.2f}\n")

            # Set the median line color to blue
            box_pesq['medians'][i - 1].set_color('blue')

            # Add a dashed blue mean line inside the box (from Q1 to Q3)
            mean_x = box_pesq['medians'][i - 1].get_xdata()  # Get x-coordinates of median line
            ax1.plot(mean_x, [mean_val, mean_val], linestyle='--', color='blue', linewidth=2)

        # Add the legend for PESQ plot
        ax1.legend(handles=[mean_line, median_line], loc='upper left', fontsize=14)

        # Plot ViSQOL scores
        box_visqol = ax2.boxplot([visqol_scores[lang] for lang in visqol_scores.keys()],
                                 labels=[lang.capitalize() for lang in visqol_scores.keys()],
                                 whis=1.0, 
                                 flierprops=dict(marker='', color='none'),
                                 boxprops=dict(color='black', linewidth=1),
                                 whiskerprops=dict(color='black', linewidth=1),  # Change whisker color and width
                                 capprops=dict(color='black', linewidth=2))  # Change cap color and width
        ax2.set_ylabel('ViSQOL Scores', fontsize=18)
        ax2.set_title('ViSQOL Score Distribution by Language', fontsize=20, pad=15)
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.set_ylim(1, 5)
        ax2.tick_params(axis='x', labelsize=18)  # Set language labels to 18pt

        # Keep the x-axis line, but remove the vertical lines at the x-tick positions
        ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)
        ax2.xaxis.set_tick_params(width=0)  # Removes the vertical tick marks

        for i, lang in enumerate(visqol_scores.keys(), 1):
            # Collect statistics for ViSQOL
            scores = np.array(visqol_scores[lang])
            min_val = np.min(scores)
            q1_val = np.percentile(scores, 25)
            median_val = np.median(scores)
            q3_val = np.percentile(scores, 75)
            max_val = np.max(scores)
            mean_val = np.mean(scores)

            # Clip the minimum and maximum whiskers to stay within 1 and 5
            min_val = max(min_val, 1)
            max_val = min(max_val, 5)

            f.write(f"{lang.capitalize()}\tViSQOL\t{min_val:.2f}\t{q1_val:.2f}\t{median_val:.2f}\t{q3_val:.2f}\t{max_val:.2f}\t{mean_val:.2f}\n")

            # Set the median line color to blue
            box_visqol['medians'][i - 1].set_color('blue')

            # Add a dashed blue mean line inside the box (from Q1 to Q3)
            mean_x = box_visqol['medians'][i - 1].get_xdata()  # Get x-coordinates of median line
            ax2.plot(mean_x, [mean_val, mean_val], linestyle='--', color='blue', linewidth=2)

            # Clip whiskers to not extend beyond 1 and 5
            whiskers = box_visqol['whiskers'][i - 1:i + 1]
            for whisker in whiskers:
                whisker.set_ydata([max(whisker.get_ydata()[0], 1), min(whisker.get_ydata()[1], 5)])

        # Add the legend for ViSQOL plot
        # ax2.legend(handles=[mean_line, median_line], loc='upper left', fontsize=14)

        # Adjust spacing and set the main title
        fig.suptitle('Score Distributions Across Languages', fontsize=22)
        fig.tight_layout(rect=[0, 0, 1, 1])  # Reduced white space under the title

        # Save the plot
        plt.savefig(os.path.join(output_dir, 'score_distributions.png'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, 'score_distributions.svg'), bbox_inches='tight')
        plt.close()

def save_score_distributions_by_metric(data, output_dir='plots', txt_filename='score_distributions_by_metric.txt'):
    """
    Creates a vertical boxplot for each metric (PESQ and ViSQOL), with the mean, median, Q1, Q3, and min/max 
    statistics saved to a PNG plot and a text file.

    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
        txt_filename (str): Name of the text file to save the statistics (default: 'all_metrics_statistics.txt')
    """
    plt.style.use('default')

    # Initialize lists to store PESQ and ViSQOL scores
    pesq_scores = []
    visqol_scores = []

    # Collect all PESQ and ViSQOL scores (no language segmentation)
    for language in data.keys():
        audio_file = list(data[language].keys())[0]
        for noise_type in data[language][audio_file].keys():
            for snr in data[language][audio_file][noise_type].keys():
                pesq_scores.append(data[language][audio_file][noise_type][snr]['PESQ'])
                visqol_scores.append(data[language][audio_file][noise_type][snr]['ViSQOL'])

    # Calculate statistics for PESQ
    pesq_min = np.min(pesq_scores)
    pesq_q1 = np.percentile(pesq_scores, 25)
    pesq_median = np.median(pesq_scores)
    pesq_q3 = np.percentile(pesq_scores, 75)
    pesq_max = np.max(pesq_scores)
    pesq_mean = np.mean(pesq_scores)

    # Calculate statistics for ViSQOL
    visqol_min = np.min(visqol_scores)
    visqol_q1 = np.percentile(visqol_scores, 25)
    visqol_median = np.median(visqol_scores)
    visqol_q3 = np.percentile(visqol_scores, 75)
    visqol_max = np.max(visqol_scores)
    visqol_mean = np.mean(visqol_scores)

    # Prepare a file to write the statistics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, txt_filename), 'w') as f:
        f.write("Metric\tMin\tQ1\tMedian\tQ3\tMax\tMean\n")

        # Write PESQ statistics
        f.write(f"PESQ\t{pesq_min:.2f}\t{pesq_q1:.2f}\t{pesq_median:.2f}\t{pesq_q3:.2f}\t{pesq_max:.2f}\t{pesq_mean:.2f}\n")

        # Write ViSQOL statistics
        f.write(f"ViSQOL\t{visqol_min:.2f}\t{visqol_q1:.2f}\t{visqol_median:.2f}\t{visqol_q3:.2f}\t{visqol_max:.2f}\t{visqol_mean:.2f}\n")

def plot_score_distribution_boxplots_by_noise(data, output_dir='plots', txt_filename='boxplot_statistics_by_noise.txt'):
    """
    Creates boxplots of PESQ and ViSQOL scores for each language, separated by noise type,
    and saves the summary statistics (mean, median, Q1, Q3, min, max) for each noise type and language.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
        txt_filename (str): Name of the text file to save the statistics (default: 'boxplot_statistics_by_noise.txt')
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
    
    # Prepare a file to write the statistics
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, txt_filename), 'w') as f:
        f.write("Noise Type\tLanguage\tMetric\tMin\tQ1\tMedian\tQ3\tMax\tMean\n")
        
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
                
                # Collect statistics for PESQ and write to file
                scores = np.array(pesq_scores[noise_type][lang])
                min_val = np.min(scores)
                q1_val = np.percentile(scores, 25)
                median_val = np.median(scores)
                q3_val = np.percentile(scores, 75)
                max_val = np.max(scores)
                mean_val = np.mean(scores)
                f.write(f"{noise_type.capitalize()}\t{lang.capitalize()}\tPESQ\t{min_val:.2f}\t{q1_val:.2f}\t{median_val:.2f}\t{q3_val:.2f}\t{max_val:.2f}\t{mean_val:.2f}\n")
        
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
                
                # Collect statistics for ViSQOL and write to file
                scores = np.array(visqol_scores[noise_type][lang])
                min_val = np.min(scores)
                q1_val = np.percentile(scores, 25)
                median_val = np.median(scores)
                q3_val = np.percentile(scores, 75)
                max_val = np.max(scores)
                mean_val = np.mean(scores)
                f.write(f"{noise_type.capitalize()}\t{lang.capitalize()}\tViSQOL\t{min_val:.2f}\t{q1_val:.2f}\t{median_val:.2f}\t{q3_val:.2f}\t{max_val:.2f}\t{mean_val:.2f}\n")

        # Add a main title
        fig.suptitle('Score Distributions by Noise Type', fontsize=14, y=0.95)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'score_distributions_by_noise.png'),
                    bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, 'score_distributions_by_noise.svg'),
                    bbox_inches='tight')
        plt.close()

def plot_score_density_distributions(data, output_dir='plots'):
    """
    Creates KDE plots of PESQ and ViSQOL scores for each language and analyzes curve characteristics.
    Includes vertical lines for local minima and maxima points.
    
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
    colors = plt.cm.Set3(np.linspace(0, 1, len(data.keys())))
    
    # Initialize text file for curve analysis
    os.makedirs(output_dir, exist_ok=True)
    analysis_file = os.path.join(output_dir, 'curve_analysis.txt')
    
    with open(analysis_file, 'w') as f:
        f.write("Score Distribution Curve Analysis\n")
        f.write("================================\n\n")
        
        # Process both PESQ and ViSQOL scores
        for score_type, scores_dict, ax in [
            ("PESQ", pesq_scores, ax1),
            ("ViSQOL", visqol_scores, ax2)
        ]:
            f.write(f"\n{score_type} Analysis\n")
            f.write("-" * (len(score_type) + 9) + "\n\n")
            
            for (language, scores), color in zip(scores_dict.items(), colors):
                # Calculate KDE manually
                scores_array = np.array(scores)
                kde = gaussian_kde(scores_array)
                
                # x_data remains fixed between 1 and 5, but avoid blank space on the left
                x_data = np.linspace(1, 5, 200)
                y_data = kde(x_data)
                
                # Create KDE plot
                ax.plot(x_data, y_data, color=color, label=language.capitalize())
                ax.fill_between(x_data, y_data, alpha=0.3, color=color)
                
                # Find local maxima and minima
                max_indices = argrelextrema(y_data, np.greater)[0]
                min_indices = argrelextrema(y_data, np.less)[0]
                
                # Add vertical lines for maxima points
                for idx in max_indices:
                    ax.vlines(x=x_data[idx], ymin=0, ymax=y_data[idx], 
                            color=color, linestyle='--', linewidth=0.8, alpha=0.7)
                    # Add a small dot at the maximum point
                    ax.plot(x_data[idx], y_data[idx], 'o', color=color, markersize=4)
                
                # Add vertical lines for minima points
                for idx in min_indices:
                    ax.vlines(x=x_data[idx], ymin=0, ymax=y_data[idx], 
                            color=color, linestyle=':', linewidth=0.8, alpha=0.7)
                    # Add a small dot at the minimum point
                    ax.plot(x_data[idx], y_data[idx], 's', color=color, markersize=4)
                
                # Calculate inflection points using second derivative
                spline = interpolate.UnivariateSpline(x_data, y_data, s=0.1)
                x_smooth = np.linspace(min(x_data), max(x_data), 1000)
                y_smooth = spline(x_smooth)
                
                # Calculate second derivative
                y_smooth_2d = spline.derivative(n=2)(x_smooth)
                inflection_indices = np.where(np.diff(np.signbit(y_smooth_2d)))[0]
                
                # Write analysis to file
                f.write(f"\nLanguage: {language.capitalize()}\n")
                
                # Write local maxima
                f.write("Local Maxima (x, y):\n")
                for idx in max_indices:
                    f.write(f"  ({x_data[idx]:.3f}, {y_data[idx]:.3f})\n")
                
                # Write local minima
                f.write("Local Minima (x, y):\n")
                for idx in min_indices:
                    f.write(f"  ({x_data[idx]:.3f}, {y_data[idx]:.3f})\n")
                
                # Write inflection points
                f.write("Inflection Points (x, y):\n")
                for idx in inflection_indices:
                    f.write(f"  ({x_smooth[idx]:.3f}, {y_smooth[idx]:.3f})\n")
                
                f.write("\n")
            
            # Set up axis labels and titles
            ax.set_xlabel(f'{score_type} Score')
            ax.set_ylabel('Density')
            ax.set_title(f'{score_type} Score Distribution by Language')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.set_xlim(1, 5)  # Fix x-axis range to 1-5
            
            # Dynamically adjust the y-axis limits based on the max of the y_data
            ax.set_ylim(bottom=0)  # Ensures the y-axis starts from 0
            ax.set_ylim(top=max(y_data) * 1.1)  # Optionally, increase the y-limit slightly to avoid clipping
            
            ax.legend()
    
    # Add a main title
    fig.suptitle('Score Density Distributions Across Languages', fontsize=14, y=1.05)
    
    # Add a small legend for the markers
    legend_elements = [
        plt.Line2D([0], [0], color='gray', linestyle='--', label='Local Maximum'),
        plt.Line2D([0], [0], color='gray', linestyle=':', label='Local Minimum'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'score_density_distributions.png'),
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'score_density_distributions.svg'),
                bbox_inches='tight')
    plt.close()

def plot_score_density_violin(data, output_dir='plots'):
    """
    Creates violin plots of PESQ and ViSQOL scores for each language, with the metrics
    shown as two sides of each violin plot.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    # Prepare data for plotting
    plot_data = {
        'Language': [],
        'Score': [],
        'Metric': []
    }
    
    # Collect scores for each language
    for language in data.keys():
        audio_file = list(data[language].keys())[0]  # Corrected here
        for noise_type in data[language][audio_file].keys():
            for snr in data[language][audio_file][noise_type].keys():
                # Add PESQ score
                plot_data['Language'].append(language.capitalize())
                plot_data['Score'].append(data[language][audio_file][noise_type][snr]['PESQ'])
                plot_data['Metric'].append('PESQ')
                
                # Add ViSQOL score
                plot_data['Language'].append(language.capitalize())
                plot_data['Score'].append(data[language][audio_file][noise_type][snr]['ViSQOL'])
                plot_data['Metric'].append('ViSQOL')
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(plot_data)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Create violin plot with color per metric (PESQ and ViSQOL)
    sns.violinplot(data=df, x='Language', y='Score', hue='Metric', split=True,
                  inner='box',  # Show box plot inside violin
                  palette=['lightblue', 'lightgreen'])
    
    # Customize the plot appearance
    plt.title('PESQ and ViSQOL Score Distributions by Language', fontsize=20, pad=20)
    plt.xlabel('', fontsize=18)
    plt.ylabel('Score', fontsize=18)
    
    # Set y-axis limits to match the original plot
    plt.ylim(1, 5)
    
    # Remove x-axis tick vertical lines below each violin plot
    plt.tick_params(axis='x', which='both', bottom=False)

    plt.xticks(fontsize=18)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust legend
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Tight layout to prevent label clipping
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'score_density_violin.png'),
                bbox_inches='tight',
                dpi=300)
    plt.savefig(os.path.join(output_dir, 'score_density_violin.svg'),
                bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir, 'score_heatmaps.svg'),
                bbox_inches='tight')
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
    plt.savefig(os.path.join(output_dir, 'radar_charts.svg'),
                bbox_inches='tight')
    plt.close()

def plot_scores_by_metric_and_gender(data, output_dir='plots'):
    """
    Creates scatter plots for PESQ and ViSQOL scores for each language, coloring points by gender.
    
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
        plt.savefig(os.path.join(output_dir, f'{language}_pesq_gender_scores.png'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, f'{language}_pesq_gender_scores.svg'), bbox_inches='tight')
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
        plt.savefig(os.path.join(output_dir, f'{language}_visqol_gender_scores.svg'), bbox_inches='tight')
        plt.close()

def plot_metric_correlation_scatter_by_language_and_gender(data, output_dir='plots'):
    """
    Creates scatter plots for PESQ vs ViSQOL scores for each language, coloring points by gender,
    and includes a cubic polynomial trend line for each gender.
    Also saves the PESQ and ViSQOL scores as tuples in a text file.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')
    
    languages = list(data.keys())
    
    # Define colors for gender
    gender_colors = {'_F_': 'red', '_M_': 'blue'}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output text file
    txt_output_path = os.path.join(output_dir, 'metric_correlation_by_gender.txt')
    with open(txt_output_path, 'w') as txt_file:
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
            
            # Write PESQ and ViSQOL tuples to text file
            female_tuples = list(zip(pesq_scores['_F_'], visqol_scores['_F_']))
            male_tuples = list(zip(pesq_scores['_M_'], visqol_scores['_M_']))
            txt_file.write(f'{language} - Female: {female_tuples}\n')
            txt_file.write(f'{language} - Male: {male_tuples}\n\n')
            
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
            plt.savefig(os.path.join(output_dir, f'{language}_metric_correlation_by_gender.png'), bbox_inches='tight', dpi=300)
            plt.savefig(os.path.join(output_dir, f'{language}_metric_correlation_by_gender.svg'), bbox_inches='tight')
            plt.close()

def plot_metric_correlation_by_gender(data, output_dir='plots'):
    """
    Creates a single line plot overlaying PESQ vs ViSQOL scores for each language,
    with separate lines for male and female speakers, using cubic polynomial trend lines.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results
        output_dir (str): Directory where plots will be saved (default: 'plots')
    """
    plt.style.use('default')

    # Define colors for each language
    language_colors = {'english': 'blue', 'turkish': 'red', 'korean': 'green'}
    gender_styles = {'_F_': '-', '_M_': '--'}  # Solid line for female, dashed for male

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define output text file
    txt_output_path = os.path.join(output_dir, 'language_specific_metric_correlation.txt')

    with open(txt_output_path, 'w') as txt_file:
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        for language, color in language_colors.items():
            pesq_scores = {'_F_': [], '_M_': []}
            visqol_scores = {'_F_': [], '_M_': []}

            # Gather data for the current language
            if language in data:
                for audio_file, noise_data in data[language].items():
                    gender = '_F_' if '_F_' in audio_file else '_M_' if '_M_' in audio_file else None
                    if not gender:
                        continue

                    for noise_type, snr_data in noise_data.items():
                        for snr, metrics in snr_data.items():
                            pesq_scores[gender].append(metrics['PESQ'])
                            visqol_scores[gender].append(metrics['ViSQOL'])

            # Write PESQ and ViSQOL tuples to text file
            female_tuples = list(zip(pesq_scores['_F_'], visqol_scores['_F_']))
            male_tuples = list(zip(pesq_scores['_M_'], visqol_scores['_M_']))
            txt_file.write(f'{language.capitalize()} - Female: {female_tuples}\n')
            txt_file.write(f'{language.capitalize()} - Male: {male_tuples}\n\n')

            # Plot PESQ vs ViSQOL scores for both genders in this language
            for gender, linestyle in gender_styles.items():
                x = np.array(visqol_scores[gender])
                y = np.array(pesq_scores[gender])

                # Fit a cubic polynomial to the data for each gender
                if len(x) > 1:  # Ensure there are enough points to fit a curve
                    coeffs = np.polyfit(x, y, 3)
                    cubic_trend_line = np.polyval(coeffs, np.sort(x))
                    ax.plot(np.sort(x), cubic_trend_line, color=color, linestyle=linestyle, linewidth=2,
                            label=f'{language.capitalize()} {"Female" if gender == "_F_" else "Male"}')

        # Add a thin dotted black line from bottom-left to top-right
        ax.plot([1, 5], [1, 5], linestyle=':', color='black', linewidth=0.5)

        # Customize the plot
        ax.set_xlabel('ViSQOL Score', fontsize=18)
        ax.set_ylabel('PESQ Score', fontsize=18)
        ax.set_title('PESQ and ViSQOL Scores by Language and Gender', fontsize=20, pad=15)
        ax.set_xlim(1, 5)
        ax.set_ylim(1, 5)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=14)

        # Save the plot
        plt.savefig(os.path.join(output_dir, 'metric_correlation_by_gender.png'), bbox_inches='tight', dpi=300)
        plt.savefig(os.path.join(output_dir, 'metric_correlation_by_gender.svg'), bbox_inches='tight')
        plt.close()

def plot_average_scores_by_language_gender_and_metric(data, output_dir='plots'):
    """
    Creates a single comprehensive plot of PESQ and ViSQOL scores across languages and genders.
    """
    plt.style.use('default')
    
    def collect_average_scores(metric):
        averages_by_language_gender = {}
        
        for language in data.keys():
            averages_by_language_gender[language] = {gender: {snr: [] for snr in snr_levels} for gender in ['male', 'female']}
            
            for audio_file, audio_data in data[language].items():
                # Determine gender based on _M_ or _F_ in filename
                gender = 'male' if '_M_' in audio_file else 'female'
                
                for snr in snr_levels:
                    scores = []
                    for noise_type in noise_types:
                        try:
                            scores.append(audio_data[noise_type][snr][metric])
                        except KeyError:
                            continue
                    
                    averages_by_language_gender[language][gender][snr] = np.mean(scores) if scores else np.nan
        
        return averages_by_language_gender
    
    # Collect averages
    pesq_averages = collect_average_scores('PESQ')
    visqol_averages = collect_average_scores('ViSQOL')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define comprehensive color and marker mapping
    color_map = {
        'english_male_PESQ': 'blue', 'english_female_PESQ': 'lightblue',
        'english_male_ViSQOL': 'darkblue', 'english_female_ViSQOL': 'lightskyblue',
        'turkish_male_PESQ': 'red', 'turkish_female_PESQ': 'lightcoral',
        'turkish_male_ViSQOL': 'darkred', 'turkish_female_ViSQOL': 'indianred',
        'korean_male_PESQ': 'green', 'korean_female_PESQ': 'lightgreen',
        'korean_male_ViSQOL': 'darkgreen', 'korean_female_ViSQOL': 'mediumseagreen'
    }
    
    # Prepare plot
    plt.figure(figsize=(15, 8))
    
    # Plot each language, gender, and metric combination
    for language in pesq_averages.keys():
        for gender in ['male', 'female']:
            # Plot PESQ (continuous line)
            snr_values = list(pesq_averages[language][gender].keys())
            pesq_scores = list(pesq_averages[language][gender].values())
            color = color_map[f'{language}_{gender}_PESQ']
            marker = 'o' if gender == 'male' else 's'
            plt.plot(snr_values, pesq_scores, marker=marker, linestyle='-', 
                     color=color, label=f'{language.capitalize()} {gender.capitalize()} - PESQ')
            
            # Plot ViSQOL (dotted line)
            visqol_scores = list(visqol_averages[language][gender].values())
            color = color_map[f'{language}_{gender}_ViSQOL']
            plt.plot(snr_values, visqol_scores, marker=marker, linestyle=':', 
                     color=color, label=f'{language.capitalize()} {gender.capitalize()} - ViSQOL')
    
    plt.xlabel('SNR Level (dB)', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('PESQ and ViSQOL Scores by Language, Gender, and SNR', fontsize=14)
    plt.ylim(1, 5)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'average_scores_by_language_gender_metric.png'), bbox_inches='tight',dpi=300)
    plt.savefig(os.path.join(output_dir, 'average_scores_by_language_gender_metric.svg'), bbox_inches='tight')
    plt.close()
