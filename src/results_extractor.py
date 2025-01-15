import json
import os
from results_logger import plot_metrics_by_language, plot_metric_correlation, plot_score_distribution_boxplots, plot_score_distributions_by_noise, plot_score_density_distributions, plot_score_heatmaps, plot_average_scores_by_snr, plot_radar_charts, plot_pca_clusters, plot_score_differences, plot_relative_performance_ratios, conduct_statistical_tests, perform_pairwise_tests, compute_effect_sizes, plot_average_scores_by_snr_level, plot_median_scores_by_snr_level, plot_gender_based_scatter, plot_metric_correlation_by_gender

def extract_json_data(file_path):
    """
    Extracts and returns the JSON data from a specified file path
    
    Args:
        file_path (str): Path to the JSON file to be read
        
    Returns:
        dict: The parsed JSON data from the file
    """
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: Could not find {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {file_path}")
        return None


if __name__ == '__main__':
    results_path = os.path.join('results', 'analysis_results.json')
    results = extract_json_data(results_path)
    
    if results:
        # plot_metrics_by_language(results)
        plot_metric_correlation(results)
        # plot_score_distribution_boxplots(results)
        # plot_score_distributions_by_noise(results)
        # plot_score_density_distributions(results)
        # plot_score_heatmaps(results)
        # plot_average_scores_by_snr(results)
        # plot_radar_charts(results)
        # plot_pca_clusters(results)
        # plot_score_differences(results)
        # plot_relative_performance_ratios(results, ['turkish', 'korean', 'spanish', 'chinese'])
        # conduct_statistical_tests(results)
        # perform_pairwise_tests(results, [('english', 'turkish'), ('english', 'korean'), ('english', 'chinese'), ('english', 'spanish'), ('turkish', 'korean')])
        # compute_effect_sizes(results, [('english', 'turkish'), ('english', 'korean'), ('english', 'chinese'), ('english', 'spanish'), ('turkish', 'korean')])
        # plot_average_scores_by_snr_level(results)
        # plot_median_scores_by_snr_level(results)
        # plot_gender_based_scatter(results)
        plot_metric_correlation_by_gender(results)
        print("Plots have been saved in the 'plots' directory.")
