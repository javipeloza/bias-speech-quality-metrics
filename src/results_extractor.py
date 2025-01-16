import json
import os
from results_logger import plot_language_metric_differences, t_test_language_bias, post_hoc_noise_type_comparison, three_way_anova, tukey_hsd_analysis_with_bonferroni, perform_anova_analysis, two_way_anova, plot_score_distributions_all_metrics, perform_one_way_anova, plot_metrics_by_language, plot_metric_correlation, plot_score_distribution_boxplots, plot_score_distributions_boxplots_by_noise, plot_score_density_distributions, plot_score_heatmaps, plot_average_scores_by_snr, plot_radar_charts, plot_pca_clusters, plot_score_differences, plot_relative_performance_ratios, conduct_statistical_tests, perform_pairwise_tests, compute_effect_sizes, plot_average_scores_by_snr_level, plot_median_scores_by_snr_level, plot_gender_based_scatter, plot_metric_correlation_by_gender, plot_pesq_visqol_overlay, metrics_overlay_average, plot_scores_by_noise_type, compute_score_differences_for_languages

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
        # Metric average/median type plots
        # plot_average_scores_by_snr(results)
        plot_average_scores_by_snr_level(results)
        # plot_median_scores_by_snr_level(results)
        # metrics_overlay_average(results)
        # plot_language_metric_differences(results)     # TBD (not ready)


        # Degradation type plots
        # plot_metrics_by_language(results)
        # plot_scores_by_noise_type(results)
        # plot_pesq_visqol_overlay(results)
        # plot_metric_correlation(results)
        

        # Distribution type plots
        # plot_score_distribution_boxplots(results)
        # plot_score_distributions_all_metrics(results)
        # plot_score_distributions_boxplots_by_noise(results)


        # Density type plots
        # plot_score_density_distributions(results)
        # plot_score_heatmaps(results)
        # plot_radar_charts(results)


        # Gender type plots
        # plot_gender_based_scatter(results)
        # plot_metric_correlation_by_gender(results)


        # Comparison plots
        # plot_score_differences(results)    # Prints avg. differences by noise & snr
        # plot_relative_performance_ratios(results, ['turkish', 'korean'])   # Only compared against english (Don't understand)
        # differences = compute_score_differences_for_languages(results, 'english', 'turkish')     # TBD


        # Statistical analysis plots
        # perform_one_way_anova(results)
        # two_way_anova(results)
        # three_way_anova(results)
        # perform_anova_analysis(results)

        # tukey_hsd_analysis_with_bonferroni(results)  # Only does visqol
        # post_hoc_noise_type_comparison(results)  # Only does visqol (results for pesq were all false)
        # perform_pairwise_tests(results, [('english', 'turkish'), ('english', 'korean'), ('turkish', 'korean')])
        # t_test_language_bias(results)
        # mann_whitney_language_bias(results)
        
        # TBD: Understand what this does
        # compute_effect_sizes(results, [('english', 'turkish'), ('english', 'korean'), ('turkish', 'korean')])
        
        # TBD (Check and understand what Kruskal Wallis test means)
        # conduct_statistical_tests(results)
        # plot_pca_clusters(results)
        
        print("Plots have been saved in the 'plots' directory.")
