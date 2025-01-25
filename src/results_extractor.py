import os
from results_logger import plot_average_scores_by_snr_level, plot_metrics_by_language_and_degradation_type, plot_metrics_by_degradation_type, plot_average_scores_by_degradation_type, plot_metric_correlation_by_language_and_degradation_type, plot_score_distribution_boxplots, save_score_distributions_by_metric, plot_score_distribution_boxplots_by_noise, plot_score_density_distributions, plot_score_density_violin, plot_score_heatmaps, plot_radar_charts, plot_metric_correlation_scatter_by_language_and_gender, plot_metric_correlation_by_gender, plot_average_scores_by_language_gender_and_metric, plot_scores_by_metric_and_gender
from statistical_analyzer import compute_score_differences_for_languages, two_way_anova, three_way_anova, perform_pairwise_statistical_tests, perform_pairwise_statistical_tests_by_noise, perform_groupwise_statistical_tests, compare_turkish_male_against_others, calculate_overall_deviation_metrics
from file_manager import extract_json_data, load_and_convert_json_results_by_gender

if __name__ == '__main__':
    results_path = os.path.join('results', 'analysis_results.json')
    results = extract_json_data(results_path)
    
    if results:
        # Plots
        plot_average_scores_by_snr_level(results)
        plot_metrics_by_language_and_degradation_type(results)
        plot_metrics_by_degradation_type(results)
        plot_average_scores_by_degradation_type(results)
        plot_metric_correlation_by_language_and_degradation_type(results)
        plot_score_distribution_boxplots(results)
        save_score_distributions_by_metric(results)
        plot_score_distribution_boxplots_by_noise(results)
        plot_score_density_distributions(results)
        plot_score_density_violin(results)
        plot_score_heatmaps(results)
        plot_radar_charts(results)
        plot_metric_correlation_scatter_by_language_and_gender(results)
        plot_metric_correlation_by_gender(results)
        plot_scores_by_metric_and_gender(results)
        plot_average_scores_by_language_gender_and_metric(results)

        # Statistical Analysis
        differences = compute_score_differences_for_languages(results, 'english', 'turkish')
        two_way_anova(results)
        three_way_anova(results)
        perform_pairwise_statistical_tests(results)
        perform_pairwise_statistical_tests_by_noise(results)
        perform_groupwise_statistical_tests(results)

        # Distribution Deviation Analysis
        correlated_scores = load_and_convert_json_results_by_gender("aggregated_scores_by_gender.json")
        compare_turkish_male_against_others(correlated_scores, correlated_scores["Turkish Male"])
        calculate_overall_deviation_metrics(correlated_scores)
        
        print("Plots have been saved in the 'plots' directory.")
