from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import numpy as np
from results_logger import plot_post_hoc_results, plot_pairwise_language_comparison, plot_comprehensive_language_bias_analysis
from itertools import combinations

def perform_post_hoc_tests(aggregated_by_metric):
    post_hoc_results = {}

    for metric, languages in aggregated_by_metric.items():
        scores = []
        labels = []

        # Collect scores and labels for each language under the current metric
        for language, score_list in languages.items():
            scores.extend(score_list)
            labels.extend([language] * len(score_list))

        scores = np.array(scores)
        labels = np.array(labels)

        # Perform Tukey's HSD test
        tukey_result = pairwise_tukeyhsd(scores, labels, alpha=0.05)

        # Store the results
        post_hoc_results[metric] = tukey_result.summary()

    # Save post-hoc results to a text file
    plot_post_hoc_results(post_hoc_results)

    return post_hoc_results


def pairwise_language_comparison(aggregated_by_metric):
    # Check if there are any metrics to compare
    if not aggregated_by_metric:
        return {}

    # Use the first metric as a reference for language keys
    first_metric = list(aggregated_by_metric.keys())[0]
    language_pairs = list(combinations(aggregated_by_metric[first_metric].keys(), 2))
    
    pairwise_comparison_results = {}
    
    for metric in aggregated_by_metric:
        pairwise_comparison_results[metric] = {}
        
        for lang1, lang2 in language_pairs:
            scores1 = aggregated_by_metric[metric][lang1]
            scores2 = aggregated_by_metric[metric][lang2]
            
            # Only perform f_oneway if both score lists have more than one element
            if len(scores1) > 1 and len(scores2) > 1:
                f_stat, p_value = f_oneway(scores1, scores2)
                
                pairwise_comparison_results[metric][(lang1, lang2)] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant_difference': p_value < 0.05
                }
            else:
                pairwise_comparison_results[metric][(lang1, lang2)] = {
                    'f_statistic': None,
                    'p_value': None,
                    'significant_difference': False,
                    'reason': 'Insufficient data for statistical test'
                }
    
    plot_pairwise_language_comparison(pairwise_comparison_results)
    
    return pairwise_comparison_results


def comprehensive_language_bias_analysis(aggregated_by_metric):
    results = {}
    
    for metric in aggregated_by_metric:
        # Prepare data
        language_scores = list(aggregated_by_metric[metric].values())
        language_names = list(aggregated_by_metric[metric].keys())
        
        # One-way ANOVA
        f_stat, p_value = f_oneway(*language_scores)
        
        # Effect size (Eta-squared)
        total_ss = np.sum((np.concatenate(language_scores) - 
                           np.mean(np.concatenate(language_scores)))**2)
        between_ss = sum(len(scores) * (np.mean(scores) - 
                         np.mean(np.concatenate(language_scores)))**2 
                         for scores in language_scores)
        eta_squared = between_ss / total_ss
        
        results[metric] = {
            'languages': language_names,
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant_bias': p_value < 0.05
        }
        
		# Print results to terminal
        print(f"\nComprehensive Language Bias Analysis for Metric: {metric}")
        print(f"Languages Compared: {', '.join(language_names)}")
        print(f"F-Statistic: {f_stat:.4f}")
        print(f"P-Value: {p_value:.4f}")
        print(f"Eta-Squared (Effect Size): {eta_squared:.4f}")
        print(f"Significant Bias Detected: {'Yes' if p_value < 0.05 else 'No'}")
        
    plot_comprehensive_language_bias_analysis(results)
    
    return results
