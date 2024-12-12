from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
import numpy as np
from results_logger import plot_post_hoc_results
from itertools import combinations
from scipy import stats

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

    # Display the results
    for metric, result in post_hoc_results.items():
        print(f"Post-hoc test results for {metric}:")
        print(result)
        print()

    return post_hoc_results


def analyze_language_bias_by_categorial_grouping(aggregated_by_metric):
    # Group languages into "popular" and "less popular" categories
    popular_languages = ['english']
    less_popular_languages = ['korean', 'turkish', 'spanish', 'chinese']

    for metric in aggregated_by_metric:
        popular_scores = [
            score for lang, score in aggregated_by_metric[metric].items() 
            if lang in popular_languages
        ]
        less_popular_scores = [
            score for lang, score in aggregated_by_metric[metric].items() 
            if lang in less_popular_languages
        ]

        f_stat, p_value = f_oneway(popular_scores, less_popular_scores)
        
        print(f"Metric {metric} Language Bias Test:")
        print(f"F-statistic: {f_stat}")
        print(f"p-value: {p_value}")
        print("Significant bias" if p_value < 0.05 else "No significant bias")
        
def pairwise_language_comparison(aggregated_by_metric):
    language_pairs = list(combinations(aggregated_by_metric['pesq'].keys(), 2))
    
    for metric in aggregated_by_metric:
        print(f"Metric: {metric}")
        for lang1, lang2 in language_pairs:
            scores1 = aggregated_by_metric[metric][lang1]
            scores2 = aggregated_by_metric[metric][lang2]
            
            f_stat, p_value = f_oneway(scores1, scores2)
            
            print(f"{lang1} vs {lang2}:")
            print(f"F-statistic: {f_stat}")
            print(f"p-value: {p_value}")
            print("Significant difference" if p_value < 0.05 else "No significant difference")
            
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
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'significant_bias': p_value < 0.05
        }
    
    return results
