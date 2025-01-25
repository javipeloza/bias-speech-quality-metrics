import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.stats import f_oneway, kruskal, ttest_ind, mannwhitneyu, ks_2samp, ranksums, spearmanr, wilcoxon, friedmanchisquare, pearsonr
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import json

noise_types = ['blue_noise', 'pink_noise', 'noisy_crowd']
snr_levels = [str(x) for x in np.arange(-25, 41, 5)]

def compute_score_differences_for_languages(data, lang1, lang2):
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
    # Helper function to collect average scores for a given metric (PESQ or ViSQOL)
    def collect_average_scores(metric):
        averages_by_language = {lang: {snr: [] for snr in snr_levels} for lang in [lang1, lang2]}
        
        for language in [lang1, lang2]:
            audio_file = list(data[language].keys())[0]  # Assuming one audio file per language
            for snr in snr_levels:
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
    
    for snr in snr_levels:
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

def two_way_anova(data, output_dir='plots', txt_filename='two_way_anova_statistics.txt'):
    """
    Perform a two-way ANOVA test on PESQ and ViSQOL scores based on language and noise type as independent variables.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results.
        output_dir (str): Directory where results will be saved (default: 'anova_results').
        txt_filename (str): Name of the text file to save the ANOVA statistics (default: 'two_way_anova_statistics.txt').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare the dataset for ANOVA
    anova_data = {'Language': [], 'Noise Type': [], 'PESQ': [], 'ViSQOL': []}
    
    # Collect scores for PESQ and ViSQOL, while also storing language and noise type
    for language in data.keys():
        audio_file = list(data[language].keys())[0]
        for noise_type in data[language][audio_file].keys():
            for snr in data[language][audio_file][noise_type].keys():
                anova_data['Language'].append(language)
                anova_data['Noise Type'].append(noise_type)
                anova_data['PESQ'].append(data[language][audio_file][noise_type][snr]['PESQ'])
                anova_data['ViSQOL'].append(data[language][audio_file][noise_type][snr]['ViSQOL'])
    
    # Convert the data to a pandas DataFrame
    df = pd.DataFrame(anova_data)
    
    # Calculate average scores for each language and noise type
    avg_pesq = df.groupby('Language')['PESQ'].mean()
    avg_visqol = df.groupby('Language')['ViSQOL'].mean()
    
    # Perform two-way ANOVA for PESQ
    pesq_results = stats.f_oneway(
        *[df[df['Language'] == lang]['PESQ'] for lang in df['Language'].unique()]
    )
    
    # Perform two-way ANOVA for ViSQOL
    visqol_results = stats.f_oneway(
        *[df[df['Noise Type'] == noise]['ViSQOL'] for noise in df['Noise Type'].unique()]
    )
    
    # Format numbers without scientific notation
    f_stat_pesq = f"{pesq_results.statistic:.4f}"
    p_value_pesq = f"{pesq_results.pvalue:.4f}"
    
    f_stat_visqol = f"{visqol_results.statistic:.4f}"
    p_value_visqol = f"{visqol_results.pvalue:.4f}"
    
    # Determine significance
    significance_threshold = 0.1
    significant_pesq = "Yes" if float(p_value_pesq) < significance_threshold else "No"
    significant_visqol = "Yes" if float(p_value_visqol) < significance_threshold else "No"
    
    # Prepare the results text
    results_text = []
    results_text.append("ANOVA Analysis Results")
    results_text.append("=" * 23 + "\n")
    
    # Average PESQ scores by language
    results_text.append("Average PESQ Scores by Language:")
    for lang, avg_score in avg_pesq.items():
        results_text.append(f"  {lang.lower()}: {avg_score:.4f}")
    
    # Average ViSQOL scores by language
    results_text.append("\nAverage ViSQOL Scores by Language:")
    for lang, avg_score in avg_visqol.items():
        results_text.append(f"  {lang.lower()}: {avg_score:.4f}")
    
    # PESQ ANOVA results
    results_text.append("\nPESQ ANOVA Results:")
    results_text.append(f"  F-statistic: {f_stat_pesq}")
    results_text.append(f"  p-value: {p_value_pesq}")
    results_text.append(f"  Significance threshold: {significance_threshold}")
    results_text.append(f"  Significant: {significant_pesq}")
    
    # ViSQOL ANOVA results
    results_text.append("\nViSQOL ANOVA Results:")
    results_text.append(f"  F-statistic: {f_stat_visqol}")
    results_text.append(f"  p-value: {p_value_visqol}")
    results_text.append(f"  Significance threshold: {significance_threshold}")
    results_text.append(f"  Significant: {significant_visqol}")
    
    # Save to text file
    with open(os.path.join(output_dir, txt_filename), 'w') as f:
        f.write("\n".join(results_text))
    
    # Return results for possible further use or verification
    return pesq_results, visqol_results

def three_way_anova(data, output_dir='plots', txt_filename="three_way_anova_results.txt"):
    """
    Performs a three-way ANOVA on the given data with factors Language, Noise_Type, and Gender (male or female).
    The gender is inferred from the filenames, where male files contain '_M_' and female files contain '_F_'.
    
    Args:
        data (dict): Parsed JSON data containing the analysis results.
        noise_types (list): List of noise types to analyze (e.g., 'blue_noise', 'pink_noise', 'noisy_crowd').
        output_filename (str): Name of the output text file to save the three-way ANOVA results (default: "plots/three_way_anova_results.txt").
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Prepare the data for the analysis
    rows = []
    for lang, lang_data in data.items():
        for audio_file, audio_data in lang_data.items():
            for noise_type in noise_types:
                if noise_type in audio_data:  # Ensure the noise type exists in the current language's data
                    # Extract gender from the filename
                    gender = 'Male' if '_M_' in audio_file else 'Female'
                    
                    for snr, snr_data in audio_data[noise_type].items():
                        rows.append({
                            'Language': lang,
                            'Noise_Type': noise_type,
                            'Gender': gender,
                            'ViSQOL_Score': snr_data['ViSQOL']
                        })

    # Create DataFrame from the rows
    df = pd.DataFrame(rows)
    
    # Perform three-way ANOVA using statsmodels
    model = ols('ViSQOL_Score ~ C(Language) * C(Noise_Type) * C(Gender)', data=df).fit()
    anova_results = anova_lm(model)

    # Set pandas display options to use float format
    pd.set_option('display.float_format', '{:0.2f}'.format)

    # Save to text file
    with open(os.path.join(output_dir, txt_filename), 'w') as f:
        f.write("Three-Way ANOVA Results (Language, Noise Type, Gender)\n")
        f.write("======================================================\n")
        f.write(anova_results.to_string(float_format='{:0.2f}'.format))

def perform_groupwise_statistical_tests(data, output_dir='plots'):
    """
    Performs ANOVA and Kruskal-Wallis tests to compare multiple languages.
    Saves all results to a single text file.

    Args:
        data (dict): Parsed JSON data containing the analysis results.
        output_dir (str): Directory where results will be saved (default: 'results').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define noise types, SNR levels, and languages
    languages = list(data.keys())

    def collect_scores(metric):
        scores_by_language = {lang: [] for lang in languages}
        
        for language in languages:
            audio_file = list(data[language].keys())[0]
            for noise_type in noise_types:
                sorted_snr_levels = sorted(snr_levels, key=lambda x: float(x))
                for snr in sorted_snr_levels:
                    try:
                        scores_by_language[language].append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
        return scores_by_language

    # Collect PESQ and ViSQOL scores
    pesq_scores_by_language = collect_scores('PESQ')
    visqol_scores_by_language = collect_scores('ViSQOL')

    # Open a single output file for all results
    output_file_path = os.path.join(output_dir, "groupwise_statistical_test_results.txt")
    with open(output_file_path, "w") as f:
        f.write("Groupwise Statistical Tests Results\n")
        f.write("=" * 70 + "\n\n")

        def perform_tests(scores_by_language, metric_name, test_name, test_func):
            f.write(f"{test_name} Results\n")
            f.write("=" * 70 + "\n")

            # Collect data for all groups
            groups = [scores_by_language[lang] for lang in languages if lang in scores_by_language and scores_by_language[lang]]
            if len(groups) < 2:
                f.write("Not enough data for statistical testing.\n")
                f.write("-" * 70 + "\n\n")
                return

            test_result = test_func(*groups)
            stat, pvalue = test_result.statistic, test_result.pvalue

            f.write(f"{metric_name}:\n")
            f.write(f"Statistic: {stat:.2f}, p-value: {pvalue:.2f}\n")
            f.write("-" * 70 + "\n\n")

        # Perform ANOVA
        perform_tests(pesq_scores_by_language, "PESQ", "ANOVA", lambda *args: f_oneway(*args))
        perform_tests(visqol_scores_by_language, "ViSQOL", "ANOVA", lambda *args: f_oneway(*args))

        # Perform Kruskal-Wallis Test
        perform_tests(pesq_scores_by_language, "PESQ", "Kruskal-Wallis Test", lambda *args: kruskal(*args))
        perform_tests(visqol_scores_by_language, "ViSQOL", "Kruskal-Wallis Test", lambda *args: kruskal(*args))

def perform_pairwise_statistical_tests(data, output_dir='plots'):
    """
    Performs various pairwise statistical tests including T-tests, Mann-Whitney U tests, 
    Kolmogorov-Smirnov tests, Wilcoxon Rank Sum tests, and Spearman correlation.
    Saves all results to a single text file.

    Args:
        data (dict): Parsed JSON data containing the analysis results.
        output_dir (str): Directory where results will be saved (default: 'plots').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    languages_to_compare = [('english', 'turkish'), ('english', 'korean'), ('turkish', 'korean')]

    def collect_scores_by_language(metric):
        scores_by_language = {lang: [] for lang in data.keys()}
        
        for language in data.keys():
            audio_file = list(data[language].keys())[0]
            for noise_type in noise_types:
                sorted_snr_levels = sorted(snr_levels, key=lambda x: float(x))  # Numeric sorting
                for snr in sorted_snr_levels:
                    try:
                        scores_by_language[language].append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
        return scores_by_language

    pesq_scores_by_language = collect_scores_by_language('PESQ')
    visqol_scores_by_language = collect_scores_by_language('ViSQOL')

    output_file_path = os.path.join(output_dir, "pairwise_statistical_test_results.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("Pairwise Statistical Tests Results\n")
        f.write("=" * 70 + "\n\n")

        def perform_tests(scores_by_language, metric_name, test_name, test_func):
            """
            Generic function to perform statistical tests and log results.

            Args:
                scores_by_language (dict): Dictionary mapping languages to their scores.
                metric_name (str): Name of the metric (e.g., PESQ or ViSQOL).
                test_name (str): Name of the test (e.g., T-Test, Spearman Correlation).
                test_func (function): Statistical test function to apply.
            """
            f.write(f"{test_name} Results\n")
            f.write("=" * 70 + "\n")

            for lang1, lang2 in languages_to_compare:
                if lang1 not in scores_by_language or lang2 not in scores_by_language:
                    continue
                
                scores1 = scores_by_language[lang1]
                scores2 = scores_by_language[lang2]

                if not scores1 or not scores2:
                    continue  # Skip if no data available

                test_result = test_func(scores1, scores2)

                if test_name == "Spearman Rank Correlation":
                    # Spearman correlation returns a different object
                    stat, pvalue = test_result.statistic, test_result.pvalue
                elif isinstance(test_result, tuple):  # For functions returning (stat, p-value)
                    stat, pvalue = test_result
                else:  # For other cases like scipy test objects
                    stat, pvalue = test_result.statistic, test_result.pvalue

                f.write(f"{metric_name} - {lang1.capitalize()} vs {lang2.capitalize()}:\n")
                f.write(f"Statistic: {stat:.2f}, p-value: {pvalue:.2f}\n")
                f.write("-" * 70 + "\n")
            
            f.write("\n")

        # Perform Statistical Tests

        # Parametric
        perform_tests(pesq_scores_by_language, "PESQ", "T-Test", lambda x, y: ttest_ind(x, y, equal_var=False))
        perform_tests(visqol_scores_by_language, "ViSQOL", "T-Test", lambda x, y: ttest_ind(x, y, equal_var=False))

        # Non-Parametric
        perform_tests(pesq_scores_by_language, "PESQ", "Mann-Whitney U Test", lambda x, y: mannwhitneyu(x, y, alternative='two-sided'))
        perform_tests(visqol_scores_by_language, "ViSQOL", "Mann-Whitney U Test", lambda x, y: mannwhitneyu(x, y, alternative='two-sided'))

        perform_tests(pesq_scores_by_language, "PESQ", "Kolmogorov-Smirnov Test", lambda x, y: ks_2samp(x, y))
        perform_tests(visqol_scores_by_language, "ViSQOL", "Kolmogorov-Smirnov Test", lambda x, y: ks_2samp(x, y))
        
        perform_tests(pesq_scores_by_language, "PESQ", "Wilcoxon Rank Sum Test", lambda x, y: ranksums(x, y, alternative='two-sided'))
        perform_tests(visqol_scores_by_language, "ViSQOL", "Wilcoxon Rank Sum Test", lambda x, y: ranksums(x, y, alternative='two-sided'))

        perform_tests(pesq_scores_by_language, "PESQ", "Spearman Rank Correlation", lambda x, y: spearmanr(x, y))
        perform_tests(visqol_scores_by_language, "ViSQOL", "Spearman Rank Correlation", lambda x, y: spearmanr(x, y))

        # Add Friedman Test
        f.write("Friedman Test Results\n")
        f.write("=" * 70 + "\n")
        
        for metric_name, scores_by_language in [('PESQ', pesq_scores_by_language), 
                                                ('ViSQOL', visqol_scores_by_language)]:
            # Prepare data for Friedman test: each row represents a block, columns are treatments
            friedman_data = []
            for language in data.keys():
                if language not in scores_by_language:
                    continue
                language_scores = scores_by_language[language]
                # Reshape scores into a 2D array
                if language_scores:
                    friedman_data.append(language_scores[:len(snr_levels)])
            
            if len(friedman_data) >= 2:  # Friedman test requires at least two blocks
                stat, pvalue = friedmanchisquare(*friedman_data)
                f.write(f"{metric_name} Friedman Test:\n")
                f.write(f"Statistic: {stat:.2f}, p-value: {pvalue:.2f}\n")
                f.write("-" * 70 + "\n")

def perform_pairwise_statistical_tests_by_noise(data, output_dir='plots'):
    """
    Performs pairwise statistical tests comparing noise types across languages.
    Saves results to a single text file.

    Args:
        data (dict): Parsed JSON data containing the analysis results.
        output_dir (str): Directory where results will be saved (default: 'plots').
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    languages = list(data.keys())

    def collect_scores_by_noise_type(metric):
        scores_by_noise_type = {lang: {noise: [] for noise in noise_types} for lang in languages}
        
        for language in languages:
            audio_file = list(data[language].keys())[0]
            for noise_type in noise_types:
                sorted_snr_levels = sorted(snr_levels, key=lambda x: float(x))  # Numeric sorting
                for snr in sorted_snr_levels:
                    try:
                        scores_by_noise_type[language][noise_type].append(data[language][audio_file][noise_type][snr][metric])
                    except KeyError:
                        continue
        return scores_by_noise_type

    pesq_scores_by_noise_type = collect_scores_by_noise_type('PESQ')
    visqol_scores_by_noise_type = collect_scores_by_noise_type('ViSQOL')

    output_file_path = os.path.join(output_dir, "pairwise_statistical_tests_results_by_noise.txt")
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write("Pairwise Noise Type Statistical Tests Results\n")
        f.write("=" * 70 + "\n\n")

        def perform_tests(scores_by_noise_type, metric_name, test_name, test_func):
            f.write(f"{test_name} Results\n")
            f.write("=" * 70 + "\n")

            noise_type_pairs = [('blue_noise', 'pink_noise'), ('blue_noise', 'noisy_crowd'), ('pink_noise', 'noisy_crowd')]

            for noise1, noise2 in noise_type_pairs:
                all_scores1, all_scores2 = [], []
                for language in languages:
                    scores1 = scores_by_noise_type[language][noise1]
                    scores2 = scores_by_noise_type[language][noise2]
                    
                    if scores1 and scores2:
                        all_scores1.extend(scores1)
                        all_scores2.extend(scores2)

                if not all_scores1 or not all_scores2:
                    continue  # Skip if no data available

                test_result = test_func(all_scores1, all_scores2)

                if test_name == "Spearman Rank Correlation":
                    stat, pvalue = test_result.statistic, test_result.pvalue
                elif isinstance(test_result, tuple):  # For functions returning (stat, p-value)
                    stat, pvalue = test_result
                else:  # For other cases like scipy test objects
                    stat, pvalue = test_result.statistic, test_result.pvalue

                f.write(f"{metric_name} - {noise1.capitalize()} vs {noise2.capitalize()}:\n")
                f.write(f"Statistic: {stat:.2f}, p-value: {pvalue:.2f}\n")
                f.write("-" * 70 + "\n")
            
            f.write("\n")

        # Perform Statistical Tests
        # Parametric
        perform_tests(pesq_scores_by_noise_type, "PESQ", "T-Test", lambda x, y: ttest_ind(x, y, equal_var=False))
        perform_tests(visqol_scores_by_noise_type, "ViSQOL", "T-Test", lambda x, y: ttest_ind(x, y, equal_var=False))

        # Non-Parametric
        perform_tests(pesq_scores_by_noise_type, "PESQ", "Mann-Whitney U Test", lambda x, y: mannwhitneyu(x, y, alternative='two-sided'))
        perform_tests(visqol_scores_by_noise_type, "ViSQOL", "Mann-Whitney U Test", lambda x, y: mannwhitneyu(x, y, alternative='two-sided'))

        perform_tests(pesq_scores_by_noise_type, "PESQ", "Kolmogorov-Smirnov Test", lambda x, y: ks_2samp(x, y))
        perform_tests(visqol_scores_by_noise_type, "ViSQOL", "Kolmogorov-Smirnov Test", lambda x, y: ks_2samp(x, y))
        
        perform_tests(pesq_scores_by_noise_type, "PESQ", "Wilcoxon Rank Sum Test", lambda x, y: ranksums(x, y, alternative='two-sided'))
        perform_tests(visqol_scores_by_noise_type, "ViSQOL", "Wilcoxon Rank Sum Test", lambda x, y: ranksums(x, y, alternative='two-sided'))

        perform_tests(pesq_scores_by_noise_type, "PESQ", "Spearman Rank Correlation", lambda x, y: spearmanr(x, y))
        perform_tests(visqol_scores_by_noise_type, "ViSQOL", "Spearman Rank Correlation", lambda x, y: spearmanr(x, y))

        # Friedman Test
        f.write("Friedman Test Results\n")
        f.write("=" * 70 + "\n")
        
        for metric_name, scores_by_noise_type in [('PESQ', pesq_scores_by_noise_type), 
                                                  ('ViSQOL', visqol_scores_by_noise_type)]:
            # Prepare data for Friedman test
            friedman_data = []
            for language in languages:
                language_scores = []
                for noise_type in noise_types:
                    noise_scores = scores_by_noise_type[language][noise_type][:len(snr_levels)]
                    if noise_scores:
                        language_scores.extend(noise_scores)
                if language_scores:
                    friedman_data.append(language_scores)
            
            if len(friedman_data) >= 2:  # Friedman test requires at least two blocks
                stat, pvalue = friedmanchisquare(*friedman_data)
                f.write(f"{metric_name} Friedman Test:\n")
                f.write(f"Statistic: {stat:.2f}, p-value: {pvalue:.2f}\n")
                f.write("-" * 70 + "\n")

def plot_pca_clusters(data, output_dir='plots'):
    """
    Performs PCA on the score profiles, visualizes clusters of languages, 
    and extracts insights to detect potential biases in metrics.

    Args:
        data (dict): Parsed JSON data containing the analysis results.
        output_dir (str): Directory where plots will be saved (default: 'plots').
    """
    plt.style.use('default')

    languages = list(data.keys())

    # Prepare the data matrix
    score_matrix = []
    for language in languages:
        audio_file = list(data[language].keys())[0]  # Assuming each language has one audio file
        scores = []
        for noise_type in noise_types:
            for snr in snr_levels:
                try:
                    # Try to extract PESQ and ViSQOL scores
                    pesq_score = data[language][audio_file][noise_type][snr]['PESQ']
                    visqol_score = data[language][audio_file][noise_type][snr]['ViSQOL']
                    scores.append(pesq_score)
                    scores.append(visqol_score)
                except KeyError:
                    # Handle missing data with NaN
                    scores.append(np.nan)
                    scores.append(np.nan)
        score_matrix.append(scores)

    # Convert to numpy array and handle missing values (replace NaN with column means)
    score_matrix = np.array(score_matrix)
    score_matrix = np.nan_to_num(score_matrix, nan=np.nanmean(score_matrix, axis=0))

    # Standardize the data
    scaler = StandardScaler()
    score_matrix_scaled = scaler.fit_transform(score_matrix)

    # Perform PCA (using two principal components)
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(score_matrix_scaled)

    # Extract explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    explained_variance_text = f"Explained Variance Ratio (PC1, PC2): {explained_variance[0]:.6f}, {explained_variance[1]:.6f}\n"

    # Extract PCA loadings
    pca_loadings = pca.components_
    pca_loadings_text = "PCA Loadings:\n"
    for i, component in enumerate(pca_loadings):
        pca_loadings_text += f"Component {i+1}: " + ", ".join([f"{val:.6f}" for val in component]) + "\n"

    # Calculate mean and variance of scores for each language
    metrics_summary = {}
    metrics_summary_text = "Metrics Summary per Language:\n"
    for i, language in enumerate(languages):
        pesq_mean = np.nanmean(score_matrix[i, ::2])  # Every other column is PESQ
        pesq_std = np.nanstd(score_matrix[i, ::2])
        visqol_mean = np.nanmean(score_matrix[i, 1::2])  # Every other column is ViSQOL
        visqol_std = np.nanstd(score_matrix[i, 1::2])
        
        metrics_summary[language] = {
            'PESQ_mean': pesq_mean,
            'PESQ_std': pesq_std,
            'ViSQOL_mean': visqol_mean,
            'ViSQOL_std': visqol_std,
        }
        metrics_summary_text += (f"{language.capitalize()}: "
                                 f"PESQ_mean={pesq_mean:.6f}, PESQ_std={pesq_std:.6f}, "
                                 f"ViSQOL_mean={visqol_mean:.6f}, ViSQOL_std={visqol_std:.6f}\n")

    # Save all results into a txt file
    os.makedirs(output_dir, exist_ok=True)
    results_txt_path = os.path.join(output_dir, 'pca_results.txt')
    with open(results_txt_path, 'w') as f:
        f.write(explained_variance_text)
        f.write(pca_loadings_text)
        f.write(metrics_summary_text)

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
    plt.savefig(os.path.join(output_dir, 'pca_clusters.png'),
                bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(output_dir, 'pca_clusters.svg'),
                bbox_inches='tight')
    plt.close()


# -------------------------- Distribution Deviation Calculations ----------------------------------------

def calculate_deviations(data):
    pesq_scores = np.array([pair[0] for pair in data])
    visqol_scores = np.array([pair[1] for pair in data])

    # Compute Differences
    differences = pesq_scores - visqol_scores

    # Compute Standard Metrics
    mae = np.mean(np.abs(differences))  # Mean Absolute Error
    rmse = np.sqrt(np.mean(differences**2))  # Root Mean Squared Error
    std_dev = np.std(differences)  # Standard Deviation of Differences
    pearson_corr, _ = pearsonr(pesq_scores, visqol_scores)  # Pearson Correlation Coefficient

    # Compute Deviation from Ideal Line y = x
    mad = np.mean(np.abs(differences))  # Mean Absolute Deviation
    rmsd = np.sqrt(np.mean(differences**2))  # Root Mean Squared Deviation
    bias = np.mean(differences)  # Mean Bias (PESQ - ViSQOL)

    # Store Results
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Standard Deviation": std_dev,
        "Pearson Correlation Coefficient": pearson_corr,
        "Mean Absolute Deviation (MAD)": mad,
        "Root Mean Squared Deviation (RMSD)": rmsd,
        "Bias (Mean Difference PESQ - ViSQOL)": bias
    }

    return metrics

def compare_turkish_male_against_others(datasets, turkish_male_data, output_dir='plots'):
    # Calculate metrics for Turkish Male
    turkish_male_metrics = calculate_deviations(turkish_male_data)
    
    # Initialize variables to calculate the average metrics for all other datasets
    avg_metrics = {key: 0 for key in turkish_male_metrics.keys()}
    num_comparisons = 0
    
    # Calculate metrics for all datasets except Turkish Male and accumulate them
    for name, data in datasets.items():
        if name != "Turkish Male":  # Skip the Turkish Male dataset itself
            metrics = calculate_deviations(data)
            for key in avg_metrics:
                avg_metrics[key] += metrics[key]
            num_comparisons += 1

    # Calculate the average of the metrics for all other datasets
    for key in avg_metrics:
        avg_metrics[key] /= num_comparisons

    # Compare Turkish Male metrics with the average of the others
    comparison_results = {
        "Pearson Correlation Difference": turkish_male_metrics["Pearson Correlation Coefficient"] - avg_metrics["Pearson Correlation Coefficient"],
        "Bias Difference": turkish_male_metrics["Bias (Mean Difference PESQ - ViSQOL)"] - avg_metrics["Bias (Mean Difference PESQ - ViSQOL)"],
        "RMSE Difference": turkish_male_metrics["RMSE"] - avg_metrics["RMSE"],
        "MAE Difference": turkish_male_metrics["MAE"] - avg_metrics["MAE"],
        "RMSD Difference": turkish_male_metrics["Root Mean Squared Deviation (RMSD)"] - avg_metrics["Root Mean Squared Deviation (RMSD)"]
    }

    output_file = os.path.join(output_dir, "turkish_male_vs_others_comparison_results.txt")
    # Save the results to the output file
    with open(output_file, "w") as file:
        # Write Turkish Male metrics
        file.write("Turkish Male Metrics:\n")
        for key, value in turkish_male_metrics.items():
            file.write(f"{key}: {value:.4f}\n")
        
        # Write the average metrics for all other datasets (excluding Turkish Male)
        file.write("\nAverage Metrics for All Datasets Except Turkish Male:\n")
        for key, value in avg_metrics.items():
            file.write(f"{key}: {value:.4f}\n")

        # Write comparison results (Turkish Male vs average of others)
        file.write("\nComparison of Turkish Male against the average of other datasets:\n")
        for metric, diff in comparison_results.items():
            file.write(f"{metric}: {diff:.4f}\n")
        
def calculate_overall_deviation_metrics(datasets, output_dir='plots'):
    # Combine all datasets into one
    all_pesq_scores = []
    all_visqol_scores = []
    
    # Collect all PESQ and ViSQOL scores from all datasets
    for name, data in datasets.items():
        pesq_scores = np.array([pair[0] for pair in data])
        visqol_scores = np.array([pair[1] for pair in data])
        
        all_pesq_scores.extend(pesq_scores)
        all_visqol_scores.extend(visqol_scores)
    
    # Convert to numpy arrays
    all_pesq_scores = np.array(all_pesq_scores)
    all_visqol_scores = np.array(all_visqol_scores)

    # Compute Differences
    differences = all_pesq_scores - all_visqol_scores

    # Compute Standard Metrics
    mae = np.mean(np.abs(differences))  # Mean Absolute Error
    rmse = np.sqrt(np.mean(differences**2))  # Root Mean Squared Error
    std_dev = np.std(differences)  # Standard Deviation of Differences
    pearson_corr, _ = pearsonr(all_pesq_scores, all_visqol_scores)  # Pearson Correlation Coefficient

    # Compute Deviation from Ideal Line y = x
    mad = np.mean(np.abs(differences))  # Mean Absolute Deviation
    rmsd = np.sqrt(np.mean(differences**2))  # Root Mean Squared Deviation
    bias = np.mean(differences)  # Mean Bias (PESQ - ViSQOL)

    # Store Results
    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Standard Deviation": std_dev,
        "Pearson Correlation Coefficient": pearson_corr,
        "Mean Absolute Deviation (MAD)": mad,
        "Root Mean Squared Deviation (RMSD)": rmsd,
        "Bias (Mean Difference PESQ - ViSQOL)": bias
    }
    output_file = os.path.join(output_dir, "deviation_metrics.txt")
    # Save the results to the output file
    with open(output_file, "w") as file:
        file.write("Overall Metrics (Across All Datasets):\n")
        for key, value in metrics.items():
            file.write(f"{key}: {value:.4f}\n")
