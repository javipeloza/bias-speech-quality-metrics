from scipy.stats import f_oneway

class StatisticalAnalyzer:
    def __init__(self, name):
        self.name = name

    """Base class for statistical analyzers"""
    def analyze(self, results, degrdataion_type, languages):
        raise NotImplementedError("Must implement analyze method")

class Anova(StatisticalAnalyzer):
    def __init__(self):
        super().__init__("ANOVA")

    def analyze(self, results, degradation_type, languages):
        """
        Perform ANOVA on the results for each language, categorized by metric type and degradation type.
        
        Args:
            results (list): A list of dictionaries containing results for different languages.
            degradation_type (str): The type of degradation to analyze (e.g., 'noise', 'distortion').
            languages (list): A list of languages corresponding to the results provided.
        
        Returns:
            dict: A dictionary with ANOVA results for each metric, including F-statistics and p-values.
        """

        aggregated_by_snr = {}

        # results: {
        #     file_name_1: {
        #         noise: {
        #             -40: {
        #                 'PESQ': 1.75,
        #                 'ViSQOL': 1.88
        #             },
        #             0: {...}
        #         }
        #     }
        # }

        for language, result in zip(languages, results):
            for file_name, degradations in result.items():
                if degradation_type not in degradations:
                    continue

                for snr, metrics in degradations[degradation_type].items():
                    for metric, score in metrics.items():
                        if metric not in aggregated_by_snr:
                            aggregated_by_snr[metric] = {}

                        if language not in aggregated_by_snr[metric]:
                            aggregated_by_snr[metric][language] = {}

                        if file_name not in aggregated_by_snr[metric][language]:
                            aggregated_by_snr[metric][language][file_name] = {}

                        aggregated_by_snr[metric][language][file_name][snr] = score

		# Aggregated Results By SNR: {
		# 	'pesq': {
		# 		'english': {
		# 			'file1': {
		# 				-40: 1.88,
		# 				0: 1.75
		# 			},
        #                  ...
		# 		},
        # 		'turkish': {...}
		# 	},
        #   (...)
		# }

        aggregated_by_file = {}

        for metric, languages in aggregated_by_snr.items():
            aggregated_by_file[metric] = {}
            for language, files in languages.items():
                aggregated_by_file[metric][language] = {}
                for file_name, snrs in files.items():
					# Collect all scores in the order they appear in the SNR dictionary
                    scores = list(snrs.values())
                    aggregated_by_file[metric][language][file_name] = scores
          
		# Aggregated Results By File: {
		# 	'pesq': {
		# 		'english': {
		# 			'file1': [scores],
		# 			'file2': [scores],
		# 			...
		# 		},
		# 		'turkish': {...}
		# 	},
		#   (...)
		# }

		# Aggregated results By Metric
        aggregated_by_metric = {}

        for metric, languages in aggregated_by_file.items():
            aggregated_by_metric[metric] = {}
            for language, files in languages.items():
				# Combine scores from all files by calculating mean at each index
                combined_scores = []
                max_length = max(len(scores) for scores in files.values())

                for i in range(max_length):
                    scores_at_i = [scores[i] for scores in files.values() if i < len(scores)]
                    combined_scores.append(sum(scores_at_i) / len(scores_at_i))

                aggregated_by_metric[metric][language] = combined_scores

		# Aggregated results By Metric: {
        #     'pesq': {
		# 		'english': [scores], # Scores should be the combined between all files in the language that underwent those degradations
        #       'turkish': [scores]
		# 	},
        #     'ViSQOL': {
        #         ...
		# 	}
		# }

        statistical_results = {}
        
        for metric in aggregated_by_metric:
            statistical_results[metric] = {}
            aggregated_scores = []
            for language in aggregated_by_metric[metric]:
                aggregated_scores.append(aggregated_by_metric[metric][language])

            f_stat, p_value = f_oneway(*aggregated_scores)
            statistical_results[metric] = {
				'F-statistic': f_stat,
				'p-value': p_value
			}
          
		# Statistical Results: {
        #     'pesq': {
		# 		'F-statistic': 0.05,
        #       'p-value': 0.98
		# 	},
        #     'ViSQOL': {
        #         ...
		# 	}
		# }

        return statistical_results
		