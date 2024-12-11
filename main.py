from analyzer import AudioQualityAnalyzer
from metrics import PESQStrategy
from degradation_types import NoiseType
from results_logger import ResultsLogger
from file_manager import FileManager
from statistical_analyzers import Anova
import os

if __name__ == '__main__':
    # Directory paths
    languages = ['turkish', 'english','korean','spanish','chinese']
    # languages = ['english','turkish','spanish']
    
    results_file = './results/analysis_results.txt'
    logger = ResultsLogger(results_file)

    analyzers = []

    metrics = [PESQStrategy()]
    degradation_types = [NoiseType()]
    
    # Create an instance of StatisticalAnalyzers
    statistical_analyzers = [Anova()]

    for language in languages:
        ref_dir = f"./audio/reference/{language}"
        deg_dir = f"./audio/degraded/{language}"
        temp_ref_dir = os.path.join(ref_dir, 'temp_ref')

        # Clean all files in the degraded directory
        FileManager.clean_directory(deg_dir)

        # Clean all files in the temp_ref directory
        FileManager.clean_directory(temp_ref_dir)

        # Initialize analyzer with statistical analyzers
        analyzer = AudioQualityAnalyzer(language, ref_dir, deg_dir)

        # Add metrics
        for metric in metrics:
            analyzer.add_metric(metric)

        # Add degradation types
        for degradation_type in degradation_types:
            analyzer.add_degradation_type(NoiseType())

        # Perform analysis
        analyzer.analyze()
        
        # Log results
        logger.log_results(analyzer)
        
        analyzers.append(analyzer)

    # Create comparative plot
    logger.plot_results(analyzers)

    results = [analyzer.get_results() for analyzer in analyzers]

    # Analyze results 
    for statistical_analyzer in statistical_analyzers:
        stats = statistical_analyzer.analyze(results, NoiseType().name, languages)
        logger.plot_results_table(stats, languages)
        for metric, result in stats.items():
            print(f"{statistical_analyzer.name} Results for {metric} under {NoiseType().name}:")
            print(result)
