from analyzer import AudioQualityAnalyzer
from metrics import PESQStrategy, ViSQOLStrategy, ViSQOLMatLabStrategy
from degradation_types import NoiseType
from results_logger import ResultsLogger, plot_analysis_results, plot_statistical_results_table
from file_manager import FileManager
from statistical_analyzers import Anova
import os

if __name__ == '__main__':
    # Directory paths
    # languages = ['turkish','english','korean','spanish','chinese']
    languages = ['turkish', 'english']
    
    results_file = './results/analysis_results.txt'
    # Clean the results file before logging
    open(results_file, 'w').close()
    logger = ResultsLogger(results_file)

    analyzers = []

    metrics = [ViSQOLStrategy()]
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
    plot_analysis_results(analyzers)

    results = [analyzer.get_results() for analyzer in analyzers]

    # Analyze results 
    for statistical_analyzer in statistical_analyzers:
        for degradation_type in degradation_types:
            stats = statistical_analyzer.analyze(results, degradation_type.name, languages)
            plot_statistical_results_table(stats, languages)
