from analyzer import AudioQualityAnalyzer
from metrics import PESQStrategy, ViSQOLStrategy
from degradation_types import NoiseType
from results_logger import ResultsLogger

if __name__ == '__main__':
    # Directory paths
    languages = ['turkish', 'english']
    
    results_file = './results/analysis_results.txt'
    logger = ResultsLogger(results_file)

    analyzers = []
    
    for language in languages:
        ref_dir = f"./audio/reference/{language}"
        deg_dir = f"./audio/degraded/{language}"

        # Initialize analyzer
        analyzer = AudioQualityAnalyzer(language, ref_dir, deg_dir)
        
        # Add metrics
        analyzer.add_metric(PESQStrategy())
        
        # Add degradation types
        analyzer.add_degradation_type(NoiseType())
        # Add more degradation types as needed
        
        # Perform analysis
        analyzer.analyze()
        
        # Log results
        logger.log_results(analyzer)
        
        analyzers.append(analyzer)

    # Create comparative plot
    logger.plot_results(analyzers)
