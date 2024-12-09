from analyzer import AudioQualityAnalyzer, PESQStrategy, ViSQOLStrategy
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

        # Initialize analyzer for PESQ
        pesq_analyzer = AudioQualityAnalyzer(language, ref_dir, deg_dir, PESQStrategy())
        
        # Perform analysis
        pesq_analyzer.analyze()
        
        # Log individual results
        logger.log_results(pesq_analyzer, metric_name='PESQ')
        
        # Add analyzer to list for comparison plot
        analyzers.append(pesq_analyzer)

    # Create comparative plot with all analyzers
    logger.plot_results(analyzers)
