from analyzer import AudioQualityAnalyzer
from metrics import PESQStrategy
from degradation_types import NoiseType
from results_logger import ResultsLogger
from file_manager import FileManager
import os

if __name__ == '__main__':
    # Directory paths
    languages = ['turkish', 'english']
    
    results_file = './results/analysis_results.txt'
    logger = ResultsLogger(results_file)

    analyzers = []
    
    for language in languages:
        ref_dir = f"./audio/reference/{language}"
        deg_dir = f"./audio/degraded/{language}"
        temp_ref_dir = os.path.join(ref_dir, 'temp_ref')

        # Clean all files in the degraded directory
        FileManager.clean_directory(deg_dir)

        # Clean all files in the temp_ref directory
        FileManager.clean_directory(temp_ref_dir)

        # Initialize analyzer
        analyzer = AudioQualityAnalyzer(language, ref_dir, deg_dir)
        
        # Add metrics
        analyzer.add_metric(PESQStrategy())

        # Add degradation types
        analyzer.add_degradation_type(NoiseType())

        # Perform analysis
        analyzer.analyze()
        
        # Log results
        logger.log_results(analyzer)
        
        analyzers.append(analyzer)

    # Create comparative plot
    logger.plot_results(analyzers)
