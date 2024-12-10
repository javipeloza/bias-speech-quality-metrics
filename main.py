from analyzer import AudioQualityAnalyzer
from metrics import PESQStrategy
from degradation_types import NoiseType
from results_logger import ResultsLogger
import os
import shutil

if __name__ == '__main__':
    # Directory paths
    languages = ['turkish', 'english']
    
    results_file = './results/analysis_results.txt'
    logger = ResultsLogger(results_file)

    analyzers = []
    
    for language in languages:
        ref_dir = f"./audio/reference/{language}"
        deg_dir = f"./audio/degraded/{language}"

        # Clean all files in the degraded directory
        for filename in os.listdir(deg_dir):
            file_path = os.path.join(deg_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

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
