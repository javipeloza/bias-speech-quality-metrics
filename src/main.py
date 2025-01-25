from analyzer import AudioQualityAnalyzer
from metrics import PESQStrategy, ViSQOLStrategy
from degradation_types import BlueNoise, PinkNoise, NoisyCrowd
from results_logger import log_analyzer_results, save_analyzer_to_txt
from file_manager import clean_directory
import os

if __name__ == '__main__':
    # Directory paths
    languages = ['turkish','korean','english']
    
    # Get the absolute path to the results directory
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    audio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio'))
    results_file = os.path.join(results_dir, 'analysis_results.txt')

    # Clean the results file before logging
    open(results_file, 'w').close()

    analyzers = []

    metrics = [PESQStrategy(), ViSQOLStrategy()]
    degradation_types = [BlueNoise(), PinkNoise(), NoisyCrowd()]

    for language in languages:
        ref_dir = os.path.join(audio_dir, 'reference', language)
        deg_dir = os.path.join(audio_dir, 'degraded', language)
        temp_ref_dir = os.path.join(ref_dir, 'temp_ref')

        # Clean all files in the degraded directory
        clean_directory(deg_dir)

        # Clean all files in the temp_ref directory
        clean_directory(temp_ref_dir)

        # Initialize analyzer with statistical analyzers
        analyzer = AudioQualityAnalyzer(language, ref_dir, deg_dir)

        # Add metrics
        for metric in metrics:
            analyzer.add_metric(metric)

        # Add degradation types
        for degradation_type in degradation_types:
            analyzer.add_degradation_type(degradation_type)

        # Perform analysis
        analyzer.analyze()

        analyzers.append(analyzer)

        # Save results
        save_analyzer_to_txt(analyzer, os.path.join(results_dir, f'analysis_results_{language}.txt'))

        # Save results to common file
        log_analyzer_results(analyzer, results_file)

        # Clean all files in the degraded directory
        clean_directory(deg_dir)

        # Clean all files in the temp_ref directory
        clean_directory(temp_ref_dir)
