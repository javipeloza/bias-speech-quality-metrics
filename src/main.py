from analyzer import AudioQualityAnalyzer
from metrics import PESQStrategy, ViSQOLStrategy
from degradation_types import BlueNoise, PinkNoise, NoisyCrowd
from results_logger import ResultsLogger, plot_analysis_results, save_analyzers_to_json, save_analyzer_to_txt, save_analyzer_to_json, json_to_analyzers
from file_manager import FileManager
from statistical_analyzers import Anova
from statistics_util import analyze_statistical_results
import os

if __name__ == '__main__':
    # Directory paths
    languages = ['turkish','korean','english','chinese','spanish']
    # languages = ['turkish', 'english']
    
    # Get the absolute path to the results directory
    results_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results'))
    audio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'audio'))
    results_file = os.path.join(results_dir, 'analysis_results.txt')

    # Clean the results file before logging
    open(results_file, 'w').close()
    logger = ResultsLogger(results_file)

    analyzers = []

    metrics = [PESQStrategy(), ViSQOLStrategy()]
    degradation_types = [BlueNoise(), PinkNoise(), NoisyCrowd()]
    
    # Create an instance of StatisticalAnalyzers
    statistical_analyzers = [Anova()]

    for language in languages:
        ref_dir = os.path.join(audio_dir, 'reference', language)
        deg_dir = os.path.join(audio_dir, 'degraded', language)
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
            analyzer.add_degradation_type(degradation_type)

        # Perform analysis
        analyzer.analyze()
        
        # Log results
        logger.log_results(analyzer)

        # Clean all files in the degraded directory
        FileManager.clean_directory(deg_dir)

        # Clean all files in the temp_ref directory
        FileManager.clean_directory(temp_ref_dir)

        # Custom file path for the analyzer using the language
        analyzer_json_file_path = os.path.join(results_dir, f'analysis_results_{language}.json')
        analyzer_txt_file_path = os.path.join(results_dir, f'analysis_results_{language}.txt')

        try:
            save_analyzer_to_txt(analyzer, analyzer_txt_file_path)
            save_analyzer_to_json(analyzer, analyzer_json_file_path)
        except Exception as e:
            print(f"Error saving analyzer for {language}: {e}")
        
        analyzers.append(analyzer)

    json_file_path = os.path.join(results_dir, 'analysis_results.json')

    # Save analyzers to a JSON file
    save_analyzers_to_json(analyzers, json_file_path)

    print("Analyzer saved to json")

    analyzers = json_to_analyzers(json_file_path)

    # Create comparative plot
    plot_analysis_results(analyzers)

    results = [analyzer.get_results() for analyzer in analyzers]

    # Analyze results 
    analyze_statistical_results(statistical_analyzers, degradation_types, results, languages)
