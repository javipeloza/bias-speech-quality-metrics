from pesq import pesq
from degrade import resample_signal, export_file
from scipy.io import wavfile
import os
import matlab.engine

class MetricStrategy:
    @property
    def name(self):
        raise NotImplementedError("Must implement name property")

    """Base class for quality metrics"""
    def calculate_score(self, ref, deg):
        raise NotImplementedError("Must implement calculate_score method")
    

class PESQStrategy(MetricStrategy):
    @property
    def name(self):
        return "PESQ"

    def calculate_score(self, ref_path, deg_path):
        _, deg = wavfile.read(deg_path)
        _, ref = wavfile.read(ref_path)
        return pesq(8000, ref, deg, 'nb')

class ViSQOLStrategy(MetricStrategy):
    @property
    def name(self):
        return "ViSQOL"

    def calculate_score(self, ref_path, deg_path):
        fs = 16000  # Sampling rate for speech mode
        # Resample reference and degraded signals to 16 kHz in memory
        ref_signal_resampled = resample_signal(ref_path, sampling_rate=fs)
        deg_signal_resampled = resample_signal(deg_path, sampling_rate=fs)

        # Save the reference and degraded signals to WAV files with _resampled suffix
        ref_path_resampled = ref_path.replace('.wav', '_resampled.wav')  # Modify the reference file path
        deg_path_resampled = deg_path.replace('.wav', '_resampled.wav')  # Modify the degraded file path
        export_file(ref_signal_resampled, ref_path_resampled)
        export_file(deg_signal_resampled, deg_path_resampled)

        # Start MATLAB engine
        eng = matlab.engine.start_matlab()

        # Load audio files and run ViSQOL
        ref_signal = eng.audioread(ref_path_resampled)
        deg_signal = eng.audioread(deg_path_resampled)

        mos_score = eng.visqol(
            deg_signal,
            ref_signal,
            fs,
            "Mode", "audio",
            "OutputMetric", "MOS"
        )

        # Stop MATLAB engine
        eng.quit()

        return mos_score
    
class ViSQOLDockerStrategy(MetricStrategy):
    @property
    def name(self):
        return "ViSQOL_docker"

    def calculate_score(self, ref_path, deg_path):
        # Resample reference and degraded signals to 16 kHz in memory
        ref_signal_resampled = resample_signal(ref_path, sampling_rate=48000)
        deg_signal_resampled = resample_signal(deg_path, sampling_rate=48000)

        # Save the reference and degraded signals to WAV files with _resampled suffix
        ref_path_resampled = ref_path.replace('.wav', '_resampled.wav')  # Modify the reference file path
        deg_path_resampled = deg_path.replace('.wav', '_resampled.wav')  # Modify the degraded file path
        export_file(ref_signal_resampled, ref_path_resampled)
        export_file(deg_signal_resampled, deg_path_resampled)

        # Run the Docker command to calculate the ViSQOL score
        import subprocess
        command = [
            "docker", "run", "-it", "-v", f"{os.path.dirname(os.getcwd())}/audio:/audio", 
            "mubtasimahasan/visqol:v3", 
            "--degraded_file", deg_path_resampled.lstrip('.').replace('\\', '/'),
            "--reference_file", ref_path_resampled.lstrip('.').replace('\\', '/')
        ]

        # command = [
        #     "docker", "run", "-it", "-v", f"{os.getcwd()}/audio:/audio", 
        #     "mubtasimahasan/visqol:v3", 
        #     "--degraded_file", deg_path_resampled.lstrip('.').replace('\\', '/'),
        #     "--reference_file", ref_path_resampled.lstrip('.').replace('\\', '/')
        # ]

        result = subprocess.run(command, capture_output=True, text=True)

        # Extract the score from the command output
        score_line = result.stdout.splitlines()[-1]  # Get the last line of the output
        score = float(score_line.split(':')[-1].strip())  # Extract the score after the colon and convert to float
        return score
