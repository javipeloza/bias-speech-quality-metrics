import os
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tkinter as tk
from tkinter import filedialog
import librosa
import webrtcvad
import numpy as np
import re

class FileManager:
    @staticmethod
    def clean_directory(directory):
        # Clean all files in the specified directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def extract_sentences_from_audio(): 
    # Initialize tkinter and get folder selection
    root = tk.Tk()
    root.withdraw()
    audio_folder = filedialog.askdirectory(title="Select Folder Containing Audio Files")
    
    if not audio_folder:  # Handle case where user cancels folder selection
        print("No folder selected. Exiting...")
        return
        
    # Create a new directory for exported audio chunks
    output_dir = os.path.join(audio_folder, "separated_sentences")
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each audio file in the selected folder
    for filename in os.listdir(audio_folder):
        if filename.endswith(".wav"):  # Process only .wav files
            audio_path = os.path.join(audio_folder, filename)  # Full path to the audio file
            audio = AudioSegment.from_file(audio_path)

            # Split the audio based on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=1000,  # Minimum length of silence (in ms) to consider as a separator
                silence_thresh=-60  # Silence threshold (in dBFS)
            )

            # Create a folder for the current audio file's subsentences
            file_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])  # Folder for subsentences
            os.makedirs(file_output_dir, exist_ok=True)  # Create the directory if it doesn't exist

            # Export each chunk as a separate audio file in the new directory
            for i, chunk in enumerate(chunks):
                chunk_length = len(chunk)  # Get the length of the chunk in milliseconds
                if 4000 <= chunk_length <= 20000:  # Check if the chunk is between 4 and 20 seconds long
                    output_path = os.path.join(file_output_dir, f"{os.path.splitext(filename)[0]}_{i + 1}.wav")  # Update path to include the new directory and naming convention
                    chunk.export(output_path, format="wav")
                    print(f"Exported: {output_path}")
                else:
                    print(f"Skipped chunk {i + 1} from {filename} (length: {chunk_length / 1000} seconds)")

def calculate_speech_activity(audio_file, sample_rate=16000, frame_duration_ms=30):
    """
    Calculate the speech activity percentage in an audio file.
    
    :param audio_file: Path to the audio file
    :param sample_rate: Sample rate for the audio (default is 16kHz)
    :param frame_duration_ms: Duration of each frame in milliseconds (default is 30ms)
    :return: Speech activity percentage
    """
    # Load the audio file in mono and resample to the desired sample rate
    y, sr = librosa.load(audio_file, sr=sample_rate, mono=True)

    # Convert the audio to 16-bit PCM
    y = np.int16(y / np.max(np.abs(y)) * 32767)  # Normalize to 16-bit PCM

    # Initialize the VAD (Voice Activity Detector)
    vad = webrtcvad.Vad(1)  # Set aggressiveness mode: 0 (least aggressive) to 3 (most aggressive)

    # Convert the audio to frames of the specified duration
    frame_length = int(sample_rate * frame_duration_ms / 1000)  # Length of each frame in samples
    frames = []

    # Break the audio into frames
    for start in range(0, len(y), frame_length):
        end = min(start + frame_length, len(y))
        frame = y[start:end]
        
        if len(frame) < frame_length:
            # If the last frame is too short, pad it
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')

        frames.append(frame)

    # Calculate how many frames contain speech
    speech_frames = 0
    for frame in frames:
        # Check if the frame contains speech
        is_speech = vad.is_speech(frame.tobytes(), sample_rate)
        if is_speech:
            speech_frames += 1

    # Calculate the percentage of speech activity
    speech_percentage = (speech_frames / len(frames)) * 100
    return speech_percentage

def calculate_speech_activity_in_folder(folder_path):
    """
    Process all audio files in a given folder and calculate speech activity percentage for each.
    
    :param folder_path: Path to the folder containing audio files
    :return: Dictionary of audio files and their corresponding speech activity percentages,
             and the average speech activity percentage
    """
    audio_files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.mp3', '.flac'))]  # List audio files
    speech_activity = {}
    total_speech_activity = 0

    for audio_file in audio_files:
        audio_file_path = os.path.join(folder_path, audio_file)
        print(f"Processing: {audio_file_path}")
        percentage = calculate_speech_activity(audio_file_path)
        speech_activity[audio_file] = percentage
        total_speech_activity += percentage
        print(f"{audio_file}: {percentage:.2f}% speech activity\n")

    # Calculate the average speech activity
    # Spanish: 98.04%
	# Turkish: 97.27%
	# Korean: 98.66%
	# English: 97.67%
	# Chinese: 95.31%
    average_speech_activity = total_speech_activity / len(audio_files) if audio_files else 0
    return speech_activity, average_speech_activity

def fix_txt_files_to_json():
    # Initialize tkinter and get folder selection
    root = tk.Tk()
    root.withdraw()
    input_folder = filedialog.askdirectory(title="Select Folder Containing Text Files")
    
    if not input_folder:  # Handle case where user cancels folder selection
        print("No folder selected. Exiting...")
        return

    # Create output folder
    output_folder = os.path.join(input_folder, "json_fixed")
    os.makedirs(output_folder, exist_ok=True)

    # Regular expression to match np.int64(number_here)
    pattern = r'np\.int64\((-?\d+)\)'

    # Process each txt file in the selected folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace('.txt', '.json'))
            
            try:
                with open(input_path, 'r') as file:
                    content = file.read()

                # Replace single quotes with double quotes first
                fixed_content = content.replace("'", '"')
                
                # Replace all occurrences of np.int64(number_here) with "number_here"
                fixed_content = re.sub(pattern, r'"\1"', fixed_content)

                with open(output_path, 'w') as file:
                    file.write(fixed_content)

                print(f"Processed: {filename} -> {os.path.basename(output_path)}")
            
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print(f"\nAll files have been processed. Fixed JSON files are in: {output_folder}")
