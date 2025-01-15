import os
import shutil
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tkinter as tk
from tkinter import filedialog

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

    @staticmethod
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
