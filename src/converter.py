import os
from pydub import AudioSegment

class AudioConverter:
    """Utility class for audio format conversions"""
    
    def opus_to_wav(dir_name: str, file_name: str) -> str:
        """
        Convert an opus file to wav format
        
        Args:
            dir_name (str): Directory path containing the opus file
            file_name (str): Name of the opus file to convert
            
        Returns:
            str: Path to the converted wav file
        """
        input_path = os.path.join(dir_name, file_name)
        output_path = input_path.replace('.opus', '.wav')

        # Load opus file
        audio = AudioSegment.from_file(input_path, format="ogg")
        
        # Export as wav
        audio.export(output_path, format="wav")
        
        return output_path
