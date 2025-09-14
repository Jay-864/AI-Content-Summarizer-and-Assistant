import whisper
import os

# Load the Whisper model
model = whisper.load_model("base")

def transcribe_audio(audio_path):
    """Transcribe audio using OpenAI Whisper with timestamps"""
    try:
        # Enable word-level timestamps
        result = model.transcribe(audio_path, word_timestamps=True)
        return result
    except Exception as e:
        raise Exception(f"Failed to transcribe audio: {str(e)}")