import PyPDF2
import os
import subprocess

def process_pdf(filepath):
    """Extract text from PDF file"""
    try:
        with open(filepath, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        raise Exception(f"Failed to process PDF: {str(e)}")

def extract_audio_from_video(video_path, output_dir):
    """Extract audio from video file using ffmpeg"""
    try:
        output_path = os.path.join(output_dir, f"audio_{os.path.basename(video_path)}.wav")
        command = [
            'ffmpeg', '-i', video_path, 
            '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', 
            output_path, '-y'
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        raise Exception(f"Failed to extract audio from video: {str(e)}")