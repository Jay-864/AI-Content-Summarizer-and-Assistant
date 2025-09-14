import os
import uuid
import threading
from flask import Flask, render_template, request, jsonify, session, send_from_directory
from werkzeug.utils import secure_filename
from utils.file_processor import process_pdf, extract_audio_from_video
from utils.whisper_transcribe import transcribe_audio
import ollama

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config.from_pyfile('config.py')

# Global variable to track processing status
processing_status = {}
chat_sessions = {}

@app.route('/')
def index():
    session_id = session.get('session_id', str(uuid.uuid4()))
    session['session_id'] = session_id
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {
            'messages': [],
            'extracted_text': '',
            'is_processing': False
        }
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session expired'}), 400
    
    # Set processing status
    chat_sessions[session_id]['is_processing'] = True
    processing_status[session_id] = "Processing your file..."
    
    try:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'pdf':
            filepath = os.path.join(app.config['UPLOAD_FOLDER_PDF'], filename)
            file.save(filepath)
            
            # Process PDF in a separate thread
            thread = threading.Thread(
                target=process_pdf_file, 
                args=(session_id, filepath)
            )
            thread.start()
            
        elif file_extension in ['mp4', 'mov', 'avi', 'mkv']:
            filepath = os.path.join(app.config['UPLOAD_FOLDER_VIDEO'], filename)
            file.save(filepath)
            
            # Process video in a separate thread
            thread = threading.Thread(
                target=process_video_file, 
                args=(session_id, filepath)
            )
            thread.start()
            
        else:
            chat_sessions[session_id]['is_processing'] = False
            return jsonify({'error': 'Unsupported file type'}), 400
            
        return jsonify({'message': 'File uploaded successfully. Processing...'})
        
    except Exception as e:
        chat_sessions[session_id]['is_processing'] = False
        return jsonify({'error': str(e)}), 500

def process_pdf_file(session_id, filepath):
    try:
        extracted_text = process_pdf(filepath)
        chat_sessions[session_id]['extracted_text'] = extracted_text
        processing_status[session_id] = "File processed successfully. You can now ask questions."
    except Exception as e:
        processing_status[session_id] = f"Error processing PDF: {str(e)}"
    finally:
        chat_sessions[session_id]['is_processing'] = False

def process_video_file(session_id, filepath):
    try:
        # Extract audio from video
        audio_path = extract_audio_from_video(
            filepath, 
            app.config['UPLOAD_FOLDER_TEMP']
        )
        
        # Transcribe audio with timestamps
        transcription_result = transcribe_audio(audio_path)
        
        # Store both the full text and the segments with timestamps
        extracted_text = transcription_result["text"]
        segments = transcription_result.get("segments", [])
        
        chat_sessions[session_id]['extracted_text'] = extracted_text
        chat_sessions[session_id]['segments'] = segments  # Store segments with timestamps
        
        processing_status[session_id] = "File processed successfully. You can now ask questions."
        
        # Clean up temporary audio file
        os.remove(audio_path)
        
    except Exception as e:
        processing_status[session_id] = f"Error processing video: {str(e)}"
    finally:
        chat_sessions[session_id]['is_processing'] = False

@app.route('/ask', methods=['POST'])
def ask_question():
    session_id = session.get('session_id')
    if not session_id or session_id not in chat_sessions:
        return jsonify({'error': 'Session expired'}), 400
    
    if chat_sessions[session_id]['is_processing']:
        return jsonify({'error': 'System is still processing your file'}), 400
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    if not chat_sessions[session_id]['extracted_text']:
        return jsonify({'error': 'No document content available. Please upload a file first.'}), 400
    
    # Add user question to chat history
    chat_sessions[session_id]['messages'].append({
        'role': 'user',
        'content': question
    })
    
    # Set processing status
    chat_sessions[session_id]['is_processing'] = True
    processing_status[session_id] = "Generating answer..."
    
    # Process question in a separate thread
    thread = threading.Thread(
        target=generate_answer, 
        args=(session_id, question)
    )
    thread.start()
    
    return jsonify({'message': 'Question received. Processing...'})

def find_text_around_timestamp(segments, target_seconds, window_seconds=30):
    """Find text around a specific timestamp"""
    if not segments:
        return "No timestamp information available."
    
    # Find the segment that contains the target timestamp
    context_text = []
    for segment in segments:
        start = segment['start']
        end = segment['end']
        
        # Check if this segment is within the window of the target timestamp
        if (start <= target_seconds <= end) or (abs(start - target_seconds) <= window_seconds):
            # Format the timestamp
            timestamp = format_timestamp(start)
            context_text.append(f"[{timestamp}] {segment['text']}")
    
    if not context_text:
        return f"No content found around timestamp {format_timestamp(target_seconds)}."
    
    return "\n".join(context_text)

def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def generate_answer(session_id, question):
    try:
        extracted_text = chat_sessions[session_id]['extracted_text']
        
        # Prepare prompt for the AI model
        prompt = f"""
        Act as if you are a chatbot and based on the following content, please answer the user's question.
        
        Content:
        {extracted_text}  # Limit context to avoid token limits
        
        Question: {question}
        
        Make sure and NEVER forget to provide a detailed answer in HTML format with proper formatting. 
        Use headings, paragraphs, bullet points, and bold text (using <b></b>) where appropriate.
        """
        
        # Get response from Ollama
        response = ollama.chat(model='gemma3:4b', messages=[
            {'role': 'user', 'content': prompt}
        ], think=False)
        
        answer = response['message']['content']
        
        # Add AI response to chat history
        chat_sessions[session_id]['messages'].append({
            'role': 'assistant',
            'content': answer
        })
        
        processing_status[session_id] = "Answer generated successfully."
        
    except Exception as e:
        processing_status[session_id] = f"Error generating answer: {str(e)}"
        
        # Add error message to chat history
        chat_sessions[session_id]['messages'].append({
            'role': 'assistant',
            'content': f"Sorry, I encountered an error while processing your question: {str(e)}"
        })
        
    finally:
        chat_sessions[session_id]['is_processing'] = False

@app.route('/status')
def get_status():
    session_id = session.get('session_id')
    if not session_id:
        return jsonify({'error': 'Session expired'}), 400
    
    status = processing_status.get(session_id, 'Ready')
    is_processing = chat_sessions.get(session_id, {}).get('is_processing', False)
    
    return jsonify({
        'status': status,
        'is_processing': is_processing
    })

@app.route('/messages')
def get_messages():
    session_id = session.get('session_id')
    if not session_id or session_id not in chat_sessions:
        return jsonify({'error': 'Session expired'}), 400
    
    return jsonify({
        'messages': chat_sessions[session_id]['messages']
    })

if __name__ == '__main__':
    # Create upload directories if they don't exist
    os.makedirs(app.config['UPLOAD_FOLDER_PDF'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_VIDEO'], exist_ok=True)
    os.makedirs(app.config['UPLOAD_FOLDER_TEMP'], exist_ok=True)
    
    app.run(debug=True)