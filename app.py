from flask import Flask, request, jsonify
import numpy as np
import tempfile
import os
import base64
from flask_cors import CORS
import asyncio
import logging

from voice import VoiceProcessingSystem
from response import ResponseGenerator, Query
from sym import SystemManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS to allow your frontend domains
ALLOWED_ORIGINS = [
    'http://localhost:3000',
    'https://ncea-ai-three.vercel.app',  # Replace with your actual Vercel URL
    'https://your-custom-domain.com'  # Replace with your custom domain if you have one
]

CORS(app, origins=ALLOWED_ORIGINS, allow_headers=['Content-Type'])

# Initialize components
try:
    # Try to load from environment variables first (for production)
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    searchapi_key = os.environ.get('SEARCHAPI_KEY', '')
    
    # Fallback to config.yaml for local development
    if not openai_api_key:
        try:
            with open("config.yaml", 'r') as f:
                import yaml
                config = yaml.safe_load(f)
                openai_api_key = config.get('openai_api_key')
                searchapi_key = config.get('searchapi_key', '')
        except FileNotFoundError:
            logger.warning("config.yaml not found, using environment variables only")
    
    if not openai_api_key:
        raise ValueError("OpenAI API key not found in environment or config")
    
    # Initialize components
    system_manager = SystemManager()
    voice_processor = VoiceProcessingSystem(openai_api_key)
    response_generator = ResponseGenerator(
        openai_api_key,
        cache_file="response_cache.json",
        voice_id=os.environ.get('VOICE_ID', 'nova')
    )
    
    logger.info("✅ Voice assistant components initialized successfully")
    
except Exception as e:
    logger.error(f"❌ Error initializing components: {e}")
    # Don't raise in production, just log the error
    system_manager = None
    voice_processor = None
    response_generator = None

@app.route('/')
def index():
    return jsonify({
        "status": "running",
        "message": "Voice Assistant API Server is running",
        "endpoints": [
            "/api/text/process",
            "/api/voice/process", 
            "/api/voice/profile",
            "/api/voice/profile/create"
        ]
    })

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({"status": "healthy"}), 200

@app.route('/api/text/process', methods=['POST', 'OPTIONS'])
def process_text():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        if not response_generator:
            return jsonify({"error": "Service not initialized"}), 503
            
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        user_text = data['text']
        logger.info(f"Processing text: {user_text[:50]}...")
        
        # Create query object
        query_obj = Query(
            text=user_text,
            intent="conversation",
            speaker_id="web_user"
        )
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(response_generator.generate_response(query_obj))
            
            # Generate audio for the response
            audio_data = loop.run_until_complete(response_generator.tts_engine.generate_speech(response.text))
            audio_base64 = base64.b64encode(audio_data).decode('utf-8') if audio_data else None
            
        finally:
            loop.close()
        
        return jsonify({
            "text": response.text,
            "source": response.source,
            "audio": audio_base64
        })
        
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        return jsonify({"error": "Failed to process request"}), 500

@app.route('/api/voice/process', methods=['POST', 'OPTIONS'])
def process_voice():
    if request.method == 'OPTIONS':
        return '', 204
        
    if not voice_processor or not response_generator:
        return jsonify({"error": "Voice service not initialized"}), 503
        
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    
    # Save the audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        audio_file.save(temp_file.name)
        temp_filename = temp_file.name
    
    try:
        # Process audio through the pipeline
        import wave
        try:
            with wave.open(temp_filename, 'rb') as wf:
                sample_rate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                audio_data = audio_data.reshape(1, -1)
        except wave.Error:
            # If not a valid WAV file, return error
            return jsonify({"error": "Invalid audio format. Please send WAV format."}), 400
        
        # Process the audio
        voice_result = voice_processor.process_voice(audio_data, sample_rate)
        
        if not voice_result or not voice_result.text:
            return jsonify({"error": "Could not process speech"}), 400
        
        # Generate response
        query_obj = Query(
            text=voice_result.text,
            intent=voice_result.intent,
            speaker_id=voice_result.speaker_id
        )
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(response_generator.generate_response(query_obj))
            audio_data = loop.run_until_complete(response_generator.tts_engine.generate_speech(response.text))
            audio_base64 = base64.b64encode(audio_data).decode('utf-8') if audio_data else None
        finally:
            loop.close()
        
        return jsonify({
            "userSpeech": voice_result.text,
            "response": {
                "text": response.text,
                "source": response.source
            },
            "audio": audio_base64
        })
    
    except Exception as e:
        logger.error(f"Error processing voice: {e}")
        return jsonify({"error": "Failed to process audio"}), 500
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

@app.route('/api/voice/profile', methods=['GET', 'OPTIONS'])
def check_profile():
    if request.method == 'OPTIONS':
        return '', 204
        
    try:
        # Check if voice profiles exist
        if os.path.exists("voice_profiles.json"):
            import json
            with open("voice_profiles.json", "r") as f:
                profiles_data = json.load(f)
                
            hasProfile = len(profiles_data) > 0
            profile = profiles_data[0] if hasProfile else None
                
            return jsonify({
                "hasProfile": hasProfile,
                "profile": {
                    "user_id": profile["user_id"],
                    "name": profile["name"],
                    "created_at": profile["created_at"]
                } if profile else None
            })
        
        return jsonify({"hasProfile": False, "profile": None})
    
    except Exception as e:
        logger.error(f"Error checking profile: {e}")
        return jsonify({"error": "Failed to check profile"}), 500

@app.route('/api/voice/profile/create', methods=['POST', 'OPTIONS'])
def create_profile():
    if request.method == 'OPTIONS':
        return '', 204
        
    if not voice_processor:
        return jsonify({"error": "Voice service not initialized"}), 503
        
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    audio_file = request.files['audio']
    name = request.form.get('name', 'User1')
    
    # Save the audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        audio_file.save(temp_file.name)
        temp_filename = temp_file.name
    
    try:
        # Load and process audio
        import wave
        with wave.open(temp_filename, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            audio_data = audio_data.reshape(1, -1)
        
        # Create profile
        profile = voice_processor.create_voice_profile(audio_data, name)
        
        if not profile:
            return jsonify({"error": "Could not create voice profile"}), 400
        
        return jsonify({
            "success": True,
            "profile": {
                "user_id": profile.user_id,
                "name": profile.name,
                "created_at": profile.created_at
            }
        })
    
    except Exception as e:
        logger.error(f"Error creating profile: {e}")
        return jsonify({"error": "Failed to create profile"}), 500
    
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except:
                pass

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)