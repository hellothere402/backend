import openai
from typing import Dict, Optional, List
from dataclasses import dataclass
import json
import time
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

@dataclass
class Query:
    text: str
    intent: str
    speaker_id: str
    context: Dict = None

@dataclass
class Response:
    text: str
    audio: bytes = None
    source: str = None
    cache_key: str = None

class ResponseGenerator:
    def __init__(self, openai_api_key: str, cache_file: str = "response_cache.json", voice_id: str = "nova"):
        self.api_key = openai_api_key
        self.cache_file = cache_file
        self.response_cache = self._load_cache()
        
        # Initialize processors
        self.local_processor = LocalProcessor()
        self.cloud_processor = CloudProcessor(api_key=openai_api_key)
        self.tts_engine = TTSEngine(api_key=openai_api_key, voice_id=voice_id)
        
        logger.info("Response generator initialized")

    def _generate_cache_key(self, query: Query) -> str:
        """Generate a unique cache key for the query"""
        try:
            key_parts = [
                query.text.lower().strip(),
                query.intent,
                query.speaker_id if query.speaker_id else "unknown"
            ]
            return "_".join(key_parts)
        except Exception as e:
            logger.error(f"Error generating cache key: {e}")
            return str(time.time())
        
    def _check_cache(self, cache_key: str) -> Optional[Response]:
        """Check if response exists in cache"""
        try:
            cached_data = self.response_cache.get(cache_key)
            if cached_data:
                return Response(
                    text=cached_data.get('text'),
                    audio=cached_data.get('audio'),
                    source=cached_data.get('source', 'cache'),
                    cache_key=cache_key
                )
            return None
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return None

    def _cache_response(self, cache_key: str, response: Response):
        """Cache a response"""
        try:
            self.response_cache[cache_key] = {
                'text': response.text,
                'source': response.source,
                'timestamp': time.time()
            }
            # Don't cache audio to save memory
            self._save_cache()
        except Exception as e:
            logger.error(f"Error caching response: {e}")

    def _load_cache(self) -> dict:
        """Load response cache from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            return {}

    def _save_cache(self):
        """Save cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.response_cache, f)
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

    async def generate_response(self, query: Query) -> Response:
        """Generate response for the given query"""
        try:
            cache_key = self._generate_cache_key(query)
            
            # Check cache for non-time-sensitive queries
            if not query.text.lower().startswith(('what time', 'what date', 'weather')):
                cached = self._check_cache(cache_key)
                if cached:
                    return cached
            
            # Try local processing first
            response = await self.local_processor.process(query)
            
            # If no local response, use cloud
            if not response:
                response = await self.cloud_processor.process(query)
            
            # Cache if appropriate
            if response and response.text:
                self._cache_response(cache_key, response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return Response(
                text="I'm having trouble understanding. Could you please try again?",
                source="error"
            )

    def cleanup(self):
        """Clean up resources"""
        self._save_cache()


class LocalProcessor:
    def __init__(self):
        """Initialize LocalProcessor with command templates"""
        self.command_templates = {
            "hello": "Hello! How can I help you today?",
            "goodbye": "Goodbye! Have a great day!",
            "thank you": "You're welcome! Is there anything else I can help you with?",
            "help": "I'm here to help. What can I assist you with?",
            "how are you": "I'm doing well, thank you! How can I assist you today?"
        }

    async def process(self, query: Query) -> Optional[Response]:
        """Process simple queries locally"""
        text_lower = query.text.lower()
        
        # Check for time query
        if 'time' in text_lower:
            current_time = datetime.now().strftime("%I:%M %p")
            return Response(
                text=f"It's currently {current_time}.",
                source="local"
            )
        
        # Check templates
        for key, response_text in self.command_templates.items():
            if key in text_lower:
                return Response(text=response_text, source="local")
        
        return None


class CloudProcessor:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)
        self.conversation_history = {}

    async def process(self, query: Query) -> Response:
        """Process queries using OpenAI's GPT"""
        try:
            # Get or create conversation history
            history = self.conversation_history.get(query.speaker_id, [])
            
            # Prepare messages
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Keep responses concise and friendly."},
                *history[-4:],  # Include last 2 exchanges
                {"role": "user", "content": query.text}
            ]
            
            # Get completion from OpenAI
            completion = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use gpt-3.5 for faster responses
                messages=messages,
                temperature=0.7,
                max_tokens=150
            )
            
            response_text = completion.choices[0].message.content
            
            # Update history
            history.extend([
                {"role": "user", "content": query.text},
                {"role": "assistant", "content": response_text}
            ])
            self.conversation_history[query.speaker_id] = history[-10:]  # Keep last 5 exchanges
            
            return Response(
                text=response_text,
                source="cloud"
            )
            
        except Exception as e:
            logger.error(f"Error in cloud processing: {e}")
            return Response(
                text="I apologize, but I'm having trouble connecting to my knowledge base. Please try again.",
                source="error"
            )


class TTSEngine:
    def __init__(self, api_key: str, voice_id: str = "nova"):
        self.client = openai.OpenAI(api_key=api_key)
        self.voice_id = voice_id
    
    async def generate_speech(self, text: str) -> bytes:
        """Generate speech from text using OpenAI's TTS"""
        try:
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice_id,
                input=text
            )
            return response.content
            
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            return None