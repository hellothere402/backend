import numpy as np
import openai
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import os
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class VoiceProfile:
    user_id: str
    embeddings: List[np.ndarray]
    name: str
    created_at: float

@dataclass
class ProcessedSpeech:
    text: str
    confidence: float
    speaker_id: Optional[str]
    intent: Optional[str]

class IntentClassifier:
    def __init__(self):
        self.intents = {
            "query": ["what", "how", "why", "when", "where", "who"],
            "command": ["set", "turn", "play", "stop", "start", "open"],
            "conversation": ["tell", "chat", "talk", "discuss"],
            "emergency": ["help", "emergency", "urgent", "pain"]
        }

    def classify(self, text: str) -> str:
        """Classify the intent of the text"""
        try:
            text = text.lower()
            
            # Check emergency intent first
            if any(word in text for word in self.intents["emergency"]):
                return "emergency"
                
            # Check other intents
            for intent, keywords in self.intents.items():
                if any(word in text.split() for word in keywords):
                    return intent
                    
            # Default to conversation
            return "conversation"
        except Exception as e:
            logger.error(f"Error in intent classification: {e}")
            return "conversation"

class VoiceProcessingSystem:
    def __init__(self, openai_api_key: str):
        logger.info("Initializing Voice Processing System...")
        
        # Initialize OpenAI client correctly (without proxies argument)
        self.client = openai.OpenAI(api_key=openai_api_key)
        
        # For cloud deployment, we'll use simplified speaker recognition
        self.speaker_recognizer = SimplifiedSpeakerRecognition()
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier()
        
        # Load voice profiles
        self.voice_profiles = self._load_voice_profiles()
        logger.info("Voice Processing System initialized")

    def process_voice(self, audio_data: np.ndarray, sample_rate: int) -> Optional[ProcessedSpeech]:
        """Process voice data through the pipeline"""
        try:
            # Ensure audio data is properly shaped
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(1, -1)
            elif audio_data.shape[0] != 1:
                audio_data = audio_data.reshape(1, -1)
            
            logger.info(f"Processing audio data with shape: {audio_data.shape}")
            
            # For now, skip speaker recognition in cloud (it's complex)
            # Just use a default speaker
            speaker_id = "web_user"
            
            # Transcribe audio using Whisper
            text, confidence = self._transcribe_audio(audio_data, sample_rate)
            
            if not text:
                logger.warning("No text transcribed")
                return None
            
            logger.info(f"Transcribed text: {text}")
            
            # Classify intent
            intent = self.intent_classifier.classify(text)
            logger.info(f"Detected intent: {intent}")
            
            return ProcessedSpeech(
                text=text,
                confidence=confidence,
                speaker_id=speaker_id,
                intent=intent
            )
            
        except Exception as e:
            logger.error(f"Error in voice processing: {e}")
            return None

    def _transcribe_audio(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[str, float]:
        """Transcribe audio using Whisper API"""
        import tempfile
        import wave
        
        temp_filename = None
        try:
            # Create a temporary WAV file
            temp_filename = f"/tmp/temp_audio_{int(time.time())}.wav"
            
            # Save audio to WAV file
            with wave.open(temp_filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                audio_int16 = (audio_data * 32767).astype(np.int16)
                wf.writeframes(audio_int16.tobytes())
            
            # Open and transcribe
            with open(temp_filename, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            
            return response, 1.0
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return "", 0.0
        finally:
            # Clean up temp file
            if temp_filename and os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    logger.warning(f"Could not remove temp file: {e}")

    def create_voice_profile(self, audio_data: np.ndarray, name: str) -> Optional[VoiceProfile]:
        """Create new voice profile from audio data"""
        try:
            logger.info("Creating voice profile...")
            
            # For cloud deployment, use simplified embeddings
            # Just store some basic statistics as "embeddings"
            embeddings = []
            
            # Create simple audio features
            segment_length = 16000 * 2  # 2 seconds
            for i in range(0, len(audio_data[0]), segment_length):
                segment = audio_data[:, i:i+segment_length]
                if len(segment[0]) >= segment_length:
                    # Simple feature: mean and std of segment
                    feature = np.array([
                        np.mean(segment),
                        np.std(segment),
                        np.max(np.abs(segment))
                    ])
                    embeddings.append(feature)
            
            if not embeddings:
                logger.error("No valid embeddings generated")
                return None
            
            user_id = f"user_{len(self.voice_profiles)}"
            
            profile = VoiceProfile(
                user_id=user_id,
                embeddings=embeddings,
                name=name,
                created_at=time.time()
            )
            
            self.voice_profiles[user_id] = profile
            self._save_voice_profiles()
            
            logger.info(f"Created profile for {name} (ID: {user_id})")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating voice profile: {e}")
            return None

    def _load_voice_profiles(self) -> Dict[str, VoiceProfile]:
        """Load voice profiles from storage"""
        try:
            if not os.path.exists("voice_profiles.json"):
                return {}
            
            with open("voice_profiles.json", "r") as f:
                profiles_data = json.load(f)
            
            profiles = {}
            for profile in profiles_data:
                profiles[profile["user_id"]] = VoiceProfile(
                    user_id=profile["user_id"],
                    embeddings=[np.array(emb) for emb in profile["embeddings"]],
                    name=profile["name"],
                    created_at=profile["created_at"]
                )
            
            logger.info(f"Loaded {len(profiles)} voice profiles")
            return profiles
            
        except Exception as e:
            logger.warning(f"Error loading voice profiles: {e}")
            return {}

    def _save_voice_profiles(self):
        """Save voice profiles to storage"""
        try:
            profiles_data = []
            for profile in self.voice_profiles.values():
                profiles_data.append({
                    "user_id": profile.user_id,
                    "embeddings": [emb.tolist() for emb in profile.embeddings],
                    "name": profile.name,
                    "created_at": profile.created_at
                })
            
            with open("voice_profiles.json", "w") as f:
                json.dump(profiles_data, f)
            
            logger.info("Voice profiles saved")
            
        except Exception as e:
            logger.error(f"Error saving voice profiles: {e}")


class SimplifiedSpeakerRecognition:
    """Simplified speaker recognition for cloud deployment"""
    
    def __init__(self, similarity_threshold: float = 0.6):
        self.similarity_threshold = similarity_threshold
        logger.info("Simplified speaker recognition initialized")

    def identify_speaker(self, audio_data: np.ndarray, profiles: Dict[str, VoiceProfile]) -> Optional[str]:
        """Simple speaker identification - just returns default for cloud"""
        # In cloud deployment, we'll skip complex speaker recognition
        # and just use a default user
        return "web_user" if profiles else None

    def generate_embedding(self, audio_data: np.ndarray) -> np.ndarray:
        """Generate simple embedding from audio"""
        # Simple features for cloud deployment
        return np.array([
            np.mean(audio_data),
            np.std(audio_data),
            np.max(np.abs(audio_data))
        ])