# audio_cloud.py - Cloud-compatible version without PyAudio
import numpy as np
from collections import deque
from queue import Queue
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class AudioInputSystem:
    """Cloud-compatible audio system - processes audio files only, no real-time capture"""
    
    def __init__(self):
        # Audio parameters
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK_DURATION_MS = 30
        self.CHUNK = int(self.RATE * self.CHUNK_DURATION_MS / 1000)
        
        # Buffer setup
        self.buffer_duration_seconds = 3
        self.buffer_size = int(self.RATE * self.buffer_duration_seconds)
        self.audio_buffer = deque(maxlen=self.buffer_size)
        
        # Queue for processing
        self.voice_segments_queue = Queue()
        
        # Control flags
        self.is_running = False
        self.is_paused = False
        
        logger.info("Cloud-compatible audio system initialized")

    def process_audio_file(self, audio_data: np.ndarray, sample_rate: int):
        """Process audio data from file upload"""
        try:
            # Resample if necessary
            if sample_rate != self.RATE:
                # Simple resampling - in production use librosa or scipy
                ratio = self.RATE / sample_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Add to buffer
            self.audio_buffer.extend(audio_data)
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            raise

    def get_audio_buffer(self):
        """Get the current contents of the audio buffer"""
        return np.array(list(self.audio_buffer))

    def clear_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer.clear()

    # Stub methods for compatibility
    def start_audio_stream(self):
        logger.warning("Real-time audio streaming not available in cloud environment")
        pass
    
    def stop(self):
        logger.info("Audio system stopped")
        pass
    
    def pause(self):
        self.is_paused = True
        
    def resume(self):
        self.is_paused = False


class AudioBufferManager:
    """Manages the audio buffer for cloud processing"""
    
    def __init__(self, max_duration_seconds: int = 3, sample_rate: int = 16000):
        self.max_duration = max_duration_seconds
        self.sample_rate = sample_rate
        self.max_samples = max_duration_seconds * sample_rate
        self.buffer = deque(maxlen=self.max_samples)
        
    def add_audio(self, audio_data: np.ndarray):
        """Add audio data to the buffer"""
        self.buffer.extend(audio_data)
    
    def get_latest_audio(self, duration_seconds: float = None) -> np.ndarray:
        """Get the latest audio from the buffer"""
        if duration_seconds is None:
            return np.array(list(self.buffer))
        
        samples = int(duration_seconds * self.sample_rate)
        return np.array(list(self.buffer)[-samples:])
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
    
    def process_wav_data(self, wav_data: bytes, sample_rate: int) -> np.ndarray:
        """Process WAV data from uploaded file"""
        try:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(wav_data, dtype=np.int16).astype(np.float32) / 32767.0
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                # Simple resampling
                ratio = self.sample_rate / sample_rate
                new_length = int(len(audio_data) * ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Error processing WAV data: {e}")
            raise