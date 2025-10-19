"""
Speaker Module
Handles text-to-speech output using pyttsx3.
"""

import pyttsx3
import threading

class Speaker:
    """
    Text-to-speech engine for voice narration.
    """
    
    def __init__(self, rate=165, volume=0.9):
        """
        Initialize TTS engine.
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        print("🔊 Initializing text-to-speech engine...")
        
        # Initialize engine
        self.engine = pyttsx3.init()
        
        # Configure voice properties
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        
        # Try to set a clear voice
        voices = self.engine.getProperty('voices')
        if voices:
            # Prefer female voice if available (often clearer)
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
        
        # Lock for thread-safe speaking
        self.speak_lock = threading.Lock()
        
        print("✅ Text-to-speech ready")
        
    def speak(self, text):
        """
        Speak the given text.
        
        Args:
            text: String to speak
        """
        if not text:
            return
        
        # Thread-safe speaking
        with self.speak_lock:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"⚠️ TTS error: {e}")
    
    def speak_async(self, text):
        """
        Speak text in a separate thread (non-blocking).
        
        Args:
            text: String to speak
        """
        thread = threading.Thread(target=self.speak, args=(text,), daemon=True)
        thread.start()
    
    def stop(self):
        """Stop current speech."""
        try:
            self.engine.stop()
        except:
            pass
    
    def set_rate(self, rate):
        """
        Change speech rate.
        
        Args:
            rate: New rate (words per minute)
        """
        self.engine.setProperty('rate', rate)
    
    def set_volume(self, volume):
        """
        Change volume.
        
        Args:
            volume: Volume level (0.0 to 1.0)
        """
        self.engine.setProperty('volume', volume)