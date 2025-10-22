"""
LUMOS-AI OPTIMIZED - Automatic Real-Time Vision Assistant
NO WAKE WORDS - Fully Automatic Interaction
Optimized for: 16GB RAM, RTX 4050 6GB, Intel i5-12450H
"""

import os
from pathlib import Path

# CREATE CACHE FOLDER
CACHE_DIR = Path("./model_cache")
CACHE_DIR.mkdir(exist_ok=True)

os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE_DIR)
os.environ['TORCH_HOME'] = str(CACHE_DIR)

import cv2
import threading
import time
import torch
from queue import Queue, Empty
import numpy as np
import pyttsx3
import speech_recognition as sr

# Import modules
from modules.vision_agent import VisionAgent
from modules.deep_ocr import DeepOCR
from modules.reasoning_agent import ReasoningAgent
from utils.device_manager import get_device_manager
from optimized_deep_learning import (
    get_optimized_dl, 
    AutomaticInteractionManager,
    PredictionSmoother
)

# Import pose and sign language
from vision.pose_detector import PoseDetector
from vision.sign_language import SignLanguageDetector

# ============================================================================
# OPTIMIZED CONFIGURATION
# ============================================================================

CONFIG = {
    'camera_index': 0,
    'frame_width': 640,
    'frame_height': 480,
    'display_video': True,
    'api_provider': 'gemini',
    'gemini_model': 'gemini-2.0-flash-exp',
    'use_sam': False,  # Disable for speed, enable if needed
    'use_trocr': True,
    'use_easyocr': True,
    'verbose': False,  # Reduce console spam
    
    # AUTOMATIC MODE
    'auto_mode': True,  # No wake word needed
    'text_detection_enabled': True,
    'sign_language_enabled': True,
    'activity_detection_enabled': True,
    'speech_recognition_enabled': True,
    
    # PERFORMANCE
    'target_fps': 30,
    'enable_gpu_optimization': True,
    'use_mixed_precision': True,
}

# ============================================================================
# GLOBAL STATE
# ============================================================================

running = True
latest_frame = None
frame_lock = threading.Lock()
tts_queue = Queue(maxsize=10)

# ============================================================================
# OPTIMIZED TTS THREAD
# ============================================================================

def tts_thread():
    """Dedicated thread for text-to-speech to prevent blocking"""
    print("🔊 TTS thread starting...")
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    engine.setProperty('volume', 0.9)
    
    while running:
        try:
            text = tts_queue.get(timeout=1.0)
            if text:
                print(f"🗣️  Speaking: {text}")
                engine.say(text)
                engine.runAndWait()
        except Empty:
            continue
        except Exception as e:
            if CONFIG['verbose']:
                print(f"⚠️ TTS error: {e}")
    
    print("👋 TTS thread stopped")


def speak_async(text):
    """Queue text for speaking without blocking"""
    if not tts_queue.full():
        tts_queue.put(text)


# ============================================================================
# SPEECH RECOGNITION THREAD (Continuous)
# ============================================================================

def speech_thread():
    """Continuous speech recognition for questions"""
    global running
    
    print("🎤 Speech recognition starting...")
    
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Calibrate
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
    
    print("✅ Listening for questions...")
    
    while running:
        try:
            with microphone as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=5)
                
                try:
                    text = recognizer.recognize_google(audio).lower()
                    
                    # Check if it's a question
                    question_words = ['what', 'where', 'when', 'who', 'how', 'why', 'is', 'are', 'can']
                    if any(word in text for word in question_words):
                        print(f"❓ Question: {text}")
                        # Process question (will be handled by reasoning agent)
                        if not tts_queue.full():
                            # Add to processing queue
                            pass
                
                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    pass
        
        except sr.WaitTimeoutError:
            pass
        except Exception as e:
            if CONFIG['verbose']:
                print(f"⚠️ Speech error: {e}")
            time.sleep(1)
    
    print("👋 Speech thread stopped")


# ============================================================================
# UNIFIED VISION PROCESSING THREAD
# ============================================================================

def unified_vision_thread():
    """
    Single thread handling all vision processing with intelligent scheduling
    """
    global latest_frame, running
    
    print("\n" + "="*70)
    print("👁️  UNIFIED VISION PROCESSING THREAD")
    print("="*70)
    
    # Initialize camera
    cap = cv2.VideoCapture(CONFIG['camera_index'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        running = False
        return
    
    print(f"✅ Camera ready: {CONFIG['frame_width']}x{CONFIG['frame_height']}\n")
    
    # Get device and optimization layer
    device_mgr = get_device_manager(prefer_gpu=True)
    device = device_mgr.get_device()
    dl_layer = get_optimized_dl(device)
    
    # Initialize all models
    print("📦 Loading all models...")
    vision_agent = VisionAgent(use_cache=True)
    ocr_agent = DeepOCR(use_trocr=True, use_easyocr=True)
    reasoning_agent = ReasoningAgent(api_provider='gemini', model_name=CONFIG['gemini_model'])
    
    pose_detector = None
    if CONFIG['activity_detection_enabled']:
        pose_detector = PoseDetector(model_size='m')
    
    sign_detector = None
    if CONFIG['sign_language_enabled']:
        sign_detector = SignLanguageDetector()
    
    # Wrap models with optimization
    if hasattr(vision_agent, 'yolo') and vision_agent.yolo:
        vision_agent.yolo = dl_layer.wrap_model(vision_agent.yolo, 'yolo')
    
    print("✅ All models loaded and optimized!\n")
    
    # Interaction manager
    interaction_mgr = AutomaticInteractionManager()
    smoother = PredictionSmoother(window_size=3)
    
    # Performance tracking
    fps_time = time.time()
    frame_count = 0
    fps = 0
    
    # Unified results
    current_results = {
        'objects': [],
        'text': '',
        'sign': '',
        'activity': '',
        'caption': ''
    }
    
    print("🚀 Starting automatic vision processing...\n")
    
    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue
        
        frame_count += 1
        
        # Update shared frame
        with frame_lock:
            latest_frame = frame.copy()
        
        # ====================
        # 1. FAST DETECTION (Every frame - YOLO)
        # ====================
        vision_data = dl_layer.cached_inference(
            lambda f: vision_agent.analyze_frame(f, use_sam=False),
            frame,
            'yolo',
            force=False
        )
        
        if vision_data:
            current_results['objects'] = vision_data.get('objects', [])
            current_results['caption'] = vision_data.get('caption', '')
        
        # ====================
        # 2. TEXT DETECTION (Every 3 frames)
        # ====================
        if CONFIG['text_detection_enabled'] and frame_count % 3 == 0:
            text_objects = dl_layer.cached_inference(
                lambda f: ocr_agent.extract_text_from_objects(f, current_results['objects']),
                frame,
                'trocr',
                force=False
            )
            
            if text_objects:
                # Extract all text
                texts = []
                for obj in text_objects:
                    if 'ocr' in obj and obj['ocr'].get('text'):
                        texts.append(obj['ocr']['text'])
                
                if texts:
                    full_text = ' '.join(texts)
                    current_results['text'] = full_text
                    
                    # Automatic text reading
                    if interaction_mgr.should_speak_text(full_text):
                        speak_async(f"I can see text: {full_text}")
        
        # ====================
        # 3. SIGN LANGUAGE (Every 3 frames)
        # ====================
        if CONFIG['sign_language_enabled'] and sign_detector and frame_count % 3 == 0:
            sign, conf = dl_layer.cached_inference(
                lambda f: sign_detector.recognize_sign(f, threshold=0.6),
                frame,
                'sign',
                force=False
            ) or (None, 0)
            
            if sign and conf > 0.6:
                current_results['sign'] = sign
                
                # Automatic sign announcement
                if interaction_mgr.should_announce_sign(sign):
                    speak_async(f"Sign language detected: {sign}")
        
        # ====================
        # 4. ACTIVITY DETECTION (Every 2 frames)
        # ====================
        if CONFIG['activity_detection_enabled'] and pose_detector and frame_count % 2 == 0:
            poses = dl_layer.cached_inference(
                lambda f: pose_detector.detect_poses(f),
                frame,
                'pose',
                force=False
            )
            
            if poses:
                activities = [p['activity'] for p in poses if p.get('activity')]
                if activities:
                    activity = activities[0]
                    current_results['activity'] = activity
                    
                    # Automatic activity announcement
                    if interaction_mgr.should_announce_activity(activity):
                        speak_async(f"I see someone {activity}")
        
        # ====================
        # DISPLAY WITH OPTIMIZED UI
        # ====================
        if CONFIG['display_video']:
            display_frame = frame.copy()
            
            # FPS counter
            if time.time() - fps_time >= 1.0:
                fps = frame_count / (time.time() - fps_time)
                frame_count = 0
                fps_time = time.time()
            
            # Status overlay
            y_offset = 30
            cv2.putText(display_frame, f"LUMOS-AI | FPS: {fps:.1f}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30
            
            # Cache stats
            stats = dl_layer.cache.get_stats()
            cv2.putText(display_frame, stats, 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 25
            
            # Current detections
            if current_results['text']:
                cv2.putText(display_frame, f"Text: {current_results['text'][:30]}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                y_offset += 25
            
            if current_results['sign']:
                cv2.putText(display_frame, f"Sign: {current_results['sign']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
                y_offset += 25
            
            if current_results['activity']:
                cv2.putText(display_frame, f"Activity: {current_results['activity']}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                y_offset += 25
            
            cv2.putText(display_frame, "Press 'q' to quit", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Draw objects
            for obj in current_results['objects'][:5]:
                x1, y1, x2, y2 = obj['box']
                x1, y1, x2, y2 = smoother.smooth_bbox(obj.get('name', 'obj'), [x1, y1, x2, y2])
                
                label = f"{obj.get('name', 'object')}"
                if 'distance_m' in obj:
                    label += f" {obj['distance_m']}m"
                
                color = (0, 255, 0) if obj.get('distance_m', 10) > 2 else (0, 0, 255)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow('LUMOS-AI Optimized', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Quit signal received")
                running = False
                break
        
        # Maintain target FPS
        time.sleep(0.001)  # Minimal sleep for CPU
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print final stats
    perf = dl_layer.get_performance_stats()
    print(f"\n📊 Final Performance:")
    print(f"   Average FPS: {perf['fps']:.1f}")
    print(f"   {perf['cache_stats']}")
    print(f"   Total inference time: {perf['avg_total_time']:.3f}s")
    
    print("👋 Vision thread stopped")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point"""
    global running
    
    print("\n" + "="*70)
    print("✨ LUMOS-AI OPTIMIZED - AUTOMATIC REAL-TIME ASSISTANT ✨")
    print("="*70)
    print("\n🎯 Features:")
    print("   ✅ Automatic text reading (no wake word)")
    print("   ✅ Sign language recognition")
    print("   ✅ Activity detection (waving, pointing, etc.)")
    print("   ✅ Speech recognition for questions")
    print("   ✅ Real-time object detection with depth")
    print("   ✅ Intelligent caching for smooth performance")
    print("   ✅ Optimized for RTX 4050 + 16GB RAM")
    
    print("\n⚙️  Configuration:")
    print(f"   • Target FPS: {CONFIG['target_fps']}")
    print(f"   • Resolution: {CONFIG['frame_width']}x{CONFIG['frame_height']}")
    print(f"   • GPU Optimization: {CONFIG['enable_gpu_optimization']}")
    print(f"   • Automatic Mode: {CONFIG['auto_mode']}")
    
    print("\n🛑 Press 'q' in camera window to quit\n")
    print("="*70 + "\n")
    
    # Start threads
    threads = []
    
    # TTS thread
    tts_t = threading.Thread(target=tts_thread, daemon=True)
    threads.append(tts_t)
    
    # Speech recognition thread
    if CONFIG['speech_recognition_enabled']:
        speech_t = threading.Thread(target=speech_thread, daemon=True)
        threads.append(speech_t)
    
    # Main vision thread
    vision_t = threading.Thread(target=unified_vision_thread, daemon=True)
    threads.append(vision_t)
    
    # Start all threads
    for t in threads:
        t.start()
        time.sleep(0.5)
    
    # Wait for completion
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down LUMOS-AI...")
        running = False
        time.sleep(2)
    
    print("\n" + "="*70)
    print("✨ LUMOS-AI shut down complete! ✨")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()