"""
LUMOS-AI Next-Gen - Main Orchestrator
ENHANCED: Live voice interaction, Gemini 2.0 Flash, Camera display
"""

import os
from pathlib import Path

# CREATE CACHE FOLDER TO STORE MODELS
CACHE_DIR = Path("./model_cache")
CACHE_DIR.mkdir(exist_ok=True)

# TELL HUGGINGFACE TO USE THIS FOLDER
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)
os.environ['HF_HOME'] = str(CACHE_DIR)
os.environ['HUGGINGFACE_HUB_CACHE'] = str(CACHE_DIR)

import cv2
import threading
import time
import json
from queue import Queue
from pathlib import Path
import numpy as np

# Import all agents
from modules.vision_agent import VisionAgent
from modules.deep_ocr import DeepOCR
from modules.feedback_agent import FeedbackAgent
from modules.reasoning_agent import ReasoningAgent
from utils.device_manager import get_device_manager

# Audio output and input
import pyttsx3
import speech_recognition as sr


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'camera_index': 0,
    'frame_width': 640,
    'frame_height': 480,
    'narration_interval': 4.0,
    'display_video': True,  # ENABLED for camera window
    'api_provider': 'gemini',
    'gemini_model': 'gemini-2.0-flash-exp',  # UPGRADED to Gemini 2.0 Flash
    'sam_enabled': True,
    'use_sam': True,
    'use_trocr': True,
    'use_easyocr': True,
    'enable_feedback': True,
    'verbose': True,
    'voice_interaction': True,  # NEW: Enable voice commands
    'wake_word': 'lumos',  # NEW: Wake word to activate
    'auto_narrate': True,  # NEW: Automatic narration mode
}


# ============================================================================
# GLOBAL STATE
# ============================================================================

running = True
latest_frame = None
frame_lock = threading.Lock()
voice_command_queue = Queue(maxsize=5)
listening_active = False


# ============================================================================
# VOICE INPUT THREAD
# ============================================================================

def voice_input_thread():
    """
    Continuously listens for voice commands.
    Activates on wake word "lumos" or continuously if auto mode.
    """
    global running, listening_active
    
    print("\n" + "="*70)
    print("🎤 VOICE INPUT THREAD STARTING")
    print("="*70)
    
    # Initialize speech recognizer
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    
    # Adjust for ambient noise
    print("🎧 Calibrating microphone for ambient noise...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=2)
    print("✅ Microphone ready!\n")
    
    wake_word = CONFIG['wake_word'].lower()
    
    print(f"💡 Say '{CONFIG['wake_word'].upper()}' to interact, or 'stop listening' to pause")
    print(f"💡 Auto-narration is {'ON' if CONFIG['auto_narrate'] else 'OFF'}\n")
    
    while running:
        try:
            with microphone as source:
                print("🎤 Listening..." if listening_active else "🎤 Ready (say wake word)...")
                
                # Listen for audio
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                try:
                    # Recognize speech
                    text = recognizer.recognize_google(audio).lower()
                    print(f"👂 Heard: '{text}'")
                    
                    # Check for wake word
                    if wake_word in text and not listening_active:
                        listening_active = True
                        print(f"✨ {CONFIG['wake_word'].upper()} activated! Listening for commands...")
                        continue
                    
                    # Check for stop command
                    if "stop listening" in text or "pause" in text:
                        listening_active = False
                        print("⏸️  Voice interaction paused. Say wake word to resume.")
                        continue
                    
                    # Process commands if listening or wake word detected
                    if listening_active or wake_word in text:
                        # Remove wake word from command
                        command = text.replace(wake_word, "").strip()
                        
                        if command:
                            if not voice_command_queue.full():
                                voice_command_queue.put(command)
                                print(f"📝 Command queued: '{command}'")
                        
                        # Auto-deactivate after command (unless in continuous mode)
                        if not CONFIG['auto_narrate']:
                            listening_active = False
                
                except sr.UnknownValueError:
                    # Speech not understood
                    pass
                except sr.RequestError as e:
                    print(f"⚠️ Speech recognition error: {e}")
                    time.sleep(2)
        
        except sr.WaitTimeoutError:
            # No speech detected within timeout
            pass
        except Exception as e:
            print(f"⚠️ Voice input error: {e}")
            time.sleep(1)
    
    print("👋 Voice input thread stopped")


# ============================================================================
# VISION PROCESSING THREAD
# ============================================================================

def vision_thread(vision_queue, ocr_queue):
    """
    Continuously captures and processes video frames.
    Runs unified vision analysis with all models.
    """
    global latest_frame, running
    
    print("\n" + "="*70)
    print("🎥 VISION THREAD STARTING")
    print("="*70)
    
    # Initialize camera
    cap = cv2.VideoCapture(CONFIG['camera_index'])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['frame_width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['frame_height'])
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        running = False
        return
    
    print(f"✅ Camera opened: {CONFIG['frame_width']}x{CONFIG['frame_height']}\n")
    
    # Initialize vision agent
    vision_agent = VisionAgent(use_cache=True)
    
    last_process_time = time.time()
    frame_count = 0
    
    # Initialize vision_output with empty data
    vision_output = {
        'objects': [],
        'caption': 'Initializing...',
        'from_cache': False
    }
    
    while running:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to capture frame")
            time.sleep(0.1)
            continue
        
        frame_count += 1
        
        # Update shared frame for display
        with frame_lock:
            latest_frame = frame.copy()
        
        # Process at intervals
        current_time = time.time()
        if current_time - last_process_time >= CONFIG['narration_interval']:
            try:
                if CONFIG['verbose']:
                    print(f"🔍 Processing frame {frame_count}...")
                
                # Run complete vision analysis
                vision_output = vision_agent.analyze_frame(
                    frame,
                    use_sam=CONFIG['use_sam']
                )
                
                # Send to OCR and reasoning queues
                if not vision_queue.full():
                    vision_queue.put({
                        'frame': frame.copy(),
                        'vision_data': vision_output,
                        'timestamp': current_time
                    })
                
                if not ocr_queue.full():
                    ocr_queue.put({
                        'frame': frame.copy(),
                        'objects': vision_output['objects'],
                        'timestamp': current_time
                    })
                
                last_process_time = current_time
                
                if CONFIG['verbose']:
                    obj_count = len(vision_output['objects'])
                    cached = vision_output.get('from_cache', False)
                    print(f"✅ Vision: {obj_count} objects detected "
                          f"{'(cached)' if cached else ''}")
                
            except Exception as e:
                print(f"❌ Vision processing error: {e}")
                import traceback
                traceback.print_exc()
        
        # Display video with enhanced UI
        if CONFIG['display_video']:
            display_frame = frame.copy()
            
            # Add status overlay (NOW SAFE - vision_output is always defined)
            obj_count = len(vision_output.get('objects', []))
            fps = int(1/(current_time - last_process_time + 0.001))
            status = f"LUMOS-AI | Objects: {obj_count} | FPS: {fps}"
            
            cv2.putText(display_frame, status, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Voice status
            voice_status = "🎤 LISTENING" if listening_active else f"💤 Say '{CONFIG['wake_word'].upper()}'"
            cv2.putText(display_frame, voice_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            cv2.putText(display_frame, "Press 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw detected objects
            if 'objects' in vision_output:
                for obj in vision_output['objects'][:5]:  # Show top 5
                    x1, y1, x2, y2 = obj['box']
                    label = f"{obj.get('name', 'object')}"
                    if 'distance_m' in obj:
                        label += f" {obj['distance_m']}m"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow('LUMOS-AI Vision', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Quit signal received")
                running = False
                break
        
        time.sleep(0.03)  # ~30 FPS
    
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Vision thread stopped")


# ============================================================================
# OCR PROCESSING THREAD
# ============================================================================

def ocr_thread(ocr_queue, text_queue):
    """
    Processes OCR on detected objects and text regions.
    """
    global running
    
    print("\n" + "="*70)
    print("📖 OCR THREAD STARTING")
    print("="*70 + "\n")
    
    # Initialize OCR
    ocr_agent = DeepOCR(
        use_trocr=CONFIG['use_trocr'],
        use_easyocr=CONFIG['use_easyocr']
    )
    
    while running:
        try:
            if not ocr_queue.empty():
                data = ocr_queue.get(timeout=1.0)
                
                frame = data['frame']
                objects = data['objects']
                
                # Extract text from objects
                ocr_results = ocr_agent.extract_text_from_objects(frame, objects)
                
                # Filter objects with text
                text_objects = [obj for obj in ocr_results if 'ocr' in obj]
                
                if text_objects:
                    if CONFIG['verbose']:
                        print(f"📝 OCR: Found text in {len(text_objects)} objects")
                    
                    # Send to reasoning thread
                    if not text_queue.full():
                        text_queue.put({
                            'text_objects': text_objects,
                            'timestamp': data['timestamp']
                        })
            else:
                time.sleep(0.1)
                
        except Exception as e:
            if CONFIG['verbose']:
                print(f"⚠️ OCR error: {e}")
            time.sleep(1)
    
    print("👋 OCR thread stopped")


# ============================================================================
# REASONING & NARRATION THREAD
# ============================================================================

def reasoning_thread(vision_queue, text_queue):
    """
    Generates natural language narration and speaks to user.
    Responds to voice commands.
    """
    global running
    
    print("\n" + "="*70)
    print("🧠 REASONING THREAD STARTING")
    print("="*70 + "\n")
    
    # Initialize reasoning agent with Gemini 2.0 Flash
    reasoning_agent = ReasoningAgent(
        api_provider=CONFIG['api_provider'],
        model_name=CONFIG['gemini_model']
    )
    
    # Initialize TTS
    print("🔊 Initializing text-to-speech...")
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 165)
    tts_engine.setProperty('volume', 0.9)
    print("✅ TTS ready\n")
    
    # Initialize feedback agent
    feedback_agent = None
    if CONFIG['enable_feedback']:
        feedback_agent = FeedbackAgent()
    
    print("🎤 LUMOS-AI is now active with voice interaction!\n")
    
    last_narration_time = time.time()
    
    while running:
        try:
            # Check for voice commands first
            if not voice_command_queue.empty():
                command = voice_command_queue.get()
                
                # Process specific commands
                if "what do you see" in command or "describe" in command:
                    print(f"🎯 Processing command: '{command}'")
                    
                    # Force immediate vision analysis
                    if not vision_queue.empty():
                        data = vision_queue.get()
                        vision_data = data['vision_data']
                        
                        # Generate narration
                        narration = reasoning_agent.generate_narration(vision_data)
                        print(f"\n🗣️  LUMOS: {narration}\n")
                        
                        try:
                            tts_engine.say(narration)
                            tts_engine.runAndWait()
                        except Exception as e:
                            print(f"⚠️ TTS error: {e}")
                
                elif "read text" in command or "what does it say" in command:
                    print(f"🎯 Processing command: '{command}'")
                    
                    # Get latest text detection
                    if not text_queue.empty():
                        text_data = text_queue.get()
                        text_objects = text_data['text_objects']
                        
                        texts = []
                        for obj in text_objects:
                            if 'ocr' in obj and obj['ocr']['text']:
                                texts.append(obj['ocr']['text'])
                        
                        if texts:
                            response = f"I can see the following text: {', '.join(texts)}"
                        else:
                            response = "I don't see any readable text right now."
                        
                        print(f"\n🗣️  LUMOS: {response}\n")
                        try:
                            tts_engine.say(response)
                            tts_engine.runAndWait()
                        except Exception as e:
                            print(f"⚠️ TTS error: {e}")
                
                elif "how many" in command or "count" in command:
                    print(f"🎯 Processing command: '{command}'")
                    
                    if not vision_queue.empty():
                        data = vision_queue.get()
                        vision_data = data['vision_data']
                        obj_count = len(vision_data.get('objects', []))
                        
                        response = f"I can see {obj_count} objects in view."
                        print(f"\n🗣️  LUMOS: {response}\n")
                        
                        try:
                            tts_engine.say(response)
                            tts_engine.runAndWait()
                        except Exception as e:
                            print(f"⚠️ TTS error: {e}")
                
                else:
                    # General query - use Gemini to respond
                    response = reasoning_agent.answer_question(command)
                    print(f"\n🗣️  LUMOS: {response}\n")
                    
                    try:
                        tts_engine.say(response)
                        tts_engine.runAndWait()
                    except Exception as e:
                        print(f"⚠️ TTS error: {e}")
            
            # Auto-narration mode
            current_time = time.time()
            if CONFIG['auto_narrate'] and current_time - last_narration_time >= CONFIG['narration_interval']:
                if not vision_queue.empty():
                    data = vision_queue.get(timeout=1.0)
                    
                    vision_data = data['vision_data']
                    frame = data['frame']
                    
                    # Check for text data
                    text_detected = ""
                    if not text_queue.empty():
                        text_data = text_queue.get()
                        text_objects = text_data['text_objects']
                        
                        # Combine text from all objects
                        texts = []
                        for obj in text_objects:
                            if 'ocr' in obj and obj['ocr']['text']:
                                texts.append(obj['ocr']['text'])
                        
                        text_detected = ' '.join(texts)
                    
                    # Add text to vision data
                    vision_data['text_detected'] = text_detected
                    
                    # Generate narration
                    if CONFIG['verbose']:
                        print("🤔 Generating narration...")
                    
                    narration = reasoning_agent.generate_narration(vision_data)
                    
                    # Speak narration
                    print(f"\n🗣️  LUMOS: {narration}\n")
                    
                    try:
                        tts_engine.say(narration)
                        tts_engine.runAndWait()
                    except Exception as e:
                        print(f"⚠️ TTS error: {e}")
                    
                    last_narration_time = current_time
            else:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            running = False
            break
        except Exception as e:
            print(f"❌ Reasoning error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)
    
    # Save feedback on exit
    if feedback_agent:
        stats = feedback_agent.get_statistics()
        print(f"\n📊 Feedback Statistics:")
        print(f"   Total feedbacks: {stats['total_feedbacks']}")
        print(f"   Adapters trained: {stats['adapters_trained']}")
    
    print("👋 Reasoning thread stopped")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main entry point - orchestrates all threads.
    """
    global running
    
    print("\n" + "="*70)
    print("✨ LUMOS-AI NEXT-GEN - VOICE-INTERACTIVE VISION ASSISTANT ✨")
    print("="*70)
    print("\n🚀 Starting system...\n")
    
    print("📋 Features:")
    print("   ✅ Multi-model vision fusion (YOLO + DETR + SAM + MiDaS + BLIP-2)")
    print("   ✅ Advanced OCR (TrOCR + EasyOCR)")
    print("   ✅ AI reasoning (Gemini 2.0 Flash)")
    print("   ✅ Voice interaction with wake word")
    print("   ✅ Self-learning with LoRA adapters")
    print("   ✅ Natural language narration")
    print("   ✅ Live camera display")
    print("   ✅ Depth perception and spatial awareness")
    
    print("\n⚙️  Configuration:")
    print(f"   • Resolution: {CONFIG['frame_width']}x{CONFIG['frame_height']}")
    print(f"   • Narration interval: {CONFIG['narration_interval']}s")
    print(f"   • API: {CONFIG['api_provider']} ({CONFIG['gemini_model']})")
    print(f"   • Wake word: '{CONFIG['wake_word'].upper()}'")
    print(f"   • Voice interaction: {CONFIG['voice_interaction']}")
    print(f"   • Camera display: {CONFIG['display_video']}")
    
    # Initialize device
    device_mgr = get_device_manager(prefer_gpu=True)
    
    print("\n🛑 Press Ctrl+C or 'q' in camera window to quit\n")
    print("="*70 + "\n")
    
    # Create queues
    vision_queue = Queue(maxsize=2)
    ocr_queue = Queue(maxsize=2)
    text_queue = Queue(maxsize=2)
    
    # Start threads
    threads = []
    
    # Voice input thread
    if CONFIG['voice_interaction']:
        voice_t = threading.Thread(
            target=voice_input_thread,
            daemon=True
        )
        threads.append(voice_t)
    
    vision_t = threading.Thread(
        target=vision_thread,
        args=(vision_queue, ocr_queue),
        daemon=True
    )
    threads.append(vision_t)
    
    ocr_t = threading.Thread(
        target=ocr_thread,
        args=(ocr_queue, text_queue),
        daemon=True
    )
    threads.append(ocr_t)
    
    reasoning_t = threading.Thread(
        target=reasoning_thread,
        args=(vision_queue, text_queue),
        daemon=True
    )
    threads.append(reasoning_t)
    
    # Start all threads
    for t in threads:
        t.start()
        time.sleep(1)  # Stagger startup
    
    # Wait for completion
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down LUMOS-AI...")
        running = False
        time.sleep(2)
    
    print("\n" + "="*70)
    print("✨ LUMOS-AI shut down complete. Stay safe! ✨")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()