"""
Optimized Deep Learning Layer
Adds intelligent caching, model optimization, and automatic real-time inference
Optimized for: 16GB RAM, RTX 4050 6GB VRAM, Intel i5 12450H
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
import pickle
import hashlib
from collections import OrderedDict
import time
from threading import Lock

class SmartCache:
    """Intelligent caching system for frames and predictions"""
    
    def __init__(self, max_size=50):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.lock = Lock()
        self.hits = 0
        self.misses = 0
    
    def _compute_hash(self, frame):
        """Fast frame hashing using downsampled image"""
        small = cv2.resize(frame, (32, 32))
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def get(self, frame, threshold=0.92):
        """Get cached result if similar frame exists"""
        with self.lock:
            frame_hash = self._compute_hash(frame)
            
            if frame_hash in self.cache:
                self.hits += 1
                # Move to end (most recently used)
                self.cache.move_to_end(frame_hash)
                return self.cache[frame_hash]
            
            self.misses += 1
            return None
    
    def put(self, frame, result):
        """Store result in cache"""
        with self.lock:
            frame_hash = self._compute_hash(frame)
            self.cache[frame_hash] = result
            self.cache.move_to_end(frame_hash)
            
            # Remove oldest if cache full
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return f"Cache: {hit_rate:.1f}% hit rate ({self.hits}/{total})"


class ModelOptimizer:
    """Optimizes models for RTX 4050 with mixed precision and TensorRT"""
    
    def __init__(self, device):
        self.device = device
        self.use_fp16 = device.type == 'cuda'
    
    def optimize_model(self, model, input_shape=None):
        """Optimize model with mixed precision and compile"""
        if not isinstance(model, nn.Module):
            return model
        
        model = model.to(self.device)
        model.eval()
        
        # Enable mixed precision for CUDA
        if self.use_fp16:
            model = model.half()
        
        # Torch compile for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode='reduce-overhead')
        except:
            pass
        
        return model
    
    def optimize_inference(self, model_fn):
        """Decorator for optimized inference with autocast"""
        def wrapper(*args, **kwargs):
            if self.use_fp16:
                with torch.cuda.amp.autocast():
                    with torch.no_grad():
                        return model_fn(*args, **kwargs)
            else:
                with torch.no_grad():
                    return model_fn(*args, **kwargs)
        return wrapper


class FramePreprocessor:
    """Fast frame preprocessing pipeline"""
    
    def __init__(self):
        self.cache = {}
    
    def preprocess(self, frame, target_size=(640, 480), normalize=True):
        """Optimized preprocessing with caching"""
        h, w = frame.shape[:2]
        cache_key = f"{w}x{h}_{target_size}"
        
        # Resize if needed
        if (w, h) != target_size:
            frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1] if requested
        if normalize:
            frame = frame.astype(np.float32) / 255.0
        
        return frame
    
    def batch_preprocess(self, frames):
        """Process multiple frames efficiently"""
        return [self.preprocess(f) for f in frames]


class InferenceScheduler:
    """Schedules model inference to prevent lag"""
    
    def __init__(self, fps_target=30):
        self.fps_target = fps_target
        self.frame_time = 1.0 / fps_target
        self.last_inference_time = {}
        self.min_interval = {
            'yolo': 0.033,      # 30 FPS
            'detr': 0.1,        # 10 FPS
            'sam': 0.2,         # 5 FPS
            'midas': 0.1,       # 10 FPS
            'blip2': 0.2,       # 5 FPS
            'trocr': 0.1,       # 10 FPS
            'pose': 0.05,       # 20 FPS
            'sign': 0.1,        # 10 FPS
        }
    
    def should_run(self, model_name):
        """Check if enough time has passed to run model"""
        current_time = time.time()
        last_time = self.last_inference_time.get(model_name, 0)
        interval = self.min_interval.get(model_name, 0.1)
        
        if current_time - last_time >= interval:
            self.last_inference_time[model_name] = current_time
            return True
        return False
    
    def update_interval(self, model_name, fps):
        """Dynamically adjust inference interval"""
        self.min_interval[model_name] = 1.0 / fps


class OptimizedDeepLearning:
    """
    Main optimization layer that wraps all models
    Provides automatic caching, scheduling, and optimization
    """
    
    def __init__(self, device):
        print("🚀 Initializing Optimized Deep Learning Layer...")
        
        self.device = device
        self.cache = SmartCache(max_size=50)
        self.optimizer = ModelOptimizer(device)
        self.preprocessor = FramePreprocessor()
        self.scheduler = InferenceScheduler(fps_target=30)
        
        # Performance tracking
        self.inference_times = {}
        self.frame_count = 0
        self.start_time = time.time()
        
        print("✅ Deep Learning Layer ready!")
    
    def wrap_model(self, model, model_name):
        """Wrap model with optimizations"""
        print(f"  ⚡ Optimizing {model_name}...")
        
        # Optimize model
        if hasattr(model, 'to'):
            model = self.optimizer.optimize_model(model)
        
        return model
    
    def cached_inference(self, model_fn, frame, model_name, force=False):
        """
        Run inference with caching and scheduling
        
        Args:
            model_fn: Function that runs model inference
            frame: Input frame
            model_name: Name for scheduling
            force: Force inference regardless of schedule
        """
        # Check cache first
        cached = self.cache.get(frame)
        if cached is not None and model_name in cached:
            return cached[model_name]
        
        # Check if should run based on schedule
        if not force and not self.scheduler.should_run(model_name):
            # Return previous result if available
            if hasattr(self, f'_last_{model_name}'):
                return getattr(self, f'_last_{model_name}')
            return None
        
        # Run inference
        start_time = time.time()
        result = model_fn(frame)
        inference_time = time.time() - start_time
        
        # Track performance
        self.inference_times[model_name] = inference_time
        
        # Cache result
        cached = self.cache.get(frame) or {}
        cached[model_name] = result
        self.cache.put(frame, cached)
        
        # Store as last result
        setattr(self, f'_last_{model_name}', result)
        
        return result
    
    def batch_inference(self, model_fn, frames, model_name):
        """Run batched inference for efficiency"""
        results = []
        
        # Process in batches
        batch_size = 4 if self.device.type == 'cuda' else 2
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i+batch_size]
            batch_results = model_fn(batch)
            results.extend(batch_results)
        
        return results
    
    def get_performance_stats(self):
        """Get performance statistics"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        stats = {
            'fps': fps,
            'cache_stats': self.cache.get_stats(),
            'inference_times': self.inference_times.copy(),
            'avg_total_time': sum(self.inference_times.values()),
        }
        
        return stats
    
    def optimize_for_rtx4050(self):
        """Specific optimizations for RTX 4050"""
        # Set optimal CUDA settings
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable memory optimization
            torch.cuda.empty_cache()
            
            # Set memory growth
            torch.cuda.set_per_process_memory_fraction(0.8, 0)
        
        print("⚡ RTX 4050 optimizations applied!")


class PredictionSmoother:
    """Smooths predictions over time to reduce jitter"""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = {}
    
    def smooth(self, key, value, weight=0.7):
        """Exponential moving average smoothing"""
        if key not in self.history:
            self.history[key] = value
            return value
        
        smoothed = weight * value + (1 - weight) * self.history[key]
        self.history[key] = smoothed
        return smoothed
    
    def smooth_bbox(self, obj_id, bbox):
        """Smooth bounding box coordinates"""
        smoothed = []
        for i, coord in enumerate(bbox):
            key = f"{obj_id}_bbox_{i}"
            smoothed.append(int(self.smooth(key, coord)))
        return smoothed


class AutomaticInteractionManager:
    """Manages automatic interaction without wake words"""
    
    def __init__(self):
        self.last_text_detected = ""
        self.last_sign_detected = ""
        self.last_activity = ""
        self.last_speech = ""
        
        self.interaction_cooldown = 2.0  # seconds
        self.last_interaction_time = {}
    
    def should_speak_text(self, text):
        """Check if should speak detected text"""
        if not text or text == self.last_text_detected:
            return False
        
        current_time = time.time()
        last_time = self.last_interaction_time.get('text', 0)
        
        if current_time - last_time >= self.interaction_cooldown:
            self.last_text_detected = text
            self.last_interaction_time['text'] = current_time
            return True
        
        return False
    
    def should_announce_sign(self, sign):
        """Check if should announce sign language"""
        if not sign or sign == self.last_sign_detected:
            return False
        
        current_time = time.time()
        last_time = self.last_interaction_time.get('sign', 0)
        
        if current_time - last_time >= self.interaction_cooldown:
            self.last_sign_detected = sign
            self.last_interaction_time['sign'] = current_time
            return True
        
        return False
    
    def should_announce_activity(self, activity):
        """Check if should announce human activity"""
        if not activity or activity == self.last_activity:
            return False
        
        # Only announce meaningful activities
        meaningful = ['waving', 'raising hand', 'pointing']
        if not any(word in activity.lower() for word in meaningful):
            return False
        
        current_time = time.time()
        last_time = self.last_interaction_time.get('activity', 0)
        
        if current_time - last_time >= self.interaction_cooldown:
            self.last_activity = activity
            self.last_interaction_time['activity'] = current_time
            return True
        
        return False


# Global instance
_dl_layer = None

def get_optimized_dl(device):
    """Get or create optimized DL layer"""
    global _dl_layer
    if _dl_layer is None:
        _dl_layer = OptimizedDeepLearning(device)
        _dl_layer.optimize_for_rtx4050()
    return _dl_layer