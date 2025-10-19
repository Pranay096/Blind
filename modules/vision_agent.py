"""
Vision Agent - Unified Multi-Model Vision System
Integrates: YOLOv8 + DETR + SAM + MiDaS + BLIP-2
Outputs unified JSON with objects, masks, depth, and captions
FIXED: DETR meta tensor loading issue + MODEL CACHING ENABLED
"""
import os
import torch
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from typing import Dict, List, Any
import hashlib

# Model imports
from ultralytics import YOLO
from transformers import (
    DetrImageProcessor, DetrForObjectDetection,
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoImageProcessor, AutoModel
)

# SAM import (optional)
try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except:
    SAM_AVAILABLE = False
    print("⚠️ Segment Anything not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")

from utils.device_manager import get_device_manager


class VisionAgent:
    """
    Unified vision system combining multiple SOTA models.
    Provides comprehensive scene understanding with caching.
    """
    
    def __init__(self, use_cache=True, cache_threshold=0.95, model_cache_dir="./model_cache"):
        """
        Initialize all vision models.
        
        Args:
            use_cache: Enable frame similarity caching
            cache_threshold: Similarity threshold for cache hits (0-1)
            model_cache_dir: Directory to cache downloaded models
        """
        print("\n🔷 Initializing Vision Agent...")
        
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        
        # Model cache directory
        self.model_cache_dir = model_cache_dir
        
        # Cache settings
        self.use_cache = use_cache
        self.cache_threshold = cache_threshold
        self.frame_cache = {}
        self.last_frame_hash = None
        
        # Initialize models
        self._init_yolo()
        self._init_detr()
        self._init_sam()
        self._init_midas()
        self._init_blip2()
        
        print("✅ Vision Agent ready!\n")
    
    def _init_yolo(self):
        """Initialize YOLOv8 for fast detection."""
        print("  📦 Loading YOLOv8...")
        try:
            self.yolo = YOLO('yolov8m.pt')
            self.yolo_conf = 0.5
            print("  ✓ YOLOv8 loaded")
        except Exception as e:
            print(f"  ⚠️ YOLO failed: {e}")
            self.yolo = None
    
    def _init_detr(self):
        """Initialize DETR for transformer-based detection - FIXED + CACHED."""
        print("  🎯 Loading DETR...")
        try:
            model_name = "facebook/detr-resnet-50"
            
            # CACHE ENABLED HERE ↓
            self.detr_processor = DetrImageProcessor.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir
            )
            
            # FIX: Load model without moving to device first + CACHE ENABLED
            self.detr_model = DetrForObjectDetection.from_pretrained(
                model_name,
                ignore_mismatched_sizes=True,
                cache_dir=self.model_cache_dir  # ← ADDED CACHING
            )
            
            # FIX: Only move to device if CUDA, otherwise just use eval
            if self.device.type == 'cuda':
                self.detr_model = self.detr_model.to(self.device)
            
            self.detr_model.eval()
            print("  ✓ DETR loaded")
        except Exception as e:
            print(f"  ⚠️ DETR failed: {e}")
            print("  → Continuing with YOLO only")
            self.detr_model = None
            self.detr_processor = None
    
    def _init_sam(self):
        """Initialize Segment Anything Model."""
        print("  🎭 Loading SAM...")
        if not SAM_AVAILABLE:
            self.sam_predictor = None
            return
            
        try:
            # Use SAM ViT-B (balance of speed/quality)
            # Check in cache directory first
            sam_checkpoint = os.path.join(self.model_cache_dir, "sam_vit_b_01ec64.pth")
            model_type = "vit_b"
            
            if not os.path.exists(sam_checkpoint):
                # Fallback to current directory
                sam_checkpoint = "sam_vit_b_01ec64.pth"
                if not os.path.exists(sam_checkpoint):
                    print(f"  ⚠️ SAM checkpoint not found: {sam_checkpoint}")
                    print("  Download with: wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
                    self.sam_predictor = None
                    return
            
            sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print("  ✓ SAM loaded")
        except Exception as e:
            print(f"  ⚠️ SAM initialization failed: {e}")
            self.sam_predictor = None
    
    def _init_midas(self):
        """Initialize MiDaS for depth estimation."""
        print("  🗺️  Loading MiDaS...")
        try:
            # Use DPT-Hybrid for quality (or MiDaS_small for speed)
            # Note: torch.hub.load doesn't support cache_dir parameter
            # It uses torch cache automatically (~/.cache/torch/hub/)
            self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", skip_validation=True)
            self.midas.to(self.device)
            self.midas.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", skip_validation=True)
            self.midas_transform = midas_transforms.dpt_transform
            print("  ✓ MiDaS loaded")
        except Exception as e:
            print(f"  ⚠️ MiDaS failed: {e}")
            self.midas = None
    
    def _init_blip2(self):
        """Initialize BLIP-2 for image captioning - CACHED."""
        print("  💬 Loading BLIP-2...")
        try:
            model_name = "Salesforce/blip2-opt-2.7b"
            
            # CACHE ENABLED HERE ↓
            self.blip2_processor = Blip2Processor.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir  # ← ADDED CACHING
            )
            self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                cache_dir=self.model_cache_dir  # ← ADDED CACHING
            )
            self.blip2_model.to(self.device)
            self.blip2_model.eval()
            print("  ✓ BLIP-2 loaded")
        except Exception as e:
            print(f"  ⚠️ BLIP-2 failed: {e}")
            print("  → Scene captions will be unavailable")
            self.blip2_model = None
            self.blip2_processor = None
    
    def _compute_frame_hash(self, frame):
        """Compute hash of frame for caching."""
        # Resize to small size for faster hashing
        small = cv2.resize(frame, (64, 64))
        return hashlib.md5(small.tobytes()).hexdigest()
    
    def _frame_similarity(self, frame1, frame2):
        """Compute similarity between two frames."""
        # Resize for faster comparison
        f1 = cv2.resize(frame1, (64, 64))
        f2 = cv2.resize(frame2, (64, 64))
        
        # Compute SSIM-like similarity
        diff = np.mean(np.abs(f1.astype(float) - f2.astype(float)))
        similarity = 1 - (diff / 255.0)
        return similarity
    
    def analyze_frame(self, frame: np.ndarray, use_sam=False) -> Dict[str, Any]:
        """
        Complete vision analysis of a frame.
        
        Args:
            frame: OpenCV BGR image
            use_sam: Whether to generate segmentation masks (slower)
            
        Returns:
            Unified JSON with all vision outputs
        """
        # Check cache
        if self.use_cache and self.last_frame_hash:
            frame_hash = self._compute_frame_hash(frame)
            if frame_hash in self.frame_cache:
                cached = self.frame_cache[frame_hash]
                cached['from_cache'] = True
                return cached
        
        # Run all vision models
        with torch.no_grad():
            # 1. Fast detection with YOLO
            yolo_objects = self._detect_yolo(frame) if self.yolo else []
            
            # 2. Fine detection with DETR (if available)
            detr_objects = self._detect_detr(frame) if self.detr_model else []
            
            # 3. Merge detections
            merged_objects = self._merge_detections(yolo_objects, detr_objects)
            
            # 4. Add segmentation masks with SAM (optional)
            if use_sam and self.sam_predictor and len(merged_objects) > 0:
                merged_objects = self._add_segmentation_masks(frame, merged_objects)
            
            # 5. Estimate depth
            depth_map = self._estimate_depth(frame) if self.midas else None
            
            # 6. Add depth to objects
            if depth_map is not None:
                merged_objects = self._add_depth_to_objects(merged_objects, depth_map)
            
            # 7. Generate caption
            caption = self._generate_caption(frame) if self.blip2_model else "Scene view"
        
        # Build unified output
        output = {
            "objects": merged_objects,
            "depth_estimation": self._serialize_depth(depth_map) if depth_map is not None else None,
            "caption": caption,
            "timestamp": datetime.now().isoformat(),
            "frame_shape": frame.shape,
            "from_cache": False
        }
        
        # Update cache
        if self.use_cache:
            frame_hash = self._compute_frame_hash(frame)
            self.frame_cache[frame_hash] = output.copy()
            self.last_frame_hash = frame_hash
            
            # Limit cache size
            if len(self.frame_cache) > 10:
                oldest = list(self.frame_cache.keys())[0]
                del self.frame_cache[oldest]
        
        return output
    
    def _detect_yolo(self, frame):
        """Run YOLOv8 detection."""
        if not self.yolo:
            return []
            
        try:
            results = self.yolo(frame, verbose=False, conf=self.yolo_conf)
            
            objects = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = self.yolo.names[cls]
                    
                    objects.append({
                        "name": name,
                        "confidence": round(conf, 3),
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                        "source": "yolo"
                    })
            
            return objects
        except Exception as e:
            print(f"⚠️ YOLO detection error: {e}")
            return []
    
    def _detect_detr(self, frame):
        """Run DETR detection."""
        if not self.detr_model or not self.detr_processor:
            return []
            
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            inputs = self.detr_processor(images=image, return_tensors="pt")
            
            # Only move to device if CUDA
            if self.device.type == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.detr_model(**inputs)
            
            # Post-process
            target_sizes = torch.tensor([image.size[::-1]])
            if self.device.type == 'cuda':
                target_sizes = target_sizes.to(self.device)
            
            results = self.detr_processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.7
            )[0]
            
            objects = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                score = float(score)
                label_id = int(label)
                label_name = self.detr_model.config.id2label[label_id]
                x1, y1, x2, y2 = box.cpu().numpy()
                
                objects.append({
                    "name": label_name,
                    "confidence": round(score, 3),
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "source": "detr"
                })
            
            return objects
        except Exception as e:
            print(f"⚠️ DETR detection error: {e}")
            return []
    
    def _merge_detections(self, yolo_objs, detr_objs):
        """Merge YOLO and DETR detections using NMS."""
        all_objects = yolo_objs + detr_objs
        
        if not all_objects:
            return []
        
        # Simple NMS based on IoU
        merged = []
        used = set()
        
        all_objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        for i, obj1 in enumerate(all_objects):
            if i in used:
                continue
            
            # Check overlaps
            overlaps = [obj1]
            for j, obj2 in enumerate(all_objects[i+1:], start=i+1):
                if j in used:
                    continue
                
                iou = self._compute_iou(obj1['box'], obj2['box'])
                if iou > 0.5:
                    overlaps.append(obj2)
                    used.add(j)
            
            # Use highest confidence detection
            best = max(overlaps, key=lambda x: x['confidence'])
            merged.append(best)
            used.add(i)
        
        return merged
    
    def _compute_iou(self, box1, box2):
        """Compute IoU between two boxes."""
        x1, y1, x2, y2 = box1
        x1p, y1p, x2p, y2p = box2
        
        xi1 = max(x1, x1p)
        yi1 = max(y1, y1p)
        xi2 = min(x2, x2p)
        yi2 = min(y2, y2p)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2p - x1p) * (y2p - y1p)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)
    
    def _add_segmentation_masks(self, frame, objects):
        """Add SAM segmentation masks to objects."""
        if not self.sam_predictor:
            return objects
            
        try:
            self.sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            for obj in objects:
                x1, y1, x2, y2 = obj['box']
                input_box = np.array([x1, y1, x2, y2])
                
                masks, scores, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                
                # Store mask as RLE or polygon for efficiency
                mask = masks[0]
                obj['mask'] = self._mask_to_rle(mask)
                obj['mask_score'] = float(scores[0])
        except Exception as e:
            print(f"⚠️ SAM segmentation error: {e}")
        
        return objects
    
    def _mask_to_rle(self, mask):
        """Convert binary mask to RLE for compact storage."""
        # Simple RLE encoding
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return runs.tolist()
    
    def _estimate_depth(self, frame):
        """Estimate depth map with MiDaS."""
        if not self.midas:
            return None
            
        try:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_batch = self.midas_transform(img_rgb).to(self.device)
            
            prediction = self.midas(input_batch)
            
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth_map = prediction.cpu().numpy()
            
            # Normalize
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            
            return depth_map
        except Exception as e:
            print(f"⚠️ MiDaS depth estimation error: {e}")
            return None
    
    def _add_depth_to_objects(self, objects, depth_map):
        """Add depth information to detected objects."""
        if depth_map is None:
            return objects
            
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            
            # Crop depth to object region
            depth_crop = depth_map[y1:y2, x1:x2]
            
            if depth_crop.size > 0:
                mean_depth = float(np.mean(depth_crop))
                # Convert to approximate distance (inverse depth)
                obj['depth_value'] = round(mean_depth, 3)
                obj['distance_m'] = round(self._depth_to_distance(mean_depth), 1)
        
        return objects
    
    def _depth_to_distance(self, normalized_depth):
        """Convert normalized depth to distance estimate."""
        # Heuristic: lower depth = farther
        if normalized_depth < 0.1:
            return 10.0
        elif normalized_depth > 0.8:
            return 0.5
        else:
            return 10.0 - (normalized_depth * 9.5)
    
    def _generate_caption(self, frame):
        """Generate scene caption with BLIP-2."""
        if not self.blip2_model or not self.blip2_processor:
            return "Scene view"
            
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            inputs = self.blip2_processor(images=image, return_tensors="pt").to(
                self.device, 
                torch.float16 if self.device.type == "cuda" else torch.float32
            )
            
            generated_ids = self.blip2_model.generate(**inputs, max_length=50)
            caption = self.blip2_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            
            return caption
        except Exception as e:
            print(f"⚠️ BLIP-2 caption error: {e}")
            return "Scene view"
    
    def _serialize_depth(self, depth_map):
        """Serialize depth map for JSON."""
        if depth_map is None:
            return None
            
        # Downsample for efficiency
        small_depth = cv2.resize(depth_map, (80, 60))
        return {
            "shape": small_depth.shape,
            "mean": float(np.mean(small_depth)),
            "std": float(np.std(small_depth))
        }
    
    def clear_cache(self):
        """Clear frame cache."""
        self.frame_cache.clear()
        self.last_frame_hash = None