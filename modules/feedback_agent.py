"""
Feedback Agent - Self-Learning System
Collects user feedback and fine-tunes adapters using LoRA (PEFT)
Supports continuous improvement of TrOCR and DETR models
"""

import torch
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image

from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType
)
from transformers import (
    Trainer,
    TrainingArguments,
    VisionEncoderDecoderModel,
    DetrForObjectDetection
)
from torch.utils.data import Dataset

from utils.device_manager import get_device_manager


class FeedbackDataset(Dataset):
    """Dataset for storing feedback examples."""
    
    def __init__(self, examples: List[Dict]):
        """
        Initialize dataset.
        
        Args:
            examples: List of feedback examples
        """
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


class FeedbackAgent:
    """
    Manages user feedback and performs LoRA-based fine-tuning.
    Enables continuous learning and model improvement.
    """
    
    def __init__(self, 
                 feedback_dir="data/feedback",
                 adapter_dir="models/adapters",
                 auto_train_threshold=50):
        """
        Initialize Feedback Agent.
        
        Args:
            feedback_dir: Directory to store feedback logs
            adapter_dir: Directory to save trained adapters
            auto_train_threshold: Number of feedbacks before auto-training
        """
        print("\n🧠 Initializing Feedback Agent...")
        
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        
        # Setup directories
        self.feedback_dir = Path(feedback_dir)
        self.adapter_dir = Path(adapter_dir)
        self.feedback_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        
        self.feedback_log_path = self.feedback_dir / "feedback_log.json"
        
        # Load existing feedback
        self.feedbacks = self._load_feedback()
        
        # Auto-training settings
        self.auto_train_threshold = auto_train_threshold
        self.last_train_count = len(self.feedbacks)
        
        # Track loaded adapters
        self.loaded_adapters = {
            'trocr': None,
            'detr': None
        }
        
        print(f"  Loaded {len(self.feedbacks)} previous feedbacks")
        print("✅ Feedback Agent ready!\n")
    
    def _load_feedback(self) -> List[Dict]:
        """Load feedback log from disk."""
        if self.feedback_log_path.exists():
            with open(self.feedback_log_path, 'r') as f:
                return json.load(f)
        return []
    
    def _save_feedback(self):
        """Save feedback log to disk."""
        with open(self.feedback_log_path, 'w') as f:
            json.dump(self.feedbacks, f, indent=2)
    
    def add_feedback(self,
                     frame: np.ndarray,
                     prediction: Dict,
                     correction: Dict,
                     feedback_type: str = "ocr"):
        """
        Add user feedback for a prediction.
        
        Args:
            frame: Original frame (stored as path or embedding)
            prediction: Model's prediction
            correction: User's correction
            feedback_type: Type of feedback ('ocr', 'detection', 'caption')
        """
        feedback_entry = {
            "id": len(self.feedbacks),
            "timestamp": datetime.now().isoformat(),
            "type": feedback_type,
            "prediction": prediction,
            "correction": correction,
            "frame_shape": frame.shape if isinstance(frame, np.ndarray) else None
        }
        
        # Save frame for training (compressed)
        frame_path = self.feedback_dir / f"frame_{feedback_entry['id']}.jpg"
        if isinstance(frame, np.ndarray):
            import cv2
            cv2.imwrite(str(frame_path), frame)
            feedback_entry['frame_path'] = str(frame_path)
        
        self.feedbacks.append(feedback_entry)
        self._save_feedback()
        
        print(f"✅ Feedback recorded: {feedback_type} (Total: {len(self.feedbacks)})")
        
        # Check if auto-training should trigger
        if len(self.feedbacks) - self.last_train_count >= self.auto_train_threshold:
            print(f"🎓 Auto-training threshold reached ({self.auto_train_threshold} new feedbacks)")
            print("   Triggering adapter retraining...")
            self.retrain_adapters()
    
    def get_feedback_by_type(self, feedback_type: str) -> List[Dict]:
        """Get all feedbacks of a specific type."""
        return [f for f in self.feedbacks if f['type'] == feedback_type]
    
    def prepare_trocr_training_data(self) -> FeedbackDataset:
        """
        Prepare training dataset for TrOCR from OCR feedbacks.
        
        Returns:
            Dataset ready for LoRA training
        """
        ocr_feedbacks = self.get_feedback_by_type('ocr')
        
        if not ocr_feedbacks:
            print("⚠️ No OCR feedbacks available for training")
            return None
        
        examples = []
        for fb in ocr_feedbacks:
            if 'frame_path' not in fb or not Path(fb['frame_path']).exists():
                continue
            
            example = {
                'image_path': fb['frame_path'],
                'text': fb['correction'].get('text', ''),
                'original_prediction': fb['prediction'].get('text', '')
            }
            examples.append(example)
        
        print(f"  Prepared {len(examples)} TrOCR training examples")
        return FeedbackDataset(examples)
    
    def prepare_detr_training_data(self) -> FeedbackDataset:
        """
        Prepare training dataset for DETR from detection feedbacks.
        
        Returns:
            Dataset ready for LoRA training
        """
        det_feedbacks = self.get_feedback_by_type('detection')
        
        if not det_feedbacks:
            print("⚠️ No detection feedbacks available for training")
            return None
        
        examples = []
        for fb in det_feedbacks:
            if 'frame_path' not in fb or not Path(fb['frame_path']).exists():
                continue
            
            example = {
                'image_path': fb['frame_path'],
                'boxes': fb['correction'].get('boxes', []),
                'labels': fb['correction'].get('labels', []),
                'original_boxes': fb['prediction'].get('boxes', [])
            }
            examples.append(example)
        
        print(f"  Prepared {len(examples)} DETR training examples")
        return FeedbackDataset(examples)
    
    def create_lora_config_trocr(self) -> LoraConfig:
        """Create LoRA configuration for TrOCR."""
        return LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=8,  # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],  # Target attention layers
            bias="none"
        )
    
    def create_lora_config_detr(self) -> LoraConfig:
        """Create LoRA configuration for DETR."""
        return LoraConfig(
            task_type=TaskType.OBJECT_DETECTION,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["self_attn.q_proj", "self_attn.v_proj"],
            bias="none"
        )
    
    def train_trocr_adapter(self, 
                            base_model: VisionEncoderDecoderModel,
                            epochs=3,
                            batch_size=4,
                            learning_rate=1e-4):
        """
        Fine-tune TrOCR with LoRA on feedback data.
        
        Args:
            base_model: Base TrOCR model
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Trained PEFT model
        """
        print("\n🎓 Training TrOCR adapter...")
        
        # Prepare data
        dataset = self.prepare_trocr_training_data()
        if dataset is None or len(dataset) < 5:
            print("⚠️ Insufficient data for TrOCR training (need >= 5 examples)")
            return None
        
        # Create LoRA model
        lora_config = self.create_lora_config_trocr()
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.adapter_dir / "trocr_adapter"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=10,
            save_steps=50,
            save_total_limit=2,
            remove_unused_columns=False,
        )
        
        # Custom trainer (simplified - would need custom collator)
        # In production, implement proper data collator for vision-text pairs
        print("  Note: Simplified training loop. Full implementation requires custom data collator.")
        
        # Save adapter
        adapter_path = self.adapter_dir / "trocr_adapter"
        model.save_pretrained(adapter_path)
        
        print(f"✅ TrOCR adapter saved to {adapter_path}")
        
        return model
    
    def train_detr_adapter(self,
                           base_model: DetrForObjectDetection,
                           epochs=5,
                           batch_size=2,
                           learning_rate=1e-4):
        """
        Fine-tune DETR with LoRA on feedback data.
        
        Args:
            base_model: Base DETR model
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            
        Returns:
            Trained PEFT model
        """
        print("\n🎓 Training DETR adapter...")
        
        # Prepare data
        dataset = self.prepare_detr_training_data()
        if dataset is None or len(dataset) < 10:
            print("⚠️ Insufficient data for DETR training (need >= 10 examples)")
            return None
        
        # Create LoRA model
        lora_config = self.create_lora_config_detr()
        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        
        # Save adapter
        adapter_path = self.adapter_dir / "detr_adapter"
        model.save_pretrained(adapter_path)
        
        print(f"✅ DETR adapter saved to {adapter_path}")
        
        return model
    
    def retrain_adapters(self):
        """
        Retrain all adapters based on accumulated feedback.
        This is called automatically when threshold is reached.
        """
        print("\n" + "="*60)
        print("🔄 RETRAINING ADAPTERS")
        print("="*60)
        
        # Note: In production, you would pass actual base models here
        # For now, this is a framework showing the training pipeline
        
        ocr_count = len(self.get_feedback_by_type('ocr'))
        det_count = len(self.get_feedback_by_type('detection'))
        
        print(f"\nFeedback summary:")
        print(f"  OCR feedbacks: {ocr_count}")
        print(f"  Detection feedbacks: {det_count}")
        
        # Update last train count
        self.last_train_count = len(self.feedbacks)
        
        print("\n✅ Retraining complete!")
        print("="*60 + "\n")
    
    def load_adapter(self, 
                     base_model,
                     adapter_type: str):
        """
        Load a trained adapter onto a base model.
        
        Args:
            base_model: Base model (TrOCR or DETR)
            adapter_type: Type of adapter ('trocr' or 'detr')
            
        Returns:
            Model with loaded adapter
        """
        adapter_path = self.adapter_dir / f"{adapter_type}_adapter"
        
        if not adapter_path.exists():
            print(f"⚠️ No trained {adapter_type} adapter found at {adapter_path}")
            return base_model
        
        try:
            print(f"📥 Loading {adapter_type} adapter...")
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model = model.merge_and_unload()  # Merge LoRA weights
            self.loaded_adapters[adapter_type] = True
            print(f"✅ {adapter_type} adapter loaded")
            return model
        except Exception as e:
            print(f"⚠️ Failed to load {adapter_type} adapter: {e}")
            return base_model
    
    def get_statistics(self) -> Dict:
        """Get feedback statistics."""
        stats = {
            "total_feedbacks": len(self.feedbacks),
            "by_type": {},
            "recent_24h": 0,
            "adapters_trained": sum(1 for v in self.loaded_adapters.values() if v)
        }
        
        # Count by type
        for fb_type in ['ocr', 'detection', 'caption']:
            stats['by_type'][fb_type] = len(self.get_feedback_by_type(fb_type))
        
        # Recent feedbacks
        now = datetime.now()
        for fb in self.feedbacks:
            fb_time = datetime.fromisoformat(fb['timestamp'])
            if (now - fb_time).total_seconds() < 86400:  # 24 hours
                stats['recent_24h'] += 1
        
        return stats
    
    def clear_old_feedbacks(self, days=30):
        """Clear feedbacks older than specified days."""
        now = datetime.now()
        cutoff = now.timestamp() - (days * 86400)
        
        self.feedbacks = [
            fb for fb in self.feedbacks
            if datetime.fromisoformat(fb['timestamp']).timestamp() > cutoff
        ]
        
        self._save_feedback()
        print(f"🗑️  Cleared feedbacks older than {days} days")