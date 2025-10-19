"""
Deep OCR - Advanced Text Recognition
Uses TrOCR (HuggingFace) with EasyOCR fallback
Handles curved, blurred, angled text from SAM-masked regions
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import easyocr

from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel
)

from utils.device_manager import get_device_manager


class DeepOCR:
    """
    Advanced OCR system with TrOCR and preprocessing.
    Supports curved, blurred, and angled text recognition.
    """
    
    def __init__(self, use_trocr=True, use_easyocr=True):
        """
        Initialize Deep OCR system.
        
        Args:
            use_trocr: Enable TrOCR (transformer-based)
            use_easyocr: Enable EasyOCR fallback
        """
        print("\n📖 Initializing Deep OCR...")
        
        self.device_manager = get_device_manager()
        self.device = self.device_manager.get_device()
        
        self.use_trocr = use_trocr
        self.use_easyocr = use_easyocr
        
        # Initialize models
        if use_trocr:
            self._init_trocr()
        
        if use_easyocr:
            self._init_easyocr()
        
        print("✅ Deep OCR ready!\n")
    
    def _init_trocr(self):
        """Initialize TrOCR model."""
        print("  🔤 Loading TrOCR...")
        model_name = "microsoft/trocr-base-handwritten"  # or trocr-large-printed
        
        self.trocr_processor = TrOCRProcessor.from_pretrained(model_name)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.trocr_model.to(self.device)
        self.trocr_model.eval()
        print("  ✓ TrOCR loaded")
    
    def _init_easyocr(self):
        """Initialize EasyOCR as fallback."""
        print("  📚 Loading EasyOCR...")
        # English only for speed; add more languages as needed
        self.easyocr_reader = easyocr.Reader(
            ['en'],
            gpu=self.device.type == "cuda"
        )
        print("  ✓ EasyOCR loaded")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced preprocessing for better OCR.
        Handles curved, blurred, and angled text.
        
        Args:
            image: Input image (BGR or RGB)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 1. Denoise with bilateral filter (preserves edges)
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # 2. Enhance contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Deskew if needed
        deskewed = self._deskew(enhanced)
        
        # 4. Adaptive thresholding for varying lighting
        binary = cv2.adaptiveThreshold(
            deskewed, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # 5. Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew image to correct text angle."""
        coords = np.column_stack(np.where(image > 0))
        if len(coords) == 0:
            return image
        
        angle = cv2.minAreaRect(coords)[-1]
        
        # Adjust angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image
        if abs(angle) > 0.5:  # Only deskew if angle significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def detect_text_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect bounding boxes of text regions.
        
        Args:
            frame: Input image
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        if not self.use_easyocr:
            return []
        
        # Use EasyOCR for detection (faster than TrOCR)
        results = self.easyocr_reader.detect(frame)
        
        boxes = []
        if results and len(results) > 0:
            horizontal_list, free_list = results
            
            for box in horizontal_list[0]:
                x_min, x_max, y_min, y_max = box
                w = x_max - x_min
                h = y_max - y_min
                boxes.append((int(x_min), int(y_min), int(w), int(h)))
        
        return boxes
    
    def read_text_trocr(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Read text using TrOCR.
        
        Args:
            image: Preprocessed text region
            
        Returns:
            (text, confidence) tuple
        """
        if not self.use_trocr:
            return "", 0.0
        
        try:
            # Convert to PIL RGB
            if len(image.shape) == 2:
                pil_image = Image.fromarray(image).convert('RGB')
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Process with TrOCR
            pixel_values = self.trocr_processor(
                images=pil_image,
                return_tensors="pt"
            ).pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values)
            
            text = self.trocr_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # TrOCR doesn't provide confidence directly
            # Use length and character diversity as proxy
            confidence = min(len(text) / 10.0, 1.0) if text else 0.0
            
            return text.strip(), confidence
            
        except Exception as e:
            print(f"⚠️ TrOCR error: {e}")
            return "", 0.0
    
    def read_text_easyocr(self, image: np.ndarray, min_confidence=0.3) -> Tuple[str, float]:
        """
        Read text using EasyOCR.
        
        Args:
            image: Input image
            min_confidence: Minimum confidence threshold
            
        Returns:
            (text, confidence) tuple
        """
        if not self.use_easyocr:
            return "", 0.0
        
        try:
            results = self.easyocr_reader.readtext(image)
            
            # Aggregate results
            texts = []
            confidences = []
            
            for (bbox, text, conf) in results:
                if conf >= min_confidence:
                    texts.append(text)
                    confidences.append(conf)
            
            if texts:
                full_text = ' '.join(texts)
                avg_confidence = sum(confidences) / len(confidences)
                return full_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            print(f"⚠️ EasyOCR error: {e}")
            return "", 0.0
    
    def read_text(self, 
                  frame: np.ndarray, 
                  masked_region: Optional[np.ndarray] = None,
                  use_preprocessing=True,
                  min_confidence=0.4) -> dict:
        """
        Main OCR function with fallback strategy.
        
        Args:
            frame: Full frame or cropped region
            masked_region: Optional SAM mask to focus on specific area
            use_preprocessing: Apply advanced preprocessing
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with text and metadata
        """
        # Apply mask if provided
        if masked_region is not None:
            frame = cv2.bitwise_and(frame, frame, mask=masked_region)
        
        # Preprocess
        if use_preprocessing:
            processed = self.preprocess_image(frame)
        else:
            processed = frame
        
        # Try TrOCR first (better for handwritten/curved text)
        text_trocr, conf_trocr = "", 0.0
        if self.use_trocr:
            text_trocr, conf_trocr = self.read_text_trocr(processed)
        
        # Try EasyOCR as fallback or complement
        text_easy, conf_easy = "", 0.0
        if self.use_easyocr:
            text_easy, conf_easy = self.read_text_easyocr(frame, min_confidence)
        
        # Choose best result
        if conf_trocr >= conf_easy and conf_trocr >= min_confidence:
            final_text = text_trocr
            final_conf = conf_trocr
            method = "trocr"
        elif conf_easy >= min_confidence:
            final_text = text_easy
            final_conf = conf_easy
            method = "easyocr"
        else:
            # Combine if both have some confidence
            if text_trocr and text_easy:
                final_text = f"{text_trocr} {text_easy}"
                final_conf = (conf_trocr + conf_easy) / 2
                method = "combined"
            else:
                final_text = ""
                final_conf = 0.0
                method = "none"
        
        return {
            "text": final_text,
            "confidence": round(final_conf, 3),
            "method": method,
            "trocr_result": text_trocr if self.use_trocr else None,
            "easyocr_result": text_easy if self.use_easyocr else None
        }
    
    def read_multiple_regions(self, 
                             frame: np.ndarray,
                             regions: List[Tuple[int, int, int, int]]) -> List[dict]:
        """
        Read text from multiple regions.
        
        Args:
            frame: Full frame
            regions: List of (x, y, w, h) bounding boxes
            
        Returns:
            List of OCR results
        """
        results = []
        
        for (x, y, w, h) in regions:
            # Crop region
            cropped = frame[y:y+h, x:x+w]
            
            if cropped.size == 0:
                continue
            
            # Read text
            result = self.read_text(cropped)
            result['bbox'] = (x, y, w, h)
            
            results.append(result)
        
        return results
    
    def extract_text_from_objects(self, 
                                   frame: np.ndarray,
                                   objects: List[dict]) -> List[dict]:
        """
        Extract text from detected objects (e.g., from Vision Agent).
        
        Args:
            frame: Full frame
            objects: List of detected objects with 'box' and optional 'mask'
            
        Returns:
            Updated objects with text information
        """
        for obj in objects:
            x1, y1, x2, y2 = obj['box']
            
            # Crop object region
            cropped = frame[y1:y2, x1:x2]
            
            # Use mask if available
            mask = None
            if 'mask' in obj:
                # Reconstruct mask from RLE (simplified)
                mask_full = np.zeros(frame.shape[:2], dtype=np.uint8)
                # TODO: Decode RLE properly
                mask = mask_full[y1:y2, x1:x2]
            
            # Read text
            ocr_result = self.read_text(cropped, masked_region=mask)
            
            # Add to object if text found
            if ocr_result['text']:
                obj['ocr'] = ocr_result
        
        return objects