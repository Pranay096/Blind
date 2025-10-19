"""
Sign Language Recognition without MediaPipe
Detects and interprets sign language gestures using Hugging Face model.
"""

import torch
import cv2
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import time

class SignLanguageDetector:
    def __init__(self, model_name='RavenOnur/Sign-Language'):
        print(f"🤟 Loading sign language model: {model_name}...")
        try:
            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.device = torch.device("cpu")
            self.model.to(self.device)
            self.model.eval()

            self.pipe = pipeline(
                "image-classification",
                model=self.model,
                image_processor=self.processor,
                device=-1  # CPU
            )
            print(f"✅ Sign language detector ready")

            self.last_sign = ""
            self.last_sign_time = 0
            self.sign_cooldown = 2.0  # seconds
        except Exception as e:
            print(f"⚠️ Failed to load sign language model: {e}")
            self.pipe = None

    def recognize_sign(self, frame, threshold=0.5):
        if self.pipe is None:
            return None, 0

        current_time = time.time()

        # Convert frame to PIL image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Run prediction
        results = self.pipe(pil_image, top_k=1)
        if not results:
            return None, 0

        top_result = results[0]
        sign = top_result['label']
        confidence = top_result['score']

        if confidence < threshold:
            return None, 0

        # Cooldown to avoid repeated detection
        if sign == self.last_sign and current_time - self.last_sign_time < self.sign_cooldown:
            return None, 0

        self.last_sign = sign
        self.last_sign_time = current_time

        return sign, confidence

# ----------------- Main Webcam Loop -----------------
if __name__ == "__main__":
    detector = SignLanguageDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("⚠️ Cannot access camera")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        sign, confidence = detector.recognize_sign(frame)

        if sign:
            text = f"{sign} ({confidence:.2f})"
            cv2.putText(frame, text, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Sign Language Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
