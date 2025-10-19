"""
Camera Module
Handles webcam capture and frame management.
"""

import cv2

class Camera:
    """
    Webcam interface for capturing frames.
    """
    
    def __init__(self, camera_index=0, width=640, height=480):
        """
        Initialize camera.
        
        Args:
            camera_index: Camera device index (0 for default)
            width: Frame width
            height: Frame height
        """
        print(f"📷 Opening camera {camera_index}...")
        
        self.cap = cv2.VideoCapture(camera_index)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_index}")
        
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Verify settings
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"✅ Camera opened: {actual_width}x{actual_height}")
        
        self.width = actual_width
        self.height = actual_height
        
    def read_frame(self):
        """
        Read a frame from the camera.
        
        Returns:
            Frame as numpy array (BGR format), or None if failed
        """
        ret, frame = self.cap.read()
        
        if not ret:
            return None
        
        return frame
    
    def release(self):
        """Release camera resources."""
        if self.cap:
            self.cap.release()
            print("📷 Camera released")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.release()