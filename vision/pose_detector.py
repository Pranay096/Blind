"""
Human Pose and Activity Detector
Detects human poses, gestures, and activities using YOLOv8-pose.
"""

from ultralytics import YOLO
import numpy as np
import cv2

class PoseDetector:
    """
    YOLOv8-pose based human pose and activity detector.
    Detects poses, gestures like waving, and human behaviors.
    """
    
    def __init__(self, model_size='m'):
        """
        Initialize pose detector.
        
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium)
        """
        print(f"🤸 Loading YOLOv8{model_size}-pose model...")
        self.model = YOLO(f'yolov8{model_size}-pose.pt')
        print(f"✅ Pose detector ready")
        
        # Keypoint indices (COCO format)
        self.KEYPOINT_NAMES = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Track previous poses for motion detection
        self.prev_poses = {}
        
    def detect_poses(self, frame):
        """
        Detect human poses in frame.
        
        Args:
            frame: OpenCV BGR image
            
        Returns:
            List of pose dictionaries with keypoints and activities
        """
        results = self.model(frame, verbose=False)
        
        poses = []
        
        for result in results:
            if result.keypoints is None:
                continue
                
            keypoints_data = result.keypoints.data
            boxes = result.boxes
            
            for idx, (kpts, box) in enumerate(zip(keypoints_data, boxes)):
                # Extract keypoints (x, y, confidence)
                kpts_np = kpts.cpu().numpy()
                
                # Get bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                
                # Build keypoint dictionary
                keypoints = {}
                for i, name in enumerate(self.KEYPOINT_NAMES):
                    if i < len(kpts_np):
                        x, y, kp_conf = kpts_np[i]
                        keypoints[name] = {
                            'x': float(x),
                            'y': float(y),
                            'confidence': float(kp_conf)
                        }
                
                # Detect activities/gestures
                activity = self._detect_activity(keypoints, idx)
                
                pose = {
                    'person_id': idx,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf,
                    'keypoints': keypoints,
                    'activity': activity
                }
                
                poses.append(pose)
        
        return poses
    
    def _detect_activity(self, keypoints, person_id):
        """
        Detect human activity/gesture from keypoints.
        
        Args:
            keypoints: Dictionary of keypoint positions
            person_id: Identifier for tracking
            
        Returns:
            Activity description string
        """
        activities = []
        
        # Check if keypoints are valid
        def is_valid(kp_name):
            return (kp_name in keypoints and 
                    keypoints[kp_name]['confidence'] > 0.3)
        
        # Detect waving (hand raised above shoulder with movement)
        if (is_valid('left_wrist') and is_valid('left_shoulder') and
            is_valid('left_elbow')):
            
            wrist = keypoints['left_wrist']
            shoulder = keypoints['left_shoulder']
            elbow = keypoints['left_elbow']
            
            # Hand raised above shoulder
            if wrist['y'] < shoulder['y'] - 50:
                # Check if arm is extended
                arm_angle = self._calculate_angle(shoulder, elbow, wrist)
                if arm_angle > 140:  # Relatively straight arm
                    activities.append('waving left hand')
        
        # Check right hand waving
        if (is_valid('right_wrist') and is_valid('right_shoulder') and
            is_valid('right_elbow')):
            
            wrist = keypoints['right_wrist']
            shoulder = keypoints['right_shoulder']
            elbow = keypoints['right_elbow']
            
            if wrist['y'] < shoulder['y'] - 50:
                arm_angle = self._calculate_angle(shoulder, elbow, wrist)
                if arm_angle > 140:
                    activities.append('waving right hand')
        
        # Detect both hands raised
        if (is_valid('left_wrist') and is_valid('right_wrist') and
            is_valid('left_shoulder') and is_valid('right_shoulder')):
            
            left_wrist = keypoints['left_wrist']
            right_wrist = keypoints['right_wrist']
            left_shoulder = keypoints['left_shoulder']
            right_shoulder = keypoints['right_shoulder']
            
            if (left_wrist['y'] < left_shoulder['y'] and
                right_wrist['y'] < right_shoulder['y']):
                activities.append('raising both hands')
        
        # Detect sitting (hips and knees visible, knees bent)
        if (is_valid('left_hip') and is_valid('left_knee') and 
            is_valid('left_ankle')):
            
            hip = keypoints['left_hip']
            knee = keypoints['left_knee']
            ankle = keypoints['left_ankle']
            
            knee_angle = self._calculate_angle(hip, knee, ankle)
            if 70 < knee_angle < 130:
                activities.append('sitting')
        
        # Detect standing (body upright, knees relatively straight)
        if (is_valid('left_shoulder') and is_valid('left_hip') and
            is_valid('left_knee') and is_valid('left_ankle')):
            
            shoulder = keypoints['left_shoulder']
            hip = keypoints['left_hip']
            knee = keypoints['left_knee']
            ankle = keypoints['left_ankle']
            
            # Check if body is upright
            body_angle = abs(shoulder['x'] - ankle['x'])
            knee_angle = self._calculate_angle(hip, knee, ankle)
            
            if body_angle < 50 and knee_angle > 150:
                activities.append('standing')
        
        # Detect arms crossed
        if (is_valid('left_wrist') and is_valid('right_wrist') and
            is_valid('left_shoulder') and is_valid('right_shoulder')):
            
            left_wrist = keypoints['left_wrist']
            right_wrist = keypoints['right_wrist']
            left_shoulder = keypoints['left_shoulder']
            right_shoulder = keypoints['right_shoulder']
            
            # Wrists near opposite shoulders
            if (abs(left_wrist['x'] - right_shoulder['x']) < 100 and
                abs(right_wrist['x'] - left_shoulder['x']) < 100):
                activities.append('arms crossed')
        
        # Detect pointing (arm extended toward camera)
        for side in ['left', 'right']:
            wrist_key = f'{side}_wrist'
            elbow_key = f'{side}_elbow'
            shoulder_key = f'{side}_shoulder'
            
            if is_valid(wrist_key) and is_valid(elbow_key) and is_valid(shoulder_key):
                wrist = keypoints[wrist_key]
                elbow = keypoints[elbow_key]
                shoulder = keypoints[shoulder_key]
                
                # Arm extended forward
                arm_angle = self._calculate_angle(shoulder, elbow, wrist)
                if arm_angle > 150 and abs(wrist['y'] - shoulder['y']) < 80:
                    activities.append(f'pointing {side} hand')
        
        # Return combined activities or default
        if activities:
            return ', '.join(activities)
        else:
            return 'standing still'
    
    def _calculate_angle(self, p1, p2, p3):
        """
        Calculate angle between three points (in degrees).
        
        Args:
            p1, p2, p3: Points with 'x' and 'y' keys
            
        Returns:
            Angle in degrees
        """
        try:
            # Vectors
            v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
            v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
            
            # Angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            
            return np.degrees(angle)
        except:
            return 0
    
    def draw_poses(self, frame, poses):
        """
        Draw pose keypoints and skeleton on frame.
        
        Args:
            frame: OpenCV BGR image
            poses: List of pose dictionaries
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Skeleton connections (COCO format)
        skeleton = [
            ('nose', 'left_eye'), ('nose', 'right_eye'),
            ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
            ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
            ('left_shoulder', 'right_shoulder'),
            ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
            ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
            ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
            ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        
        for pose in poses:
            keypoints = pose['keypoints']
            activity = pose['activity']
            
            # Draw skeleton
            for start, end in skeleton:
                if start in keypoints and end in keypoints:
                    start_kp = keypoints[start]
                    end_kp = keypoints[end]
                    
                    if start_kp['confidence'] > 0.3 and end_kp['confidence'] > 0.3:
                        pt1 = (int(start_kp['x']), int(start_kp['y']))
                        pt2 = (int(end_kp['x']), int(end_kp['y']))
                        cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
            
            # Draw keypoints
            for name, kp in keypoints.items():
                if kp['confidence'] > 0.3:
                    center = (int(kp['x']), int(kp['y']))
                    cv2.circle(annotated, center, 4, (0, 0, 255), -1)
            
            # Draw activity label
            x1, y1, x2, y2 = pose['bbox']
            label = f"Person: {activity}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return annotated