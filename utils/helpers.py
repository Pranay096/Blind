"""
Helper Utilities
Miscellaneous helper functions for LUMOS Vision.
"""

import cv2
import numpy as np

def draw_detections(frame, detections):
    """
    Draw bounding boxes and labels on frame.
    
    Args:
        frame: OpenCV BGR image
        detections: List of detection dictionaries
        
    Returns:
        Annotated frame
    """
    annotated = frame.copy()
    
    for det in detections:
        bbox = det['bbox']
        name = det['name']
        confidence = det.get('confidence', 0)
        distance = det.get('distance_m', None)
        direction = det.get('direction', '')
        
        x1, y1, x2, y2 = bbox
        
        # Color based on distance (red = close, green = far)
        if distance:
            if distance < 1.5:
                color = (0, 0, 255)  # Red
            elif distance < 3.0:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green
        else:
            color = (255, 0, 0)  # Blue
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{name} {confidence:.0%}"
        if distance:
            label += f" | {distance}m {direction}"
        
        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(annotated, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return annotated


def draw_depth_overlay(frame, depth_map, alpha=0.5):
    """
    Overlay depth map on frame.
    
    Args:
        frame: OpenCV BGR image
        depth_map: Normalized depth map
        alpha: Transparency (0 = invisible, 1 = opaque)
        
    Returns:
        Frame with depth overlay
    """
    # Convert depth map to color (closer = warmer colors)
    depth_colored = cv2.applyColorMap(
        (depth_map * 255).astype(np.uint8), 
        cv2.COLORMAP_JET
    )
    
    # Blend with original frame
    overlay = cv2.addWeighted(frame, 1 - alpha, depth_colored, alpha, 0)
    
    return overlay


def draw_text_boxes(frame, text_regions):
    """
    Draw boxes around detected text regions.
    
    Args:
        frame: OpenCV BGR image
        text_regions: List of (x, y, w, h) tuples
        
    Returns:
        Frame with text boxes
    """
    annotated = frame.copy()
    
    for (x, y, w, h) in text_regions:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
    
    return annotated


def format_distance(distance_m):
    """
    Format distance for speech.
    
    Args:
        distance_m: Distance in meters
        
    Returns:
        Formatted string
    """
    if distance_m < 1.0:
        return f"{int(distance_m * 100)} centimeters"
    else:
        return f"{distance_m:.1f} meters"


def prioritize_objects(detections):
    """
    Prioritize objects for narration (closest and most important first).
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Sorted list of detections
    """
    # Priority weights for different object types
    priority_weights = {
        'person': 10,
        'car': 9,
        'truck': 9,
        'bus': 9,
        'motorcycle': 8,
        'bicycle': 7,
        'dog': 6,
        'cat': 6,
        'chair': 3,
        'table': 3,
    }
    
    def get_priority(det):
        # Base priority on object type
        obj_priority = priority_weights.get(det['name'], 5)
        
        # Adjust by distance (closer = higher priority)
        distance = det.get('distance_m', 10)
        distance_factor = max(0, 10 - distance)
        
        return obj_priority + distance_factor
    
    # Sort by priority (highest first)
    sorted_detections = sorted(detections, key=get_priority, reverse=True)
    
    return sorted_detections