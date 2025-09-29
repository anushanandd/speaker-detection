"""
Face Detection Module
====================

Centralized face detection utilities using MediaPipe with
non-maximum suppression and robust face tracking.
"""

import cv2
import numpy as np
import mediapipe as mp
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .config_manager import VideoConfig, DetectionConfig

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """Represents a detected face with bounding box and confidence."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    center_x: int
    center_y: int
    
    def __post_init__(self):
        """Calculate center coordinates after initialization."""
        self.center_x = self.x + self.width // 2
        self.center_y = self.y + self.height // 2


class FaceDetector:
    """Centralized face detection utilities."""
    
    def __init__(self, video_config: VideoConfig, detection_config: DetectionConfig):
        """
        Initialize face detector.
        
        Args:
            video_config: Video configuration
            detection_config: Detection configuration
        """
        self.video_config = video_config
        self.detection_config = detection_config
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for close-range, 1 for full-range
            min_detection_confidence=video_config.face_confidence_threshold
        )
        
        # Performance tracking
        self.frame_count = 0
        self.detection_skip_counter = 0
        
        logger.info("FaceDetector initialized")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the video frame using MediaPipe.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        try:
            # Skip frames for performance if configured
            if hasattr(self, 'skip_frames') and self.skip_frames:
                self.detection_skip_counter += 1
                if self.detection_skip_counter < self.video_config.face_detection_skip_frames:
                    return []
                self.detection_skip_counter = 0
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            faces = []
            if results.detections:
                h, w, _ = frame.shape
                
                for detection in results.detections:
                    try:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * w)
                        y = int(bbox.ymin * h)
                        width = int(bbox.width * w)
                        height = int(bbox.height * h)
                        confidence = detection.score[0]
                        
                        # Validate bounding box
                        if (x >= 0 and y >= 0 and 
                            x + width <= w and y + height <= h and
                            width > 0 and height > 0):
                            
                            face = FaceDetection(
                                x=x, y=y, width=width, height=height,
                                confidence=confidence,
                                center_x=0, center_y=0  # Will be calculated in __post_init__
                            )
                            faces.append(face)
                            
                    except Exception as e:
                        logger.warning(f"Error processing individual face: {e}")
                        continue
            
            # Apply non-maximum suppression to remove overlapping faces
            faces = self.non_max_suppression(faces)
            
            # Limit number of faces
            if len(faces) > self.detection_config.max_faces:
                faces = sorted(faces, key=lambda x: x.confidence, reverse=True)[:self.detection_config.max_faces]
            
            self.frame_count += 1
            return faces
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def non_max_suppression(self, faces: List[FaceDetection], 
                           overlap_threshold: Optional[float] = None) -> List[FaceDetection]:
        """
        Remove overlapping face detections using non-maximum suppression.
        
        Args:
            faces: List of detected faces
            overlap_threshold: Maximum overlap ratio to allow
            
        Returns:
            List of non-overlapping faces
        """
        if len(faces) <= 1:
            return faces
        
        if overlap_threshold is None:
            overlap_threshold = self.video_config.overlap_threshold
        
        # Sort by confidence (highest first)
        faces = sorted(faces, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        for face in faces:
            # Check if this face overlaps significantly with any kept face
            should_keep = True
            for kept_face in keep:
                overlap = self.calculate_overlap(face, kept_face)
                if overlap > overlap_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep.append(face)
        
        return keep
    
    def calculate_overlap(self, face1: FaceDetection, face2: FaceDetection) -> float:
        """
        Calculate overlap ratio between two face bounding boxes.
        
        Args:
            face1: First face detection
            face2: Second face detection
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        try:
            # Calculate intersection rectangle
            x1 = max(face1.x, face2.x)
            y1 = max(face1.y, face2.y)
            x2 = min(face1.x + face1.width, face2.x + face2.width)
            y2 = min(face1.y + face1.height, face2.y + face2.height)
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = face1.width * face1.height
            area2 = face2.width * face2.height
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overlap: {e}")
            return 0.0
    
    def draw_faces(self, frame: np.ndarray, faces: List[FaceDetection], 
                   active_face: Optional[FaceDetection] = None) -> np.ndarray:
        """
        Draw face detections on the frame.
        
        Args:
            frame: Video frame to draw on
            faces: List of detected faces
            active_face: Currently active face (if any)
            
        Returns:
            Frame with face detections drawn
        """
        try:
            for face in faces:
                # Determine color and thickness based on state
                if active_face and face == active_face:
                    color = (0, 255, 0)  # Green: active speaker
                    thickness = 3
                    status = "ACTIVE"
                else:
                    color = (128, 128, 128)  # Gray: detected face
                    thickness = 2
                    status = "FACE"
                
                # Draw face rectangle
                cv2.rectangle(frame, (face.x, face.y), 
                             (face.x + face.width, face.y + face.height), 
                             color, thickness)
                
                # Draw status text
                cv2.putText(frame, status, (face.x, face.y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw confidence score
                cv2.putText(frame, f"{face.confidence:.2f}", 
                           (face.x, face.y + face.height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing faces: {e}")
            return frame
    
    def set_skip_frames(self, skip: bool) -> None:
        """
        Enable or disable frame skipping for performance.
        
        Args:
            skip: Whether to skip frames
        """
        self.skip_frames = skip
        self.detection_skip_counter = 0
        logger.info(f"Frame skipping {'enabled' if skip else 'disabled'}")
    
    def get_detection_stats(self) -> dict:
        """Get face detection statistics."""
        return {
            'frames_processed': self.frame_count,
            'skip_frames_enabled': getattr(self, 'skip_frames', False),
            'skip_counter': self.detection_skip_counter
        }
    
    def close(self) -> None:
        """Close MediaPipe face detection."""
        try:
            if hasattr(self, 'face_detection'):
                self.face_detection.close()
            logger.info("FaceDetector closed")
        except Exception as e:
            logger.error(f"Error closing FaceDetector: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.close()
