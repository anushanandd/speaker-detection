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
    """Represents a detected face with bounding box, confidence, and mouth state."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    center_x: int
    center_y: int
    is_mouth_open: bool = False
    mouth_openness: float = 0.0
    landmarks: Optional[List[Tuple[float, float]]] = None
    
    def __post_init__(self):
        """Calculate center coordinates after initialization."""
        self.center_x = self.x + self.width // 2
        self.center_y = self.y + self.height // 2


class FaceDetector:
    """Centralized face detection utilities with mouth detection."""
    
    def __init__(self, video_config: VideoConfig, detection_config: DetectionConfig):
        """
        Initialize face detector with FaceMesh for mouth detection.
        
        Args:
            video_config: Video configuration
            detection_config: Detection configuration
        """
        self.video_config = video_config
        self.detection_config = detection_config
        
        # Initialize MediaPipe FaceMesh for detailed facial landmarks
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=detection_config.max_faces,
            refine_landmarks=True,
            min_detection_confidence=video_config.face_confidence_threshold,
            min_tracking_confidence=0.5
        )
        
        # Mouth landmark indices (MediaPipe FaceMesh) - using central points for stability
        self.UPPER_LIP_CENTER = 13  # Central upper lip point
        self.LOWER_LIP_CENTER = 14  # Central lower lip point
        
        # Face reference points for normalization
        self.LEFT_EYE = 33   # Left eye corner
        self.RIGHT_EYE = 362 # Right eye corner
        self.LEFT_MOUTH = 61  # Left mouth corner
        self.RIGHT_MOUTH = 291 # Right mouth corner
        
        # Mouth detection threshold (normalized ratio, distance-invariant)
        self.mouth_open_threshold = getattr(video_config, 'mouth_open_threshold', 0.15)
        
        # Performance tracking
        self.frame_count = 0
        self.detection_skip_counter = 0
        
        logger.info("FaceDetector initialized with FaceMesh for mouth detection")
    
    def detect_mouth_openness(self, landmarks) -> Tuple[bool, float]:
        """
        Detect if mouth is open and calculate openness level using distance-invariant normalization.
        
        Uses central lip landmarks (13, 14) and normalizes by face width for scale invariance.
        
        Args:
            landmarks: MediaPipe face landmarks
            
        Returns:
            Tuple of (is_mouth_open, normalized_mouth_openness_ratio)
        """
        try:
            # Get central lip landmarks (most stable points)
            upper_lip = landmarks[self.UPPER_LIP_CENTER]
            lower_lip = landmarks[self.LOWER_LIP_CENTER]
            
            # Calculate vertical distance between lips
            mouth_vertical_distance = abs(upper_lip.y - lower_lip.y)
            
            # Get face reference points for normalization
            left_eye = landmarks[self.LEFT_EYE]
            right_eye = landmarks[self.RIGHT_EYE]
            left_mouth = landmarks[self.LEFT_MOUTH]
            right_mouth = landmarks[self.RIGHT_MOUTH]
            
            # Calculate face width (eye distance as primary reference)
            eye_distance = abs(right_eye.x - left_eye.x)
            mouth_width = abs(right_mouth.x - left_mouth.x)
            
            # Use the larger of eye distance or mouth width for normalization
            # This ensures we have a stable reference even if one measurement is poor
            face_reference = max(eye_distance, mouth_width)
            
            # Avoid division by zero
            if face_reference < 1e-6:
                logger.warning("Face reference distance too small for normalization")
                return False, 0.0
            
            # Normalize mouth openness by face reference (distance-invariant ratio)
            normalized_mouth_openness = mouth_vertical_distance / face_reference
            
            # Determine if mouth is open based on normalized threshold
            is_mouth_open = normalized_mouth_openness > self.mouth_open_threshold
            
            return is_mouth_open, normalized_mouth_openness
            
        except Exception as e:
            logger.warning(f"Error detecting mouth openness: {e}")
            return False, 0.0
    
    def get_face_bounding_box(self, landmarks, frame_shape) -> Tuple[int, int, int, int]:
        """
        Calculate face bounding box from landmarks.
        
        Args:
            landmarks: MediaPipe face landmarks
            frame_shape: (height, width, channels) of the frame
            
        Returns:
            Tuple of (x, y, width, height)
        """
        try:
            # Get all landmark coordinates
            x_coords = [landmark.x for landmark in landmarks]
            y_coords = [landmark.y for landmark in landmarks]
            
            # Convert to pixel coordinates
            h, w = frame_shape[:2]
            x_pixels = [int(x * w) for x in x_coords]
            y_pixels = [int(y * h) for y in y_coords]
            
            # Calculate bounding box
            x = max(0, min(x_pixels))
            y = max(0, min(y_pixels))
            width = min(w - x, max(x_pixels) - x)
            height = min(h - y, max(y_pixels) - y)
            
            return x, y, width, height
            
        except Exception as e:
            logger.warning(f"Error calculating bounding box: {e}")
            return 0, 0, 0, 0
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in the video frame using MediaPipe FaceMesh with mouth detection.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of detected faces with bounding boxes, confidence scores, and mouth state
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
            results = self.face_mesh.process(rgb_frame)
            
            faces = []
            if results.multi_face_landmarks:
                h, w, _ = frame.shape
                
                for face_landmarks in results.multi_face_landmarks:
                    try:
                        # Calculate bounding box from landmarks
                        x, y, width, height = self.get_face_bounding_box(face_landmarks.landmark, frame.shape)
                        
                        # Validate bounding box
                        if (x >= 0 and y >= 0 and 
                            x + width <= w and y + height <= h and
                            width > 0 and height > 0):
                            
                            # Detect mouth openness
                            is_mouth_open, mouth_openness = self.detect_mouth_openness(face_landmarks.landmark)
                            
                            # Convert landmarks to list of tuples for storage
                            landmarks_list = [(lm.x, lm.y) for lm in face_landmarks.landmark]
                            
                            # Create face detection with mouth state
                            face = FaceDetection(
                                x=x, y=y, width=width, height=height,
                                confidence=0.9,  # FaceMesh doesn't provide confidence, use high default
                                center_x=0, center_y=0,  # Will be calculated in __post_init__
                                is_mouth_open=is_mouth_open,
                                mouth_openness=mouth_openness,
                                landmarks=landmarks_list
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
    
    def draw_face_mesh(self, frame: np.ndarray, show_full_mesh: bool = False) -> np.ndarray:
        """
        Draw face mesh with landmarks on the frame.
        
        Args:
            frame: Video frame to draw on
            show_full_mesh: Whether to show full face mesh or just landmarks
            
        Returns:
            Frame with face mesh drawn
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    
                    if show_full_mesh:
                        # Draw full face mesh contours
                        self.mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_CONTOURS,
                            None,
                            self.mp_drawing_styles.get_default_face_mesh_contours_style()
                        )
                        
                        # Draw face mesh tesselation
                        self.mp_drawing.draw_landmarks(
                            frame,
                            face_landmarks,
                            self.mp_face_mesh.FACEMESH_TESSELATION,
                            None,
                            self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                        )
                    
                    # Always highlight mouth opening landmarks (central lip points)
                    mouth_landmarks = [self.UPPER_LIP_CENTER, self.LOWER_LIP_CENTER]
                    for idx in mouth_landmarks:
                        if idx < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)  # Red dots for central lip points
                    
                    # Also highlight reference points for normalization
                    reference_landmarks = [self.LEFT_EYE, self.RIGHT_EYE, self.LEFT_MOUTH, self.RIGHT_MOUTH]
                    for idx in reference_landmarks:
                        if idx < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(frame, (x, y), 2, (255, 255, 0), -1)  # Yellow dots for reference points
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing face mesh: {e}")
            return frame

    def draw_faces(self, frame: np.ndarray, faces: List[FaceDetection], 
                   active_face: Optional[FaceDetection] = None) -> np.ndarray:
        """
        Draw face detections on the frame with mouth state indicators.
        
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
                is_speaking = active_face and face == active_face
                
                if is_speaking and face.is_mouth_open:
                    # Green: speaking AND mouth open
                    color = (0, 255, 0)
                    status = "SPEAKING"
                    thickness = 3
                elif is_speaking and not face.is_mouth_open:
                    # Blue: speaking but mouth closed
                    color = (255, 0, 0)
                    status = "SPEAKING"
                    thickness = 3
                elif not is_speaking and face.is_mouth_open:
                    # Yellow: mouth open but not speaking
                    color = (0, 255, 255)
                    status = "MOUTH OPEN"
                    thickness = 2
                else:
                    # Red: mouth closed and not speaking
                    color = (0, 0, 255)
                    status = "FACE"
                    thickness = 2
                
                # Draw face rectangle
                cv2.rectangle(frame, (face.x, face.y), 
                             (face.x + face.width, face.y + face.height), 
                             color, thickness)
                
                # Draw status text
                cv2.putText(frame, status, (face.x, face.y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Draw mouth openness level
                mouth_text = f"Mouth: {face.mouth_openness:.3f}"
                cv2.putText(frame, mouth_text, (face.x, face.y + face.height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                
                # Draw mouth state indicator
                mouth_indicator = "OPEN" if face.is_mouth_open else "CLOSED"
                cv2.putText(frame, mouth_indicator, (face.x, face.y + face.height + 40), 
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
        """Close MediaPipe face mesh."""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
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
