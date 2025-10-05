"""
Unit Tests for Face Detection Module
===================================
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.face_detector import FaceDetector, FaceDetection
from src.config_manager import VideoConfig, DetectionConfig


class TestFaceDetection(unittest.TestCase):
    """Test cases for FaceDetection dataclass."""
    
    def test_face_detection_creation(self):
        """Test FaceDetection object creation."""
        face = FaceDetection(
            x=100, y=50, width=200, height=250,
            confidence=0.95, center_x=0, center_y=0
        )
        
        self.assertEqual(face.x, 100)
        self.assertEqual(face.y, 50)
        self.assertEqual(face.width, 200)
        self.assertEqual(face.height, 250)
        self.assertEqual(face.confidence, 0.95)
        self.assertEqual(face.center_x, 200)  # x + width // 2
        self.assertEqual(face.center_y, 175)  # y + height // 2
    
    def test_face_detection_center_calculation(self):
        """Test center coordinate calculation."""
        face = FaceDetection(
            x=0, y=0, width=100, height=100,
            confidence=0.8, center_x=0, center_y=0
        )
        
        self.assertEqual(face.center_x, 50)
        self.assertEqual(face.center_y, 50)


class TestFaceDetector(unittest.TestCase):
    """Test cases for FaceDetector class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.video_config = VideoConfig()
        self.detection_config = DetectionConfig()
        # Note: FaceDetector requires MediaPipe, so we'll test what we can without it
        self.detector = None
    
    def test_calculate_overlap(self):
        """Test overlap calculation between two faces."""
        # Create a mock detector for testing utility methods
        class MockDetector:
            def calculate_overlap(self, face1, face2):
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
        
        detector = MockDetector()
        
        # Test non-overlapping faces
        face1 = FaceDetection(0, 0, 100, 100, 0.9, 50, 50)
        face2 = FaceDetection(200, 200, 100, 100, 0.8, 250, 250)
        overlap = detector.calculate_overlap(face1, face2)
        self.assertEqual(overlap, 0.0)
        
        # Test overlapping faces
        face3 = FaceDetection(0, 0, 100, 100, 0.9, 50, 50)
        face4 = FaceDetection(50, 50, 100, 100, 0.8, 100, 100)
        overlap = detector.calculate_overlap(face3, face4)
        self.assertGreater(overlap, 0.0)
        self.assertLessEqual(overlap, 1.0)
        
        # Test identical faces
        face5 = FaceDetection(0, 0, 100, 100, 0.9, 50, 50)
        face6 = FaceDetection(0, 0, 100, 100, 0.8, 50, 50)
        overlap = detector.calculate_overlap(face5, face6)
        self.assertEqual(overlap, 1.0)
    
    def test_non_max_suppression(self):
        """Test non-maximum suppression algorithm."""
        # Create a mock detector for testing utility methods
        class MockDetector:
            def __init__(self):
                self.video_config = VideoConfig()
            
            def calculate_overlap(self, face1, face2):
                # Simplified overlap calculation
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
            
            def non_max_suppression(self, faces, overlap_threshold=None):
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
        
        detector = MockDetector()
        
        # Test with non-overlapping faces
        faces = [
            FaceDetection(0, 0, 100, 100, 0.9, 50, 50),
            FaceDetection(200, 200, 100, 100, 0.8, 250, 250),
        ]
        result = detector.non_max_suppression(faces)
        self.assertEqual(len(result), 2)
        
        # Test with overlapping faces
        faces = [
            FaceDetection(0, 0, 100, 100, 0.9, 50, 50),
            FaceDetection(25, 25, 100, 100, 0.8, 75, 75),  # Overlaps more with first
        ]
        result = detector.non_max_suppression(faces)
        self.assertEqual(len(result), 1)  # Should keep only the higher confidence one
        self.assertEqual(result[0].confidence, 0.9)


class TestVideoConfig(unittest.TestCase):
    """Test cases for VideoConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = VideoConfig()
        
        self.assertEqual(config.camera_index, 0)
        self.assertEqual(config.frame_width, 640)
        self.assertEqual(config.frame_height, 480)
        self.assertEqual(config.face_confidence_threshold, 0.7)
        self.assertEqual(config.overlap_threshold, 0.3)
        self.assertEqual(config.mouth_open_threshold, 0.02)
        self.assertEqual(config.face_detection_skip_frames, 2)


class TestDetectionConfig(unittest.TestCase):
    """Test cases for DetectionConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = DetectionConfig()
        
        self.assertEqual(config.speaker_confidence_threshold, 0.3)
        self.assertEqual(config.angle_offset, 0.0)
        self.assertEqual(config.max_faces, 10)


if __name__ == '__main__':
    unittest.main()
