"""
Audio-Visual Speaker Detection System
====================================

A real-time system that combines ReSpeaker microphone array audio direction detection
with webcam face detection to identify the active speaker.

Author: AudioVisual Detection Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "AudioVisual Detection Team"
__email__ = "contact@audiovisual-detection.com"

from .config_manager import ConfigManager
from .audio_processor import AudioProcessor
from .face_detector import FaceDetector
from .speaker_detector import AudioVisualSpeakerDetector

__all__ = [
    "ConfigManager",
    "AudioProcessor", 
    "FaceDetector",
    "AudioVisualSpeakerDetector"
]
