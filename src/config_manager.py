"""
Configuration Management System
==============================

Centralized configuration management using YAML files with validation
and environment variable support.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    device_index: int = 0
    sample_rate: int = 16000
    n_channels: int = 6
    mic_indexes: List[int] = field(default_factory=lambda: [2, 3, 4, 5])
    frame_samples: int = 1024
    hop_samples: int = 512
    mic_spacing: float = 0.035
    sound_speed: float = 343.0
    activity_threshold: float = 0.003


@dataclass
class VideoConfig:
    """Video processing configuration."""
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    face_confidence_threshold: float = 0.7
    overlap_threshold: float = 0.3


@dataclass
class DetectionConfig:
    """Detection algorithm configuration."""
    speaker_confidence_threshold: float = 0.3
    angle_offset: float = 0.0
    max_faces: int = 10


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/speaker_detection.log"


@dataclass
class PerformanceConfig:
    """Performance optimization configuration."""
    audio_buffer_size: int = 4
    face_detection_skip_frames: int = 2
    max_processing_time: float = 0.1


class ConfigManager:
    """Centralized configuration management system."""
    
    def __init__(self, config_file: str = "config/default.yaml"):
        """
        Initialize configuration manager.
        
        Args:
            config_file: Path to configuration file
        """
        self.config_file = Path(config_file)
        self.config_data = {}
        self.audio_config = AudioConfig()
        self.video_config = VideoConfig()
        self.detection_config = DetectionConfig()
        self.logging_config = LoggingConfig()
        self.performance_config = PerformanceConfig()
        self.mic_pairs: List[Tuple[int, int, float]] = []
        
        self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {self.config_file}")
            else:
                logger.warning(f"Configuration file {self.config_file} not found, using defaults")
                self.config_data = {}
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            self.config_data = {}
        
        self._parse_config()
    
    def _parse_config(self) -> None:
        """Parse configuration data into dataclasses."""
        # Audio configuration
        audio_data = self.config_data.get('audio', {})
        self.audio_config = AudioConfig(
            device_index=audio_data.get('device_index', self.audio_config.device_index),
            sample_rate=audio_data.get('sample_rate', self.audio_config.sample_rate),
            n_channels=audio_data.get('n_channels', self.audio_config.n_channels),
            mic_indexes=audio_data.get('mic_indexes', self.audio_config.mic_indexes),
            frame_samples=audio_data.get('frame_samples', self.audio_config.frame_samples),
            hop_samples=audio_data.get('hop_samples', self.audio_config.hop_samples),
            mic_spacing=audio_data.get('mic_spacing', self.audio_config.mic_spacing),
            sound_speed=audio_data.get('sound_speed', self.audio_config.sound_speed),
            activity_threshold=audio_data.get('activity_threshold', self.audio_config.activity_threshold)
        )
        
        # Video configuration
        video_data = self.config_data.get('video', {})
        self.video_config = VideoConfig(
            camera_index=video_data.get('camera_index', self.video_config.camera_index),
            frame_width=video_data.get('frame_width', self.video_config.frame_width),
            frame_height=video_data.get('frame_height', self.video_config.frame_height),
            face_confidence_threshold=video_data.get('face_confidence_threshold', self.video_config.face_confidence_threshold),
            overlap_threshold=video_data.get('overlap_threshold', self.video_config.overlap_threshold)
        )
        
        # Detection configuration
        detection_data = self.config_data.get('detection', {})
        self.detection_config = DetectionConfig(
            speaker_confidence_threshold=detection_data.get('speaker_confidence_threshold', self.detection_config.speaker_confidence_threshold),
            angle_offset=detection_data.get('angle_offset', self.detection_config.angle_offset),
            max_faces=detection_data.get('max_faces', self.detection_config.max_faces)
        )
        
        # Logging configuration
        logging_data = self.config_data.get('logging', {})
        self.logging_config = LoggingConfig(
            level=logging_data.get('level', self.logging_config.level),
            format=logging_data.get('format', self.logging_config.format),
            file=logging_data.get('file', self.logging_config.file)
        )
        
        # Performance configuration
        performance_data = self.config_data.get('performance', {})
        self.performance_config = PerformanceConfig(
            audio_buffer_size=performance_data.get('audio_buffer_size', self.performance_config.audio_buffer_size),
            face_detection_skip_frames=performance_data.get('face_detection_skip_frames', self.performance_config.face_detection_skip_frames),
            max_processing_time=performance_data.get('max_processing_time', self.performance_config.max_processing_time)
        )
        
        # Microphone pairs
        mic_pairs_data = self.config_data.get('mic_pairs', [])
        self.mic_pairs = [tuple(pair) for pair in mic_pairs_data]
        
        # Override with environment variables if present
        self._apply_env_overrides()
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_mappings = {
            'AUDIO_DEVICE_INDEX': ('audio_config', 'device_index', int),
            'AUDIO_SAMPLE_RATE': ('audio_config', 'sample_rate', int),
            'CAMERA_INDEX': ('video_config', 'camera_index', int),
            'LOG_LEVEL': ('logging_config', 'level', str),
        }
        
        for env_var, (config_attr, field_name, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setattr(getattr(self, config_attr), field_name, type_func(value))
                    logger.info(f"Override from {env_var}: {field_name} = {value}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {value} ({e})")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        log_file = Path(self.logging_config.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, self.logging_config.level.upper()),
            format=self.logging_config.format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logger.info("Logging system initialized")
    
    def get_mic_pairs(self) -> List[Tuple[int, int, float]]:
        """Get microphone pairs for TDOA calculation."""
        if not self.mic_pairs:
            # Generate default pairs if not configured
            spacing = self.audio_config.mic_spacing
            self.mic_pairs = [
                (0, 3, 3 * spacing),  # Outermost pair
                (0, 2, 2 * spacing),
                (1, 3, 2 * spacing),
                (1, 2, 1 * spacing),
                (0, 1, 1 * spacing),
                (2, 3, 1 * spacing),
            ]
        return self.mic_pairs
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        errors = []
        
        # Validate audio config
        if self.audio_config.sample_rate <= 0:
            errors.append("Sample rate must be positive")
        if self.audio_config.n_channels <= 0:
            errors.append("Number of channels must be positive")
        if len(self.audio_config.mic_indexes) != 4:
            errors.append("Must specify exactly 4 microphone indexes")
        if self.audio_config.mic_spacing <= 0:
            errors.append("Microphone spacing must be positive")
        
        # Validate video config
        if self.video_config.frame_width <= 0 or self.video_config.frame_height <= 0:
            errors.append("Frame dimensions must be positive")
        if not 0 <= self.video_config.face_confidence_threshold <= 1:
            errors.append("Face confidence threshold must be between 0 and 1")
        
        # Validate detection config
        if not 0 <= self.detection_config.speaker_confidence_threshold <= 1:
            errors.append("Speaker confidence threshold must be between 0 and 1")
        
        if errors:
            for error in errors:
                logger.error(f"Configuration validation error: {error}")
            return False
        
        logger.info("Configuration validation passed")
        return True
    
    def save_config(self, output_file: str = None) -> None:
        """Save current configuration to YAML file."""
        if output_file is None:
            output_file = self.config_file
        
        config_dict = {
            'audio': {
                'device_index': self.audio_config.device_index,
                'sample_rate': self.audio_config.sample_rate,
                'n_channels': self.audio_config.n_channels,
                'mic_indexes': self.audio_config.mic_indexes,
                'frame_samples': self.audio_config.frame_samples,
                'hop_samples': self.audio_config.hop_samples,
                'mic_spacing': self.audio_config.mic_spacing,
                'sound_speed': self.audio_config.sound_speed,
                'activity_threshold': self.audio_config.activity_threshold
            },
            'video': {
                'camera_index': self.video_config.camera_index,
                'frame_width': self.video_config.frame_width,
                'frame_height': self.video_config.frame_height,
                'face_confidence_threshold': self.video_config.face_confidence_threshold,
                'overlap_threshold': self.video_config.overlap_threshold
            },
            'detection': {
                'speaker_confidence_threshold': self.detection_config.speaker_confidence_threshold,
                'angle_offset': self.detection_config.angle_offset,
                'max_faces': self.detection_config.max_faces
            },
            'logging': {
                'level': self.logging_config.level,
                'format': self.logging_config.format,
                'file': self.logging_config.file
            },
            'performance': {
                'audio_buffer_size': self.performance_config.audio_buffer_size,
                'face_detection_skip_frames': self.performance_config.face_detection_skip_frames,
                'max_processing_time': self.performance_config.max_processing_time
            },
            'mic_pairs': [list(pair) for pair in self.mic_pairs]
        }
        
        try:
            with open(output_file, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"ConfigManager(config_file={self.config_file})"
    
    def __repr__(self) -> str:
        """Detailed string representation of configuration."""
        return (f"ConfigManager(\n"
                f"  audio_config={self.audio_config},\n"
                f"  video_config={self.video_config},\n"
                f"  detection_config={self.detection_config},\n"
                f"  logging_config={self.logging_config},\n"
                f"  performance_config={self.performance_config}\n"
                f")")
