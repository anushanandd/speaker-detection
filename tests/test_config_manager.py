"""
Unit Tests for Configuration Management Module
============================================
"""

import unittest
import tempfile
import os
import yaml
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config_manager import ConfigManager, AudioConfig, VideoConfig, DetectionConfig


class TestConfigDataclasses(unittest.TestCase):
    """Test cases for configuration dataclasses."""
    
    def test_audio_config_defaults(self):
        """Test AudioConfig default values."""
        config = AudioConfig()
        
        self.assertEqual(config.device_index, 0)
        self.assertEqual(config.sample_rate, 16000)
        self.assertEqual(config.n_channels, 6)
        self.assertEqual(config.mic_indexes, [2, 3, 4, 5])
        self.assertEqual(config.frame_samples, 1024)
        self.assertEqual(config.hop_samples, 512)
        self.assertEqual(config.mic_spacing, 0.035)
        self.assertEqual(config.sound_speed, 343.0)
        self.assertEqual(config.activity_threshold, 0.003)
    
    def test_video_config_defaults(self):
        """Test VideoConfig default values."""
        config = VideoConfig()
        
        self.assertEqual(config.camera_index, 0)
        self.assertEqual(config.frame_width, 640)
        self.assertEqual(config.frame_height, 480)
        self.assertEqual(config.face_confidence_threshold, 0.7)
        self.assertEqual(config.overlap_threshold, 0.3)
    
    def test_detection_config_defaults(self):
        """Test DetectionConfig default values."""
        config = DetectionConfig()
        
        self.assertEqual(config.speaker_confidence_threshold, 0.3)
        self.assertEqual(config.angle_offset, 0.0)
        self.assertEqual(config.max_faces, 10)


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, "test_config.yaml")
    
    def tearDown(self):
        """Clean up after tests."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_config(self, config_data):
        """Create a test configuration file."""
        with open(self.config_file, 'w') as f:
            yaml.dump(config_data, f)
    
    def test_default_config_loading(self):
        """Test loading default configuration."""
        # Test with non-existent config file (should use defaults)
        config_manager = ConfigManager("non_existent.yaml")
        
        # Should have default values
        self.assertEqual(config_manager.audio_config.sample_rate, 16000)
        self.assertEqual(config_manager.video_config.frame_width, 640)
        self.assertEqual(config_manager.detection_config.max_faces, 10)
    
    def test_custom_config_loading(self):
        """Test loading custom configuration."""
        config_data = {
            'audio': {
                'sample_rate': 44100,
                'device_index': 1,
                'mic_spacing': 0.05
            },
            'video': {
                'frame_width': 1280,
                'frame_height': 720,
                'face_confidence_threshold': 0.8
            },
            'detection': {
                'speaker_confidence_threshold': 0.5,
                'max_faces': 5
            }
        }
        
        self.create_test_config(config_data)
        config_manager = ConfigManager(self.config_file)
        
        # Check that custom values are loaded
        self.assertEqual(config_manager.audio_config.sample_rate, 44100)
        self.assertEqual(config_manager.audio_config.device_index, 1)
        self.assertEqual(config_manager.audio_config.mic_spacing, 0.05)
        self.assertEqual(config_manager.video_config.frame_width, 1280)
        self.assertEqual(config_manager.video_config.frame_height, 720)
        self.assertEqual(config_manager.video_config.face_confidence_threshold, 0.8)
        self.assertEqual(config_manager.detection_config.speaker_confidence_threshold, 0.5)
        self.assertEqual(config_manager.detection_config.max_faces, 5)
    
    def test_partial_config_loading(self):
        """Test loading partial configuration (some values missing)."""
        config_data = {
            'audio': {
                'sample_rate': 22050
                # Other audio values should use defaults
            },
            'video': {
                'frame_width': 800
                # Other video values should use defaults
            }
        }
        
        self.create_test_config(config_data)
        config_manager = ConfigManager(self.config_file)
        
        # Check custom values
        self.assertEqual(config_manager.audio_config.sample_rate, 22050)
        self.assertEqual(config_manager.video_config.frame_width, 800)
        
        # Check default values are preserved
        self.assertEqual(config_manager.audio_config.device_index, 0)
        self.assertEqual(config_manager.video_config.frame_height, 480)
        self.assertEqual(config_manager.detection_config.max_faces, 10)
    
    def test_mic_pairs_loading(self):
        """Test loading microphone pairs configuration."""
        config_data = {
            'mic_pairs': [
                [0, 1, 0.035],
                [1, 2, 0.035],
                [2, 3, 0.035]
            ]
        }
        
        self.create_test_config(config_data)
        config_manager = ConfigManager(self.config_file)
        
        expected_pairs = [(0, 1, 0.035), (1, 2, 0.035), (2, 3, 0.035)]
        self.assertEqual(config_manager.mic_pairs, expected_pairs)
    
    def test_get_mic_pairs_default(self):
        """Test getting default microphone pairs when not configured."""
        config_manager = ConfigManager("non_existent.yaml")
        pairs = config_manager.get_mic_pairs()
        
        # Should generate default pairs
        self.assertGreater(len(pairs), 0)
        self.assertEqual(len(pairs[0]), 3)  # Each pair should have (i, j, distance)
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid configuration
        config_manager = ConfigManager("non_existent.yaml")
        self.assertTrue(config_manager.validate_config())
        
        # Test invalid configuration
        config_manager.audio_config.sample_rate = -1  # Invalid
        self.assertFalse(config_manager.validate_config())
        
        config_manager.audio_config.sample_rate = 16000  # Reset
        config_manager.audio_config.mic_indexes = [1, 2, 3]  # Wrong length
        self.assertFalse(config_manager.validate_config())
    
    def test_save_config(self):
        """Test saving configuration to file."""
        config_manager = ConfigManager("non_existent.yaml")
        
        # Modify some values
        config_manager.audio_config.sample_rate = 44100
        config_manager.video_config.frame_width = 1280
        
        # Save to file
        output_file = os.path.join(self.temp_dir, "output_config.yaml")
        config_manager.save_config(output_file)
        
        # Load and verify
        new_config_manager = ConfigManager(output_file)
        self.assertEqual(new_config_manager.audio_config.sample_rate, 44100)
        self.assertEqual(new_config_manager.video_config.frame_width, 1280)
    
    def test_environment_variable_overrides(self):
        """Test environment variable overrides."""
        # Set environment variables
        os.environ['AUDIO_DEVICE_INDEX'] = '2'
        os.environ['AUDIO_SAMPLE_RATE'] = '22050'
        os.environ['CAMERA_INDEX'] = '1'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        
        try:
            config_manager = ConfigManager("non_existent.yaml")
            
            # Check that environment variables override defaults
            self.assertEqual(config_manager.audio_config.device_index, 2)
            self.assertEqual(config_manager.audio_config.sample_rate, 22050)
            self.assertEqual(config_manager.video_config.camera_index, 1)
            self.assertEqual(config_manager.logging_config.level, 'DEBUG')
        finally:
            # Clean up environment variables
            for var in ['AUDIO_DEVICE_INDEX', 'AUDIO_SAMPLE_RATE', 'CAMERA_INDEX', 'LOG_LEVEL']:
                if var in os.environ:
                    del os.environ[var]
    
    def test_invalid_environment_variables(self):
        """Test handling of invalid environment variables."""
        # Set invalid environment variable
        os.environ['AUDIO_DEVICE_INDEX'] = 'invalid'
        
        try:
            config_manager = ConfigManager("non_existent.yaml")
            # Should use default value when environment variable is invalid
            self.assertEqual(config_manager.audio_config.device_index, 0)
        finally:
            if 'AUDIO_DEVICE_INDEX' in os.environ:
                del os.environ['AUDIO_DEVICE_INDEX']


if __name__ == '__main__':
    unittest.main()
