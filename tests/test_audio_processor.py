"""
Unit Tests for Audio Processing Module
=====================================
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.audio_processor import AudioProcessor
from src.config_manager import AudioConfig, PerformanceConfig


class TestAudioProcessor(unittest.TestCase):
    """Test cases for AudioProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.audio_config = AudioConfig()
        self.perf_config = PerformanceConfig()
        self.processor = AudioProcessor(self.audio_config, self.perf_config)
    
    def tearDown(self):
        """Clean up after tests."""
        self.processor.stop_stream()
    
    def test_gcc_phat_basic(self):
        """Test basic GCC-PHAT functionality."""
        # Create test signals with known delay
        fs = 16000
        t = np.linspace(0, 1, fs)
        sig1 = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        delay_samples = 100
        sig2 = np.roll(sig1, delay_samples)  # Delayed version
        
        tau, cc = AudioProcessor.gcc_phat(sig1, sig2, fs)
        
        # Check that delay is detected correctly (within reasonable tolerance)
        expected_tau = delay_samples / fs
        self.assertAlmostEqual(tau, expected_tau, places=3)
    
    def test_gcc_phat_no_delay(self):
        """Test GCC-PHAT with no delay."""
        fs = 16000
        t = np.linspace(0, 0.1, int(fs * 0.1))
        sig = np.sin(2 * np.pi * 440 * t)
        
        tau, cc = AudioProcessor.gcc_phat(sig, sig, fs)
        
        # Should detect no delay
        self.assertAlmostEqual(tau, 0.0, places=3)
    
    def test_tdoa_to_angle(self):
        """Test TDOA to angle conversion."""
        # Test with known values
        tau = 0.001  # 1ms delay
        distance = 0.035  # 3.5cm spacing
        sound_speed = 343.0
        
        angle = AudioProcessor.tdoa_to_angle(tau, distance, sound_speed)
        
        # Calculate expected angle
        expected_angle = np.degrees(np.arcsin((sound_speed * tau) / distance))
        expected_angle = abs(expected_angle)
        
        self.assertAlmostEqual(angle, expected_angle, places=1)
    
    def test_tdoa_to_angle_edge_cases(self):
        """Test TDOA to angle conversion edge cases."""
        # Test with zero distance (should handle gracefully)
        angle = AudioProcessor.tdoa_to_angle(0.001, 0.0, 343.0)
        self.assertEqual(angle, 0.0)
        
        # Test with very large delay (should be clipped)
        angle = AudioProcessor.tdoa_to_angle(1.0, 0.035, 343.0)
        self.assertLessEqual(angle, 90.0)
    
    def test_process_audio_block(self):
        """Test audio block processing."""
        # Create test audio data (4 channels, 1024 samples)
        audio_data = np.random.randn(1024, 4).astype(np.float32)
        
        # Define microphone pairs
        mic_pairs = [
            (0, 1, 0.035),
            (1, 2, 0.035),
            (2, 3, 0.035),
        ]
        
        azimuth = self.processor.process_audio_block(audio_data, mic_pairs)
        
        # Should return a valid angle
        self.assertIsInstance(azimuth, float)
        self.assertGreaterEqual(azimuth, 0.0)
        self.assertLessEqual(azimuth, 180.0)
    
    def test_audio_activity_detection(self):
        """Test audio activity detection."""
        # Test with silent audio
        silent_audio = np.zeros((1024, 4), dtype=np.float32)
        self.processor.process_audio_block(silent_audio, [])
        self.assertFalse(self.processor.is_active())
        
        # Test with loud audio
        loud_audio = np.ones((1024, 4), dtype=np.float32) * 0.01
        self.processor.process_audio_block(loud_audio, [])
        self.assertTrue(self.processor.is_active())
    
    def test_getters(self):
        """Test getter methods."""
        # Test initial values
        self.assertEqual(self.processor.get_current_angle(), 0.0)
        self.assertEqual(self.processor.get_audio_level(), 0.0)
        self.assertFalse(self.processor.is_active())


class TestAudioConfig(unittest.TestCase):
    """Test cases for AudioConfig class."""
    
    def test_default_values(self):
        """Test default configuration values."""
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


if __name__ == '__main__':
    unittest.main()
