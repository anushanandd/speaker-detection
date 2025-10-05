"""
Audio Processing Module
======================

Centralized audio processing utilities including GCC-PHAT algorithm,
TDOA calculation, and real-time audio stream management.
"""

import numpy as np
import sounddevice as sd
import threading
import logging
from collections import deque
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass

from .config_manager import AudioConfig, PerformanceConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    """Represents an audio frame with metadata."""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    channels: int


class AudioProcessor:
    """Centralized audio processing utilities."""
    
    def __init__(self, audio_config: AudioConfig, performance_config: PerformanceConfig):
        """
        Initialize audio processor.
        
        Args:
            audio_config: Audio configuration
            performance_config: Performance configuration
        """
        self.config = audio_config
        self.perf_config = performance_config
        
        # Audio processing state
        self.audio_buffer = deque(maxlen=performance_config.audio_buffer_size)
        self.current_angle = 0.0
        self.audio_level = 0.0
        self.is_audio_active = False
        self.audio_lock = threading.Lock()
        
        # Stream management
        self.stream = None
        self.is_streaming = False
        self.callback = None
        
        logger.info("AudioProcessor initialized")
    
    @staticmethod
    def gcc_phat(sig: np.ndarray, ref: np.ndarray, fs: int, 
                 max_tau: Optional[float] = None, interp: int = 8) -> Tuple[float, np.ndarray]:
        """
        Generalized Cross Correlation with Phase Transform (GCC-PHAT).
        
        Returns the time delay of arrival (TDOA) between two signals.
        
        Args:
            sig: Reference signal
            ref: Signal to compare against reference
            fs: Sample rate
            max_tau: Maximum expected delay in seconds
            interp: Interpolation factor for higher resolution
            
        Returns:
            tau: Time delay in seconds
            cc: Cross-correlation function
        """
        try:
            n = sig.shape[0] + ref.shape[0]
            nfft = 1 << (n - 1).bit_length()
            
            # Compute FFTs
            SIG = np.fft.rfft(sig, n=nfft)
            REF = np.fft.rfft(ref, n=nfft)
            
            # Cross-power spectrum
            R = SIG * np.conj(REF)
            
            # PHAT weighting
            denom = np.abs(R)
            denom[denom == 0] = 1e-15
            R /= denom
            
            # Inverse FFT with interpolation
            cc = np.fft.irfft(R, n=nfft * interp)
            max_shift = int(interp * nfft / 2)
            cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))
            
            # Find peak
            if max_tau is not None:
                max_shift_lim = min(int(interp * fs * max_tau), max_shift)
                mid = max_shift
                cc = cc[mid - max_shift_lim: mid + max_shift_lim + 1]
                shift = np.argmax(np.abs(cc)) - max_shift_lim
            else:
                shift = np.argmax(np.abs(cc)) - max_shift
            
            tau = shift / float(interp * fs)
            return tau, cc
            
        except Exception as e:
            logger.error(f"Error in GCC-PHAT calculation: {e}")
            return 0.0, np.array([])
    
    @staticmethod
    def tdoa_to_angle(tau: float, distance: float, sound_speed: float = 343.0) -> float:
        """
        Convert Time Difference of Arrival (TDOA) to azimuth angle.
        
        Args:
            tau: Time delay in seconds
            distance: Distance between microphones in meters
            sound_speed: Speed of sound in m/s
            
        Returns:
            angle: Azimuth angle in degrees [0, 180]
        """
        try:
            val = np.clip((sound_speed * tau) / max(distance, 1e-6), -1.0, 1.0)
            theta = np.degrees(np.arcsin(val))
            return abs(theta)
        except Exception as e:
            logger.error(f"Error in TDOA to angle conversion: {e}")
            return 0.0
    
    def process_audio_block(self, audio_data: np.ndarray, mic_pairs: List[Tuple[int, int, float]]) -> float:
        """
        Process audio block and return azimuth angle.
        
        Args:
            audio_data: Audio data from microphone array
            mic_pairs: List of microphone pairs for TDOA calculation
            
        Returns:
            azimuth: Azimuth angle in degrees
        """
        try:
            angles = []
            
            for (i, j, distance) in mic_pairs:
                if i < audio_data.shape[1] and j < audio_data.shape[1]:
                    tau, _ = self.gcc_phat(
                        audio_data[:, i],
                        audio_data[:, j],
                        self.config.sample_rate,
                        max_tau=distance / self.config.sound_speed,
                        interp=8
                    )
                    angle = self.tdoa_to_angle(tau, distance, self.config.sound_speed)
                    angles.append(angle)
            
            if not angles:
                return 0.0
            
            angles = np.array(angles)
            
            # Robust averaging (trim extremes)
            if angles.size >= 4:
                angles = np.sort(angles)[1:-1]  # Drop min/max
            
            azimuth = np.mean(angles)
            
            # Check for audio activity
            self.audio_level = np.mean(np.abs(audio_data))
            self.is_audio_active = self.audio_level > self.config.activity_threshold
            
            return azimuth
            
        except Exception as e:
            logger.error(f"Error processing audio block: {e}")
            return 0.0
    
    def audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """
        Audio stream callback for real-time processing.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Audio status
        """
        try:
            if status:
                logger.warning(f"Audio status: {status}")
            
            # Extract microphone data and convert to float32
            mic_data = indata[:, self.config.mic_indexes].astype(np.float32)
            self.audio_buffer.append(mic_data)
            
            # Concatenate buffered audio for processing
            audio_buffer = np.concatenate(list(self.audio_buffer), axis=0)
            
            # Process audio in chunks
            while audio_buffer.shape[0] >= self.config.frame_samples:
                frame = audio_buffer[:self.config.frame_samples, :]
                audio_buffer = audio_buffer[self.config.hop_samples:, :]
                
                # Get microphone pairs from config (will be passed from main detector)
                # For now, use default pairs
                mic_pairs = [
                    (0, 3, 3 * self.config.mic_spacing),
                    (0, 2, 2 * self.config.mic_spacing),
                    (1, 3, 2 * self.config.mic_spacing),
                    (1, 2, 1 * self.config.mic_spacing),
                    (0, 1, 1 * self.config.mic_spacing),
                    (2, 3, 1 * self.config.mic_spacing),
                ]
                
                azimuth = self.process_audio_block(frame, mic_pairs)
                
                with self.audio_lock:
                    self.current_angle = azimuth
                
                # Call external callback if provided
                if self.callback:
                    try:
                        self.callback(azimuth, self.audio_level, self.is_audio_active)
                    except Exception as e:
                        logger.error(f"Error in audio callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
    
    def start_stream(self, callback: Optional[Callable] = None) -> bool:
        """
        Start audio stream.
        
        Args:
            callback: Optional callback function for audio events
            
        Returns:
            True if stream started successfully, False otherwise
        """
        try:
            if self.is_streaming:
                logger.warning("Audio stream already running")
                return True
            
            self.callback = callback
            
            self.stream = sd.InputStream(
                device=self.config.device_index,
                channels=self.config.n_channels,
                samplerate=self.config.sample_rate,
                dtype='float32',
                blocksize=self.config.hop_samples,
                callback=self.audio_callback
            )
            
            self.stream.start()
            self.is_streaming = True
            logger.info("Audio stream started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            return False
    
    def stop_stream(self) -> None:
        """Stop audio stream."""
        try:
            if self.stream and self.is_streaming:
                self.stream.stop()
                self.stream.close()
                self.is_streaming = False
                logger.info("Audio stream stopped")
        except Exception as e:
            logger.error(f"Error stopping audio stream: {e}")
    
    def get_current_angle(self) -> float:
        """Get current audio direction angle."""
        with self.audio_lock:
            return self.current_angle
    
    def get_audio_level(self) -> float:
        """Get current audio level."""
        return self.audio_level
    
    def is_active(self) -> bool:
        """Check if audio is currently active."""
        return self.is_audio_active
    
    def find_respeaker_device(self) -> Optional[int]:
        """
        Automatically find ReSpeaker device by name.
        
        Returns:
            Device index if found, None otherwise
        """
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                device_name = device['name'].lower()
                if ('respeaker' in device_name and 
                    device['max_input_channels'] >= 6):
                    logger.info(f"Found ReSpeaker device: {i}: {device['name']} (inputs: {device['max_input_channels']})")
                    return i
            
            logger.warning("ReSpeaker device not found. Available input devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.warning(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
            return None
            
        except Exception as e:
            logger.error(f"Error finding ReSpeaker device: {e}")
            return None
    
    def query_devices(self) -> None:
        """Query and log available audio devices."""
        try:
            devices = sd.query_devices()
            logger.info("Available audio devices:")
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    logger.info(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
        except Exception as e:
            logger.error(f"Error querying audio devices: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_stream()
    
    def __del__(self):
        """Destructor to ensure stream is stopped."""
        self.stop_stream()
