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
import webrtcvad

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
    
    def __init__(self, audio_config: AudioConfig, performance_config: PerformanceConfig, mic_pairs: List[Tuple[int, int, float]] = None):
        """
        Initialize audio processor.
        
        Args:
            audio_config: Audio configuration
            performance_config: Performance configuration
            mic_pairs: Microphone pairs for TDOA calculation
        """
        self.config = audio_config
        self.perf_config = performance_config
        self.mic_pairs = mic_pairs or []
        
        # Audio processing state
        self.audio_buffer = deque(maxlen=performance_config.audio_buffer_size)
        self.current_angle = 0.0
        self.audio_level = 0.0
        self.is_audio_active = False
        self.audio_lock = threading.Lock()
        
        # VAD (Voice Activity Detection)
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 0-3 (2 is balanced)
        
        # DOA tracking and smoothing
        self.angle_history = deque(maxlen=10)  # Keep last 10 angle estimates
        self.vad_history = deque(maxlen=5)     # Keep last 5 VAD decisions
        self.smoothed_angle = 0.0
        self.confidence_threshold = 0.7        # Minimum confidence for stable DOA
        
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
    
    def detect_voice_activity(self, audio_frame: np.ndarray) -> bool:
        """
        Detect voice activity using WebRTC VAD.
        
        Args:
            audio_frame: Audio frame (mono, 16kHz, 16-bit)
            
        Returns:
            True if voice activity detected, False otherwise
        """
        try:
            # Convert to 16-bit PCM for WebRTC VAD
            if audio_frame.dtype != np.int16:
                audio_frame = (audio_frame * 32767).astype(np.int16)
            
            # WebRTC VAD expects 10ms, 20ms, or 30ms frames
            # We'll use 20ms frames (320 samples at 16kHz)
            frame_size = 320
            if len(audio_frame) >= frame_size:
                # Take the first channel if multi-channel
                if audio_frame.ndim > 1:
                    mono_frame = audio_frame[:, 0]
                else:
                    mono_frame = audio_frame
                
                # Ensure frame is exactly the right size
                if len(mono_frame) > frame_size:
                    mono_frame = mono_frame[:frame_size]
                elif len(mono_frame) < frame_size:
                    # Pad with zeros if too short
                    mono_frame = np.pad(mono_frame, (0, frame_size - len(mono_frame)))
                
                # Convert to bytes for WebRTC VAD
                audio_bytes = mono_frame.tobytes()
                
                # Detect voice activity
                is_speech = self.vad.is_speech(audio_bytes, self.config.sample_rate)
                return is_speech
            
            return False
            
        except Exception as e:
            logger.warning(f"Error in VAD detection: {e}")
            return False
    
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
    
    def smooth_angle_estimate(self, angle: float, vad_decision: bool) -> float:
        """
        Apply smoothing and tracking to angle estimates.
        
        Args:
            angle: Raw angle estimate
            vad_decision: VAD decision for this frame
            
        Returns:
            Smoothed angle estimate
        """
        try:
            # Add to history
            self.angle_history.append(angle)
            self.vad_history.append(vad_decision)
            
            # Only smooth if we have enough history and recent VAD activity
            if len(self.angle_history) >= 3:
                # Check if we have recent voice activity
                recent_vad_activity = sum(self.vad_history) >= 2  # At least 2 out of 5 recent frames
                
                if recent_vad_activity:
                    # Apply median filter for robustness against outliers
                    angles = list(self.angle_history)
                    angles.sort()
                    median_angle = angles[len(angles) // 2]
                    
                    # Apply circular mean for better angle averaging
                    # Convert to complex numbers for circular mean
                    complex_angles = [np.exp(1j * np.radians(ang)) for ang in angles]
                    mean_complex = np.mean(complex_angles)
                    circular_mean_angle = np.degrees(np.angle(mean_complex))
                    
                    # Weighted combination of median and circular mean
                    self.smoothed_angle = 0.7 * circular_mean_angle + 0.3 * median_angle
                else:
                    # No recent voice activity, keep previous estimate
                    pass  # self.smoothed_angle remains unchanged
            
            return self.smoothed_angle
            
        except Exception as e:
            logger.warning(f"Error in angle smoothing: {e}")
            return angle
    
    def process_audio_block(self, audio_data: np.ndarray, mic_pairs: List[Tuple[int, int, float]]) -> float:
        """
        Process audio block with VAD + DOA + tracking pipeline.
        
        Args:
            audio_data: Audio data from microphone array
            mic_pairs: List of microphone pairs for TDOA calculation
            
        Returns:
            azimuth: Smoothed azimuth angle in degrees
        """
        try:
            # Step 1: Voice Activity Detection (VAD)
            # Use first microphone for VAD
            vad_audio = audio_data[:, 0] if audio_data.shape[1] > 0 else audio_data.flatten()
            vad_decision = self.detect_voice_activity(vad_audio)
            
            # Step 2: Direction of Arrival (DOA) - only if VAD is positive
            raw_angle = 0.0
            if vad_decision:
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
                
                if angles:
                    angles = np.array(angles)
                    
                    # Robust averaging (trim extremes)
                    if angles.size >= 4:
                        angles = np.sort(angles)[1:-1]  # Drop min/max
                    
                    raw_angle = np.mean(angles)
            
            # Step 3: Smoothing and Tracking
            smoothed_angle = self.smooth_angle_estimate(raw_angle, vad_decision)
            
            # Update audio level and activity status
            self.audio_level = np.mean(np.abs(audio_data))
            
            # Audio is active if VAD detects speech
            self.is_audio_active = vad_decision
            
            # Update current angle with smoothed estimate
            with self.audio_lock:
                self.current_angle = smoothed_angle
            
            return smoothed_angle
            
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
                
                azimuth = self.process_audio_block(frame, self.mic_pairs)
                
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
    
    def get_vad_status(self) -> dict:
        """Get VAD and tracking status for debugging."""
        return {
            'vad_history': list(self.vad_history),
            'angle_history': list(self.angle_history),
            'smoothed_angle': self.smoothed_angle,
            'recent_vad_activity': sum(self.vad_history) >= 2 if len(self.vad_history) >= 2 else False
        }
    
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
