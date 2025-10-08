"""
Main Speaker Detection System
============================

Combines audio direction detection with face detection to identify
active speakers in real-time.
"""

import cv2
import numpy as np
import logging
import time
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .config_manager import ConfigManager
from .audio_processor import AudioProcessor
from .face_detector import FaceDetector, FaceDetection

logger = logging.getLogger(__name__)


@dataclass
class SpeakerDetection:
    """Represents an active speaker detection result."""
    face: FaceDetection
    audio_angle: float
    confidence: float
    is_speaking: bool
    timestamp: float


class AudioVisualSpeakerDetector:
    """Main audio-visual speaker detection system."""
    
    def __init__(self, config_file: str = "config/default.yaml", show_face_mesh: bool = False):
        """
        Initialize the audio-visual speaker detection system.
        
        Args:
            config_file: Path to configuration file
            show_face_mesh: Whether to show MediaPipe face mesh with landmarks
        """
        # Load configuration
        self.config_manager = ConfigManager(config_file)
        self.show_face_mesh = show_face_mesh
        
        if not self.config_manager.validate_config():
            raise ValueError("Invalid configuration")
        
        # Initialize components
        self.audio_processor = AudioProcessor(
            self.config_manager.audio_config,
            self.config_manager.performance_config,
            self.config_manager.get_mic_pairs()
        )
        
        self.face_detector = FaceDetector(
            self.config_manager.video_config,
            self.config_manager.detection_config
        )
        
        # Video processing state
        self.cap = None
        self.current_faces: List[FaceDetection] = []
        self.last_active_speaker: Optional[SpeakerDetection] = None
        
        # Performance tracking
        self.frame_times = []
        self.processing_times = []
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        logger.info("AudioVisualSpeakerDetector initialized")
    
    def map_audio_angle_to_screen_position(self, audio_angle: float) -> int:
        """
        Map audio direction angle to screen position.
        
        Args:
            audio_angle: Audio direction angle in degrees
            
        Returns:
            Screen x-coordinate corresponding to the audio direction
        """
        try:
            screen_center_x = self.config_manager.video_config.frame_width // 2
            angle_range = 90.0
            
            # Normalize angle to [-1, 1] range
            normalized_angle = (audio_angle - 90.0) / angle_range
            screen_x = screen_center_x + (normalized_angle * screen_center_x)
            
            # Clamp to screen bounds
            screen_x = max(0, min(self.config_manager.video_config.frame_width - 1, int(screen_x)))
            
            return screen_x
            
        except Exception as e:
            logger.error(f"Error mapping audio angle to screen position: {e}")
            return self.config_manager.video_config.frame_width // 2
    
    def find_active_speaker(self, faces: List[FaceDetection], audio_angle: float) -> Optional[SpeakerDetection]:
        """
        Find the active speaker by correlating audio direction with face positions.
        
        Args:
            faces: List of detected faces
            audio_angle: Current audio direction angle
            
        Returns:
            SpeakerDetection object if active speaker found, None otherwise
        """
        try:
            if not faces:
                return None
            
            audio_screen_x = self.map_audio_angle_to_screen_position(audio_angle)
            
            best_face = None
            min_distance = float('inf')
            
            # Find the face closest to the audio direction
            for face in faces:
                distance = abs(face.center_x - audio_screen_x)
                if distance < min_distance:
                    min_distance = distance
                    best_face = face
            
            if best_face:
                max_distance = self.config_manager.video_config.frame_width // 2
                confidence = max(0.0, 1.0 - (min_distance / max_distance))
                
                # Only consider someone a speaker if:
                # 1. Audio is active (not silent)
                # 2. Confidence is above threshold
                # 3. Face is close enough to audio direction
                is_speaking = (confidence > self.config_manager.detection_config.speaker_confidence_threshold and 
                             self.audio_processor.is_active())
                
                # Only return a SpeakerDetection if there's actually audio activity
                if self.audio_processor.is_active():
                    return SpeakerDetection(
                        face=best_face,
                        audio_angle=audio_angle,
                        confidence=confidence,
                        is_speaking=is_speaking,
                        timestamp=time.time()
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error finding active speaker: {e}")
            return None
    
    def draw_status_info(self, frame: np.ndarray, active_speaker: Optional[SpeakerDetection]) -> np.ndarray:
        """
        Draw minimal system status information on the frame.
        
        Args:
            frame: Video frame to draw on
            active_speaker: Current active speaker detection
            
        Returns:
            Frame with status information drawn
        """
        try:
            h, w = frame.shape[:2]
            
            # Direction indicator (top left)
            angle = self.audio_processor.get_current_angle()
            # Ensure angle is a valid number
            if np.isnan(angle) or np.isinf(angle):
                angle = 0.0
            cv2.putText(frame, f"Direction: {angle:.0f} deg", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # FPS counter (top right)
            fps_color = (0, 255, 0) if self.current_fps >= 20 else (0, 255, 255) if self.current_fps >= 10 else (0, 0, 255)
            cv2.putText(frame, f"FPS: {self.current_fps:.1f}", 
                       (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
            
            # Processing time (top left, below direction)
            if self.processing_times:
                avg_time = np.mean(self.processing_times[-10:])  # Last 10 frames
                cv2.putText(frame, f"Proc: {avg_time*1000:.1f}ms", 
                           (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Audio activity (top left, below processing time)
            audio_status = "VAD: ACTIVE" if self.audio_processor.is_active() else "VAD: SILENT"
            color = (0, 0, 0)  # Black text for better visibility
            cv2.putText(frame, f"{audio_status}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error drawing status info: {e}")
            return frame
    
    def update_fps(self) -> None:
        """Update FPS counter."""
        self.fps_counter += 1
        current_time = time.time()
        time_diff = current_time - self.fps_start_time
        
        # Update FPS every second
        if time_diff >= 1.0:
            self.current_fps = self.fps_counter / time_diff
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture.
        
        Returns:
            True if camera initialized successfully, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.config_manager.video_config.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config_manager.video_config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config_manager.video_config.frame_height)
            
            if not self.cap.isOpened():
                logger.error("Could not open camera")
                return False
            
            logger.info("Camera initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[FaceDetection], Optional[SpeakerDetection]]:
        """
        Process a single video frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (detected_faces, active_speaker)
        """
        start_time = time.time()
        
        try:
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            self.current_faces = faces
            
            # Get current audio direction
            audio_angle = self.audio_processor.get_current_angle()
            
            # Find active speaker
            active_speaker = self.find_active_speaker(faces, audio_angle)
            
            # Update last active speaker
            if active_speaker:
                self.last_active_speaker = active_speaker
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Keep only recent processing times
            if len(self.processing_times) > 100:
                self.processing_times = self.processing_times[-50:]
            
            return faces, active_speaker
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return [], None
    
    def run(self) -> None:
        """Main execution loop for the audio-visual speaker detection system."""
        logger.info("🎤🎥 Audio-Visual Speaker Detection System")
        logger.info("=" * 50)
        logger.info("Hardware: ReSpeaker 4 Mic Array (UAC1.0) + Webcam")
        logger.info("Algorithm: WebRTC VAD + GCC-PHAT DOA + Tracking + MediaPipe Face Detection")
        
        # Initialize camera
        if not self.initialize_camera():
            return
        
        # Auto-detect ReSpeaker device
        respeaker_device = self.audio_processor.find_respeaker_device()
        if respeaker_device is not None:
            self.config_manager.audio_config.device_index = respeaker_device
            logger.info(f"✅ Automatically selected ReSpeaker device: {respeaker_device}")
        else:
            logger.warning(f"⚠️  ReSpeaker not found, using configured device: {self.config_manager.audio_config.device_index}")
        
        # Query audio devices for reference
        self.audio_processor.query_devices()
        logger.info(f"Using samplerate: {self.config_manager.audio_config.sample_rate}, "
                   f"channels: {self.config_manager.audio_config.n_channels}, "
                   f"device index: {self.config_manager.audio_config.device_index}")
        
        # Start audio stream
        try:
            with self.audio_processor:
                if not self.audio_processor.start_stream():
                    logger.error("Failed to start audio stream")
                    return
                
                logger.info("✅ Audio stream started successfully!")
                logger.info("🎮 Controls: Press 'q' to quit")
                logger.info("🎨 Visual indicators:")
                logger.info("  🟢 GREEN: Speaking + Mouth Open")
                logger.info("  🔵 BLUE:  Speaking + Mouth Closed")
                logger.info("  🟡 YELLOW: Mouth Open (not speaking)")
                logger.info("  🔴 RED:   Mouth Closed (not speaking)")
                logger.info("🚀 System ready! Start speaking to test...")
                
                # Main processing loop
                while True:
                    try:
                        ret, frame = self.cap.read()
                        if not ret:
                            logger.error("Could not read frame from camera")
                            break
                        
                        # Process frame
                        faces, active_speaker = self.process_frame(frame)
                        
                        # Update FPS counter
                        self.update_fps()
                        
                        # Draw detections based on visualization mode
                        if self.show_face_mesh:
                            # Show MediaPipe face mesh with full contours and tessellation
                            frame = self.face_detector.draw_face_mesh(frame, show_full_mesh=True)
                        else:
                            # Show only mouth landmarks (no full mesh)
                            frame = self.face_detector.draw_face_mesh(frame, show_full_mesh=False)
                        
                        # Always draw face rectangles with speaker status
                        frame = self.face_detector.draw_faces(frame, faces, 
                                                            active_speaker.face if active_speaker else None)
                        
                        # Draw status information
                        frame = self.draw_status_info(frame, active_speaker)
                        
                        # Display frame
                        cv2.imshow('Audio-Visual Speaker Detection', frame)
                        
                        # Handle key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                            
                    except Exception as e:
                        logger.error(f"Error in main loop: {e}")
                        break
                
        except KeyboardInterrupt:
            logger.info("👋 Stopped by user")
        except Exception as e:
            logger.error(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up system resources."""
        try:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            self.face_detector.close()
            self.audio_processor.stop_stream()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_performance_stats(self) -> dict:
        """Get system performance statistics."""
        return {
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'max_processing_time': max(self.processing_times) if self.processing_times else 0,
            'frames_processed': len(self.processing_times),
            'face_detection_stats': self.face_detector.get_detection_stats()
        }
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()
