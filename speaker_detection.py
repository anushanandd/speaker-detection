"""
Audio-Visual Speaker Detection System
====================================

A real-time system that combines ReSpeaker microphone array audio direction detection
with webcam face detection to identify the active speaker.

Features:
- Direction of Arrival (DOA) detection using GCC-PHAT algorithm
- Real-time face detection using MediaPipe
- Audio-visual correlation for speaker identification
- Clean, simple visual feedback

Hardware Requirements:
- ReSpeaker 4 Mic Array (UAC1.0)
- Webcam
"""

import numpy as np
import cv2
import sounddevice as sd
import threading
from collections import deque
import mediapipe as mp
from dataclasses import dataclass

# ========= CONFIGURATION =========
# Audio Configuration
DEVICE_INDEX = 0       # ReSpeaker 4 Mic Array (UAC1.0)
SAMPLE_RATE = 16000
N_CHANNELS = 6         # 6 channels as reported by macOS
MIC_INDEXES = [2, 3, 4, 5]   # The 4 real microphones on ReSpeaker UAC1.0
FRAME_SAMPLES = 1024   # Analysis window (64 ms @16k)
HOP_SAMPLES = 512      # 50% overlap

# Microphone Array Geometry
MIC_SPACING = 0.035    # Meters between adjacent mics (~3.5 cm)
SOUND_SPEED = 343.0    # Speed of sound in m/s

# Microphone pairs for robust TDOA averaging
MIC_PAIRS = [
    (0, 3, 3*MIC_SPACING),  # Outermost pair
    (0, 2, 2*MIC_SPACING),
    (1, 3, 2*MIC_SPACING),
    (1, 2, 1*MIC_SPACING),
    (0, 1, 1*MIC_SPACING),
    (2, 3, 1*MIC_SPACING),
]

# Video Configuration
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Detection Parameters
FACE_CONFIDENCE_THRESHOLD = 0.7
AUDIO_ACTIVITY_THRESHOLD = 0.003
SPEAKER_CONFIDENCE_THRESHOLD = 0.3

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

@dataclass
class SpeakerDetection:
    """Represents an active speaker detection result."""
    face: FaceDetection
    audio_angle: float
    confidence: float
    is_speaking: bool


class AudioVisualSpeakerDetector:
    def __init__(self):
        """Initialize the audio-visual speaker detection system."""
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for close-range, 1 for full-range
            min_detection_confidence=FACE_CONFIDENCE_THRESHOLD
        )
        
        # Audio processing state
        self.audio_buffer = deque(maxlen=4)
        self.current_audio_angle = 0.0
        self.audio_lock = threading.Lock()
        self.is_audio_active = False
        self.audio_level = 0.0
        
        # Video processing state
        self.cap = None
        self.current_faces = []

    def gcc_phat(self, sig, ref, fs, max_tau=None, interp=8):
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
        n = sig.shape[0] + ref.shape[0]
        nfft = 1 << (n - 1).bit_length()
        SIG = np.fft.rfft(sig, n=nfft)
        REF = np.fft.rfft(ref, n=nfft)
        R = SIG * np.conj(REF)
        denom = np.abs(R)
        denom[denom == 0] = 1e-15
        R /= denom

        cc = np.fft.irfft(R, n=nfft*interp)
        max_shift = int(interp * nfft / 2)
        cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

        if max_tau is not None:
            max_shift_lim = min(int(interp * fs * max_tau), max_shift)
            mid = max_shift
            cc = cc[mid - max_shift_lim: mid + max_shift_lim + 1]
            shift = np.argmax(np.abs(cc)) - max_shift_lim
        else:
            shift = np.argmax(np.abs(cc)) - max_shift
        tau = shift / float(interp * fs)
        return tau, cc

    def tdoa_to_angle(self, tau, distance):
        """
        Convert Time Difference of Arrival (TDOA) to azimuth angle.
        
        Args:
            tau: Time delay in seconds
            distance: Distance between microphones in meters
            
        Returns:
            angle: Azimuth angle in degrees [0, 180]
        """
        val = np.clip((SOUND_SPEED * tau) / max(distance, 1e-6), -1.0, 1.0)
        theta = np.degrees(np.arcsin(val))
        return abs(theta)

    def process_audio_block(self, audio_data):
        """
        Process audio block and return azimuth angle.
        
        Args:
            audio_data: Audio data from microphone array
            
        Returns:
            azimuth: Azimuth angle in degrees
        """
        angles = []
        for (i, j, distance) in MIC_PAIRS:
            tau, _ = self.gcc_phat(
                audio_data[:, i], 
                audio_data[:, j], 
                SAMPLE_RATE, 
                max_tau=distance/SOUND_SPEED, 
                interp=8
            )
            angles.append(self.tdoa_to_angle(tau, distance))
        
        angles = np.array(angles)
        if angles.size >= 4:
            angles = np.sort(angles)[1:-1]  # Drop min/max for robustness
        azimuth = np.mean(angles) if angles.size else 0.0
        
        # Check for audio activity
        self.audio_level = np.mean(np.abs(audio_data))
        self.is_audio_active = self.audio_level > AUDIO_ACTIVITY_THRESHOLD
        
        return azimuth

    def audio_callback(self, indata, frames, time_info, status):
        """
        Audio stream callback for real-time processing.
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Audio status
        """
        if status:
            print("Audio status:", status)
        
        # Extract microphone data and convert to float32
        mic_data = indata[:, MIC_INDEXES].astype(np.float32)
        self.audio_buffer.append(mic_data)
        
        # Concatenate buffered audio for processing
        audio_buffer = np.concatenate(list(self.audio_buffer), axis=0)
        
        # Process audio in chunks
        while audio_buffer.shape[0] >= FRAME_SAMPLES:
            frame = audio_buffer[:FRAME_SAMPLES, :]
            audio_buffer = audio_buffer[HOP_SAMPLES:, :]
            azimuth = self.process_audio_block(frame)
            
            with self.audio_lock:
                self.current_audio_angle = azimuth

    def detect_faces(self, frame):
        """
        Detect faces in the video frame using MediaPipe.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of detected faces with bounding boxes and confidence scores
        """
        try:
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
                        
                        # Only keep high-confidence detections
                        if confidence > FACE_CONFIDENCE_THRESHOLD:
                            face = FaceDetection(
                                x=x, y=y, width=width, height=height,
                                confidence=confidence,
                                center_x=x + width // 2,
                                center_y=y + height // 2
                            )
                            faces.append(face)
                            
                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue
            
            # Apply non-maximum suppression to remove overlapping faces
            return self.non_max_suppression(faces)
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []

    def non_max_suppression(self, faces, overlap_threshold=0.3):
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

    def calculate_overlap(self, face1, face2):
        """
        Calculate overlap ratio between two face bounding boxes.
        
        Args:
            face1: First face detection
            face2: Second face detection
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
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

    def map_audio_angle_to_screen_position(self, audio_angle):
        """
        Map audio direction angle to screen position.
        
        Args:
            audio_angle: Audio direction angle in degrees
            
        Returns:
            Screen x-coordinate corresponding to the audio direction
        """
        screen_center_x = FRAME_WIDTH // 2
        angle_range = 90.0
        
        normalized_angle = (audio_angle - 90.0) / angle_range
        screen_x = screen_center_x + (normalized_angle * screen_center_x)
        screen_x = max(0, min(FRAME_WIDTH - 1, int(screen_x)))
        
        return screen_x

    def find_active_speaker(self, faces, audio_angle):
        """
        Find the active speaker by correlating audio direction with face positions.
        
        Args:
            faces: List of detected faces
            audio_angle: Current audio direction angle
            
        Returns:
            SpeakerDetection object if active speaker found, None otherwise
        """
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
            max_distance = FRAME_WIDTH // 2
            confidence = max(0.0, 1.0 - (min_distance / max_distance))
            
            # Determine if the person is speaking
            is_speaking = confidence > SPEAKER_CONFIDENCE_THRESHOLD and self.is_audio_active
            
            return SpeakerDetection(
                face=best_face,
                audio_angle=audio_angle,
                confidence=confidence,
                is_speaking=is_speaking
            )
        
        return None

    def draw_detections(self, frame, faces, active_speaker):
        """
        Draw face detections and status information on the frame.
        
        Args:
            frame: Video frame to draw on
            faces: List of detected faces
            active_speaker: Active speaker detection result
        """
        # Draw all detected faces
        for face in faces:
            # Determine color and thickness based on state
            if active_speaker and face == active_speaker.face:
                if self.is_audio_active:
                    color = (0, 255, 0)  # Green: speaking (audio detected)
                    status = "SPEAKING"
                else:
                    color = (255, 0, 0)  # Blue: face detected but no audio
                    status = "DETECTED"
                thickness = 3
            else:
                color = (128, 128, 128)  # Gray: face detected
                status = "FACE"
                thickness = 2
            
            # Draw face rectangle
            cv2.rectangle(frame, (face.x, face.y), 
                         (face.x + face.width, face.y + face.height), 
                         color, thickness)
            
            # Draw status text
            cv2.putText(frame, status, (face.x, face.y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw system status information
        cv2.putText(frame, f"Audio Direction: {self.current_audio_angle:.1f}°", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Audio Level: {self.audio_level:.3f}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Audio Active: {'YES' if self.is_audio_active else 'NO'}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if active_speaker:
            cv2.putText(frame, f"Speaker Confidence: {active_speaker.confidence:.2f}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def run(self):
        """
        Main execution loop for the audio-visual speaker detection system.
        """
        print("🎤🎥 Audio-Visual Speaker Detection System")
        print("=" * 50)
        print("Hardware: ReSpeaker 4 Mic Array (UAC1.0) + Webcam")
        print("Algorithm: GCC-PHAT DOA + MediaPipe Face Detection")
        print("")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("✅ Camera initialized successfully")
        print("Audio devices:")
        print(sd.query_devices())
        print(f"Using samplerate: {SAMPLE_RATE}, channels: {N_CHANNELS}, device index: {DEVICE_INDEX}")
        
        # Start audio stream
        try:
            with sd.InputStream(device=DEVICE_INDEX,
                              channels=N_CHANNELS,
                              samplerate=SAMPLE_RATE,
                              dtype='float32',
                              blocksize=HOP_SAMPLES,
                              callback=self.audio_callback):
                
                print("✅ Audio stream started successfully!")
                print("")
                print("🎮 Controls:")
                print("  Press 'q' to quit")
                print("")
                print("🎨 Visual indicators:")
                print("  🟢 GREEN: Speaking (audio detected)")
                print("  🔵 BLUE:  Face detected (no audio)")
                print("  ⚪ GRAY:  Face detected")
                print("")
                print("📊 Status display:")
                print("  Audio Direction: Current direction angle")
                print("  Audio Level: Current audio input level")
                print("  Audio Active: YES/NO for speech detection")
                print("  Speaker Confidence: Correlation confidence score")
                print("")
                print("🚀 System ready! Start speaking to test...")
                
                # Main processing loop
                while True:
                    try:
                        ret, frame = self.cap.read()
                        if not ret:
                            print("❌ Error: Could not read frame from camera")
                            break
                        
                        # Detect faces in current frame
                        faces = self.detect_faces(frame)
                        self.current_faces = faces
                        
                        # Get current audio direction
                        with self.audio_lock:
                            audio_angle = self.current_audio_angle
                        
                        # Find active speaker by correlating audio and visual
                        active_speaker = self.find_active_speaker(faces, audio_angle)
                        
                        # Draw detections and status
                        self.draw_detections(frame, faces, active_speaker)
                        
                        # Display frame
                        cv2.imshow('Audio-Visual Speaker Detection', frame)
                        
                        # Handle key presses
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    except Exception as e:
                        print(f"❌ Error in main loop: {e}")
                        break
                
        except KeyboardInterrupt:
            print("\n👋 Stopped by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up system resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.face_detection.close()
        print("🧹 Cleanup completed")


def main():
    """Main entry point for the application."""
    detector = AudioVisualSpeakerDetector()
    detector.run()


if __name__ == "__main__":
    main()
