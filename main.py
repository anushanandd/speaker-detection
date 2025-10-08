#!/usr/bin/env python3
"""
Main Entry Point for Audio-Visual Speaker Detection System
=========================================================

Command-line interface for the audio-visual speaker detection system.
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.speaker_detector import AudioVisualSpeakerDetector
from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Audio-Visual Speaker Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run with default config
  python main.py --config config/custom.yaml       # Use custom config
  python main.py --device 1 --camera 1             # Use specific devices
  python main.py --verbose                          # Enable debug logging
  python main.py --test                             # Test system components
  python main.py --show-face-mesh                   # Show MediaPipe face mesh
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config/default.yaml',
        help='Path to configuration file (default: config/default.yaml)'
    )
    
    # Device selection
    parser.add_argument(
        '--device', '-d',
        type=int,
        help='Audio device index (overrides config)'
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        help='Camera device index (overrides config)'
    )
    
    # Logging options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress console output (ERROR level only)'
    )
    
    # System options
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test system components and exit'
    )
    
    parser.add_argument(
        '--list-devices',
        action='store_true',
        help='List available audio and video devices'
    )
    
    parser.add_argument(
        '--validate-config',
        action='store_true',
        help='Validate configuration file and exit'
    )
    
    # Demo modes
    parser.add_argument(
        '--audio-only',
        action='store_true',
        help='Run audio-only DOA demo (no video)'
    )
    
    parser.add_argument(
        '--face-detection-only',
        action='store_true',
        help='Run face detection demo only (no audio)'
    )
    
    # Performance options
    parser.add_argument(
        '--skip-frames',
        type=int,
        help='Skip frames for performance (overrides config)'
    )
    
    parser.add_argument(
        '--buffer-size',
        type=int,
        help='Audio buffer size (overrides config)'
    )
    
    # Visualization options
    parser.add_argument(
        '--show-face-mesh',
        action='store_true',
        help='Show MediaPipe face mesh with landmarks (default: simple rectangles)'
    )
    
    return parser


def setup_logging(verbose: bool, quiet: bool) -> None:
    """Setup logging configuration."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    if not verbose:
        logging.getLogger('mediapipe').setLevel(logging.WARNING)
        logging.getLogger('sounddevice').setLevel(logging.WARNING)


def list_devices() -> None:
    """List available audio and video devices."""
    print("🎤 Audio Devices:")
    print("-" * 40)
    
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
    except ImportError:
        print("  ❌ sounddevice not available")
    except Exception as e:
        print(f"  ❌ Error querying audio devices: {e}")
    
    print("\n📹 Video Devices:")
    print("-" * 40)
    
    try:
        import cv2
        for i in range(5):  # Check first 5 devices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"  {i}: Camera (resolution: {frame.shape[1]}x{frame.shape[0]})")
                cap.release()
    except ImportError:
        print("  ❌ opencv-python not available")
    except Exception as e:
        print(f"  ❌ Error querying video devices: {e}")


def run_audio_only_demo(config_manager) -> None:
    """Run audio-only DOA demo."""
    print("🎤 Audio-Only DOA Demo")
    print("=" * 30)
    print("This will show audio direction detection without video")
    print("Press Ctrl+C to stop")
    print("")
    
    try:
        from src.audio_processor import AudioProcessor
        
        audio_processor = AudioProcessor(
            config_manager.audio_config,
            config_manager.performance_config,
            config_manager.get_mic_pairs()
        )
        
        # Auto-detect ReSpeaker device
        respeaker_device = audio_processor.find_respeaker_device()
        if respeaker_device is not None:
            config_manager.audio_config.device_index = respeaker_device
            print(f"✅ Using ReSpeaker device: {respeaker_device}")
        else:
            print(f"⚠️  Using configured device: {config_manager.audio_config.device_index}")
        
        # Start audio stream
        with audio_processor:
            if not audio_processor.start_stream():
                print("❌ Failed to start audio stream")
                return
            
            print("✅ Audio stream started!")
            print("🚀 Start speaking to see direction detection...")
            
            try:
                while True:
                    angle = audio_processor.get_current_angle()
                    level = audio_processor.get_audio_level()
                    active = audio_processor.is_active()
                    
                    status = "ACTIVE" if active else "SILENT"
                    print(f"Direction: {angle:6.1f}° | Level: {level:.3f} | Status: {status}", end="\r")
                    
                    import time
                    time.sleep(0.1)
                    
            except KeyboardInterrupt:
                print("\n👋 Stopped by user")
                
    except Exception as e:
        print(f"❌ Error in audio demo: {e}")


def run_face_detection_demo(config_manager) -> None:
    """Run face detection demo only."""
    print("👥 Face Detection Demo")
    print("=" * 30)
    print("This will show face detection with landmarks")
    print("Press 'q' to quit")
    print("")
    
    try:
        import cv2
        from src.face_detector import FaceDetector
        
        face_detector = FaceDetector(
            config_manager.video_config,
            config_manager.detection_config
        )
        
        # Initialize camera
        cap = cv2.VideoCapture(config_manager.video_config.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config_manager.video_config.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config_manager.video_config.frame_height)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("✅ Camera initialized successfully")
        print("📹 Starting face detection...")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error: Could not read frame from camera")
                    break
                
                # Draw face mesh with landmarks
                frame = face_detector.draw_face_mesh(frame)
                
                # Show frame
                cv2.imshow('Face Detection Demo', frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n👋 Stopped by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            face_detector.close()
            print("🧹 Cleanup completed")
            
    except Exception as e:
        print(f"❌ Error in face detection demo: {e}")


def test_system_components() -> bool:
    """Test system components and return success status."""
    print("🧪 Testing System Components")
    print("=" * 40)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Configuration loading
    total_tests += 1
    try:
        config = ConfigManager()
        if config.validate_config():
            print("✅ Configuration loading and validation")
            tests_passed += 1
        else:
            print("❌ Configuration validation failed")
    except Exception as e:
        print(f"❌ Configuration loading error: {e}")
    
    # Test 2: Audio device access
    total_tests += 1
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        if len(devices) > 0:
            print("✅ Audio device access")
            tests_passed += 1
        else:
            print("❌ No audio devices found")
    except Exception as e:
        print(f"❌ Audio device test failed: {e}")
    
    # Test 3: Camera access
    total_tests += 1
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera access")
                tests_passed += 1
            else:
                print("❌ Camera opened but couldn't read frame")
            cap.release()
        else:
            print("❌ Could not open camera")
    except Exception as e:
        print(f"❌ Camera test failed: {e}")
    
    # Test 4: MediaPipe face detection
    total_tests += 1
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()
        print("✅ MediaPipe face detection")
        tests_passed += 1
        face_detection.close()
    except Exception as e:
        print(f"❌ MediaPipe test failed: {e}")
    
    # Test 5: NumPy operations
    total_tests += 1
    try:
        import numpy as np
        test_array = np.random.rand(100, 4)
        result = np.mean(test_array)
        if result > 0:
            print("✅ NumPy operations")
            tests_passed += 1
        else:
            print("❌ NumPy operations failed")
    except Exception as e:
        print(f"❌ NumPy test failed: {e}")
    
    print(f"\n📊 Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! System is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False


def apply_cli_overrides(args, config_manager: ConfigManager) -> None:
    """Apply command-line argument overrides to configuration."""
    if args.device is not None:
        config_manager.audio_config.device_index = args.device
        logger.info(f"Override: audio device index = {args.device}")
    
    if args.camera is not None:
        config_manager.video_config.camera_index = args.camera
        logger.info(f"Override: camera index = {args.camera}")
    
    if args.skip_frames is not None:
        config_manager.performance_config.face_detection_skip_frames = args.skip_frames
        logger.info(f"Override: skip frames = {args.skip_frames}")
    
    if args.buffer_size is not None:
        config_manager.performance_config.audio_buffer_size = args.buffer_size
        logger.info(f"Override: buffer size = {args.buffer_size}")


def main() -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.quiet)
    
    logger.info("🎤🎥 Audio-Visual Speaker Detection System v2.0")
    logger.info("=" * 60)
    
    # Handle special commands
    if args.list_devices:
        list_devices()
        return 0
    
    if args.test:
        success = test_system_components()
        return 0 if success else 1
    
    if args.validate_config:
        try:
            config = ConfigManager(args.config)
            if config.validate_config():
                print("✅ Configuration is valid")
                return 0
            else:
                print("❌ Configuration validation failed")
                return 1
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            return 1
    
    # Load configuration for demo modes and main application
    config_manager = ConfigManager(args.config)
    
    # Apply CLI overrides
    apply_cli_overrides(args, config_manager)
    
    # Handle demo modes
    if args.audio_only:
        run_audio_only_demo(config_manager)
        return 0
    
    if args.face_detection_only:
        run_face_detection_demo(config_manager)
        return 0
    
    # Main application
    try:
        
        # Validate final configuration
        if not config_manager.validate_config():
            logger.error("Configuration validation failed")
            return 1
        
        # Create and run detector
        with AudioVisualSpeakerDetector(args.config, show_face_mesh=args.show_face_mesh) as detector:
            detector.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("👋 Application stopped by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
