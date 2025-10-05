#!/usr/bin/env python3
"""
Setup script for Audio-Visual Speaker Detection
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def test_audio_devices():
    """Test audio device detection"""
    print("\nTesting audio devices...")
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        print("Available audio devices:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"  {i}: {device['name']} (inputs: {device['max_input_channels']})")
        return True
    except ImportError:
        print("❌ sounddevice not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing audio devices: {e}")
        return False

def test_camera():
    """Test camera access"""
    print("\nTesting camera...")
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera working - captured frame of size:", frame.shape)
                cap.release()
                return True
            else:
                print("❌ Camera opened but couldn't read frame")
                cap.release()
                return False
        else:
            print("❌ Could not open camera")
            return False
    except ImportError:
        print("❌ opencv-python not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing camera: {e}")
        return False

def test_mediapipe():
    """Test MediaPipe face detection"""
    print("\nTesting MediaPipe...")
    try:
        import mediapipe as mp
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection()
        print("✅ MediaPipe face detection initialized successfully")
        face_detection.close()
        return True
    except ImportError:
        print("❌ mediapipe not installed")
        return False
    except Exception as e:
        print(f"❌ Error testing MediaPipe: {e}")
        return False

def main():
    """Main setup function"""
    print("🎤🎥 Audio-Visual Speaker Detection Setup")
    print("=" * 50)
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Warning: Not in a virtual environment. Consider using one.")
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed at package installation")
        return
    
    # Test components
    audio_ok = test_audio_devices()
    camera_ok = test_camera()
    mediapipe_ok = test_mediapipe()
    
    print("\n" + "=" * 50)
    print("Setup Summary:")
    print(f"  Audio devices: {'✅' if audio_ok else '❌'}")
    print(f"  Camera: {'✅' if camera_ok else '❌'}")
    print(f"  MediaPipe: {'✅' if mediapipe_ok else '❌'}")
    
    if audio_ok and camera_ok and mediapipe_ok:
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update device_index in config/default.yaml with your ReSpeaker device index")
        print("2. Run: python test_webcam.py (to test camera and face detection)")
        print("3. Run: python main.py (to start the full system)")
        print("4. Run: python main.py --audio-only (for audio-only demo)")
        print("5. Run: python main.py --face-detection-only (for face detection demo)")
    else:
        print("\n❌ Setup incomplete. Please fix the issues above before proceeding.")

if __name__ == "__main__":
    main()
