#!/usr/bin/env python3
"""
Simple webcam test script to verify camera and MediaPipe face detection work
"""

import cv2
import mediapipe as mp

def test_webcam():
    """Test webcam and face detection"""
    print("Testing webcam and face detection...")
    
    # Initialize MediaPipe
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # 0 for close-range, 1 for full-range
        min_detection_confidence=0.5
    )
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully. Press 'q' to quit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            # Draw face detections
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)
            
            # Show frame
            cv2.imshow('Webcam Test - Face Detection', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detection.close()
        print("Test completed")

if __name__ == "__main__":
    test_webcam()
