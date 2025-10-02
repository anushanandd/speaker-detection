#!/usr/bin/env python3
"""
Simple Face Detector with MediaPipe FaceMesh
============================================

A focused face detector that shows facial landmarks/points using MediaPipe FaceMesh.
This will help us see the actual 468 facial landmarks and implement proper mouth detection.
"""

import cv2
import mediapipe as mp
import numpy as np

def main():
    """Simple face detector with landmark visualization."""
    print("🎭 Simple Face Detector with MediaPipe FaceMesh")
    print("=" * 50)
    print("This will show facial landmarks/points in real-time")
    print("Press 'q' to quit")
    print("")
    
    # Initialize MediaPipe FaceMesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Try different initialization approaches
    print("🔧 Initializing MediaPipe FaceMesh...")
    
    # Initialize FaceMesh (should work with MediaPipe 0.10.5)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("✅ FaceMesh initialized successfully")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    print("✅ Camera initialized successfully")
    print("📹 Starting face detection...")
    print("")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame from camera")
                break
            
            frame_count += 1
            
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with FaceMesh
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_CONTOURS,
                        None,
                        mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Draw face mesh tesselation
                    mp_drawing.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        None,
                        mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Count landmarks
                    landmark_count = len(face_landmarks.landmark)
                    
                    # Draw landmark count
                    cv2.putText(frame, f"Landmarks: {landmark_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Highlight mouth opening landmarks (correct indices for mouth opening detection)
                    # Upper lip: 13, 14 (top of upper lip)
                    # Lower lip: 17, 18 (bottom of lower lip)
                    mouth_landmarks = [13, 14, 17, 18]
                    
                    h, w, _ = frame.shape
                    for idx in mouth_landmarks:
                        if idx < len(face_landmarks.landmark):
                            landmark = face_landmarks.landmark[idx]
                            x = int(landmark.x * w)
                            y = int(landmark.y * h)
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)  # Red dots for mouth
                    
                    # Draw mouth landmark count
                    cv2.putText(frame, f"Mouth Points: {len(mouth_landmarks)}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Calculate mouth openness using correct landmarks
                    try:
                        # Get upper lip landmarks (13, 14) and lower lip landmarks (17, 18)
                        upper_lip_y = np.mean([face_landmarks.landmark[i].y for i in [13, 14]])
                        lower_lip_y = np.mean([face_landmarks.landmark[i].y for i in [17, 18]])
                        
                        # Calculate vertical distance between upper and lower lip
                        mouth_openness = abs(lower_lip_y - upper_lip_y)
                        
                        # Convert to pixel distance for threshold
                        h, w, _ = frame.shape
                        mouth_openness_pixels = mouth_openness * h
                        
                        # Set threshold (adjust based on testing)
                        mouth_open_threshold_pixels = 15
                        is_mouth_open = mouth_openness_pixels > mouth_open_threshold_pixels
                        mouth_status = "OPEN" if is_mouth_open else "CLOSED"
                        
                        cv2.putText(frame, f"Mouth: {mouth_status} ({mouth_openness_pixels:.1f}px)", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                                   (0, 255, 0) if is_mouth_open else (255, 0, 0), 2)
                    except Exception as e:
                        cv2.putText(frame, f"Mouth: Error calculating", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            else:
                cv2.putText(frame, "No faces detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw frame info
            cv2.putText(frame, f"Frame: {frame_count}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show frame
            cv2.imshow('Simple Face Detector - MediaPipe Landmarks', frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_mesh.close()
        print("🧹 Cleanup completed")

if __name__ == "__main__":
    main()
