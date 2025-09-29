#!/bin/bash
# Activation script for Vision virtual environment

echo "🎤🎥 Activating Vision virtual environment..."
source Vision/bin/activate

echo "✅ Virtual environment activated!"
echo ""
echo "Available commands:"
echo "  python test_webcam.py              - Test camera and face detection"
echo "  python audio_visual_speaker_detection.py - Run full audio-visual system"
echo "  python doa_demo.py                 - Run original DOA demo"
echo ""
echo "To deactivate, type: deactivate"
echo ""
