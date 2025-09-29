# Audio-Visual Speaker Detection System v2.0

A real-time system that combines ReSpeaker microphone array audio direction detection with webcam face detection to identify the active speaker.

## 🚀 Features

- **🎤 Direction of Arrival (DOA) Detection**: Uses GCC-PHAT algorithm for accurate audio source localization
- **👥 Real-time Face Detection**: MediaPipe-based face detection with non-maximum suppression
- **🔗 Audio-Visual Correlation**: Correlates audio direction with face positions to identify active speakers
- **🎨 Clean Visual Feedback**: Simple, intuitive color-coded interface
- **🛡️ Robust Processing**: Handles multiple faces and overlapping detections
- **⚙️ Centralized Configuration**: YAML-based configuration management
- **📊 Comprehensive Logging**: Detailed logging with configurable levels
- **🧪 Full Test Suite**: Unit tests for all major components
- **🔧 CLI Interface**: Command-line interface with multiple options
- **📈 Performance Monitoring**: Built-in performance tracking and optimization

## 🖥️ Hardware Requirements

- **ReSpeaker 6 Mic Array (UAC1.0)**: 6-microphone linear array for audio direction detection
- **Webcam**: Standard USB webcam for face detection
- **macOS**: Compatible with Apple Silicon (ARM64) and Intel Macs
- **Python 3.8+**: Required for all features

## 📦 Installation

### Quick Start

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd AudioVisual-Detection
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv Vision
   source Vision/bin/activate  # On macOS/Linux
   # or
   Vision\Scripts\activate     # On Windows
   ```

3. **Install dependencies**
   ```bash
   # For development (includes testing tools)
   pip install -r requirements.txt
   
   # For production (minimal dependencies)
   pip install -r requirements-prod.txt
   ```

4. **Run the system**
   ```bash
   python main.py
   ```

### Advanced Installation

```bash
# Test system components
python main.py --test

# List available devices
python main.py --list-devices

# Use custom configuration
python main.py --config config/custom.yaml

# Enable verbose logging
python main.py --verbose
```

## 🎮 Usage

### Basic Usage

```bash
# Run with default settings
python main.py

# Run with specific audio device
python main.py --device 1

# Run with specific camera
python main.py --camera 1

# Run with custom configuration
python main.py --config config/custom.yaml
```

### Command Line Options

```bash
python main.py --help
```

**Available options:**
- `--config, -c`: Path to configuration file
- `--device, -d`: Audio device index
- `--camera`: Camera device index
- `--verbose, -v`: Enable debug logging
- `--quiet, -q`: Suppress console output
- `--test`: Test system components
- `--list-devices`: List available devices
- `--validate-config`: Validate configuration file

### Controls
- **Press 'q'**: Quit the application

### Visual Indicators
- **🟢 GREEN**: Speaking (audio detected + face correlation)
- **🔵 BLUE**: Face detected (no audio activity)
- **⚪ GRAY**: Face detected (background)

### Status Display
- **Audio Direction**: Current direction angle in degrees
- **Audio Level**: Current audio input level
- **Audio Active**: YES/NO for speech detection
- **Speaker Confidence**: Correlation confidence score
- **Avg Processing**: Average processing time per frame

## ⚙️ Configuration

The system uses YAML configuration files for easy customization. The default configuration is in `config/default.yaml`:

```yaml
# Audio Configuration
audio:
  device_index: 0
  sample_rate: 16000
  mic_spacing: 0.035
  sound_speed: 343.0

# Video Configuration
video:
  camera_index: 0
  frame_width: 640
  frame_height: 480
  face_confidence_threshold: 0.7

# Detection Parameters
detection:
  speaker_confidence_threshold: 0.3
  max_faces: 10
```

### Environment Variables

You can override configuration using environment variables:

```bash
export AUDIO_DEVICE_INDEX=1
export CAMERA_INDEX=1
export LOG_LEVEL=DEBUG
python main.py
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src

# Run specific test file
python -m pytest tests/test_audio_processor.py

# Run tests with verbose output
python -m pytest tests/ -v
```

### Manual Testing

```bash
# Test system components
python main.py --test

# Test webcam only
python test_webcam.py

# Test audio only
python doa_demo.py
```

### Key Components

1. **ConfigManager**: Centralized configuration management with validation
2. **AudioProcessor**: GCC-PHAT algorithm and audio stream management
3. **FaceDetector**: MediaPipe face detection with NMS
4. **AudioVisualSpeakerDetector**: Main system orchestrator

## 🔬 How It Works

### 1. Audio Processing
- Captures audio from ReSpeaker's 4 microphones
- Uses GCC-PHAT algorithm to calculate time differences between microphone pairs
- Converts time differences to azimuth angles (0-180°)
- Implements robust averaging across multiple microphone pairs

### 2. Face Detection
- Uses MediaPipe for real-time face detection
- Applies non-maximum suppression to remove overlapping detections
- Tracks face positions and confidence scores
- Optimizes performance with configurable frame skipping

### 3. Speaker Identification
- Maps audio direction to screen coordinates
- Finds the face closest to the audio direction
- Correlates audio activity with face position to determine active speaker
- Provides confidence scoring for detection quality

## 📊 Performance

### Optimization Features
- **Frame Skipping**: Skip face detection frames when no audio activity
- **Buffer Management**: Efficient audio buffering with configurable size
- **Processing Time Monitoring**: Track and display processing performance
- **Resource Management**: Proper cleanup of audio/video resources

### Performance Metrics
- **Processing Time**: Typically < 10ms per frame
- **Memory Usage**: Optimized for real-time operation
- **CPU Usage**: Efficient algorithms with minimal overhead

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 🐛 Troubleshooting

### Common Issues

**Audio device not found:**
```bash
python main.py --list-devices
# Update device_index in config/default.yaml
```

**Camera not working:**
```bash
python test_webcam.py
# Check camera permissions and device index
```

**Poor detection accuracy:**
- Adjust `face_confidence_threshold` in configuration
- Calibrate microphone spacing if needed
- Check audio levels and background noise

**Performance issues:**
- Enable frame skipping: `face_detection_skip_frames: 2`
- Reduce frame resolution
- Increase audio buffer size

### Debug Mode

```bash
python main.py --verbose
```

This enables detailed logging to help diagnose issues.

## 📈 Technical Details

### Audio Configuration
- **Sample Rate**: 16 kHz
- **Channels**: 6 (using 4 microphones from ReSpeaker)
- **Microphone Spacing**: 3.5 cm between adjacent mics
- **Sound Speed**: 343 m/s

### Detection Parameters
- **Face Confidence Threshold**: 0.7
- **Audio Activity Threshold**: 0.003
- **Speaker Confidence Threshold**: 0.3

### Algorithms
- **GCC-PHAT**: Generalized Cross Correlation with Phase Transform
- **TDOA**: Time Difference of Arrival calculation
- **NMS**: Non-Maximum Suppression for face detection