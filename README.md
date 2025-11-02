# Background Extraction and Object Detection

A Python-based computer vision application for real-time background subtraction and motion detection using OpenCV.

## Features

- Real-time background modeling using median filtering
- Motion detection with background subtraction
- Configurable parameters for different environments
- Support for camera input and video files
- Modular architecture for easy extension

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ragna1204/Background-extraction-and-object-detection.git
cd Background-extraction-and-object-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface (Recommended)

The application provides a comprehensive CLI with multiple commands:

```bash
# Capture and display background model
python src/cli.py background

# Run motion detection with live camera
python src/cli.py detect

# Process a video file
python src/cli.py detect --video path/to/video.mp4

# Get information about a video file
python src/cli.py info --video path/to/video.mp4

# List available cameras
python src/cli.py cameras

# Show help
python src/cli.py --help
```

### Direct Script Execution

#### Background Capture
```bash
python src/main.py
```

#### Motion Detection
```bash
python src/motion_detection.py
```

## Project Structure

```
├── src/                    # Source code
│   ├── main.py            # Background capture script
│   └── motion_detection.py # Motion detection script
├── config/                # Configuration files
│   └── config.py          # Application settings
├── tests/                 # Unit tests
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Configuration

Edit `config/config.py` to adjust:
- Video source and resolution
- Background modeling parameters
- Motion detection thresholds
- Output settings

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.
