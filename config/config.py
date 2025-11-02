"""
Configuration settings for Background Extraction and Object Detection
"""

# Video settings
VIDEO_SOURCE = 0  # 0 for default camera, or path to video file
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

# Background modeling settings
NUM_BACKGROUND_FRAMES = 200
BACKGROUND_METHOD = 'median'  # 'median', 'mean', or 'gmm'

# Motion detection settings
MOTION_THRESHOLD = 30
BLUR_SIZE = 5
MIN_CONTOUR_AREA = 500

# Output settings
SHOW_PREVIEW = True
SAVE_VIDEO = False
OUTPUT_VIDEO_PATH = 'output.avi'

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FILE = 'motion_detection.log'
