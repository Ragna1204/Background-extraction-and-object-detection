"""
Motion detection application using modular classes
"""

import cv2
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from background_model import create_background_model
from motion_detector import ContourMotionDetector
from config.config import (
    NUM_BACKGROUND_FRAMES, VIDEO_SOURCE, BACKGROUND_METHOD
)
from logger import logger


def run_motion_detection():
    """Run complete motion detection pipeline"""
    # Create and build background model
    bg_model = create_background_model(
        method=BACKGROUND_METHOD,
        video_source=VIDEO_SOURCE,
        num_frames=NUM_BACKGROUND_FRAMES
    )
    background = bg_model.get_background()

    # Display background before starting detection
    cv2.imshow('Estimated Background (Before Detection)', background)
    logger.info("Showing estimated background. Press any key to continue to motion detection...")
    cv2.waitKey(0)
    cv2.destroyWindow('Estimated Background (Before Detection)')

    # Create motion detector and start detection
    detector = ContourMotionDetector(background, VIDEO_SOURCE)
    detector.start_detection()


if __name__ == "__main__":
    run_motion_detection()
