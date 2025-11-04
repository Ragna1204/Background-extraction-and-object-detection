"""
Background capture and display using modular classes
"""

import cv2
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from background_model import create_background_model
from config.config import NUM_BACKGROUND_FRAMES, VIDEO_SOURCE, BACKGROUND_METHOD


def show_background(background):
    """Display the computed background image"""
    cv2.imshow('Estimated Background', background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Create background model using specified method
    bg_model = create_background_model(
        method=BACKGROUND_METHOD,
        video_source=VIDEO_SOURCE,
        num_frames=NUM_BACKGROUND_FRAMES
    )

    # Build and get background
    background = bg_model.get_background()

    # Display background
    show_background(background)
