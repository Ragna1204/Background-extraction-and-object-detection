"""
Background modeling classes for motion detection
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from logger import logger
from config.config import FRAME_WIDTH, FRAME_HEIGHT


class BackgroundModel(ABC):
    """Abstract base class for background modeling algorithms"""

    def __init__(self, video_source=0, num_frames=100):
        self.video_source = video_source
        self.num_frames = num_frames
        self.background = None
        self.frames = []

    @abstractmethod
    def compute_background(self, frames):
        """Compute background from frames"""
        pass

    def capture_frames(self):
        """Capture frames from video source"""
        logger.info(f"Capturing {self.num_frames} frames to build background model...")

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")

        self.frames = []
        frame_count = 0

        try:
            while frame_count < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Could not read frame from video source")
                    break

                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                self.frames.append(frame)
                frame_count += 1

                cv2.imshow('Capturing Background Frames', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Background capture interrupted by user")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        if len(self.frames) == 0:
            raise ValueError("No frames captured for background modeling")

        logger.info(f"Captured {len(self.frames)} frames for background modeling")
        return self.frames

    def build_background(self):
        """Build background model from captured frames"""
        if not self.frames:
            self.capture_frames()

        logger.info("Computing background model...")
        self.background = self.compute_background(self.frames)
        logger.info("Background model computed successfully")

        return self.background

    def get_background(self):
        """Get the computed background"""
        if self.background is None:
            self.build_background()
        return self.background


class MedianBackgroundModel(BackgroundModel):
    """Background model using median filtering"""

    def compute_background(self, frames):
        frames_np = np.array(frames)
        return np.median(frames_np, axis=0).astype(np.uint8)


class MeanBackgroundModel(BackgroundModel):
    """Background model using mean filtering"""

    def compute_background(self, frames):
        frames_np = np.array(frames)
        return np.mean(frames_np, axis=0).astype(np.uint8)


class GaussianMixtureBackgroundModel(BackgroundModel):
    """Background model using Gaussian Mixture Model (placeholder)"""

    def __init__(self, video_source=0, num_frames=100, history=500):
        super().__init__(video_source, num_frames)
        self.history = history
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=self.history, varThreshold=16, detectShadows=True)

    def compute_background(self, frames):
        # For GMM, we need to apply the model to frames
        # This is a simplified implementation
        background_frames = []
        for frame in frames:
            fgmask = self.fgbg.apply(frame)
            background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(fgmask))
            background_frames.append(background)

        if background_frames:
            return np.median(np.array(background_frames), axis=0).astype(np.uint8)
        else:
            return np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)


def create_background_model(method='median', **kwargs):
    """Factory function to create background model instances"""
    models = {
        'median': MedianBackgroundModel,
        'mean': MeanBackgroundModel,
        'gmm': GaussianMixtureBackgroundModel
    }

    if method not in models:
        raise ValueError(f"Unknown background modeling method: {method}")

    return models[method](**kwargs)
