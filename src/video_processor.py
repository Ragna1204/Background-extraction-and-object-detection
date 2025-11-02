"""
Video processing utilities for input/output operations
"""

import cv2
import os
from logger import logger
from config.config import FRAME_WIDTH, FRAME_HEIGHT, FPS, OUTPUT_VIDEO_PATH


class VideoProcessor:
    """Class for handling video input and output operations"""

    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.is_video_file = isinstance(source, str) and os.path.isfile(source)
        self.fps = FPS
        self.frame_size = (FRAME_WIDTH, FRAME_HEIGHT)

    def open_video_source(self):
        """Open video source (camera or file)"""
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.source}")

        # Get video properties
        if self.is_video_file:
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_size = (width, height)
            logger.info(f"Opened video file: {self.source} ({width}x{height} @ {self.fps} FPS)")
        else:
            logger.info(f"Opened camera source: {self.source}")

        return self.cap

    def read_frame(self):
        """Read a single frame from video source"""
        if self.cap is None:
            self.open_video_source()

        ret, frame = self.cap.read()
        if not ret:
            return None

        return frame

    def get_frame_count(self):
        """Get total frame count (for video files)"""
        if self.cap and self.is_video_file:
            return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return -1

    def get_current_frame_number(self):
        """Get current frame number"""
        if self.cap:
            return int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return -1

    def set_frame_position(self, frame_number):
        """Set video position to specific frame"""
        if self.cap and self.is_video_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            return True
        return False

    def close(self):
        """Close video source"""
        if self.cap:
            self.cap.release()
            self.cap = None


class VideoWriter:
    """Class for writing processed video output"""

    def __init__(self, output_path=OUTPUT_VIDEO_PATH, fps=FPS, frame_size=(FRAME_WIDTH, FRAME_HEIGHT)):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.is_recording = False

    def start_recording(self):
        """Start video recording"""
        if self.writer is not None:
            logger.warning("Video recording already started")
            return

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)

        if not self.writer.isOpened():
            raise ValueError(f"Could not create video writer: {self.output_path}")

        self.is_recording = True
        logger.info(f"Started video recording to: {self.output_path}")

    def write_frame(self, frame):
        """Write a frame to video file"""
        if not self.is_recording or self.writer is None:
            logger.warning("Video recording not started")
            return

        # Ensure frame is the correct size
        if frame.shape[:2] != self.frame_size[::-1]:
            frame = cv2.resize(frame, self.frame_size)

        self.writer.write(frame)

    def stop_recording(self):
        """Stop video recording"""
        if self.writer:
            self.writer.release()
            self.writer = None

        self.is_recording = False
        logger.info(f"Stopped video recording: {self.output_path}")

    def __del__(self):
        """Cleanup on destruction"""
        self.stop_recording()


class VideoCaptureSession:
    """Context manager for video capture sessions"""

    def __init__(self, source=0):
        self.processor = VideoProcessor(source)

    def __enter__(self):
        self.processor.open_video_source()
        return self.processor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.processor.close()


def list_available_cameras(max_cameras=10):
    """List available camera devices"""
    available_cameras = []

    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                available_cameras.append(i)
            cap.release()

    return available_cameras


def get_video_info(video_path):
    """Get information about a video file"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        return {
            'fps': fps,
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration': duration,
            'path': video_path
        }
    finally:
        cap.release()
