"""
Unit tests for motion detector classes
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from src.motion_detector import MotionDetector, ContourMotionDetector


class TestMotionDetector:
    """Test MotionDetector class"""

    def test_initialization(self):
        """Test MotionDetector initialization"""
        background = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detector = MotionDetector(background, video_source=0)

        assert detector.video_source == 0
        assert detector.cap is None
        assert detector.is_running is False
        assert detector.frame_count == 0
        assert detector.fps == 0
        np.testing.assert_array_equal(detector.background, background.astype(np.int16))

    def test_process_frame(self):
        """Test frame processing for motion detection"""
        background = np.ones((50, 50, 3), dtype=np.uint8) * 100
        detector = MotionDetector(background)

        # Create a frame with some motion (different from background)
        frame = np.ones((50, 50, 3), dtype=np.uint8) * 150

        motion_frame, mask = detector.process_frame(frame)

        # Check that processing returns valid results
        assert isinstance(motion_frame, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert motion_frame.shape == frame.shape
        assert mask.shape[:2] == frame.shape[:2]
        assert mask.dtype == np.uint8

    def test_detect_contours(self):
        """Test contour detection"""
        background = np.zeros((50, 50, 3), dtype=np.uint8)
        detector = MotionDetector(background)

        # Create a mask with a rectangle
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        contours = detector.detect_contours(mask)

        # Should detect at least one contour
        assert len(contours) >= 1

    def test_draw_bounding_boxes(self):
        """Test bounding box drawing"""
        background = np.zeros((50, 50, 3), dtype=np.uint8)
        detector = MotionDetector(background)

        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        contours = [np.array([[10, 10], [20, 10], [20, 20], [10, 20]])]

        result = detector.draw_bounding_boxes(frame, contours)

        # Check that result is different from input (bounding box drawn)
        assert not np.array_equal(result, frame)
        assert result.shape == frame.shape


class TestContourMotionDetector:
    """Test ContourMotionDetector class"""

    def test_initialization(self):
        """Test ContourMotionDetector initialization"""
        background = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        detector = ContourMotionDetector(background, video_source=0)

        assert isinstance(detector, MotionDetector)
        assert detector.video_source == 0

    def test_display_results_with_contours(self):
        """Test display results with contour detection"""
        background = np.zeros((50, 50, 3), dtype=np.uint8)
        detector = ContourMotionDetector(background)

        # Mock cv2.imshow to avoid GUI
        with patch('cv2.imshow'), patch('cv2.putText'):
            original_frame = np.zeros((50, 50, 3), dtype=np.uint8)
            motion_frame = np.ones((50, 50, 3), dtype=np.uint8) * 255
            mask = np.zeros((50, 50), dtype=np.uint8)
            mask[10:20, 10:20] = 255  # Add a contour

            # This should not raise an exception
            detector.display_results(original_frame, motion_frame, mask)
