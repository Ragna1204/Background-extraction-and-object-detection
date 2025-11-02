"""
Unit tests for background model classes
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from src.background_model import (
    BackgroundModel, MedianBackgroundModel, MeanBackgroundModel,
    GaussianMixtureBackgroundModel, create_background_model
)


class TestBackgroundModel:
    """Test base BackgroundModel class"""

    def test_initialization(self):
        """Test BackgroundModel initialization"""
        model = BackgroundModel(video_source=0, num_frames=10)
        assert model.video_source == 0
        assert model.num_frames == 10
        assert model.background is None
        assert model.frames == []

    @patch('cv2.VideoCapture')
    def test_capture_frames_success(self, mock_cv2):
        """Test successful frame capture"""
        # Mock video capture
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [(True, np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)) for _ in range(5)] + [(False, None)]
        mock_cv2.return_value = mock_cap

        model = BackgroundModel(video_source=0, num_frames=5)
        frames = model.capture_frames()

        assert len(frames) == 5
        assert all(isinstance(frame, np.ndarray) for frame in frames)
        mock_cap.release.assert_called_once()

    @patch('cv2.VideoCapture')
    def test_capture_frames_no_frames(self, mock_cv2):
        """Test frame capture with no frames available"""
        mock_cap = Mock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cv2.return_value = mock_cap

        model = BackgroundModel(video_source=0, num_frames=5)

        with pytest.raises(ValueError, match="No frames captured"):
            model.capture_frames()


class TestMedianBackgroundModel:
    """Test MedianBackgroundModel class"""

    def test_compute_background(self):
        """Test median background computation"""
        model = MedianBackgroundModel()

        # Create test frames
        frames = [
            np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=np.uint8),
            np.array([[[2, 3, 4], [5, 6, 7]], [[8, 9, 10], [11, 12, 13]]], dtype=np.uint8),
            np.array([[[3, 4, 5], [6, 7, 8]], [[9, 10, 11], [12, 13, 14]]], dtype=np.uint8)
        ]

        background = model.compute_background(frames)

        # Check that background is median of frames
        expected = np.median(np.array(frames), axis=0).astype(np.uint8)
        np.testing.assert_array_equal(background, expected)


class TestMeanBackgroundModel:
    """Test MeanBackgroundModel class"""

    def test_compute_background(self):
        """Test mean background computation"""
        model = MeanBackgroundModel()

        # Create test frames
        frames = [
            np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.uint8),
            np.array([[[2, 3, 4]], [[5, 6, 7]]], dtype=np.uint8),
            np.array([[[3, 4, 5]], [[6, 7, 8]]], dtype=np.uint8)
        ]

        background = model.compute_background(frames)

        # Check that background is mean of frames
        expected = np.mean(np.array(frames), axis=0).astype(np.uint8)
        np.testing.assert_array_equal(background, expected)


class TestCreateBackgroundModel:
    """Test background model factory function"""

    def test_create_median_model(self):
        """Test creating median background model"""
        model = create_background_model('median')
        assert isinstance(model, MedianBackgroundModel)

    def test_create_mean_model(self):
        """Test creating mean background model"""
        model = create_background_model('mean')
        assert isinstance(model, MeanBackgroundModel)

    def test_create_gmm_model(self):
        """Test creating GMM background model"""
        model = create_background_model('gmm')
        assert isinstance(model, GaussianMixtureBackgroundModel)

    def test_create_invalid_model(self):
        """Test creating invalid background model"""
        with pytest.raises(ValueError, match="Unknown background modeling method"):
            create_background_model('invalid')

    def test_create_model_with_params(self):
        """Test creating model with custom parameters"""
        model = create_background_model('median', video_source=1, num_frames=20)
        assert model.video_source == 1
        assert model.num_frames == 20
