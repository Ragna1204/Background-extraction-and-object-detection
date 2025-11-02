"""
Integration tests for end-to-end functionality
"""

import pytest
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock
from src.background_model import create_background_model
from src.motion_detector import ContourMotionDetector
from src.alert_system import alert_system, MotionEvent


class TestIntegration:
    """Integration tests for the complete motion detection pipeline"""

    def test_background_model_creation_and_usage(self):
        """Test creating and using a background model"""
        # Create mock frames
        frames = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(5)]

        # Create background model
        model = create_background_model('median', num_frames=5)

        # Mock the frame capture
        model.frames = frames
        background = model.compute_background(frames)

        # Verify background is computed
        assert isinstance(background, np.ndarray)
        assert background.shape == (100, 100, 3)
        assert background.dtype == np.uint8

    def test_motion_detector_initialization(self):
        """Test motion detector initialization and basic functionality"""
        # Create test background
        background = np.ones((50, 50, 3), dtype=np.uint8) * 100

        # Create motion detector
        detector = ContourMotionDetector(background, video_source=0)

        # Verify initialization
        assert detector.video_source == 0
        assert not detector.is_running
        assert detector.frame_count == 0
        np.testing.assert_array_equal(detector.background, background.astype(np.int16))

    def test_frame_processing_pipeline(self):
        """Test the complete frame processing pipeline"""
        # Create test background and frame
        background = np.ones((50, 50, 3), dtype=np.uint8) * 100
        frame = np.ones((50, 50, 3), dtype=np.uint8) * 150  # Different from background

        detector = ContourMotionDetector(background, video_source=0)

        # Process frame
        motion_frame, mask = detector.process_frame(frame)

        # Verify results
        assert isinstance(motion_frame, np.ndarray)
        assert isinstance(mask, np.ndarray)
        assert motion_frame.shape == frame.shape
        assert mask.shape[:2] == frame.shape[:2]

    def test_contour_detection(self):
        """Test contour detection functionality"""
        background = np.zeros((50, 50, 3), dtype=np.uint8)
        detector = ContourMotionDetector(background, video_source=0)

        # Create a mask with a clear rectangle
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        contours = detector.detect_contours(mask)

        # Should detect contours
        assert len(contours) >= 1

    def test_alert_system_integration(self):
        """Test alert system integration"""
        # Clear existing events
        alert_system.events.clear()

        # Create a test event
        event = alert_system.on_motion_detected(
            object_count=2,
            confidence=0.85,
            bounding_boxes=[{'x': 10, 'y': 10, 'width': 20, 'height': 20}]
        )

        # Verify event creation
        assert isinstance(event, MotionEvent)
        assert event.object_count == 2
        assert event.confidence == 0.85
        assert len(event.bounding_boxes) == 1

        # Verify event is stored
        assert len(alert_system.events) == 1
        assert alert_system.events[0] == event

    def test_statistics_calculation(self):
        """Test statistics calculation"""
        # Clear events and add some test events
        alert_system.events.clear()

        # Add test events
        alert_system.on_motion_detected(1, 0.8)
        alert_system.on_motion_detected(3, 0.9)
        alert_system.on_motion_detected(2, 0.7)

        stats = alert_system.get_statistics()

        # Verify statistics
        assert stats['total_events'] == 3
        assert stats['total_objects_detected'] == 6
        assert abs(stats['average_objects_per_event'] - 2.0) < 0.01

    @patch('cv2.imwrite')
    def test_data_export_functionality(self, mock_imwrite):
        """Test data export functionality"""
        # Clear events and add test data
        alert_system.events.clear()
        alert_system.on_motion_detected(2, 0.8)

        # Test JSON export
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file = f.name

        try:
            result = alert_system.export_events_json(temp_file)
            assert result == temp_file
            assert os.path.exists(temp_file)

            # Verify file contains expected data
            with open(temp_file, 'r') as f:
                import json
                data = json.load(f)
                assert len(data) == 1
                assert data[0]['object_count'] == 2

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_cli_command_structure(self):
        """Test that CLI commands can be imported and structured properly"""
        # This test ensures the CLI module can be imported without errors
        # and has the expected structure
        from src import cli

        # Verify main functions exist
        assert hasattr(cli, 'create_parser')
        assert hasattr(cli, 'main')

        # Verify parser can be created
        parser = cli.create_parser()
        assert parser is not None

        # Verify expected commands are available
        help_text = parser.format_help()
        expected_commands = ['background', 'detect', 'info', 'cameras', 'alerts', 'stats']
        for cmd in expected_commands:
            assert cmd in help_text


if __name__ == "__main__":
    pytest.main([__file__])
