"""
Motion detection classes for background subtraction
"""

import cv2
import numpy as np
import time
import os
import sys

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logger import logger
from config.config import (
    FRAME_WIDTH, FRAME_HEIGHT, MOTION_THRESHOLD,
    BLUR_SIZE, MIN_CONTOUR_AREA, SHOW_PREVIEW
)
from alert_system import alert_system


class MotionDetector:
    """Class for detecting motion using background subtraction"""

    def __init__(self, background_image, video_source=0):
        self.background = cv2.resize(background_image, (FRAME_WIDTH, FRAME_HEIGHT)).astype(np.int16)
        self.video_source = video_source
        self.cap = None
        self.is_running = False

        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0

    def start_detection(self):
        """Start real-time motion detection"""
        logger.info("Starting real-time motion detection... Press 'q' to quit.")

        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.video_source}")

        self.is_running = True
        self.frame_count = 0
        self.start_time = time.time()

        try:
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    logger.warning("Could not read frame from video source")
                    break

                # Process frame for motion detection
                processed_frame, motion_mask = self.process_frame(frame)

                # Update performance metrics
                self.frame_count += 1
                elapsed_time = time.time() - self.start_time
                if elapsed_time > 0:
                    self.fps = self.frame_count / elapsed_time

                # Display results
                if SHOW_PREVIEW:
                    self.display_results(frame, processed_frame, motion_mask)

                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logger.info("Motion detection stopped by user")
                    break

        finally:
            self.stop_detection()

    def process_frame(self, frame):
        """Process a single frame for motion detection"""
        # Resize frame
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        frame_int = frame.astype(np.int16)

        # Subtract background
        diff = cv2.absdiff(frame_int, self.background).astype(np.uint8)

        # Convert to grayscale and threshold
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Clean up noise
        mask = cv2.medianBlur(mask, BLUR_SIZE)

        # Extract moving objects
        moving_objects = cv2.bitwise_and(frame, frame, mask=mask)

        return moving_objects, mask

    def detect_contours(self, mask):
        """Detect contours in motion mask"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by area
        significant_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > MIN_CONTOUR_AREA:
                significant_contours.append(contour)

        return significant_contours

    def draw_bounding_boxes(self, frame, contours):
        """Draw bounding boxes around detected motion"""
        result_frame = frame.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return result_frame

    def display_results(self, original_frame, motion_frame, mask):
        """Display detection results"""
        # Create display frames
        display_frames = []

        # Original frame
        display_frames.append(('Original', original_frame))

        # Motion detection result
        display_frames.append(('Moving Objects', motion_frame))

        # Motion mask
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        display_frames.append(('Motion Mask', mask_colored))

        # Add FPS text
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(display_frames[0][1], fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show all frames
        for title, frame in display_frames:
            cv2.imshow(title, frame)

    def get_fps(self):
        """Get current FPS"""
        return self.fps

    def stop_detection(self):
        """Stop motion detection and cleanup"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Motion detection stopped")


class ContourMotionDetector(MotionDetector):
    """Motion detector with contour detection and bounding boxes"""

    def display_results(self, original_frame, motion_frame, mask):
        """Display results with bounding boxes"""
        # Detect contours
        contours = self.detect_contours(mask)

        # Trigger alert if motion detected
        if len(contours) > 0:
            # Create bounding boxes data
            bounding_boxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                bounding_boxes.append({'x': x, 'y': y, 'width': w, 'height': h})

            # Trigger alert
            alert_system.on_motion_detected(
                object_count=len(contours),
                confidence=0.8,  # Simple confidence based on contour count
                bounding_boxes=bounding_boxes
            )

        # Draw bounding boxes on original frame
        frame_with_boxes = self.draw_bounding_boxes(original_frame, contours)

        # Create display frames
        display_frames = [
            ('Motion Detection', frame_with_boxes),
            ('Moving Objects', motion_frame),
            ('Motion Mask', cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        ]

        # Add FPS and contour count
        fps_text = f"FPS: {self.fps:.1f} | Objects: {len(contours)}"
        cv2.putText(display_frames[0][1], fps_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Show all frames
        for title, frame in display_frames:
            cv2.imshow(title, frame)
