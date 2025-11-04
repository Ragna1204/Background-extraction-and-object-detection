"""
Alert system for motion detection events
"""

import json
import csv
import time
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from logger import logger
from config.config import LOG_FILE


class MotionEvent:
    """Represents a motion detection event"""

    def __init__(self, timestamp: float, object_count: int, confidence: float = 0.0,
                 bounding_boxes: List[Dict] = None):
        self.timestamp = timestamp
        self.object_count = object_count
        self.confidence = confidence
        self.bounding_boxes = bounding_boxes or []
        self.datetime = datetime.fromtimestamp(timestamp)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary"""
        return {
            'timestamp': self.timestamp,
            'datetime': self.datetime.isoformat(),
            'object_count': self.object_count,
            'confidence': self.confidence,
            'bounding_boxes': self.bounding_boxes
        }

    def __str__(self):
        return f"MotionEvent({self.datetime}, objects={self.object_count})"


class AlertSystem:
    """Alert system for motion detection"""

    def __init__(self, enable_sound: bool = True, enable_log: bool = True,
                 enable_csv: bool = False, csv_file: str = 'motion_events.csv'):
        self.enable_sound = enable_sound
        self.enable_log = enable_log
        self.enable_csv = enable_csv
        self.csv_file = csv_file
        self.events: List[MotionEvent] = []

        # Initialize CSV file if needed
        if self.enable_csv:
            self._init_csv_file()

    def _init_csv_file(self):
        """Initialize CSV file with headers"""
        try:
            with open(self.csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['timestamp', 'datetime', 'object_count', 'confidence'])
            logger.info(f"Initialized CSV file: {self.csv_file}")
        except Exception as e:
            logger.error(f"Failed to initialize CSV file: {e}")

    def on_motion_detected(self, object_count: int, confidence: float = 0.0,
                          bounding_boxes: List[Dict] = None) -> MotionEvent:
        """Handle motion detection event"""
        event = MotionEvent(time.time(), object_count, confidence, bounding_boxes)
        self.events.append(event)

        # Log the event
        if self.enable_log:
            logger.info(f"Motion detected: {object_count} objects (confidence: {confidence:.2f})")

        # Save to CSV
        if self.enable_csv:
            self._save_to_csv(event)

        # Trigger sound alert
        if self.enable_sound:
            self._play_alert_sound()

        return event

    def _save_to_csv(self, event: MotionEvent):
        """Save event to CSV file"""
        try:
            with open(self.csv_file, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    event.timestamp,
                    event.datetime.isoformat(),
                    event.object_count,
                    event.confidence
                ])
        except Exception as e:
            logger.error(f"Failed to save event to CSV: {e}")

    def _play_alert_sound(self):
        """Play alert sound (simple beep)"""
        try:
            # Simple console beep - in a real implementation, you'd use pygame or similar
            print('\a', end='', flush=True)  # ASCII bell character
        except Exception as e:
            logger.error(f"Failed to play alert sound: {e}")

    def get_recent_events(self, hours: int = 1) -> List[MotionEvent]:
        """Get events from the last N hours"""
        cutoff_time = time.time() - (hours * 3600)
        return [event for event in self.events if event.timestamp > cutoff_time]

    def export_events_json(self, filename: str = None) -> str:
        """Export all events to JSON"""
        if filename is None:
            filename = f"motion_events_{int(time.time())}.json"

        try:
            with open(filename, 'w') as f:
                json.dump([event.to_dict() for event in self.events], f, indent=2)
            logger.info(f"Exported {len(self.events)} events to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to export events to JSON: {e}")
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about motion events"""
        if not self.events:
            return {'total_events': 0}

        total_events = len(self.events)
        total_objects = sum(event.object_count for event in self.events)
        avg_objects = total_objects / total_events if total_events > 0 else 0

        # Events per hour (last 24 hours)
        recent_events = self.get_recent_events(24)
        events_per_hour = len(recent_events) / 24 if len(recent_events) > 0 else 0

        return {
            'total_events': total_events,
            'total_objects_detected': total_objects,
            'average_objects_per_event': avg_objects,
            'events_per_hour': events_per_hour,
            'first_event': self.events[0].datetime.isoformat() if self.events else None,
            'last_event': self.events[-1].datetime.isoformat() if self.events else None
        }


# Global alert system instance
alert_system = AlertSystem()
