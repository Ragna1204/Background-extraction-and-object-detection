"""
Command-line interface for the motion detection application
"""

import argparse
import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from background_model import create_background_model
from motion_detector import ContourMotionDetector
from video_processor import get_video_info, list_available_cameras
from alert_system import alert_system
from config.config import (
    NUM_BACKGROUND_FRAMES, VIDEO_SOURCE, BACKGROUND_METHOD,
    FRAME_WIDTH, FRAME_HEIGHT
)
from logger import logger


def create_parser():
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description="Background Extraction and Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py background          # Capture and display background
  python cli.py detect              # Run motion detection
  python cli.py detect --video test.mp4  # Process video file
  python cli.py info --video test.mp4    # Get video information
  python cli.py cameras             # List available cameras
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Background capture command
    bg_parser = subparsers.add_parser('background', help='Capture background model')
    bg_parser.add_argument('--frames', type=int, default=NUM_BACKGROUND_FRAMES,
                          help=f'Number of frames to capture (default: {NUM_BACKGROUND_FRAMES})')
    bg_parser.add_argument('--method', choices=['median', 'mean', 'gmm'],
                          default=BACKGROUND_METHOD, help='Background modeling method')
    bg_parser.add_argument('--source', type=str, default=VIDEO_SOURCE,
                          help='Video source (camera index or file path)')

    # Motion detection command
    detect_parser = subparsers.add_parser('detect', help='Run motion detection')
    detect_parser.add_argument('--video', type=str, help='Video file to process (optional)')
    detect_parser.add_argument('--frames', type=int, default=NUM_BACKGROUND_FRAMES,
                              help=f'Number of frames for background (default: {NUM_BACKGROUND_FRAMES})')
    detect_parser.add_argument('--method', choices=['median', 'mean', 'gmm'],
                              default=BACKGROUND_METHOD, help='Background modeling method')
    detect_parser.add_argument('--threshold', type=int, default=30,
                              help='Motion detection threshold (default: 30)')
    detect_parser.add_argument('--min-area', type=int, default=500,
                              help='Minimum contour area (default: 500)')

    # Video info command
    info_parser = subparsers.add_parser('info', help='Get video file information')
    info_parser.add_argument('--video', type=str, required=True, help='Video file path')

    # Cameras command
    cameras_parser = subparsers.add_parser('cameras', help='List available cameras')

    # Alerts command
    alerts_parser = subparsers.add_parser('alerts', help='Show motion detection alerts')
    alerts_parser.add_argument('--export', type=str, help='Export alerts to JSON file')
    alerts_parser.add_argument('--hours', type=int, default=24,
                              help='Show alerts from last N hours (default: 24)')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show motion detection statistics')

    return parser


def handle_background_command(args):
    """Handle background capture command"""
    try:
        # Create background model
        bg_model = create_background_model(
            method=args.method,
            video_source=args.source,
            num_frames=args.frames
        )

        # Build background
        background = bg_model.get_background()

        # Display background
        import cv2
        cv2.imshow('Estimated Background', background)
        logger.info("Press any key to exit...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        logger.info("Background capture completed successfully")

    except Exception as e:
        logger.error(f"Background capture failed: {e}")
        sys.exit(1)


def handle_detect_command(args):
    """Handle motion detection command"""
    try:
        # Determine video source
        video_source = args.video if args.video else VIDEO_SOURCE

        # Create background model
        bg_model = create_background_model(
            method=args.method,
            video_source=video_source,
            num_frames=args.frames
        )
        background = bg_model.get_background()

        # Update config for motion detection
        import config.config as cfg
        cfg.MOTION_THRESHOLD = args.threshold
        cfg.MIN_CONTOUR_AREA = args.min_area

        # Display background before detection
        import cv2
        cv2.imshow('Estimated Background (Before Detection)', background)
        logger.info("Press any key to start motion detection...")
        cv2.waitKey(0)
        cv2.destroyWindow('Estimated Background (Before Detection)')

        # Start motion detection
        detector = ContourMotionDetector(background, video_source)
        detector.start_detection()

        logger.info("Motion detection completed")

    except Exception as e:
        logger.error(f"Motion detection failed: {e}")
        sys.exit(1)


def handle_info_command(args):
    """Handle video info command"""
    try:
        info = get_video_info(args.video)
        print("Video Information:")
        print(f"  Path: {info['path']}")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS: {info['fps']}")
        print(f"  Frame Count: {info['frame_count']}")
        print(".2f")
        print(f"  Duration: {info['duration']:.2f} seconds")

    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        sys.exit(1)


def handle_cameras_command(args):
    """Handle cameras list command"""
    try:
        cameras = list_available_cameras()
        if cameras:
            print("Available cameras:")
            for cam_id in cameras:
                print(f"  Camera {cam_id}")
        else:
            print("No cameras found")

    except Exception as e:
        logger.error(f"Failed to list cameras: {e}")
        sys.exit(1)


def handle_alerts_command(args):
    """Handle alerts command"""
    try:
        # Get recent events
        recent_events = alert_system.get_recent_events(args.hours)

        if not recent_events:
            print(f"No motion events detected in the last {args.hours} hours.")
            return

        print(f"Motion Detection Events (last {args.hours} hours):")
        print("-" * 60)

        for event in recent_events:
            print(f"Time: {event.datetime}")
            print(f"Objects detected: {event.object_count}")
            print(f"Confidence: {event.confidence:.2f}")
            if event.bounding_boxes:
                print(f"Bounding boxes: {len(event.bounding_boxes)}")
            print("-" * 30)

        # Export if requested
        if args.export:
            filename = alert_system.export_events_json(args.export)
            if filename:
                print(f"\nEvents exported to: {filename}")

    except Exception as e:
        logger.error(f"Failed to show alerts: {e}")
        sys.exit(1)


def handle_stats_command(args):
    """Handle stats command"""
    try:
        stats = alert_system.get_statistics()

        print("Motion Detection Statistics:")
        print("-" * 40)
        print(f"Total events: {stats['total_events']}")
        print(f"Total objects detected: {stats['total_objects_detected']}")

        if stats['total_events'] > 0:
            print(f"Average objects per event: {stats['average_objects_per_event']:.1f}")
            print(f"Events per hour: {stats['events_per_hour']:.2f}")

        if stats['first_event']:
            print(f"First event: {stats['first_event']}")
        if stats['last_event']:
            print(f"Last event: {stats['last_event']}")

    except Exception as e:
        logger.error(f"Failed to show statistics: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate handler
    handlers = {
        'background': handle_background_command,
        'detect': handle_detect_command,
        'info': handle_info_command,
        'cameras': handle_cameras_command,
        'alerts': handle_alerts_command,
        'stats': handle_stats_command
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
