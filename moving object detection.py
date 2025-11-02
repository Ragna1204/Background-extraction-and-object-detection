import cv2
import numpy as np

def capture_background(video_source=0, num_frames=100):
    cap = cv2.VideoCapture(video_source)

    frames = []
    frame_count = 0

    print("Capturing frames to build background model...")

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)
        frame_count += 1

        cv2.imshow('Capturing Background Frames', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Calculating median background...")

    frames_np = np.array(frames)
    median_frame = np.median(frames_np, axis=0).astype(np.uint8)

    return median_frame


def subtract_background_and_show(background_image, video_source=0):
    cap = cv2.VideoCapture(video_source)

    background = cv2.resize(background_image, (640, 480)).astype(np.int16)

    # 🆕 Display the background before starting
    cv2.imshow('Estimated Background (Before Detection)', background_image)
    print("Showing estimated background. Press any key to continue to foreground detection...")
    cv2.waitKey(0)
    cv2.destroyWindow('Estimated Background (Before Detection)')

    print("[INFO] Starting real-time foreground detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame_int = frame.astype(np.int16)

        # Subtract background
        diff = cv2.absdiff(frame_int, background).astype(np.uint8)

        # Convert to grayscale and threshold
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

        # Clean up noise
        mask = cv2.medianBlur(mask, 5)

        # Extracting only moving objects
        moving_objects = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Moving Objects Only', moving_objects)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    background = capture_background(video_source=0, num_frames=600)
    subtract_background_and_show(background)