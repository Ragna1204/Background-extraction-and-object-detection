import cv2
import numpy as np

def capture_background(video_source=0, num_frames=100):
    cap = cv2.VideoCapture(video_source)

    frames = []
    frame_count = 0

    print("Capturing frames to build background model")

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frames.append(frame)
        frame_count += 1

        cv2.imshow('Capturing Frames', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("Calculating median background")

    # Convert list of frames to numpy array and compute median
    frames_np = np.array(frames)
    median_frame = np.median(frames_np, axis=0).astype(dtype=np.uint8)

    return median_frame


def show_background(background):
    cv2.imshow('Estimated Background', background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    background = capture_background(video_source=0, num_frames=200)
    show_background(background)