import cv2
import numpy as np
from ultralytics import YOLO

# Constants
MODEL_PATH = "./Drone-Detector/best (1)2.pt"
REAL_WIDTH = 3.5
FOCAL_LENGTH = 200
SMOOTHING_FACTOR = 0.2
VIDEO_DEVICE = '/dev/video0'  # Define the video device explicitly
WINDOW_NAME = "Drone Detector"

# Load model
model = YOLO(MODEL_PATH)

def calculate_distance(real_width, focal_length, pixel_width):
    if pixel_width == 0:
        return None
    return (real_width * focal_length) / pixel_width

def smooth_centers(current_centers, previous_centers):
    if not previous_centers:
        return current_centers
    smoothed_centers = [
        tuple((1 - SMOOTHING_FACTOR) * np.array(prev) + SMOOTHING_FACTOR * np.array(curr)).astype(int)
        for prev, curr in zip(previous_centers, current_centers)
    ]
    return smoothed_centers

def process_frame(frame, previous_centers):
    results = model(frame)
    current_centers = []

    annotated_frame = frame.copy()
    if results and len(results) > 0:
        for result in results:
            for box in result.boxes.xyxy:
                x_min, y_min, x_max, y_max = map(int, box.tolist())
                pixel_width = x_max - x_min
                pixel_height = y_max - y_min

                x_center = (x_min + x_max) // 2
                y_center = (y_min + y_max) // 2
                current_centers.append((x_center, y_center))

                distance = calculate_distance(REAL_WIDTH, FOCAL_LENGTH, pixel_width)
                if distance:
                    cv2.putText(annotated_frame, f"{distance:.2f}m", (x_min, y_min - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    smoothed_centers = smooth_centers(current_centers, previous_centers)
    return annotated_frame, smoothed_centers

def main():
    previous_centers = []

    # Open video capture
    cap = cv2.VideoCapture(VIDEO_DEVICE)
    if not cap.isOpened():
        print(f"Failed to open {VIDEO_DEVICE}")
        return
    
    print(f"Successfully opened {VIDEO_DEVICE}. Press 'q' to quit.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process the current frame
        frame, previous_centers = process_frame(frame, previous_centers)
        
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
