from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("./best (1)2.pt")

video_path = "video_2025-01-02_21-06-08.mp4"

# Camera Calibration parameters
real_width = 3.5  # Real width of the object in meters
focal_length = 200  # Focal length of the camera, needs calibration

# Initialize the video capture
cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_FPS, 60)
window_name = "Car detector"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 464, 848)

# Initialize a list to store the centers of detected objects
previous_centers = []

def calculate_distance(real_width, focal_length, pixel_width):
    if pixel_width == 0:
        return None
    return (real_width * focal_length) / pixel_width

def smooth_center(current_centers, previous_centers, smoothing_factor=0.2):
    smoothed_centers = []
    for i, current in enumerate(current_centers):
        if previous_centers:
            prev = previous_centers[i] if i < len(previous_centers) else current
            smoothed = (1 - smoothing_factor) * np.array(prev) + smoothing_factor * np.array(current)
            smoothed_centers.append(tuple(smoothed.astype(int)))
        else:
            smoothed_centers.append(current)
    return smoothed_centers

while cap.isOpened():
    success, frame = cap.read()
    try:
        results = model(frame)

        current_centers = []
        if results and len(results) > 0:
            annotated_frame = frame.copy()
            for result in results:
                for box in result.boxes.xyxy:
                    x_min, y_min, x_max, y_max = map(int, box.tolist())
                    pixel_width = x_max - x_min
                    pixel_height = y_max - y_min

                    x_center = (x_min + x_max) // 2
                    y_center = (y_min + y_max) // 2

                    current_centers.append((x_center, y_center))

                    distance = calculate_distance(real_width, focal_length, pixel_width)
                    if distance is not None:
                        cv2.putText(annotated_frame, f"{distance:.2f}m", (x_min, y_min - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1)

                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Smooth the detected centers to stabilize the box position
            smoothed_centers = smooth_center(current_centers, previous_centers)
            previous_centers = smoothed_centers  # Update previous_centers for next iteration

        cv2.imshow(window_name, annotated_frame)
    except Exception as e:
        print(f"Error during inference: {e}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()