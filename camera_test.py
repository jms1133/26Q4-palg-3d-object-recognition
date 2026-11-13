import os
from datetime import datetime

import cv2

from src import hsv_sampler

# Folder where captured frames are saved (created on first capture).
CAPTURE_DIR = "images/camera_test"

# Initialize the video capture object. 
# 0 is usually the built-in Mac FaceTime camera. 
# 1 is usually the first plugged-in USB webcam (your Aukey).
camera_index = 0
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Cannot open camera at index {camera_index}.")
    print("Try changing camera_index to 0 or 2.")
    exit()

print(
    "Camera successfully opened. "
    "Press 'c' to capture a frame, 'w' to toggle the warp preview, 'q' to quit."
)

# When True, every frame is run through the calibration warp and shown in a
# second window so the real camera position can be validated live.
show_warp = False

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot receive frame from camera stream.")
        break
        
    # Display the raw frame
    cv2.imshow("Aukey Webcam Test", frame)

    # Optionally warp the live frame so calibration can be checked in real time.
    # A green title means the 4 corner brackets were found; red means fallback.
    if show_warp:
        warped, ok = hsv_sampler.warp_if_possible(frame)
        title = "Warped (calibration OK)" if ok else "Warped (NO corners - raw)"
        cv2.imshow("Warp Preview", warped)
        cv2.setWindowTitle("Warp Preview", title)

    # Wait briefly for a key press. waitKey returns -1 if no key was pressed.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('w'):
        show_warp = not show_warp
        if not show_warp:
            cv2.destroyWindow("Warp Preview")
    if key == ord('c'):
        # Save the current frame with a timestamped filename so captures never
        # overwrite each other. Use the raw (un-annotated) frame for later CV.
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.png")
        cv2.imwrite(path, frame)
        print(f"Captured frame -> {path}")

# Clean up resources
cap.release()
cv2.destroyAllWindows()