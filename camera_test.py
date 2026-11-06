import os
from datetime import datetime

import cv2

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

print("Camera successfully opened. Press 'c' to capture a frame, 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot receive frame from camera stream.")
        break
        
    # Display the frame
    cv2.imshow("Aukey Webcam Test", frame)
    
    # Wait briefly for a key press. waitKey returns -1 if no key was pressed.
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
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