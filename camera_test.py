import cv2

# Initialize the video capture object. 
# 0 is usually the built-in Mac FaceTime camera. 
# 1 is usually the first plugged-in USB webcam (your Aukey).
camera_index = 1 
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"Error: Cannot open camera at index {camera_index}.")
    print("Try changing camera_index to 0 or 2.")
    exit()

print("Camera successfully opened. Press 'q' to quit.")

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Cannot receive frame from camera stream.")
        break
        
    # Display the frame
    cv2.imshow("Aukey Webcam Test", frame)
    
    # Wait for 1 millisecond and check if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up resources
cap.release()
cv2.destroyAllWindows()