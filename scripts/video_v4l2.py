import cv2

# Open the video device. Ensure the device exists and has correct permissions.
cap = cv2.VideoCapture('/dev/video45')

if not cap.isOpened():
    print("Error: Could not open video device /dev/video45")
    exit()

# Optionally, you can set capture properties (e.g., resolution)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Video Feed", frame)

    # Exit the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
