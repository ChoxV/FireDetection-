from ultralytics import YOLO
import cvzone
import math
from picamera.array import PiRGBArray
from picamera import PiCamera
import time

# Initialize PiCamera
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Allow the camera to warm up
time.sleep(0.1)

# Load YOLO model
model = YOLO('fire.pt')

# Class names
classnames = ['fire']

# Capture frames from the PiCamera
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array

    # Resize the image (if needed)
    image = cv2.resize(image, (640, 480))

    # Run the YOLO model on the image
    results = model(image, stream=True)

    # Process the results
    for info in results:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(image, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    # Display the image with the bounding boxes
    cv2.imshow('frame', image)

    # Clear the stream in preparation for the next frame
    rawCapture.truncate(0)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
