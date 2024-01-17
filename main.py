import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.object_recognition import load_object_detection_model, perform_object_detection

# Load pre-trained object detection model (example: MobileNet SSD)
prototxt_path = '/Users/niladridas/jupyter-object-recognition-cam/models/deploy.prototxt'
model_path = '/Users/niladridas/jupyter-object-recognition-cam/models/yolov3.weights'
net = load_object_detection_model(prototxt_path, model_path)              

# Initialize webcam
cap = cv2.VideoCapture(0)

# Adjustable confidence threshold
confidence_threshold = 0.2

# Loop to capture frames and perform object recognition
while True:
    # Capture frame from webcam
    ret, frame = cap.read()

    # Display frame resolution
    frame_resolution = f"Resolution: {frame.shape[1]} x {frame.shape[0]}"
    cv2.putText(frame, frame_resolution, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Perform object detection
    detected_objects = perform_object_detection(frame, net, confidence_threshold)

    # Loop over the detected objects and draw bounding boxes
    for obj in detected_objects:
        (startX, startY, endX, endY) = obj['box']
        color = np.random.randint(0, 255, size=3)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color.tolist(), 2)

        label = f"{obj['label']}: {obj['confidence']:.2f}"
        cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.tolist(), 2)

    # Display the frame with matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
