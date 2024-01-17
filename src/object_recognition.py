import cv2
import numpy as np

def load_object_detection_model(prototxt_path, model_path):
    """
    Load the object detection model.

    Parameters:
    - prototxt_path (str): Path to the deploy prototxt file.
    - model_path (str): Path to the model weights file.

    Returns:
    - net: Loaded object detection model.
    """
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    return net

def perform_object_detection(frame, net, confidence_threshold=0.2):
    """
    Perform object detection on a given frame using the loaded model.

    Parameters:
    - frame (numpy.ndarray): Input frame from the webcam.
    - net: Loaded object detection model.
    - confidence_threshold (float): Confidence threshold for object detection.

    Returns:
    - List of dictionaries containing information about detected objects.
      Each dictionary has keys 'label', 'confidence', 'box' (bounding box coordinates).
    """
    # Preprocess the frame for object detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    
    # Perform inference and get predictions
    detections = net.forward()

    # Process detections
    detected_objects = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Add information about the detected object to the list
            detected_object = {
                'label': f'Object {i+1}',
                'confidence': confidence,
                'box': (startX, startY, endX, endY)
            }
            detected_objects.append(detected_object)

    return detected_objects
