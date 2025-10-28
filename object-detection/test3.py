import cv2
import numpy as np
import time
# Load YOLO
# Load YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

# Get the output layer indices
layer_names = net.getLayerNames()
output_layer_indices = net.getUnconnectedOutLayers()

# If using OpenCV 4.x or later, convert from 1-based to 0-based indexing
output_layers = [
  layer_names[i - 1] for i in output_layer_indices
] # use .flatten() to convert to 1D array

# Check output layers
print(output_layers)


# Detect faces in an image
def detect_faces(image):
  height, width = image.shape[:2]
  blob = cv2.dnn.blobFromImage(
    image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
  )
  net.setInput(blob)
  outputs = net.forward(output_layers)
  boxes, confidences, class_ids = [], [], []
  for output in outputs:
    for detection in output:
      scores = detection[5:]
      class_id = np.argmax(scores)
      confidence = scores[class_id]
      if confidence > 0.5: # Confidence threshold
        center_x, center_y = int(detection[0] * width), int(
          detection[1] * height
        )
        w, h = int(detection[2] * width), int(detection[3] * height)
        x, y = int(center_x - w / 2), int(center_y - h / 2)
        boxes.append([x, y, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)
  for i in range(len(boxes)):
    class_id = class_ids[i]
    x = int(boxes[i] * image.shape[1])
    y = int(boxes[i] * image.shape[0])
    w = int(boxes[i] * image.shape[1])
    h = int(boxes[i] * image.shape[0])
    # Draw the class label on the image
    cv2.putText(
      image,
      class_names[class_id],
      (x, y - 5),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
    )
    # Display confidence score below the label
    cv2.putText(
      image,
      str(confidences[i]),
      (x, y + 25),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
    )

    # Draw a rectangle around the detected object
    cv2.rectangle(
      image,
      (x, y),
      (x + w, y + h),
      (
        0,
        float(confidences[i]) * 255,
        (1 - float(confidences[i])) * 255,
      ),
      2,
    )
  cv2.imshow(image)
  time.sleep(100)
  return boxes, confidences, class_ids


detections = detect_faces(
  cv2.imread(r"C:\Users\Student\Downloads\woman-1851459_640.webp")
)
