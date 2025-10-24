import numpy as np
import tensorflow as tf
import cv2 as cv

# Check TensorFlow version
print(tf.__version__)

# Load the frozen graph
with tf.io.gfile.GFile("frozen_inference_graph.pb", "rb") as f:
  graph_def = tf.compat.v1.GraphDef()
  graph_def.ParseFromString(f.read())


# Create a function to run the model
def run_inference(image_path):
  # Load and preprocess the image
  img = cv.imread(image_path)
  assert img is not None
  rows, cols = img.shape[:2]
  inp = cv.resize(img, (300, 300))
  inp = inp[:, :, [2, 1, 0]] # BGR to RGB
  inp = np.expand_dims(inp, axis=0)

  # Create a new computation graph context
  tf.import_graph_def(graph_def, name="")

  # Use tf.compat.v1.get_default_graph() if needed
  detection_graph = tf.compat.v1.get_default_graph()

  # Load the tensor information
  tensor_dict = {
    "num_detections": detection_graph.get_tensor_by_name("num_detections:0"),
    "detection_boxes": detection_graph.get_tensor_by_name("detection_boxes:0"),
    "detection_scores": detection_graph.get_tensor_by_name("detection_scores:0"),
    "detection_classes": detection_graph.get_tensor_by_name("detection_classes:0"),
  }

  # Run inference
  with tf.compat.v1.Session() as sess:
    out = sess.run(tensor_dict, feed_dict={"image_tensor:0": inp})

  # Visualize detected bounding boxes
  num_detections = int(out["num_detections"][0])
  for i in range(num_detections):
    classId = int(out["detection_classes"][0][i])
    score = float(out["detection_scores"][0][i])
    bbox = [float(v) for v in out["detection_boxes"][0][i]]
    if score > 0.3: # Confidence threshold
      x = bbox[1] * cols
      y = bbox[0] * rows
      right = bbox[3] * cols
      bottom = bbox[2] * rows
      cv.rectangle(
        img,
        (int(x), int(y)),
        (int(right), int(bottom)),
        (125, 255, 51),
        thickness=2,
      )

  # Display the output image with detections
  cv.imshow("TensorFlow MobileNet-SSD", img)
  cv.waitKey(0)


# Run the inference on the example image
run_inference(r"C:\Users\Student\Downloads\woman-1851459_640.webp")
