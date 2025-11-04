import misc, time, requests
import os
import eel
import cv2
from typing import Any
from threading import Thread
import base64
# A blank image encoded in base64, used as a placeholder
BLANK_IMAGE = (
  "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="
)

# Initialize Eel, a Python library for creating simple Electron-like desktop apps
eel.init("../web")

# Variable to hold the capture object; initially set to 0
cap: Any = 0

# Flag to determine whether to save the current frame
saveFrame = False
# Index of the camera to use
capidx = 0

# Set the minimum confidence level for object detection
minconfidence = 0.5
# Flag to indicate whether to blur detected people
blurPeople = False
# Flag to send data when a person enters the frame
sendOnPersonEnter = False


# Log messages to the console and the front end
def log(*msgs):
  print(*msgs)
  eel.print(*msgs)


# Expose a function to stop video capture from the camera
@eel.expose
def stopCapture():
  global cap
  if cap:
    log("stopping capture")
    cap = None # Release the camera resource


# Expose a function to quit the application completely
@eel.expose
def quitApp():
  global cap
  if cap:
    cap.release() # Release the camera resource if it's open
  os._exit(0) # Exit the application


# Expose a JavaScript function to save the current frame
@eel.expose
def jsSaveFrame():
  global saveFrame
  saveFrame = True # Set the flag to save the frame


# Expose JavaScript function to set minimum confidence level for detection
@eel.expose
def jsSetminconfidence(val):
  global minconfidence
  minconfidence = float(val) # Update minimum confidence with the new value
  log("minconfidence set to " + str(val))


# Expose JavaScript function to set whether to blur detected people
@eel.expose
def jsSetblurPeople(val):
  global blurPeople
  blurPeople = bool(val) # Update blurring preference
  log("blurPeople set to " + str(val))


# Expose JavaScript function to set whether to send alerts on person entry
@eel.expose
def jsSetsendOnPersonEnter(val):
  global sendOnPersonEnter
  sendOnPersonEnter = bool(val) # Update the flag
  log("sendOnPersonEnter set to " + str(val))


# Expose function to request updated settings/data to be sent to JavaScript
@eel.expose
def requestUpdatedData():
  # Send current configuration to the front end
  eel.loadData(
    {
      "setblurPeopleInput": blurPeople,
      "setsendOnPersonEnterInput": sendOnPersonEnter,
      "captureIdx": capidx,
      "setminconfidenceInput": minconfidence,
    }
  )


# Expose a function to start capturing video from the specified camera
@eel.expose
def startCapture(idx):
  global cap, capidx
  idx = int(idx) # Convert the input index to an integer
  stopCapture() # Stop any existing capture
  log(f"Attempting to start capture on camera index: {idx}")
  capidx = idx # Set the camera index to the global variable
  cap = cv2.VideoCapture(idx) # Initialize the VideoCapture object

  if not cap.isOpened():
    log(
      f"Failed to open camera with index {idx}. Please check the index and try again."
    ) # Log error if camera fails to open
  else:
    log(f"camera with index {idx} was successfully opened") # Log success


# Function to send the current video frame to the front end
def send_frame(frame):
  # Convert the frame to a format suitable for web
  _, buffer = cv2.imencode(".jpg", frame) # Encode frame as JPEG
  frame_bytes = buffer.tobytes() # Get bytes from the buffer
  encoded_frame = base64.b64encode(frame_bytes).decode("utf-8") # Base64 encode
  # Send the encoded frame to the JavaScript frontend
  eel.receive_frame("data:image/jpeg;base64," + encoded_frame)


# Function to send a blank frame to avoid blank display
def sendBlankFrame():
  eel.receive_frame(BLANK_IMAGE) # Send the blank image
  time.sleep(0.1) # Sleep briefly to reduce CPU load


# Start the Eel application in a new thread
Thread(
  target=lambda: eel.start(
    mode=None, port=15674, close_callback=lambda *x: os._exit(0), shutdown_delay=10
  )
).start()
os.system("start http://127.0.0.1:15674/object-detection.html")


def sendPersonEntered(frame):
  log("person just entered the frame")
  return
  # requests.post(
  #   "https://ntfy.sh/test",
  #   data="person entered the frame".encode(encoding="utf-8"),
  # )


# Function to format numbers into a specific format
def toPlaces(num: Any, pre=0, post=0, func=round):
  """Function to format numbers into a specific format

  Args:
    num (Any): number to format
    pre (int, optional): about of places before .. Defaults to 0.
    post (int, optional): amount of places after .. Defaults to 0.
    func (func, optional): function to use for trimming decimal places. Defaults to round.

  Returns:
    str: of the number formatted to the desired place counts
  """
  # Split the number into integer and decimal parts
  num = str(num).split(".")

  if len(num) == 1:
    num.append("") # Add empty decimal part if not present

  if pre is not None:
    # Keep only the last 'pre' digits of the integer part
    num[0] = num[0][-pre:]
    while len(num[0]) < pre: # Pad with zeros
      num[0] = "0" + num[0]

  # Extract the relevant decimal digit based on 'post'
  temp = num[1][post : post + 1] if len(num[1]) > post else "0"
  num[1] = num[1][:post] # Keep only first 'post' digits

  # Pad decimal part with zeros
  while len(num[1]) < post:
    num[1] += "0"

  if post > 0:
    # Round the last digit of the decimal part
    temp = func(float(num[1][-1] + "." + temp))
    num[1] = list(num[1])
    num[1][-1] = str(temp)
    num[1] = "".join(num[1])
    num = ".".join(num) # Combine back into single string
  else:
    num = num[0]

  return num


# Function to resize an image while maintaining aspect ratio
def resize_with_aspect_ratio(image, target_width, target_height):
  h, w = image.shape[:2]
  scale = min(target_width / w, (target_height - 26) / h) # Calculate scale factor
  new_w, new_h = int(w * scale), int(h * scale) # New dimensions
  return cv2.resize(image, (new_w, new_h)) # Resize image


# Initialize variables for frame rate calculation
prev_time = time.time()
# Load pre-trained neural network model
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "graph.pbtxt")
# Define class names for object detection
class_names = {
  1: "person",
  2: "bicycle",
  3: "car",
  4: "motorcycle",
  5: "airplane",
  6: "bus",
  7: "train",
  8: "truck",
  9: "boat",
  10: "traffic light",
  11: "fire hydrant",
  13: "stop sign",
  14: "parking meter",
  15: "bench",
  16: "bird",
  17: "cat",
  18: "dog",
  19: "horse",
  20: "sheep",
  21: "cow",
  22: "elephant",
  23: "bear",
  24: "zebra",
  25: "giraffe",
  27: "backpack",
  28: "umbrella",
  31: "handbag",
  32: "tie",
  33: "suitcase",
  34: "frisbee",
  35: "skis",
  36: "snowboard",
  37: "sports ball",
  38: "kite",
  39: "baseball bat",
  40: "baseball glove",
  41: "skateboard",
  42: "surfboard",
  43: "tennis racket",
  44: "bottle",
  46: "wine glass",
  47: "cup",
  48: "fork",
  49: "knife",
  50: "spoon",
  51: "bowl",
  52: "banana",
  53: "apple",
  54: "sandwich",
  55: "orange",
  56: "broccoli",
  57: "carrot",
  58: "hot dog",
  59: "pizza",
  60: "donut",
  61: "cake",
  62: "chair",
  63: "couch",
  64: "potted plant",
  65: "bed",
  67: "dining table",
  70: "toilet",
  72: "tv",
  73: "laptop",
  74: "mouse",
  75: "remote",
  76: "keyboard",
  77: "cell phone",
  78: "microwave",
  79: "oven",
  80: "toaster",
  81: "sink",
  82: "refrigerator",
  84: "book",
  85: "clock",
  86: "vase",
  87: "scissors",
  88: "teddy bear",
  89: "hair drier",
  90: "toothbrush",
}


personExistdLastFrame = 0
while True:
  if not cap or not cap.isOpened():
    sendBlankFrame()
    continue
  personExistdThisFrame = 0
  # Capture a frame from the camera
  ret, frame = cap.read()
  if not ret:
    log(frame, ret)
    continue
  frame = cv2.flip(frame, 1) # Flip the frame for a mirror effect
  gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Convert to grayscale
  # Create a blob from the image for the neural network
  blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=False)
  net.setInput(blob) # Set the input for the network
  detections = net.forward() # Perform forward pass (object detection)

  objcount = 0 # Initialize object count

  # Loop through detected objects
  for i in range(detections.shape[2]):
    class_id = int(detections[0, 0, i, 1]) # Get the class ID
    if not class_id:
      continue # Continue if class ID is 0 (not detected)

    confidence = detections[0, 0, i, 2] # Get confidence score
    if confidence < minconfidence:
      continue # Skip detections below confidence threshold

    objcount += 1 # Increment object count
    # Get bounding box coordinates
    x = int(detections[0, 0, i, 3] * frame.shape[1])
    y = int(detections[0, 0, i, 4] * frame.shape[0])
    w = int(detections[0, 0, i, 5] * frame.shape[1])
    h = int(detections[0, 0, i, 6] * frame.shape[0])
    if class_id == 1 and sendOnPersonEnter:
      personExistdThisFrame += 1
    # Blur detected people if the flag is set
    if class_id == 1 and blurPeople:
      frame[y : y + h, x : x + w] = cv2.GaussianBlur(
        frame[y : y + h, x : x + w], (15, 15), 0
      )
    # Draw the class label on the frame
    cv2.putText(
      frame,
      class_names[class_id],
      (x, y - 5),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
    )
    # Display confidence score below the label
    cv2.putText(
      frame,
      str(confidence),
      (x, y + 25),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
    )

    # Draw a rectangle around the detected object
    cv2.rectangle(
      frame,
      (x, y),
      (x + w, y + h),
      (
        0,
        float(confidence) * 255,
        (1 - float(confidence)) * 255,
      ),
      2,
    )

  # Calculate and display frames per second (FPS)
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time # Update previous time for next FPS calculation

  cv2.putText(
    frame,
    "FPS: " + toPlaces(fps, 2, 3),
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
  )

  # Display the total count of detected objects
  cv2.putText(
    frame,
    "OBJECT COUNT: " + str(objcount),
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
  )

  send_frame(frame)
  if personExistdThisFrame > personExistdLastFrame:
    sendPersonEntered(frame)
  personExistdLastFrame = personExistdThisFrame
  if saveFrame:
    cv2.imwrite("./img/frame.png", frame) # Save the current frame as an image
    log("frame saved")
    saveFrame = False
