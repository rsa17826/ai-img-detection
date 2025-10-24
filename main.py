import cv2
from typing import Any
import misc, time

import cv2

cap = cv2.VideoCapture(1)

screen_w, screen_h = 1920, 1080
minconfidence = 0.95
blurPeople = False


def toPlaces(num: Any, pre, post=0, func=round):
  num = str(num).split(".")

  if len(num) == 1:
    num.append("")

  if pre is not None:
    num[0] = num[0][-pre:]
    while len(num[0]) < pre:
      num[0] = "0" + num[0]

  temp = num[1][post : post + 1] if len(num[1]) > post else "0"
  num[1] = num[1][:post]

  while len(num[1]) < post:
    num[1] += "0"

  if post > 0:
    temp = func(float(num[1][-1] + "." + temp))
    num[1] = list(num[1])
    num[1][-1] = str(temp)  # Replace last character
    num[1] = "".join(num[1])
    num = ".".join(num)
  else:
    num = num[0]

  return num

def resize_with_aspect_ratio(image, target_width, target_height):
  h, w = image.shape[:2]
  scale = min(target_width / w, (target_height - 26) / h)
  new_w, new_h = int(w * scale), int(h * scale)
  return cv2.resize(image, (new_w, new_h))


prev_time = time.time()
net = cv2.dnn.readNetFromTensorflow("frozen_inference_graph.pb", "graph.pbtxt")
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

while True:
  ret, frame = cap.read()
  frame = cv2.flip(frame, 1)
  gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  blob = cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=False)
  net.setInput(blob)
  detections = net.forward()
  objcount = 0
  for i in range(detections.shape[2]):
    class_id = int(detections[0, 0, i, 1])
    if not class_id:
      continue
    confidence = detections[0, 0, i, 2]
    if confidence < minconfidence:
      continue
    objcount += 1
    x = int(detections[0, 0, i, 3] * frame.shape[1])
    y = int(detections[0, 0, i, 4] * frame.shape[0])
    w = int(detections[0, 0, i, 5] * frame.shape[1])
    h = int(detections[0, 0, i, 6] * frame.shape[0])
    cv2.putText(
      frame,
      class_names[class_id],
      (x, y - 5),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
    )
    cv2.putText(
      frame,
      str(confidence),
      (x, y + 25),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (255, 255, 255),
      2,
    )
    if class_id == 1 and blurPeople:
      frame[y : y + h, x : x + w] = cv2.GaussianBlur(
        frame[y : y + h, x : x + w], (15, 15), 0
      )
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
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time

  cv2.putText(
    frame,
    "FPS: " + toPlaces(fps, 2, 3),
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
  )

  cv2.putText(
    frame,
    "OBJECT COUNT: " + str(objcount),
    (20, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
  )
  # cv2.imshow("Webcam", frame)
  cv2.imshow("Webcam", resize_with_aspect_ratio(frame, screen_w, screen_h))

  match chr(cv2.waitKey(1) & 0xFF):
    case "q":
      break
    case "w":
      cv2.imwrite("./img/frame.png", frame)

cap.release()
cv2.destroyAllWindows()


# find frame in other frame
# cv2.drawMatches()
