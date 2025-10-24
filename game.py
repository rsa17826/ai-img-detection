import cv2
from typing import Any
import random, time, re
from biasRand import BalancedRand

from misc import f

import cv2

screen_w, screen_h = 1920, 1080


def resize_with_aspect_ratio(image, target_width, target_height):
  h, w = image.shape[:2]
  scale = min(target_width / w, (target_height - 26) / h)
  new_w, new_h = int(w * scale), int(h * scale)
  return cv2.resize(image, (new_w, new_h))


def collides(x, y, w, h, face):
  x2, y2, w2, h2 = face
  return not (x >= x2 + w2 or x + w <= x2 or y >= y2 + h2 or y + h <= y2)


cap = cv2.VideoCapture(1)
autoReset = False
lastFace: Any = 0
death: Any = 0
speed: Any = 0
score: Any = 0
deathPosRand: Any = 0
spawnNewDeathRand: Any = 0
stopped: Any = 0
highScore: Any = 0
size: Any = 0


def comstr(item: int | float) -> str:
  reg = [r"(?<=\d)(\d{3}(?=(?:\d{3})*(?:$|\.)))", r",\g<0>"]
  if item is float:
    return (
      re.sub(reg[0], reg[1], str(item).split(".")[0])
      + "."
      + str(item).split(".")[1]
    )
  return re.sub(reg[0], reg[1], str(item))


def reset():
  global lastFace, death, speed, score, deathPosRand, spawnNewDeathRand, stopped, highScore, size
  lastFace = None
  death = []
  speed = 5
  score = 0
  deathPosRand = BalancedRand()
  spawnNewDeathRand = BalancedRand(0, 1, 0.1, 0.5)
  stopped = False
  highScore = int(f.read("./highScore.txt", "0"))
  size = 25


prev_time = time.time()


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


reset()

while True:
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  prev_time = curr_time
  if not stopped:
    size += 0.01
    speed += 0.01
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    height, width = frame.shape[:2]
    deathPosRand.maxNum = width
    face_cascade = cv2.CascadeClassifier(
      cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 1:
      lastFace = faces[0]

    if lastFace is not None:
      a = spawnNewDeathRand.next()
      print(a)
      if a < 0.001:
        randPos = int(deathPosRand.next())
        death.append([randPos, 5, int(size), int(size)])
      for deathBox in death:
        deathBox[1] += speed
      death = [
        *filter(lambda deathBox: deathBox[1] + deathBox[3] < height, death)
      ]

      for x, y, w, h in death:
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        if collides(x, y, w, h, lastFace):
          stopped = True
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.line(frame, (x + w, y), (x, y + h), (0, 0, 255), 2)
      x, y, w, h = lastFace
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.line(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
      cv2.line(frame, (x + w, y), (x, y + h), (0, 255, 0), 2)
      score += int(lastFace[3] / 3)
      if score > highScore:
        f.write("./highScore.txt", str(score))
      cv2.putText(
        frame,
        "FPS: " + toplaces(fps, 2, 3),
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
      )
      cv2.putText(
        frame,
        "SCORE: " + comstr(score),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
      )
      cv2.putText(
        frame,
        "HIGH SCORE: " + comstr(highScore),
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
      )

    # cv2.imshow("Webcam", frame)
    cv2.imshow("Webcam", resize_with_aspect_ratio(frame, screen_w, screen_h))
  else:
    if autoReset:
      reset()
  match chr(cv2.waitKey(1) & 0xFF):
    case "q":
      break
    case "w":
      cv2.imwrite("./img/frame.png", frame)
    case "r":
      reset()
    case "t":
      autoReset = not autoReset
      if stopped:
        reset()

cap.release()
cv2.destroyAllWindows()


# find frame in other frame
# cv2.drawMatches()
