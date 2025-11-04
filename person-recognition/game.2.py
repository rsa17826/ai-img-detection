import torch
import cv2
from typing import Any
import random, time, re
from biasRand import BalancedRand
import eel
from threading import Thread
import os, base64
import enroll_faces
from facenet_pytorch import MTCNN, InceptionResnetV1 # type: ignore
import numpy as np
from pathlib import Path

print("changing dir to ", os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# normalize known vectors for cosine sim
def l2norm(x):
  return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


# F
class f:
  @staticmethod
  def read(
    file,
    default="",
    asbinary=False,
    buffering: int = -1,
    encoding: Any = None,
    errors: Any = None,
    newline: Any = None,
    closefd: bool = True,
    opener=None,
  ):
    if Path(file).exists():
      with open(
        file,
        "r" + ("b" if asbinary else ""),
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
      ) as f:
        text = f.read()
      if text:
        return text
      return default
    else:
      with open(
        file,
        "w" + ("b" if asbinary else ""),
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
      ) as f:
        f.write(default)
      return default

  @staticmethod
  def writeCsv(file, rows):
    with open(file, "w", encoding="utf-8", newline="") as f:
      w = csv.writer(f)
      w.writerows(rows)
    return rows

  @staticmethod
  def write(
    file,
    text,
    asbinary=False,
    buffering: int = -1,
    encoding: Any = None,
    errors: Any = None,
    newline: Any = None,
    closefd: bool = True,
    opener=None,
  ):
    with open(
      file,
      "w" + ("b" if asbinary else ""),
      buffering=buffering,
      encoding=encoding,
      errors=errors,
      newline=newline,
      closefd=closefd,
      opener=opener,
    ) as f:
      f.write(text)
    return text

  @staticmethod
  def append(
    file,
    text,
    asbinary=False,
    buffering: int = -1,
    encoding: Any = None,
    errors: Any = None,
    newline: Any = None,
    closefd: bool = True,
    opener=None,
  ):
    with open(
      file,
      "a",
      buffering=buffering,
      encoding=encoding,
      errors=errors,
      newline=newline,
      closefd=closefd,
      opener=opener,
    ) as f:
      f.write(text)
    return text

  @staticmethod
  def writeline(
    file,
    text,
    buffering: int = -1,
    encoding: Any = None,
    errors: Any = None,
    newline: Any = None,
    closefd: bool = True,
    opener=None,
  ):
    with open(
      file,
      "a",
      buffering=buffering,
      encoding=encoding,
      errors=errors,
      newline=newline,
      closefd=closefd,
      opener=opener,
    ) as f:
      f.write("\n" + text)
    return text


# A blank image encoded in base64, used as a placeholder
BLANK_IMAGE = (
  "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="
)

# Initialize Eel, a Python library for creating simple Electron-like desktop apps
eel.init("gameWeb")

# Variable to hold the capture object; initially set to 0
cap: Any = 0
MATCH_THRESHOLD = 0.55
DB_PATH = "data/embeddings_db.npz"
TARGET_CONFIDENCE = 0.75
mtcnn: Any = None
# Flag to determine whether to save the current frame
saveFrame = False
# Index of the camera to use
capidx = 1

# Set the minimum confidence level for object detection
minconfidence = 0.5


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


# Expose function to request updated settings/data to be sent to JavaScript
@eel.expose
def requestUpdatedData():
  # Send current configuration to the front end
  eel.loadData(
    {
      "captureIdx": capidx,
      "setminconfidenceInput": minconfidence,
    }
  )


# Expose a function to start capturing video from the specified camera
@eel.expose
def startCapture(idx):
  global cap, capidx
  stopCapture()
  idx = int(idx) # Convert the input index to an integer
  log(f"Attempting to start capture on camera index: {idx}")
  capidx = idx # Set the camera index to the global variable
  cap = cv2.VideoCapture(idx) # Initialize the VideoCapture object
  if not cap.isOpened():
    log(
      f"Failed to open camera with index {idx}. Please check the index and try again."
    ) # Log error if camera fails to open
  else:
    log(f"camera with index {idx} was successfully opened") # Log success


def get_embedding(face_img_rgb):
  """
  face_img_rgb: np array (H,W,3) RGB cropped face region
  returns: (512,) embedding or None if detection failed
  """
  # detect/align just this face via mtcnn on the crop
  face_tensor = mtcnn(face_img_rgb)

  if face_tensor is None:
    return None

  # Convert to proper shape
  face_tensor = face_tensor.squeeze(0).to(device) # Remove the batch dimension

  with torch.no_grad():
    emb = resnet(
      face_tensor.unsqueeze(0)
    ) # Add the batch dimension back for processing
  emb = emb.squeeze(0).cpu().numpy()
  return emb


def match_identity(embedding_vec):
  """
  Compare embedding_vec (512,) to known embeddings via cosine similarity.
  Return (best_name, best_score) or (None, None)
  """
  # normalize the candidate to unit length
  cand = embedding_vec / (np.linalg.norm(embedding_vec) + 1e-10)
  # cosine sim = dot product since both normalized
  sims = known_norm.dot(cand) # shape (N,)
  best_idx = np.argmax(sims)
  best_score = sims[best_idx]
  best_name = known_labels[best_idx]
  if best_score >= MATCH_THRESHOLD:
    return best_name, float(best_score)
  else:
    return None, None


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
os.system("start http://127.0.0.1:15674")


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


def reset():
  global lastFace, deathBoxList, speed, score, deathPosRand, stopped, highScore, size, dirRand
  lastFace = None
  deathBoxList = []
  speed = 3
  gameScore = 0
  deathPosRand = BalancedRand()
  # spawnNewDeathRand = BalancedRand(0, 1, 0.1, 0.5)
  dirRand = BalancedRand(0, 3, 0.1, 0.5)
  stopped = False
  try:
    highScore = int(f.read("./highScore.txt", "0"))
  except Exception as e:
    highScore = -1
  size = 35


autoReset = False
lastFace: Any = 0
deathBoxList: Any = 0
speed: Any = 0
gameScore: Any = 0
deathPosRand: Any = 0
# spawnNewDeathRand: Any = 0
stopped: Any = 0
highScore: Any = 0
size: Any = 0
dirRand: Any = 0

reset()


def collides(x, y, w, h, face):
  x2, y2, w2, h2 = face
  h2 -= y2
  w2 -= x2
  return not (x >= x2 + w2 or x + w <= x2 or y >= y2 + h2 or y + h <= y2)


def updateFacesList():
  global mtcnn, known_norm, resnet, device, db, known_embeddings, known_labels
  try:
    enroll_faces.init(log, eel.setProg)
    db = np.load(DB_PATH)
    known_embeddings = db["embeddings"] # shape (N,512)
    known_labels = db["labels"] # shape (N,)
    # load models
    known_norm = l2norm(known_embeddings)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
  except Exception as e:
    log(e)
  eel.hideProg()


def comstr(item: Any) -> str:
  reg = [r"(?<=\d)(\d{3}(?=(?:\d{3})*(?:$|\.)))", r",\g<0>"]
  if item is float:
    return (
      re.sub(reg[0], reg[1], str(item).split(".")[0])
      + "."
      + str(item).split(".")[1]
    )
  return re.sub(reg[0], reg[1], str(item))


def say(msg):
  log(msg)


highScoreOwner = f.read("./highScorename.txt", "")
faceName = None
updateFacesList()
prev_time: float = time.time()
gameScores: Any = {}
shouldSayNewHighScores: Any = {}
spawnCount = 0.0
while True:
  if os.path.exists("updateGameUserList"):
    os.remove("updateGameUserList")
    updateFacesList()
  if not cap or not cap.isOpened():
    sendBlankFrame()
    continue
  curr_time = time.time()
  fps = 1 / (curr_time - prev_time)
  delta = curr_time - prev_time
  prev_time = curr_time
  # Capture a frame from the camera
  ret, frame = cap.read()
  if not ret:
    log(frame, ret)
    continue
  spawnCount += 0.1
  frame = cv2.flip(frame, 1) # Flip the frame for a mirror effect
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # rawframe_bgr = frame.copy()
  # detect all faces in the *full* frame
  # boxes: [[x1,y1,x2,y2], ...]
  # probs: confidence per face
  facePos = None
  height, width = frame.shape[:2]
  cv2.putText(
    frame,
    "FPS: " + toPlaces(fps, 2, 3),
    (20, 50),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2,
  )
  eel.setHighscoreMessage( # type: ignore
    "HIGH SCORE: " + comstr(highScore) + " by " + highScoreOwner,
  )
  deathPosRand.maxNum = width
  while spawnCount > 1:
    spawnCount -= 1
    randPos = int(deathPosRand.next())
    diridx = int(round(dirRand.next()))
    dir = [[0, 1], [0, -1], [1, 0], [-1, 0]][diridx]
    deathBox = [
      0,
      0,
      int(size),
      int(size),
      dir,
      speed,
    ]
    if diridx == 0:
      deathBox[1] = 0
      deathBox[0] = randPos
    elif diridx == 1:
      deathBox[1] = height - size
      deathBox[0] = randPos
    elif diridx == 2:
      deathBox[0] = 0
      deathBox[1] = randPos
    elif diridx == 3:
      deathBox[0] = width - size
      deathBox[1] = randPos
    s = deathBox[4]
    s[0] *= deathBox[5]
    s[1] *= deathBox[5]
    deathBoxList.append(deathBox)
  for deathBox in deathBoxList:
    s = deathBox[4]
    deathBox[0] += s[0]
    deathBox[1] += s[1]
    x, y, w, h, dir, speed = deathBox
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.line(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.line(frame, (x + w, y), (x, y + h), (0, 0, 255), 2)
  deathBoxList = [
    deathBox
    for deathBox in deathBoxList
    if (
      not (
        deathBox[1] > height # down
        or deathBox[1] + deathBox[2] < 0 # up
        or deathBox[0] > width # right
        or deathBox[0] + deathBox[3] < 0
      )
    )
  ]
  if mtcnn:
    boxes, probs = mtcnn.detect(frame_rgb)

    if boxes is not None:
      for box, prob in zip(boxes, probs):
        if prob is None:
          continue
        x1, y1, x2, y2 = [int(v) for v in box]

        # crop face region
        face_crop_rgb = frame_rgb[y1:y2, x1:x2]
        if face_crop_rgb.size == 0:
          continue

        emb = None
        try:
          emb = get_embedding(face_crop_rgb)
        except Exception as e:
          continue
        if emb is None:
          continue

        try:
          name, score = match_identity(emb)
        except Exception as e:
          log(e)
          continue
        label_text = "Unknown"
        color = (0, 0, 255) # red in BGR
        if name is not None:
          label_text = f"{name} ({score:.2f})"
          color = (0, 255, 0) # green

        facePos = [x1, y1, x2, y2]
        collision = False
        if name:
          for x, y, w, h, dir, speed in deathBoxList:
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            if collides(x, y, w, h, facePos):
              gameScores[name] = 0
              shouldSayNewHighScores[name] = True
              collision = True
          if not collision:
            if name not in gameScores:
              gameScores[name] = 0
            gameScores[name] += (facePos[3] / 3) * delta
          cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
          cv2.putText(
            frame,
            name + ": " + toPlaces(score, 1, 2),
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
          )
          cv2.putText(
            frame,
            str(int(gameScores[name])),
            (x1, y1 - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (150, 150, 150),
            2,
          )
        for scorereName, gameScore in gameScores.items():
          if int(gameScore) > highScore:
            highScore = gameScore
            if highScoreOwner:
              if scorereName != highScoreOwner:
                say(
                  scorereName
                  + " overtook "
                  + highScoreOwner
                  + " with a score of "
                  + str(int(gameScore))
                )
              else:
                if (
                  scorereName in shouldSayNewHighScores
                  and shouldSayNewHighScores[scorereName]
                ):
                  say(
                    scorereName
                    + " got a new high score of "
                    + str(int(gameScore))
                  )
                  shouldSayNewHighScores[scorereName] = False
            highScoreOwner = str(scorereName)
            f.write("./highScore.txt", str(int(gameScore)))
            f.write("./highScorename.txt", str(scorereName))
        # if (
        #   enableAutoCapture
        #   and name
        #   and score
        #   and score < TARGET_CONFIDENCE
        #   and score > MATCH_THRESHOLD
        # ):
        #   faceName = name
        # cv2.putText(
        #   frame,
        #   label_text,
        #   (x1, y1 - 10),
        #   cv2.FONT_HERSHEY_SIMPLEX,
        #   0.6,
        #   color,
        #   2,
        # )
  print(gameScores)
  send_frame(frame)
  # if faceName:
  #   i = 0
  #   path = f"./enrolled/{faceName}/{i}.png"
  #   os.makedirs(f"./enrolled/{faceName}", exist_ok=True)
  #   while os.path.exists(path):
  #     i += 1
  #     path = f"./enrolled/{faceName}/{i}.png"
  #   log("adding image for ", faceName, "idx: ", i)
  #   if facePos:
  #     frame_rgb_cropped = rawframe_bgr[
  #       max(0, facePos[1] - 15) : min(facePos[3] + 15, rawframe_bgr.shape[0]),
  #       max(0, facePos[0] - 15) : min(facePos[2] + 15, rawframe_bgr.shape[1]),
  #     ]
  #   else:
  #     frame_rgb_cropped = rawframe_bgr

  #   cv2.imwrite(path, frame_rgb_cropped) # Save the current frame as an image
  #   faceName = None
  #   updateFacesList()


reset()
