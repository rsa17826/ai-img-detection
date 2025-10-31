import cv2
import torch
import numpy as np
import time
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from typing import Any
import eel
from threading import Thread
import base64, time
import enroll_faces
import subprocess, sys
from pathlib import Path
import re, hashlib
from typing import Dict, List


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


def progress_bar(iteration, total, prefix="", length=40, fill="█"):
  percent = iteration / total
  filled_length = int(length * percent)
  bar = fill * filled_length + "-" * (length - filled_length)
  sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1%}")
  sys.stdout.flush()


print("changing dir to ", os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Index of the camera to use
capidx = 0
os.makedirs("data", exist_ok=True)
os.makedirs("enrolled", exist_ok=True)
os.makedirs("frames", exist_ok=True)
os.makedirs("outFrames", exist_ok=True)
os.makedirs("outFramesStep2", exist_ok=True)
# threshold tuning:
# cosine similarity ranges roughly -1 to 1
# same-person pairs are usually high (e.g. 0.6-0.9+),
# different people are lower. You will tune this.
MATCH_THRESHOLD = 0.6
DB_PATH = "data/embeddings_db.npz"
TARGET_CONFIDENCE = 0.75
mtcnn: Any = None


# normalize known vectors for cosine sim
def l2norm(x):
  return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


def updateFacesList():
  global mtcnn, known_norm, resnet, device, db, known_embeddings, known_labels
  try:
    enroll_faces.init(print)
    db = np.load(DB_PATH)
    known_embeddings = db["embeddings"] # shape (N,512)
    known_labels = db["labels"] # shape (N,)
    # load models
    known_norm = l2norm(known_embeddings)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
  except Exception as e:
    print(e)


updateFacesList()
# init attendance memory
last_seen: dict[Any, Any] = {} # name -> unix timestamp
# prepare attendance print file if missing


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


# sys.argv.append(r"C:\Users\Student\Hi Me In 10 Years [F0OkwXKcPSE].mp4")
# print(sys.argv)
import subprocess
import sys

# Ensure the script receives a video file path as the first argument
if len(sys.argv) < 2:
  print("Usage: python script.py <video_file>")
  sys.exit(1)

video_file = sys.argv[1]
if not os.path.exists("./frames/00000000000000000001.png"):
  output_pattern = "frames/%020d.png" # Define how to name your frames

  # Construct the command
  command = ["ffmpeg", "-i", video_file, output_pattern]

  # Run the command
  try:
    subprocess.run(command, check=True)
    print("Frames generated successfully.")
  except subprocess.CalledProcessError as e:
    print(f"An error occurred: {e}")

sorted_files = sorted(os.listdir("./frames"), key=lambda x: int(x.split(".")[0]))
maxProg = len(sorted_files)
colors = {}
prog = 0
enableAutoCapture = True


def text_to_color(text):
  """Convert text to a color using its hash."""
  # Create a hash of the text
  hash_object = hashlib.md5(
    text.encode()
  ) # You can use sha256 or any other hash function
  hash_hex = hash_object.hexdigest()

  # Convert the first 6 characters of the hex to RGB
  r = int(hash_hex[:2], 16) # red
  g = int(hash_hex[2:4], 16) # green
  b = int(hash_hex[4:6], 16) # blue
  return (r, g, b)


def saveFace(name, facePos: Any = None):
  i = 0
  path = f"./enrolled/{name}/{i}.png"
  os.makedirs(f"./enrolled/{name}", exist_ok=True)
  while os.path.exists(path):
    i += 1
    path = f"./enrolled/{name}/{i}.png"
  print("adding image for ", name, "idx: ", i)
  if facePos:
    frame_rgb_cropped = rawframe_bgr[
      max(0, facePos[1] - 15) : min(facePos[3] + 15, rawframe_bgr.shape[0]),
      max(0, facePos[0] - 15) : min(facePos[2] + 15, rawframe_bgr.shape[1]),
    ]
  else:
    frame_rgb_cropped = rawframe_bgr

  cv2.imwrite(path, frame_rgb_cropped) # Save the current frame as an image


prog = 0

peopleList: set[Any] = set()
for frameFileName in sorted_files:
  prog += 1
  if os.path.exists(os.path.join("./outFrames", frameFileName)):
    continue
  thisFramePeopleList = set()
  progress_bar(prog, maxProg, prefix="(1/2) detecting faces")
  frame = os.path.join("./frames", frameFileName)
  #   print(frame)

  frame_bgr = cv2.imread(frame)

  frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  rawframe_bgr = frame_bgr.copy()
  # detect all faces in the *full* frame
  # boxes: [[x1,y1,x2,y2], ...]
  # probs: confidence per face
  facePos = None
  faceFails = True
  while faceFails:
    faceFails = False
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
            print(e)
            continue
          label_text = "Unknown"
          color = (0, 0, 255) # red in BGR
          facePos = [x1, y1, x2, y2]
          if name is not None:
            label_text = f"{name} ({score:.2f})"
            if name not in colors:
              colors[name] = text_to_color(name)
            color = colors[name]
            thisFramePeopleList.add(name)
          else:
            faceFails = True
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
            cv2.imshow("a", frame_bgr)
            cv2.waitKey(1)
            name = input("\nwho is this? ")
            cv2.destroyAllWindows()
            if not name:
              faceFails = False
              continue
            else:
              thisFramePeopleList.add(name)
              saveFace(name, facePos)
              updateFacesList()
              continue
          if (
            enableAutoCapture
            and name
            and score
            and score < TARGET_CONFIDENCE
            and score > MATCH_THRESHOLD
          ):
            saveFace(name, facePos)
            updateFacesList()
          # draw bbox + label
          cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
          cv2.putText(
            frame_bgr,
            label_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
          )
    else:
      print("error, model not loaded")
      os._exit(-1)
  cv2.imwrite(
    f"./outFrames/{frameFileName}", frame_bgr
  ) # Save the current frame as an image
  data = []
  if thisFramePeopleList != peopleList:
    for person in thisFramePeopleList:
      if person not in peopleList:
        data.append(f'person "{person}" entered at {frameFileName}')
    for person in peopleList:
      if person not in thisFramePeopleList:
        data.append(f'person "{person}" exited at {frameFileName}')
    peopleList = thisFramePeopleList
    f.writeline("./log.txt", "\n" + "\n".join(data))


# load all people that are in the video and when in the video they are
actionList: Dict[int, List[List[Any]]] = {}
names: set[str] = set()

for line in f.read("./log.txt").split("\n"):
  if line and '"' in line and ".png" in line:
    # Extract the name
    name_match = re.search(r'"([^"]+)"', line)
    if name_match:
      name = name_match.group(1)

      # Check if "entered" is in the line
      entered = "entered" in line

      # Add name to the set
      if name not in names:
        names.add(name)

      # Extract frame number
      frame_match = re.search(r"(\d+)\.png", line)
      if frame_match:
        frame: int = int(frame_match.group(1))
        foundSameBefore = False
        if frame - 1 in actionList:
          i = 0
          for action in actionList[frame - 1]:
            if action[0] == name:
              del actionList[frame - 1][i]
              foundSameBefore = True
            i += 1
          if not actionList[frame - 1]:
            del actionList[(frame - 1)]
        # Append the [name, entered] list to actionList
        if not foundSameBefore:
          # Initialize frame list if not present
          if frame not in actionList:
            actionList[frame] = []
          actionList[frame].append([name, entered])


def findNextAction(targetFrame, name):
  for frame, actions in actionList.items():
    if frame <= targetFrame:
      continue
    found = False
    for action in actions:
      if action[0] != name:
        continue
      else:
        found = True
        break
    if found:
      return frame
  return int(sorted_files[-1].replace(".png", ""))


def rerange(val, low1, high1, low2, high2):
  return ((val - low1) / (high1 - low1)) * (high2 - low2) + low2


def toTime(frame_number):
  """Convert frame number to time in HH:MM:SS format."""
  total_seconds = frame_number / fps
  hours = int(total_seconds // 3600)
  minutes = int((total_seconds % 3600) // 60)
  seconds = int(total_seconds % 60)
  return f"{hours:02}:{minutes:02}:{seconds:02}"


output_video_file = re.sub(r"(\.[^.]+$)", " - updated\\1", video_file)

result = subprocess.run(
  ["ffmpeg", "-i", video_file],
  stderr=subprocess.PIPE,
  stdout=subprocess.PIPE,
  text=True,
  check=False,
)
# Use regex to find the frame rate
fps_match = re.search(r"(\d+(\.\d+)?)\s*fps", result.stderr)

if fps_match:
  fps = float(fps_match.group(1)) # Extract frame rate value
  print(f"Extracted frame rate: {fps} fps")
else:
  print("Frame rate could not be determined.")
  sys.exit(1)

print(fps, "fps")


activeActions: Any = {}
for name in names:
  activeActions[name] = {
    "entered": False,
    "nextActionFrame": 0,
    "nextActionTime": 0,
    "lastEndFrame": 0,
  }
  if name not in colors:
    colors[name] = text_to_color(name)
prog = 0
for frameFileName in sorted_files:
  if os.path.exists("./outFramesStep2/" + frameFileName):
    continue
  frameName = int(frameFileName.replace(".png", ""))
  prog += 1
  if frameName in actionList:
    for personName, action in activeActions.items():
      nextActionFrame = findNextAction(frameName, personName)
      activeActions[personName]["nextActionFrame"] = nextActionFrame
      activeActions[personName]["nextActionTime"] = "N/A"
      activeActions[personName]["entered"] = False
      if nextActionFrame in actionList:
        for a in actionList[nextActionFrame]:
          activeActions[a[0]]["entered"] = a[1]
          activeActions[a[0]]["nextActionFrame"] = nextActionFrame
          activeActions[a[0]]["nextActionTime"] = toTime(nextActionFrame)
          activeActions[a[0]]["lastEndFrame"] = frameName
  progress_bar(prog, maxProg, prefix="(2/2) generating progress bars")
  frame_bgr = cv2.imread(os.path.join("./outFrames", str(frameFileName)))
  y = 0
  for name, temp in activeActions.items():
    if temp["nextActionTime"] == "N/A":
      continue
    progress = int(
      (frameName - temp["lastEndFrame"])
      / (temp["nextActionFrame"] - temp["lastEndFrame"])
      * 100
    )
    y += 20
    cv2.putText(
      frame_bgr,
      name,
      (15, y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      colors[name],
      2,
    )
    cv2.rectangle(frame_bgr, (15, y + 10), (115, y + 20), (0, 0, 0), -1)
    if temp["entered"]:
      cv2.rectangle(
        frame_bgr,
        (15, y + 10),
        (int(15 + progress), y + 20),
        colors[name],
        -1,
      )
    else:
      cv2.rectangle(
        frame_bgr,
        (int(15 + progress), y + 10),
        (115, y + 20),
        colors[name],
        -1,
      )
    y += 20
    cv2.putText(
      frame_bgr,
      str(progress)
      + "%"
      + " - "
      + ("entering" if temp["entered"] else "exiting")
      + " at "
      + temp["nextActionTime"],
      (130, y),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      colors[name],
      2,
    )
  cv2.imwrite(f"./outFramesStep2/{frameFileName}", frame_bgr)


# # Replace frame numbers with converted time
# f.write(
#   "./log.txt",
#   re.sub(
#     r"(\d+)\.png",
#     lambda x: toTime(int(x.group(1))),
#     f.read("./log.txt"),
#   ),
# )

# Step 4: Construct the command to create a video from frames and retain audio
combine_command = [
  "ffmpeg",
  "-framerate",
  str(fps), # Use extracted frame rate
  "-i",
  "outFramesStep2/%020d.png", # Input frame pattern
  "-i",
  video_file, # Input original video file for audio
  "-c:v",
  "libx264", # Set video codec
  "-c:a",
  "aac", # Set audio codec
  "-b:a",
  "192k", # Set audio bitrate
  "-pix_fmt",
  "yuv420p", # Set pixel format
  "-shortest", # Ensure the video ends when the shortest stream ends
  str(output_video_file), # Output video file name
  "-y",
]


# Run the command to combine frames into a video
try:
  subprocess.run(combine_command, check=True)
  print(f"Video '{output_video_file}' created successfully.")
except subprocess.CalledProcessError as e:
  print(f"An error occurred while combining frames into a video: {e}")
