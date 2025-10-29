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

print("changing dir to ", os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Index of the camera to use
capidx = 0
os.makedirs("data", exist_ok=True)
os.makedirs("enrolled", exist_ok=True)
os.makedirs("frames", exist_ok=True)
# threshold tuning:
# cosine similarity ranges roughly -1 to 1
# same-person pairs are usually high (e.g. 0.6-0.9+),
# different people are lower. You will tune this.
MATCH_THRESHOLD = 0.6
TARGET_CONFIDENCE = 0.7


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


# init attendance memory
last_seen: dict[Any, Any] = {} # name -> unix timestamp
# prepare attendance print file if missing
mtcnn: Any = None


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


print(sys.argv)
# subprocess.run("ffmpeg vidToFrames " + sys.argv[1])
sorted_files = sorted(os.listdir("./frames"), key=lambda x: int(x.split(".")[0]))
maxProg = len(sorted_files)
prog = 0
for frame in sorted_files:
  prog += 1
  print(str(int(prog / maxProg * 100)) + "%")
  frame = os.path.join("./frames", frame)
  print(frame)

  frame_bgr = cv2.imread(frame)

  frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
  rawframe_bgr = frame_bgr.copy()
  # detect all faces in the *full* frame
  # boxes: [[x1,y1,x2,y2], ...]
  # probs: confidence per face
  facePos = None
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
        if name is not None:
          label_text = f"{name} ({score:.2f})"
          color = (0, 255, 0) # green

          now = time.time()
          last_time = last_seen.get(name, 0)
          # print attendance
          last_seen[name] = now
          ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
          print(f"[print] {ts_str} - {name} present")

          # append to CSV
          print([{"timestamp": ts_str, "name": name}])
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
  if frame_bgr is None:
    print(f"Error loading image: {frame}")
  else:
    frame_bgr = cv2.flip(frame_bgr, 1) # Flip the frame for a mirror effect
    cv2.imshow("a", frame_bgr)
    cv2.waitKey(100)
