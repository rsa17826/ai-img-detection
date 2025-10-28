import cv2
import torch
import numpy as np
import time
import pandas as pd
from facenet_pytorch import MTCNN, InceptionResnetV1
import os
from typing import Any
import enroll_faces
print("changing dir to ", os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.makedirs("data", exist_ok=True)
os.makedirs("enrolled/personname", exist_ok=True)
enroll_faces.init()

DB_PATH = "data/embeddings_db.npz"
ATTEND_LOG = "data/attendance_log.csv"

# threshold tuning:
# cosine similarity ranges roughly -1 to 1
# same-person pairs are usually high (e.g. 0.6-0.9+),
# different people are lower. You will tune this.
MATCH_THRESHOLD = 0.6

# debounce (seconds) to avoid re-logging same person constantly
DEBOUNCE_SECONDS = 60

# load embeddings db
db = np.load(DB_PATH)
known_embeddings = db["embeddings"] # shape (N,512)
known_labels = db["labels"] # shape (N,)


# normalize known vectors for cosine sim
def l2norm(x):
  return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)


known_norm = l2norm(known_embeddings)

# init attendance memory
last_seen: dict[Any, Any] = {} # name -> unix timestamp

# prepare attendance log file if missing

if not os.path.exists(ATTEND_LOG):
  df = pd.DataFrame(columns=["timestamp", "name"])
  df.to_csv(ATTEND_LOG, index=False)

# load models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


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


cap = cv2.VideoCapture(1)

print("[INFO] Starting camera. Press 'q' to quit.")
while True:
  ret, frame_bgr = cap.read()
  if not ret:
    print("[WARN] Could not read from camera.")
    break

  frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

  # detect all faces in the *full* frame
  # boxes: [[x1,y1,x2,y2], ...]
  # probs: confidence per face
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

      emb = get_embedding(face_crop_rgb)
      if emb is None:
        continue

      name, score = match_identity(emb)
      label_text = "Unknown"
      color = (0, 0, 255) # red in BGR
      if name is not None:
        label_text = f"{name} ({score:.2f})"
        color = (0, 255, 0) # green

        now = time.time()
        last_time = last_seen.get(name, 0)
        if (now - last_time) > DEBOUNCE_SECONDS:
          # log attendance
          last_seen[name] = now
          ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
          print(f"[LOG] {ts_str} - {name} present")

          # append to CSV
          df_row = pd.DataFrame([{"timestamp": ts_str, "name": name}])
          df_row.to_csv(ATTEND_LOG, mode="a", header=False, index=False)

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

  cv2.imshow("Attendance Camera", frame_bgr)
  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap.release()
cv2.destroyAllWindows()
