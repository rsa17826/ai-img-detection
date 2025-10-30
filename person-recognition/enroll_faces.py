# enroll_faces.py
import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1
from typing import Any


def init(log, setProg):
  ENROLL_DIR = "enrolled"
  DB_PATH = "data/embeddings_db.npz"

  # 1. Load face detector + embedder
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)
  resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

  all_embeddings: Any = []
  all_labels: Any = []
  prog = 0
  maxProg = len(os.listdir(ENROLL_DIR))
  for person_name in os.listdir(ENROLL_DIR):
    person_folder = os.path.join(ENROLL_DIR, person_name)
    prog += 1
    setProg(prog, maxProg, person_name)
    if not os.path.isdir(person_folder):
      continue

    log(f"[INFO] Processing {person_name}")
    for img_file in os.listdir(person_folder):
      img_path = os.path.join(person_folder, img_file)

      # read image with cv2 (BGR -> RGB)
      img_bgr = cv2.imread(img_path)
      if img_bgr is None:
        log(f"[WARN] Could not read {img_path}")
        continue
      img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

      # 2. Detect & crop face (returns PIL Image or None)
      face = mtcnn(img_rgb)
      if face is None:
        log(f"[WARN] No face found in {img_path}")
        os.remove(img_path)
        continue

      # face is a torch tensor [3,160,160]
      face = face.unsqueeze(0).to(device) # [1,3,160,160]

      # 3. Get embedding (512-d vector)
      with torch.no_grad():
        emb = resnet(face) # [1,512]
      emb = emb.squeeze(0).cpu().numpy() # [512]

      all_embeddings.append(emb)
      all_labels.append(person_name)

  # convert to arrays
  all_embeddings = np.array(all_embeddings)
  all_labels = np.array(all_labels)

  # log(f"[INFO] Total faces enrolled: {len(all_labels)}")

  os.makedirs("data", exist_ok=True)
  np.savez(DB_PATH, embeddings=all_embeddings, labels=all_labels)

  # log(f"[INFO] Saved database to {DB_PATH}")
