"""Optional: build data.csv from a folder of labelled hand-sign IMAGES.

Expected layout (folder name = label):
    dataset/
        A/  img1.jpg img2.jpg ...
        B/  ...
        1/  ...

Usage:
    python images_to_csv.py path/to/dataset
    python images_to_csv.py path/to/dataset --max 300 --flip

Check the dataset's license before using it. Images where no hand is found are skipped.
Use --flip if the dataset images are NOT mirrored like your webcam view
(try with and without, and keep whichever gives better live accuracy).
"""
import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp

from common import hands_to_features

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--max", type=int, default=300, help="max images per sign")
parser.add_argument("--flip", action="store_true", help="mirror images horizontally")
parser.add_argument("--out", default="data.csv")
args = parser.parse_args()

exts = {".jpg", ".jpeg", ".png", ".bmp"}
mp_hands = mp.solutions.hands
saved = {}

with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                    min_detection_confidence=0.5) as hands, \
        open(args.out, "a", newline="") as out:
    writer = csv.writer(out)
    for folder in sorted(p for p in Path(args.dataset).iterdir() if p.is_dir()):
        label = folder.name.lower() if len(folder.name) == 1 else folder.name
        saved[label] = 0
        for img_path in sorted(folder.iterdir()):
            if saved[label] >= args.max:
                break
            if img_path.suffix.lower() not in exts:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            if args.flip:
                img = cv2.flip(img, 1)
            res = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            feats = hands_to_features(res.multi_hand_landmarks)
            if feats is not None:
                writer.writerow([label] + feats.tolist())
                saved[label] += 1
        print(f"{label}: {saved[label]} samples")

print("Done. Now run: python train.py")
