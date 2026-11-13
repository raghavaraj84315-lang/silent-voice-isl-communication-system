"""Step 1: record training samples (auto-capture version).

Show a sign to the webcam so the green hand skeleton appears, then press the key
for that sign (e.g. 'a'). The script then records SAMPLES_PER_SIGN samples by itself
while you slowly move/turn your hand a little. ESC quits.
"""
import csv
import os
from collections import Counter

import cv2
import mediapipe as mp

from common import hands_to_features

DATA_FILE = "data.csv"
SAMPLES_PER_SIGN = 150   # samples recorded per key press
SKIP_FRAMES = 3          # save every 3rd frame so samples are not near-identical

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

counts = Counter()
if os.path.exists(DATA_FILE):
    with open(DATA_FILE) as f:
        for row in csv.reader(f):
            if row:
                counts[row[0]] += 1

recording_label, remaining, frame_no = None, 0, 0

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6) as hands, \
        open(DATA_FILE, "a", newline="") as out:
    writer = csv.writer(out)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view; must match recognize.py
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        features = hands_to_features(result.multi_hand_landmarks)
        for h in result.multi_hand_landmarks or []:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if recording_label is None and 32 < key < 127:
            recording_label, remaining, frame_no = chr(key), SAMPLES_PER_SIGN, 0

        if recording_label is not None and features is not None:
            frame_no += 1
            if frame_no % SKIP_FRAMES == 0:
                writer.writerow([recording_label] + features.tolist())
                out.flush()
                counts[recording_label] += 1
                remaining -= 1
                if remaining <= 0:
                    recording_label = None

        if recording_label:
            status = f"RECORDING '{recording_label}': {SAMPLES_PER_SIGN - remaining}/{SAMPLES_PER_SIGN} - move your hand slowly"
            color = (0, 0, 255)
        else:
            status = "Show a sign, then press its key to auto-record | ESC = quit"
            color = (0, 255, 0)
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        summary = "  ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        cv2.putText(frame, summary[-70:], (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.imshow("Collect data", frame)

cap.release()
cv2.destroyAllWindows()
