"""Record signs that involve MOVEMENT (words / short phrases).

Usage:   python collect_sequences.py hello
         python collect_sequences.py good_morning
         python collect_sequences.py idle        <- hands up doing random things (not a sign)

Press SPACE to start. The script then runs REPS repetitions by itself:
  "GET READY" (hands down) -> "SIGN NOW" (do the sign) -> repeat.
ESC stops. Run it again later to add more samples to the same word.
"""
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from common import holistic_features, open_camera, prepare_sequence

if len(sys.argv) < 2:
    raise SystemExit("Usage: python collect_sequences.py <word>   e.g. python collect_sequences.py hello")

word = sys.argv[1].lower()
REPS = 25             # repetitions per session
PREP_SECONDS = 1.5    # hands down, get ready
RECORD_SECONDS = 2.5  # time window to perform the sign

folder = os.path.join("sequences", word)
os.makedirs(folder, exist_ok=True)


def saved_count():
    return len([f for f in os.listdir(folder) if f.endswith(".npy")])


mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

state, phase_start, rep, note = "wait", 0.0, 0, ""
times, feats = [], []

cap = open_camera()
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view; must match recognize_words.py
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        now = time.time()
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

        color = (0, 255, 0)
        if state == "wait":
            msg = f"'{word}': {saved_count()} saved. SPACE = record {REPS} repetitions"
            if key == 32:
                state, phase_start, rep, note = "prep", now, 0, ""
        elif state == "prep":
            msg = f"GET READY (hands down)  rep {rep + 1}/{REPS}"
            color = (0, 200, 255)
            if now - phase_start >= PREP_SECONDS:
                state, phase_start, times, feats = "record", now, [], []
        elif state == "record":
            msg = "SIGN NOW!"
            color = (0, 0, 255)
            f = holistic_features(res)
            if f is not None:
                times.append(now)
                feats.append(f)
            if now - phase_start >= RECORD_SECONDS:
                seq = prepare_sequence(times, feats)
                if seq is None:
                    note = "Hands not detected - repeating"
                else:
                    np.save(os.path.join(folder, f"{saved_count():04d}.npy"), seq)
                    rep += 1
                    note = ""
                state = "wait" if rep >= REPS else "prep"
                phase_start = now

        cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        if note:
            cv2.putText(frame, note, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Collect signs with movement", frame)

cap.release()
cv2.destroyAllWindows()
print(f"'{word}' now has {saved_count()} samples.")
