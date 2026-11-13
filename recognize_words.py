"""Live sign -> sentence.

Sign one word, LOWER YOUR HANDS, sign the next word. Each word is added to the sentence.
Keys: BACKSPACE = delete last word | c = clear | ESC = quit
"""
import time

import cv2
import joblib
import mediapipe as mp

from common import holistic_features, hands_visible, open_camera, prepare_sequence

MIN_CONF = 0.5       # ignore uncertain predictions
END_GAP = 0.7        # seconds with hands down = the sign has ended
MAX_SEGMENT = 5.0    # safety limit for one sign (seconds)

model = joblib.load("words_model.pkl")
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

sentence, last_info, last_top3 = [], "Raise your hands and sign", ""
active, seg_t, seg_f, seg_start, last_hand = False, [], [], 0.0, 0.0

cap = open_camera()
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        mp_draw.draw_landmarks(frame, res.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_draw.draw_landmarks(frame, res.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_draw.draw_landmarks(frame, res.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        now = time.time()
        f = holistic_features(res)
        if f is not None and hands_visible(f):
            if not active:
                active, seg_t, seg_f, seg_start = True, [], [], now
            last_hand = now

        if active:
            if f is not None:
                seg_t.append(now)
                seg_f.append(f)
            if now - last_hand >= END_GAP or now - seg_start >= MAX_SEGMENT:
                active = False
                seq = prepare_sequence(seg_t, seg_f)
                if seq is None:
                    last_info = "Too short - try again"
                else:
                    probs = model.predict_proba([seq.flatten()])[0]
                    best = probs.argmax()
                    last_top3 = "  ".join(f"{model.classes_[i]}:{probs[i]:.0%}"
                                          for i in probs.argsort()[::-1][:3])
                    word, conf = model.classes_[best], probs[best]
                    if word == "idle":
                        last_info = "(no sign)"
                    elif conf >= MIN_CONF:
                        sentence.append(word.replace("_", " "))
                        last_info = f"{word} ({conf:.0%})"
                    else:
                        last_info = f"not sure ({word}? {conf:.0%})"

        status = "Signing..." if active else last_info
        cv2.putText(frame, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (0, 0, 255) if active else (0, 255, 0), 2)
        cv2.putText(frame, f"last guesses: {last_top3 or '-'}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(frame, " ".join(sentence)[-55:], (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.imshow("Sign to sentence", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 8 and sentence:
            sentence.pop()
        elif key == ord("c"):
            sentence = []

cap.release()
cv2.destroyAllWindows()
