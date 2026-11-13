"""Step 3: live sign -> text.

Hold a sign steady for ~HOLD_FRAMES frames and the letter is added to the text.
Keys: SPACE = add space | BACKSPACE = delete | c = clear | ESC = quit
"""
import cv2
import joblib
import mediapipe as mp

from common import hands_to_features

HOLD_FRAMES = 20        # how long a sign must be held to be accepted
MIN_CONFIDENCE = 0.6    # ignore uncertain predictions

model = joblib.load("model.pkl")
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

text = ""
current, streak, armed = None, 0, True

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6,
                    min_tracking_confidence=0.6) as hands:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        result = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        pred, conf = None, 0.0
        features = hands_to_features(result.multi_hand_landmarks)
        if features is not None:
            for h in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
            probs = model.predict_proba([features])[0]
            best = probs.argmax()
            if probs[best] >= MIN_CONFIDENCE:
                pred, conf = model.classes_[best], probs[best]

        # Accept a letter only after it has been held steadily
        if pred is not None and pred == current:
            streak += 1
        else:
            current, streak, armed = pred, 1 if pred is not None else 0, True
        if pred is not None and streak >= HOLD_FRAMES and armed:
            text += pred
            armed = False  # must change sign or drop hand before repeating

        if pred:
            cv2.putText(frame, f"{pred} ({conf:.0%})", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]),
                      (0, 0, 0), -1)
        cv2.putText(frame, text[-40:], (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Sign to text", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == 32:
            text += " "
        elif key == 8:
            text = text[:-1]
        elif key == ord("c"):
            text = ""

cap.release()
cv2.destroyAllWindows()