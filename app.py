"""Silent Voice - one app for letters AND words.

Keys:
    L = letters mode (hold a letter sign steady for about a second)
    W = words mode   (sign a word, then LOWER YOUR HANDS; sign the next word)
    TAB = switch mode | SPACE = add space | BACKSPACE = delete (a letter, or a whole word in words mode)
    C = clear the text | ESC = quit

Needs model.pkl (letters, from train.py) and/or words_model.pkl (words, from train_words.py).
A second small window shows the text in English, Hindi and Kannada (translations.csv).
Only the tracker of the active mode runs, so it is as fast as the separate programs.
"""
import os
import time

import cv2
import joblib
import mediapipe as mp

from captions import CaptionWindow
from common import holistic_features, open_camera
from engine import LetterMode, WordMode, add_letter, add_space, add_word, delete_last
from translate import load_translations, translate
from tts import speak, stop


def main():
    letters_model = joblib.load("model.pkl") if os.path.exists("model.pkl") else None
    words_model = joblib.load("words_model.pkl") if os.path.exists("words_model.pkl") else None
    if letters_model is None and words_model is None:
        raise SystemExit("No models found. Run train.py (letters) and/or train_words.py (words) first.")
    if letters_model is None:
        print("model.pkl not found - letters mode is off.")
    if words_model is None:
        print("words_model.pkl not found - words mode is off.")

    letter_mode = LetterMode(letters_model) if letters_model else None
    word_mode = WordMode(words_model) if words_model else None
    mode = "letters" if letter_mode else "words"

    mp_hands, mp_holistic, mp_draw = mp.solutions.hands, mp.solutions.holistic, mp.solutions.drawing_utils
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) \
        if letter_mode else None
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) \
        if word_mode else None

    table = load_translations()
    if not table:
        print("translations.csv not found - the caption window will show English only.")
    captions = CaptionWindow()
    if not captions.available:
        print("Caption window could not be opened - continuing without it.")
    shown = None

    text = ""
    cap = open_camera()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()

            label, top3 = None, ""
            if mode == "letters":
                result = hands.process(rgb)
                for h in result.multi_hand_landmarks or []:
                    mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
                pred, conf, top3, accepted = letter_mode.update(result.multi_hand_landmarks)
                if accepted:
                    text = add_letter(text, accepted)
                    speak(accepted, "english")
                status = f"{pred.upper()} ({conf:.0%})" if pred else "Show a letter sign"
            else:
                result = holistic.process(rgb)
                mp_draw.draw_landmarks(frame, result.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_draw.draw_landmarks(frame, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_draw.draw_landmarks(frame, result.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                status, top3, accepted = word_mode.update(holistic_features(result), now)
                if accepted:
                    text = add_word(text, accepted)
                    speak(accepted.replace("_", " "), "english")

            banner = f"MODE: {mode.upper()}   (L=letters, W=words, 1/2/3=speak EN/HI/KN)"
            cv2.putText(frame, banner, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(frame, status, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"top guesses: {top3 or '-'}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
            cv2.rectangle(frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(frame, text[-45:], (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            cv2.imshow("Silent Voice", frame)

            key = cv2.waitKey(1) & 0xFF
            wanted = mode
            if key == 27:
                break
            elif key == 9:
                wanted = "words" if mode == "letters" else "letters"
            elif key in (ord("l"), ord("L")):
                wanted = "letters"
            elif key in (ord("w"), ord("W")):
                wanted = "words"
            elif key == 32:
                text = add_space(text)
            elif key == 8:
                text = delete_last(text, mode)
            elif key in (ord("c"), ord("C")):
                text = ""
            elif key in (ord("s"), ord("S"), ord("1")):
                if text.strip():
                    speak(text, "english", interrupt=True)
            elif key == ord("2"):
                hi_text = translate(text, "hindi", table)
                if hi_text.strip():
                    speak(hi_text, "hindi", interrupt=True)
            elif key == ord("3"):
                kn_text = translate(text, "kannada", table)
                if kn_text.strip():
                    speak(kn_text, "kannada", interrupt=True)

            if text != shown:                                  # keep the caption window up to date
                captions.show(text, translate(text, "hindi", table), translate(text, "kannada", table))
                shown = text
            captions.pump()

            if wanted != mode:
                if (wanted == "letters" and letter_mode) or (wanted == "words" and word_mode):
                    mode = wanted
                    if letter_mode:
                        letter_mode.reset()
                    if word_mode:
                        word_mode.reset()
    finally:
        stop()
        captions.close()
        cap.release()
        cv2.destroyAllWindows()
        if hands:
            hands.close()
        if holistic:
            holistic.close()


if __name__ == "__main__":
    main()
