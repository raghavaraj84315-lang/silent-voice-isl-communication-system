"""The recognition engine shared by the apps (no windows in here).

- LetterMode : hold a letter sign steady and it is accepted once
- WordMode   : hands up = a sign starts, hands down = it ends and is recognized
- Session    : owns the trackers + text; call session.process(frame) for every camera frame
"""
import cv2

from common import hands_to_features, hands_visible, holistic_features, prepare_sequence

HOLD_FRAMES = 20          # letters: frames a sign must be held
LETTER_MIN_CONF = 0.4
WORD_MIN_CONF = 0.5
WORD_END_GAP = 0.7        # words: seconds with hands down = the sign has ended
WORD_MAX_SEGMENT = 5.0


def top3_text(model, probs):
    order = probs.argsort()[::-1][:3]
    return "  ".join(f"{model.classes_[i]}:{probs[i]:.0%}" for i in order)


class LetterMode:
    """Hold a letter sign steady and it is accepted once."""

    def __init__(self, model):
        self.model = model
        self.current, self.streak, self.armed = None, 0, True

    def reset(self):
        self.current, self.streak, self.armed = None, 0, True

    def update(self, hand_landmarks):
        """-> (label, confidence, top3 text, accepted letter or None)"""
        pred, conf, top3 = None, 0.0, ""
        features = hands_to_features(hand_landmarks)
        if features is not None:
            probs = self.model.predict_proba([features])[0]
            best = int(probs.argmax())
            top3 = top3_text(self.model, probs)
            if probs[best] >= LETTER_MIN_CONF:
                pred, conf = str(self.model.classes_[best]), float(probs[best])

        accepted = None
        if pred is not None and pred == self.current:
            self.streak += 1
        else:
            self.current, self.streak, self.armed = pred, (1 if pred is not None else 0), True
        if pred is not None and self.streak >= HOLD_FRAMES and self.armed:
            accepted = pred
            self.armed = False       # change sign or drop the hand before the same letter repeats
        return pred, conf, top3, accepted


class WordMode:
    """Hands up = a sign starts, hands down for a moment = the sign ends and is recognized."""

    def __init__(self, model):
        self.model = model
        self.reset()
        self.info, self.top3 = "Raise your hands and sign", ""

    def reset(self):
        self.active, self.t, self.f = False, [], []
        self.seg_start = self.last_hand = 0.0

    def update(self, features, now):
        """features = holistic_features(...) or None -> (status text, top3 text, accepted word or None)"""
        if features is not None and hands_visible(features):
            if not self.active:
                self.active, self.t, self.f, self.seg_start = True, [], [], now
            self.last_hand = now

        accepted = None
        if self.active:
            if features is not None:
                self.t.append(now)
                self.f.append(features)
            if now - self.last_hand >= WORD_END_GAP or now - self.seg_start >= WORD_MAX_SEGMENT:
                self.active = False
                seq = prepare_sequence(self.t, self.f)
                if seq is None:
                    self.info = "Too short - try again"
                else:
                    probs = self.model.predict_proba([seq.flatten()])[0]
                    best = int(probs.argmax())
                    word, conf = str(self.model.classes_[best]), float(probs[best])
                    self.top3 = top3_text(self.model, probs)
                    if word == "idle":
                        self.info = "(no sign)"
                    elif conf >= WORD_MIN_CONF:
                        accepted = word
                        self.info = f"{word} ({conf:.0%})"
                    else:
                        self.info = f"not sure ({word}? {conf:.0%})"
        status = "Signing..." if self.active else self.info
        return status, self.top3, accepted


def add_letter(text, letter):
    return text + letter


def add_word(text, word):
    spoken = word.replace("_", " ")
    if text and not text.endswith(" "):
        text += " "
    return text + spoken + " "


def add_space(text):
    return text if (not text or text.endswith(" ")) else text + " "


def delete_last(text, mode):
    if mode == "words":
        text = text.rstrip()
        return text[: text.rfind(" ") + 1] if " " in text else ""
    return text[:-1]


class Session:
    """Camera frame in -> annotated frame, status line and the growing text out."""

    def __init__(self, letters_model=None, words_model=None):
        import mediapipe as mp
        self.mp = mp
        self.letter_mode = LetterMode(letters_model) if letters_model else None
        self.word_mode = WordMode(words_model) if words_model else None
        if self.letter_mode is None and self.word_mode is None:
            raise ValueError("Need at least one model (model.pkl or words_model.pkl).")
        self.mode = "letters" if self.letter_mode else "words"
        self.text = ""
        self.last_accepted = None
        self.last_holistic_features = None
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6) \
            if self.letter_mode else None
        self.holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5) if self.word_mode else None

    def available_modes(self):
        return [m for m, ok in (("letters", self.letter_mode), ("words", self.word_mode)) if ok]

    def set_mode(self, mode):
        if mode == self.mode or mode not in self.available_modes():
            return False
        self.mode = mode
        for m in (self.letter_mode, self.word_mode):
            if m:
                m.reset()
        return True

    def toggle_mode(self):
        modes = self.available_modes()
        return self.set_mode(modes[(modes.index(self.mode) + 1) % len(modes)]) if len(modes) > 1 else False

    def add_space(self):
        self.text = add_space(self.text)

    def backspace(self):
        self.text = delete_last(self.text, self.mode)

    def clear(self):
        self.text = ""

    def process(self, frame, now, mirror=False):
        """-> (annotated BGR frame, status text, top guesses text)"""
        if mirror:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        draw = self.mp.solutions.drawing_utils
        if self.mode == "letters":
            conns = self.mp.solutions.hands.HAND_CONNECTIONS
            result = self.hands.process(rgb)
            for h in result.multi_hand_landmarks or []:
                draw.draw_landmarks(frame, h, conns)
            pred, conf, top3, accepted = self.letter_mode.update(result.multi_hand_landmarks)
            if accepted:
                self.text = add_letter(self.text, accepted)
            status = f"{pred.upper()} ({conf:.0%})" if pred else "Show a letter sign"
        else:
            holo = self.mp.solutions.holistic
            result = self.holistic.process(rgb)
            draw.draw_landmarks(frame, result.pose_landmarks, holo.POSE_CONNECTIONS)
            draw.draw_landmarks(frame, result.left_hand_landmarks, holo.HAND_CONNECTIONS)
            draw.draw_landmarks(frame, result.right_hand_landmarks, holo.HAND_CONNECTIONS)
            feats = holistic_features(result)
            self.last_holistic_features = feats
            status, top3, accepted = self.word_mode.update(feats, now)
            if accepted:
                self.text = add_word(self.text, accepted)
        self.last_accepted = accepted
        return frame, status, top3

    def close(self):
        for tracker in (self.hands, self.holistic):
            if tracker:
                tracker.close()
