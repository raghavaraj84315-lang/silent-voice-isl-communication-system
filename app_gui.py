"""Silent Voice - ONE window: camera, recognized text and language choice together.

The text is recognized in English. Pick Hindi or Kannada from the language box (or press 1 / 2 / 3)
and the same text is shown in that language right away. The default is English.

Keys:  L = letters   W = words   TAB = switch mode   1 / 2 / 3 = English / Hindi / Kannada
       SPACE = add space   BACKSPACE = delete (a letter, or a whole word in words mode)
       C = clear   ESC = quit
Needs model.pkl (letters, from train.py) and/or words_model.pkl (words, from train_words.py).
"""
import os
import time
import traceback
import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

import cv2
import joblib

from common import open_camera
from engine import Session
from translate import load_translations, translate
from tts import speak, stop

try:
    from PIL import Image, ImageTk
except Exception:                      # Pillow missing: fall back to Tk's own image format
    Image = ImageTk = None

BG, PANEL, TEXT_BG = "#070a13", "#0e1422", "#0b101c"
ACCENT, FG, MUTED = "#0284c7", "#ffffff", "#94a3b8"
LANGUAGES = [("English", "english"), ("हिन्दी  (Hindi)", "hindi"), ("ಕನ್ನಡ  (Kannada)", "kannada")]
PREFERRED_FONTS = ["Nirmala UI", "Noto Sans Devanagari", "Noto Sans Kannada", "Mangal", "Tunga",
                   "Arial Unicode MS"]


class App:
    def __init__(self):
        letters_model = joblib.load("model.pkl") if os.path.exists("model.pkl") else None
        words_model = joblib.load("words_model.pkl") if os.path.exists("words_model.pkl") else None
        if letters_model is None and words_model is None:
            raise SystemExit("No models found. Run train.py (letters) and/or train_words.py (words) first.")
        self.table = load_translations()
        if not self.table:
            print("translations.csv not found - only English will be available.")
        self.session = Session(letters_model, words_model)
        self.language = "english"
        self.shown = None
        self.running = True

        self.root = tk.Tk()
        self.root.title("Silent Voice - Sign Language Translation")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)
        try:
            if os.path.exists("static/favicon.png"):
                icon_img = tk.PhotoImage(file="static/favicon.png")
                self.root.iconphoto(False, icon_img)
        except Exception:
            pass
        families = set(tkfont.families(self.root))
        script_font = next((f for f in PREFERRED_FONTS if f in families), "TkDefaultFont")
        self.text_font = (script_font, 22)
        self.ui_font = ("Segoe UI" if "Segoe UI" in families else "TkDefaultFont", 11)
        self.ui_bold = (self.ui_font[0], 11, "bold")
        self._build_ui()

        self.cap = open_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Key>", self.on_key)
        self.root.focus_set()
        self.tick()

    # UI layout
    def _build_ui(self):
        left = tk.Frame(self.root, bg=BG)
        left.grid(row=0, column=0, padx=(12, 6), pady=12, sticky="n")
        self.blank = tk.PhotoImage(width=640, height=480)      # black placeholder until the first frame
        self.video = tk.Label(left, image=self.blank, bg="black", borderwidth=0)
        self.video.pack()

        side = tk.Frame(self.root, bg=PANEL)
        side.grid(row=0, column=1, padx=(6, 12), pady=12, sticky="n")

        tk.Label(side, text="Mode", font=self.ui_bold, fg=MUTED, bg=PANEL).grid(
            row=0, column=0, sticky="w", padx=14, pady=(14, 2))
        modes = tk.Frame(side, bg=PANEL)
        modes.grid(row=1, column=0, sticky="w", padx=14)
        self.mode_var = tk.StringVar(value=self.session.mode)
        for label, value in (("Letters  (L)", "letters"), ("Words  (W)", "words")):
            button = tk.Radiobutton(
                modes, text=label, value=value, variable=self.mode_var, indicatoron=False,
                font=self.ui_font, fg=FG, bg="#27324a", selectcolor=ACCENT, activebackground=ACCENT,
                activeforeground=FG, padx=14, pady=6, takefocus=0, command=self.on_mode_button,
                state="normal" if value in self.session.available_modes() else "disabled")
            button.pack(side="left", padx=(0, 8))

        tk.Label(side, text="Language", font=self.ui_bold, fg=MUTED, bg=PANEL).grid(
            row=2, column=0, sticky="w", padx=14, pady=(14, 2))
        self.lang_var = tk.StringVar(value=LANGUAGES[0][0])
        names = [n for n, _ in LANGUAGES] if self.table else [LANGUAGES[0][0]]
        self.lang_box = ttk.Combobox(side, textvariable=self.lang_var, values=names, state="readonly",
                                     width=24, font=self.text_font[:1] + (14,), takefocus=0)
        self.lang_box.grid(row=3, column=0, sticky="w", padx=14)
        self.lang_box.bind("<<ComboboxSelected>>", self.on_language_box)

        tk.Label(side, text="Detected now", font=self.ui_bold, fg=MUTED, bg=PANEL).grid(
            row=4, column=0, sticky="w", padx=14, pady=(14, 0))
        self.status_var = tk.StringVar(value="")
        tk.Label(side, textvariable=self.status_var, font=(self.ui_font[0], 18, "bold"), fg="#4cd137",
                 bg=PANEL, anchor="w").grid(row=5, column=0, sticky="w", padx=14)
        self.guess_var = tk.StringVar(value="")
        tk.Label(side, textvariable=self.guess_var, font=(self.ui_font[0], 9), fg="#e1b12c", bg=PANEL,
                 anchor="w").grid(row=6, column=0, sticky="w", padx=14)

        tk.Label(side, text="Text", font=self.ui_bold, fg=MUTED, bg=PANEL).grid(
            row=7, column=0, sticky="w", padx=14, pady=(12, 2))
        self.text_label = tk.Label(side, text="", font=self.text_font, fg=FG, bg=TEXT_BG, anchor="nw",
                                   justify="left", wraplength=400, width=21, height=4, padx=8, pady=6)
        self.text_label.grid(row=8, column=0, padx=14, sticky="w")

        buttons = tk.Frame(side, bg=PANEL)
        buttons.grid(row=9, column=0, sticky="w", padx=14, pady=(10, 0))
        for label, command, color in (("Space", self.session.add_space, "#27324a"),
                                      ("Backspace", self.session.backspace, "#27324a"),
                                      ("Clear", self.session.clear, "#27324a"),
                                      ("Speak (S)", self.speak_current, "#20bf6b")):
            tk.Button(buttons, text=label, font=self.ui_font, fg=FG, bg=color, activebackground=ACCENT,
                      activeforeground=FG, padx=10, pady=4, takefocus=0, relief="flat",
                      command=lambda c=command: (c(), self.root.focus_set())).pack(side="left", padx=(0, 6))

        tts_frame = tk.Frame(side, bg=PANEL)
        tts_frame.grid(row=10, column=0, sticky="w", padx=14, pady=(8, 0))
        self.auto_speak_var = tk.BooleanVar(value=True)
        tk.Checkbutton(tts_frame, text="Auto-speak recognized signs", variable=self.auto_speak_var,
                       font=(self.ui_font[0], 9), fg=FG, bg=PANEL, selectcolor="#27324a",
                       activebackground=PANEL, activeforeground=FG, takefocus=0,
                       command=lambda: self.root.focus_set()).pack(side="left")

        tk.Label(side, text="Keys: L / W mode   1 2 3 language   S speak   SPACE   BACKSPACE   C   ESC",
                 font=(self.ui_font[0], 8), fg=MUTED, bg=PANEL).grid(row=11, column=0, sticky="w",
                                                                      padx=14, pady=(8, 0))

    # Actions
    def speak_current(self):
        current_text = translate(self.session.text, self.language, self.table)
        if current_text and current_text.strip():
            speak(current_text, self.language, interrupt=True)
        self.root.focus_set()

    def set_language(self, key):
        names = {k: n for n, k in LANGUAGES}
        if key != "english" and not self.table:
            return
        self.language = key
        self.lang_var.set(names[key])
        self.shown = None                   # redraw the text in the new language

    def on_language_box(self, _event=None):
        key = dict(LANGUAGES).get(self.lang_var.get(), "english")
        self.set_language(key)
        self.root.focus_set()

    def on_mode_button(self):
        self.session.set_mode(self.mode_var.get())
        self.root.focus_set()

    def on_key(self, event):
        key = (event.keysym or "").lower()
        if key == "escape":
            self.close()
        elif key == "tab":
            self.session.toggle_mode()
        elif key == "l":
            self.session.set_mode("letters")
        elif key == "w":
            self.session.set_mode("words")
        elif key == "space":
            self.session.add_space()
        elif key == "backspace":
            self.session.backspace()
        elif key == "c":
            self.session.clear()
        elif key == "s":
            self.speak_current()
        elif key in ("1", "2", "3"):
            self.set_language(LANGUAGES[int(key) - 1][1])
        self.mode_var.set(self.session.mode)
        return "break"

    # Frame display loop
    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if Image is not None:
            self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        else:
            h, w = rgb.shape[:2]
            self.photo = tk.PhotoImage(width=w, height=h, data=b"P6 %d %d 255\n" % (w, h) + rgb.tobytes(),
                                       format="PPM")
        self.video.configure(image=self.photo)

    def tick(self):
        if not self.running:
            return
        try:
            ok, frame = self.cap.read()
            if ok:
                frame, status, top3 = self.session.process(frame, time.time())
                if self.session.last_accepted and self.auto_speak_var.get():
                    word = self.session.last_accepted.replace("_", " ")
                    if self.language != "english":
                        word = translate(word, self.language, self.table)
                    speak(word, self.language, interrupt=False)
                self.show_frame(frame)
                self.status_var.set(status)
                self.guess_var.set(f"top guesses: {top3}" if top3 else "")
                self.render_text()
            else:
                self.status_var.set("The camera is not giving pictures")
        except Exception:
            traceback.print_exc()
            self.status_var.set("Error - see the terminal window")
            self.running = False
            return
        self.root.after(5, self.tick)

    def render_text(self):
        marker = (self.session.text, self.language)
        if marker != self.shown:
            shown = translate(self.session.text, self.language, self.table)
            self.text_label.configure(text=shown if shown else "")
            self.shown = marker

    def close(self):
        self.running = False
        try:
            stop()
            self.session.close()
            self.cap.release()
        finally:
            self.root.destroy()


def main():
    App().root.mainloop()


if __name__ == "__main__":
    main()
