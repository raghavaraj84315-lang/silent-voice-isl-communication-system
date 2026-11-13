"""A small caption window that shows the text in English, Hindi and Kannada.

OpenCV cannot draw Hindi/Kannada properly, so the app uses this separate window (Tkinter, which
comes with Python and uses Windows' own text engine, so the letters are joined correctly).
If the window cannot be created, the app keeps working without it.
"""
try:
    import tkinter as tk
    from tkinter import font as tkfont
except Exception:                       # Tkinter missing
    tk = None

PREFERRED_FONTS = ["Nirmala UI", "Noto Sans Devanagari", "Noto Sans Kannada", "Mangal", "Tunga",
                   "Arial Unicode MS"]


class CaptionWindow:
    def __init__(self):
        self.available = False
        self.root = None
        if tk is None:
            return
        try:
            self.root = tk.Tk()
        except Exception:               # no display / Tk problem
            return
        self.root.title("Silent Voice - captions")
        self.root.geometry("820x420+40+40")
        self.root.configure(bg="#10151c")
        try:
            self.root.attributes("-topmost", True)
        except Exception:
            pass
        families = set(tkfont.families(self.root))
        name = next((f for f in PREFERRED_FONTS if f in families), None)
        big = (name, 26) if name else ("TkDefaultFont", 26)
        small = ("TkDefaultFont", 11, "bold")
        self.labels = {}
        for key, title in (("english", "English"), ("hindi", "हिन्दी  (Hindi)"), ("kannada", "ಕನ್ನಡ  (Kannada)")):
            tk.Label(self.root, text=title, font=small, fg="#7fb2ff", bg="#10151c",
                     anchor="w").pack(fill="x", padx=18, pady=(10, 0))
            label = tk.Label(self.root, text="", font=big, fg="white", bg="#10151c", anchor="w",
                             justify="left", wraplength=780)
            label.pack(fill="x", padx=18)
            self.labels[key] = label
        self.available = True
        self.pump()

    def show(self, english, hindi, kannada):
        if not self.available:
            return
        for key, value in (("english", english), ("hindi", hindi), ("kannada", kannada)):
            self.labels[key].config(text=value)

    def pump(self):
        """Call once per camera frame so the window stays responsive."""
        if not self.available:
            return
        try:
            self.root.update()
        except Exception:               # window was closed
            self.available = False

    def close(self):
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
        self.available = False
