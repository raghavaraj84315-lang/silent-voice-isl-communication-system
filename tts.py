"""Text-to-Speech (TTS) module for Silent Voice.

Supports speech synthesis in English, Hindi, and Kannada.
Features:
- Non-blocking background worker thread (camera feed and UI never stutter)
- Local disk caching with safe ASCII filenames (zero-latency replay, 100% offline once cached)
- Pre-caching support for all vocabulary in translations.csv
- Safe fallback for English via pyttsx3/SAPI5
"""
import hashlib
import os
import queue
import re
import threading
import time

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AUDIO = True
except Exception as e:
    PYGAME_AUDIO = False
    print(f"Warning: pygame audio could not be initialized: {e}")

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

LANG_MAP = {
    "english": "en",
    "en": "en",
    "hindi": "hi",
    "hi": "hi",
    "kannada": "kn",
    "kn": "kn",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "audio_cache")


def _safe_ascii(s):
    try:
        return str(s).encode("ascii", errors="replace").decode("ascii")
    except Exception:
        return "..."


class TextToSpeech:
    def __init__(self, cache_dir=CACHE_DIR):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.queue = queue.Queue()
        self.current_playing = False
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()

        # Offline fallback for English
        self._pyttsx3_engine = None
        if pyttsx3 is not None:
            try:
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty("rate", 150)
            except Exception:
                self._pyttsx3_engine = None

        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _get_cache_path(self, text, lang_code):
        text_hash = hashlib.sha1(f"{lang_code}:{text}".encode("utf-8")).hexdigest()[:12]
        clean_slug = re.sub(r"[^a-zA-Z0-9_-]", "", text.strip().lower())
        if clean_slug:
            filename = f"{lang_code}_{clean_slug[:16]}_{text_hash}.mp3"
        else:
            filename = f"{lang_code}_{text_hash}.mp3"
        return os.path.join(self.cache_dir, filename)

    def synthesize(self, text, lang="english"):
        """Generate audio file for the text if not already cached. Returns file path or None."""
        text = text.strip()
        if not text:
            return None

        lang_code = LANG_MAP.get(lang.lower(), "en")
        cache_path = self._get_cache_path(text, lang_code)

        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
            return cache_path

        # Generate via gTTS
        if gTTS is not None:
            try:
                tts = gTTS(text=text, lang=lang_code, slow=False)
                temp_path = cache_path + ".tmp"
                tts.save(temp_path)
                if os.path.exists(cache_path):
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
                os.rename(temp_path, cache_path)
                return cache_path
            except Exception as e:
                print(f"[TTS] gTTS error for '{_safe_ascii(text)}' ({lang_code}): {e}")

        # English offline fallback via pyttsx3
        if lang_code == "en" and self._pyttsx3_engine is not None:
            wav_path = cache_path.replace(".mp3", ".wav")
            try:
                self._pyttsx3_engine.save_to_file(text, wav_path)
                self._pyttsx3_engine.runAndWait()
                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                    return wav_path
            except Exception as e:
                print(f"[TTS] pyttsx3 fallback error: {e}")

        return None

    def _play_file(self, filepath):
        if not PYGAME_AUDIO or not filepath or not os.path.exists(filepath):
            return

        with self._lock:
            try:
                pygame.mixer.music.load(filepath)
                pygame.mixer.music.play()
                self.current_playing = True
            except Exception as e:
                print(f"[TTS] Playback error: {e}")
                self.current_playing = False
                return

        while pygame.mixer.music.get_busy() and not self._stop_flag.is_set():
            time.sleep(0.04)

        try:
            pygame.mixer.music.unload()
        except Exception:
            pass

        self.current_playing = False

    def _worker_loop(self):
        while True:
            item = self.queue.get()
            if item is None:
                break
            text, lang = item
            self._stop_flag.clear()

            audio_file = self.synthesize(text, lang)
            if audio_file and not self._stop_flag.is_set():
                self._play_file(audio_file)

            self.queue.task_done()

    def speak(self, text, lang="english", interrupt=False):
        """Queue text to be spoken in the specified language ('english', 'hindi', 'kannada')."""
        if not text or not text.strip():
            return
        if interrupt:
            self.stop()
        self.queue.put((text.strip(), lang))

    def stop(self):
        """Immediately stop currently playing speech and clear pending queue."""
        self._stop_flag.set()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                break
        if PYGAME_AUDIO:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.unload()
            except Exception:
                pass
        self.current_playing = False

    def is_speaking(self):
        return self.current_playing or not self.queue.empty()

    def precache_vocabulary(self, csv_path="translations.csv"):
        """Pre-generate speech audio for all words in translations.csv in all 3 languages."""
        import csv
        if not os.path.exists(csv_path):
            print(f"[TTS] Cannot precache: {csv_path} not found.")
            return

        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        total = len(rows) * 3
        print(f"[TTS] Pre-caching audio for {len(rows)} vocabulary entries across 3 languages ({total} files)...")
        count = 0
        for i, row in enumerate(rows, 1):
            en = (row.get("english") or "").strip()
            hi = (row.get("hindi") or "").strip()
            kn = (row.get("kannada") or "").strip()

            for text, lang in [(en, "english"), (hi, "hindi"), (kn, "kannada")]:
                if text:
                    path = self.synthesize(text, lang)
                    if path:
                        count += 1
            if i % 10 == 0 or i == len(rows):
                print(f"[TTS] Progress: {i}/{len(rows)} words processed...")

        print(f"[TTS] Pre-caching complete! {count}/{total} files cached in {self.cache_dir}")


_global_tts = None


def get_tts():
    global _global_tts
    if _global_tts is None:
        _global_tts = TextToSpeech()
    return _global_tts


def speak(text, lang="english", interrupt=False):
    return get_tts().speak(text, lang=lang, interrupt=interrupt)


def stop():
    return get_tts().stop()


def is_speaking():
    return get_tts().is_speaking()


if __name__ == "__main__":
    tts = get_tts()
    print("Testing Silent Voice TTS...")
    csv_file = os.path.join(os.path.dirname(__file__), "translations.csv")
    tts.precache_vocabulary(csv_file)

    print("\nPlaying audio test samples...")
    print("Speaking English: 'Hello'...")
    tts.speak("Hello", "english")
    time.sleep(1.5)
    print("Speaking Hindi sample...")
    tts.speak("नमस्ते", "hindi")
    time.sleep(1.5)
    print("Speaking Kannada sample...")
    tts.speak("ನಮಸ್ಕಾರ", "kannada")
    time.sleep(2.0)
    print("TTS test completed successfully!")
