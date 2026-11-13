# 🧏 Silent Voice — Indian Sign Language to Multilingual Speech Hub

**Silent Voice** is an AI-powered Indian Sign Language (ISL) recognition, real-time multilingual translation, and Text-to-Speech (TTS) platform. It recognizes both static finger-spelling (letters) and dynamic continuous gestures (words/phrases), translating them into **English**, **Hindi (हिन्दी)**, and **Kannada (ಕನ್ನಡ)** with synthesized voice audio.

---

##  Key Features

1. **Dual Recognition Engine**:
   - **Letters Mode**: MediaPipe Hands tracking 21 keypoints per hand (126 features) + Random Forest classifier (`model.pkl`).
   - **Words Mode**: MediaPipe Holistic pose & hand tracking across 30-frame temporal sequences (98 features/frame) + ExtraTrees classifier (`words_model.pkl`) trained on 40 ISL vocabulary words.
2. **Role-Based Access Control (RBAC) & Authentication**:
   - **User Mode**: Signs in to access the full live AI recognition hub, real-time multilingual translation, TTS voice playback, and vocabulary explorer.
   - **Admin Console**: Elevated management panel for superusers to manage user accounts, add/edit vocabulary entries in `translations.csv`, trigger zero-latency audio pre-caching, and monitor ML model health.
   - **Pre-seeded Accounts**: `raghav` / `admin123` (Admin) and `user` / `user123` (User), with self-registration for new users.
3. **🔄 Two-Way Communication (Hearing ➔ Deaf)**:
   - Hearing individuals can speak in English, Hindi, or Kannada using the integrated Web Speech API.
   - Real-time speech transcription displays corresponding ISL gesture video clips or fingerspelled letter guides on-screen for the deaf individual.
4. **🤖 AI Natural Sentence Auto-Smoothing**:
   - Automatically converts raw, literal sign tokens (e.g., "water drink please" or "hospital where") into natural, grammatically fluent sentences ("Please give me some water to drink." or "Where is the nearest hospital?").
   - Inflects translations accurately across English, Hindi, and Kannada.
5. **🎥 Web Camera Device Switcher**:
   - Switch between built-in webcams, USB cameras, and virtual cameras directly from the dashboard.
6. **📄 Session Transcript History & Export (PDF / CSV / TXT)**:
   - Session logs are permanently stored in `users.db` tied to user accounts.
   - One-click export to printable PDF reports (with multi-language Indic font support via Nirmala UI), Excel-ready CSV (with UTF-8 BOM), or plain text TXT.
7. **🛠️ In-Browser Gesture Collector & Retrainer (Admin Panel)**:
   - Record 30-frame temporal gesture sequences directly from the webcam inside the Admin Console.
   - One-click retrain trigger for ExtraTrees and SVM models with live progress, validation accuracy, and hot-reload.
8. **Real-time Multilingual Translation & Voice (TTS)**:
   - English, Hindi (`हिन्दी`), and Kannada (`ಕನ್ನಡ`) translations via dictionary mapping and phonetic transliteration fallback.
   - Zero-latency cached audio playback for the full 40-word vocabulary in all three languages.
10. **Modern Glassmorphic Web Dashboard**:
    - 30 FPS live webcam stream with HUD overlay, confidence meter, top guesses, vocabulary explorer, and shortcuts.
11. **Desktop GUI App**:
    - Fast, unified Tkinter interface with webcam feed and voice controls.

---

##  Quick Start

### 1. Launch the Web Application (Recommended)
Double-click `run_web.bat` or run:
```powershell
& ".\venv\Scripts\python.exe" web_app.py
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your web browser.

### 2. Launch the Desktop GUI
Double-click `run_gui.bat` or run:
```powershell
& ".\venv\Scripts\python.exe" app_gui.py
```

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `L` | Switch to **Letters Mode** |
| `W` | Switch to **Words Mode** |
| `Tab` | Toggle between Letters & Words |
| `1` | Voice current text in **English** |
| `2` | Voice current text in **Hindi** |
| `3` | Voice current text in **Kannada** |
| `S` | Speak current text |
| `Space` | Add space to current sentence |
| `Backspace` | Delete last word / letter |
| `C` | Clear all text |
| `Esc` | Exit application |

---

##  Project Structure

```
Silent Voice/
├── web_app.py               # FastAPI server, streaming backend & RBAC routes
├── auth.py                  # User authentication, SQLite database & PBKDF2 hashing
├── users.db                 # SQLite database storing users and secure sessions
├── app_gui.py               # Tkinter desktop GUI
├── app.py                   # OpenCV live captioning application
├── engine.py                # Dual recognition engine (LetterMode, WordMode, Session)
├── tts.py                   # Text-to-Speech manager & audio cache
├── common.py                # Landmark extraction, normalization & camera utilities
├── translate.py             # Multilingual dictionary & transliteration
├── translations.csv         # English, Hindi & Kannada vocabulary database
├── model.pkl                # Trained Random Forest model for letters
├── words_model.pkl          # Trained ExtraTrees model for 40 dynamic words
├── audio_cache/             # Pre-cached zero-latency MP3 audio files
├── templates/
│   ├── index.html           # Tailwind CSS glassmorphic user recognition dashboard
│   ├── login.html           # Glassmorphic auth portal (Sign in / Registration)
│   └── admin.html           # Full superuser management console
├── static/
│   ├── css/style.css        # Custom styles, animations & themes
│   ├── js/app.js            # Client-side state polling & interactions
│   └── sign_chart.png       # ISL finger-spelling alphabet chart
├── run_web.bat              # One-click web launcher
└── run_gui.bat              # One-click desktop launcher
```

---

##  How Signing Works

- **Letters Mode**: Show your hand sign in front of the camera and hold steady for about 1 second.
- **Words Mode**: Raise your hands into camera view to begin signing. Complete the motion and **lower your hands**. The sign is automatically recognized and voiced when your hands drop!
