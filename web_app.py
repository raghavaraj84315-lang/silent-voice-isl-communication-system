"""Web application server and API endpoints for Silent Voice."""
import csv
import logging
import os
import threading
import time
from typing import Optional

import cv2
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import auth
from common import augment, list_available_cameras, open_camera, prepare_sequence
from engine import Session
from pdf_export import generate_transcript_csv, generate_transcript_pdf, generate_transcript_txt
from retrain import get_training_status, run_training_pipeline
from smoother import smooth_sentence
from translate import load_translations, translate
from tts import get_tts, is_speaking, speak, stop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("silent_voice")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
AUDIO_CACHE_DIR = os.path.join(BASE_DIR, "audio_cache")
WORD_VIDEOS_DIR = os.path.join(BASE_DIR, "word_videos")

app = FastAPI(title="Silent Voice API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
if os.path.exists(WORD_VIDEOS_DIR):
    app.mount("/word_videos", StaticFiles(directory=WORD_VIDEOS_DIR), name="word_videos")

templates = Jinja2Templates(directory=TEMPLATES_DIR)


class CameraWorker:
    """Background worker for webcam capture and frame inference."""
    def __init__(self):
        letters_model = joblib.load("model.pkl") if os.path.exists("model.pkl") else None
        words_model = joblib.load("words_model.pkl") if os.path.exists("words_model.pkl") else None
        if letters_model is None and words_model is None:
            raise RuntimeError("No models found. Please train models first.")

        self.session = Session(letters_model, words_model)
        self.table = load_translations(os.path.join(BASE_DIR, "translations.csv"))
        self.cap = None
        self.running = False
        self.lock = threading.Lock()

        # Camera selection & Flip controls
        cam_txt = os.path.join(BASE_DIR, "camera.txt")
        if os.path.exists(cam_txt):
            try:
                self.camera_index = int(open(cam_txt).read().strip())
            except Exception:
                self.camera_index = 0
        else:
            self.camera_index = 0

        self.mirror = False
        self.recording_gesture = None  # Gesture recording state

        self.current_frame = None
        self.current_jpeg = None
        self.status = "Initializing..."
        self.top3 = ""
        self.auto_speak = True
        self.selected_lang = "english"
        self.last_spoken_word = None

        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self):
        if not self.running:
            self.running = True
            self.thread.start()

    def switch_camera(self, index: int):
        with self.lock:
            self.camera_index = index
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None
            try:
                with open(os.path.join(BASE_DIR, "camera.txt"), "w") as f:
                    f.write(str(index))
            except Exception:
                pass
            self.status = f"Switched to Camera {index}"

    def reload_words_model(self, model):
        with self.lock:
            if self.session.word_mode:
                self.session.word_mode.model = model
            print("[CameraWorker] Hot-reloaded newly trained words model into active session.")

    def start_recording_gesture(self, word: str, prep_sec: float = 2.0, record_sec: float = 2.5):
        clean_word = word.strip().lower().replace(" ", "_")
        with self.lock:
            self.session.set_mode("words")
            now = time.time()
            self.recording_gesture = {
                "word": clean_word,
                "start_time": now + prep_sec,
                "end_time": now + prep_sec + record_sec,
                "times": [],
                "feats": [],
                "state": "prep",
                "message": "Get ready with hands down...",
                "sample_count": 0
            }
            return {"status": "ok", "word": clean_word}

    def get_recording_status(self):
        with self.lock:
            if not self.recording_gesture:
                return {"state": "idle", "message": "Ready to record"}
            return {
                "state": self.recording_gesture["state"],
                "message": self.recording_gesture["message"],
                "word": self.recording_gesture["word"],
                "sample_count": self.recording_gesture.get("sample_count", 0)
            }

    def _capture_loop(self):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        # Initial fallback placeholder frame
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Silent Voice Live Stream", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(placeholder, "Camera initializing or in use...", (140, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)
        _, p_jpeg = cv2.imencode(".jpg", placeholder, encode_param)
        with self.lock:
            self.current_jpeg = p_jpeg.tobytes()

        try:
            self.cap = open_camera(self.camera_index)
            self.status = "Camera Active"
        except BaseException as e:
            print(f"[CameraWorker] Could not open camera {self.camera_index}: {e}")
            self.status = "Camera Busy / Reconnecting"
            self.cap = None

        last_reconnect_attempt = time.time()

        while self.running:
            if self.cap is None or not self.cap.isOpened():
                now = time.time()
                if now - last_reconnect_attempt > 3.0:
                    last_reconnect_attempt = now
                    try:
                        self.cap = open_camera(self.camera_index)
                        self.status = f"Camera {self.camera_index} Connected"
                    except BaseException:
                        self.cap = None
                time.sleep(0.1)
                continue

            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.03)
                continue

            now = time.time()
            annotated_frame, status, top3 = self.session.process(frame, now, mirror=False)

            # In-Browser Gesture Recording Logic
            if self.recording_gesture and self.recording_gesture["state"] in ("prep", "recording"):
                rec = self.recording_gesture
                if now < rec["start_time"]:
                    remain = max(1, int(rec["start_time"] - now + 0.9))
                    rec["message"] = f"Get Ready: {remain}s (Hands down)"
                    cv2.putText(annotated_frame, f"GET READY: {remain}", (180, 240),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
                elif now < rec["end_time"]:
                    rec["state"] = "recording"
                    feats = getattr(self.session, "last_holistic_features", None)
                    if feats is not None:
                        rec["times"].append(now)
                        rec["feats"].append(feats)
                    count_collected = len(rec["feats"])
                    rec["message"] = f"SIGN NOW! Collected {min(count_collected, 30)}/30 frames..."
                    cv2.putText(annotated_frame, f"COLLECTING - {min(count_collected, 30)}/30 FRAMES", (80, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2)
                else:
                    rec["state"] = "processing"
                    seq = prepare_sequence(rec["times"], rec["feats"])
                    folder = os.path.join(BASE_DIR, "sequences", rec["word"])
                    os.makedirs(folder, exist_ok=True)
                    if seq is None:
                        rec["state"] = "failed"
                        rec["message"] = "Hands not clearly visible. Please try again."
                    else:
                        existing = [f for f in os.listdir(folder) if f.endswith(".npy")]
                        start_idx = len(existing)
                        # Save 30 sequence frames (1 base + 29 augmented variations)
                        rng = np.random.default_rng(start_idx)
                        np.save(os.path.join(folder, f"{start_idx:04d}.npy"), seq)
                        for aug_i in range(1, 30):
                            aug_seq = augment(seq, rng)
                            np.save(os.path.join(folder, f"{(start_idx + aug_i):04d}.npy"), aug_seq)

                        all_files = [f for f in os.listdir(folder) if f.endswith(".npy")]
                        rec["state"] = "completed"
                        rec["sample_count"] = len(all_files)
                        rec["message"] = f"Successfully collected 30 frames for '{rec['word']}'! Total: {len(all_files)}"

            # Auto-speak newly accepted words
            if self.session.last_accepted:
                accepted = self.session.last_accepted.replace("_", " ")
                if self.auto_speak and accepted != self.last_spoken_word:
                    lang = getattr(self, "selected_lang", "english")
                    word_to_speak = accepted
                    if lang != "english":
                        translated = translate(accepted, lang, self.table)
                        if translated:
                            word_to_speak = translated
                    speak(word_to_speak, lang, interrupt=False)
                    self.last_spoken_word = accepted
            else:
                self.last_spoken_word = None

            ok, jpeg = cv2.imencode(".jpg", annotated_frame, encode_param)
            if ok:
                with self.lock:
                    self.current_jpeg = jpeg.tobytes()
                    self.status = status
                    self.top3 = top3

            time.sleep(0.01)

    def get_state(self):
        with self.lock:
            text = self.session.text
            hi = translate(text, "hindi", self.table)
            kn = translate(text, "kannada", self.table)

            return {
                "mode": self.session.mode,
                "status": self.status,
                "top3": self.top3,
                "text": text,
                "camera_index": self.camera_index,
                "selected_lang": getattr(self, "selected_lang", "english"),
                "translations": {
                    "english": text,
                    "hindi": hi if hi else "",
                    "kannada": kn if kn else ""
                },
                "auto_speak": self.auto_speak,
                "is_speaking": is_speaking()
            }

    def close(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
        self.session.close()


selected_language = "english"
camera_worker = None


@app.on_event("startup")
def startup_event():
    global camera_worker
    auth.init_db()
    try:
        camera_worker = CameraWorker()
        camera_worker.start()
        print("[Silent Voice] Camera and recognition engine started.")
    except Exception as e:
        print(f"[Silent Voice] Startup error: {e}")


@app.on_event("shutdown")
def shutdown_event():
    global camera_worker
    stop()
    if camera_worker:
        camera_worker.close()


# Auth helpers & role verification
def get_current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get("session_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1].strip()
    if token:
        return auth.get_user_by_session(token)
    return None


def require_user(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required. Please sign in.")
    return user


def require_admin(request: Request) -> dict:
    user = require_user(request)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required for this action.")
    return user


# Video streaming generator
def generate_frames():
    global camera_worker
    while True:
        if camera_worker and camera_worker.current_jpeg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + camera_worker.current_jpeg + b"\r\n")
        time.sleep(0.033)  # ~30 FPS


# Request schemas
class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str
    password: str
    role: str = "user"


class UpdateRoleRequest(BaseModel):
    role: str


class VocabularyItemRequest(BaseModel):
    english: Optional[str] = None
    hindi: str
    kannada: str
    category: Optional[str] = "General"


class ModeRequest(BaseModel):
    mode: str


class ActionRequest(BaseModel):
    action: str


class SpeakRequest(BaseModel):
    text: str
    lang: Optional[str] = "english"


class LanguageRequest(BaseModel):
    lang: str


class AutoSpeakRequest(BaseModel):
    enabled: bool


class CameraSwitchRequest(BaseModel):
    index: int


class SmoothRequest(BaseModel):
    text: str


class HearingSpeakRequest(BaseModel):
    text: str
    lang: Optional[str] = "english"
    auto_voice: Optional[bool] = False


class TranscriptItemRequest(BaseModel):
    english: str
    hindi: Optional[str] = ""
    kannada: Optional[str] = ""
    speaker: Optional[str] = "Signer"
    raw_text: Optional[str] = ""
    is_smoothed: Optional[bool] = False


class RecordGestureRequest(BaseModel):
    word: str
    hindi: Optional[str] = ""
    kannada: Optional[str] = ""
    category: Optional[str] = "General"



# Authentication routes
@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    user = get_current_user(request)
    if user:
        if user.get("role") == "admin":
            return RedirectResponse(url="/admin", status_code=303)
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request=request, name="login.html")


@app.post("/api/auth/login")
def api_login(req: LoginRequest, response: Response):
    user = auth.authenticate_user(req.username, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = auth.create_session(user["id"])
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=604800,  # 7 days
        path="/",
        samesite="lax"
    )
    return {"status": "ok", "user": user, "role": user["role"]}


@app.post("/api/auth/register")
def api_register(req: RegisterRequest, response: Response):
    try:
        user = auth.register_user(req.username, req.password, role="user")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = auth.create_session(user["id"])
    response.set_cookie(
        key="session_token",
        value=token,
        httponly=True,
        max_age=604800,
        path="/",
        samesite="lax"
    )
    return {"status": "ok", "user": user, "role": user["role"]}


@app.get("/logout")
def logout(request: Request):
    token = request.cookies.get("session_token")
    if token:
        auth.delete_session(token)
    res = RedirectResponse(url="/login", status_code=303)
    res.delete_cookie("session_token", path="/")
    return res


@app.post("/api/auth/logout")
def api_logout(request: Request, response: Response):
    token = request.cookies.get("session_token")
    if token:
        auth.delete_session(token)
    response.delete_cookie("session_token", path="/")
    return {"status": "ok"}


@app.get("/api/auth/me")
def api_me(request: Request):
    user = get_current_user(request)
    if not user:
        return {"authenticated": False, "user": None}
    return {"authenticated": True, "user": user}


# Core views
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return templates.TemplateResponse(request=request, name="index.html", context={"user": user})


@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    if user.get("role") != "admin":
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request=request, name="admin.html", context={"current_user": user})



@app.get("/favicon.ico")
def favicon():
    fav_path = os.path.join(STATIC_DIR, "favicon.png")
    if os.path.exists(fav_path):
        return FileResponse(fav_path, media_type="image/png")
    return Response(status_code=204)


@app.get("/video_feed")
def video_feed(request: Request):
    require_user(request)
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


# Recognition and audio controls
@app.get("/api/state")
def get_state(request: Request):
    require_user(request)
    if not camera_worker:
        raise HTTPException(status_code=503, detail="Camera worker initializing")
    return camera_worker.get_state()


@app.post("/api/mode")
def set_mode(req: ModeRequest, request: Request):
    require_user(request)
    if not camera_worker:
        raise HTTPException(status_code=503, detail="Camera worker not ready")
    if req.mode not in ("letters", "words"):
        raise HTTPException(status_code=400, detail="Invalid mode")
    camera_worker.session.set_mode(req.mode)
    return {"status": "ok", "mode": camera_worker.session.mode}


@app.post("/api/action")
def perform_action(req: ActionRequest, request: Request):
    require_user(request)
    if not camera_worker:
        raise HTTPException(status_code=503, detail="Camera worker not ready")
    act = req.action.lower()
    if act == "space":
        camera_worker.session.add_space()
    elif act == "backspace":
        camera_worker.session.backspace()
    elif act == "clear":
        camera_worker.session.clear()
    else:
        raise HTTPException(status_code=400, detail="Unknown action")
    return {"status": "ok", "text": camera_worker.session.text}


@app.post("/api/speak")
def trigger_speak(req: SpeakRequest, request: Request):
    require_user(request)
    text = (req.text or "").strip()
    if text:
        speak(text, req.lang or "english", interrupt=True)
    return {"status": "ok", "speaking": text, "lang": req.lang}


@app.post("/api/language")
def set_language(req: LanguageRequest, request: Request):
    require_user(request)
    if not camera_worker:
        raise HTTPException(status_code=503, detail="Camera worker not ready")
    lang = req.lang.lower()
    if lang not in ("english", "hindi", "kannada"):
        raise HTTPException(status_code=400, detail="Invalid language")
    camera_worker.selected_lang = lang
    return {"status": "ok", "language": camera_worker.selected_lang}


@app.post("/api/auto_speak")
def set_auto_speak(req: AutoSpeakRequest, request: Request):
    require_user(request)
    if camera_worker:
        camera_worker.auto_speak = req.enabled
    return {"status": "ok", "auto_speak": req.enabled}


# Camera configuration
@app.get("/api/camera/devices")
def get_camera_devices(request: Request):
    require_user(request)
    curr_idx = camera_worker.camera_index if camera_worker else 0
    devices = list_available_cameras(curr_idx)
    return {
        "devices": devices,
        "current": curr_idx
    }


@app.post("/api/camera/switch")
def switch_camera_device(req: CameraSwitchRequest, request: Request):
    require_user(request)
    if not camera_worker:
        raise HTTPException(status_code=503, detail="Camera worker not ready")
    camera_worker.switch_camera(req.index)
    return {"status": "ok", "current": camera_worker.camera_index}


# Linguistic smoothing
@app.post("/api/smooth")
def api_smooth_sentence(req: SmoothRequest, request: Request):
    require_user(request)
    text = (req.text or "").strip()
    table = camera_worker.table if camera_worker else None
    result = smooth_sentence(text, table)
    return result


# Two-way sign and speech communication
@app.get("/api/signs/reference/{word}")
def get_sign_reference(word: str, request: Request):
    require_user(request)
    w = word.strip().lower().replace(" ", "_")
    target_dir = os.path.join(WORD_VIDEOS_DIR, w)

    if os.path.isdir(target_dir):
        mp4s = [f for f in os.listdir(target_dir) if f.endswith(".mp4")]
        if mp4s:
            video_url = f"/word_videos/{w}/{mp4s[0]}"
            hi = translate(w.replace("_", " "), "hindi", camera_worker.table) if camera_worker else ""
            kn = translate(w.replace("_", " "), "kannada", camera_worker.table) if camera_worker else ""
            return {
                "found": True,
                "type": "video",
                "word": w.replace("_", " "),
                "video_url": video_url,
                "hindi": hi,
                "kannada": kn,
                "category": CATEGORY_MAP.get(w.replace("_", " "), "General")
            }

    # Fallback to letters fingerspelling guide
    chars = [c.upper() for c in word if c.isalpha()]
    return {
        "found": False,
        "type": "fingerspelling",
        "word": word,
        "letters": chars,
        "sign_chart_url": "/static/sign_chart.png"
    }


@app.post("/api/hearing/speak")
def hearing_person_speech(req: HearingSpeakRequest, request: Request):
    user = require_user(request)
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    table = camera_worker.table if camera_worker else None
    smoothed = smooth_sentence(raw_text, table)
    en_sentence = smoothed["translations"]["english"]
    hi_sentence = smoothed["translations"]["hindi"]
    kn_sentence = smoothed["translations"]["kannada"]

    # Save to user transcripts history as 'Hearing Speaker'
    auth.add_transcript(
        user_id=user["id"],
        speaker="Hearing Speaker",
        english=en_sentence,
        hindi=hi_sentence,
        kannada=kn_sentence,
        raw_text=raw_text,
        is_smoothed=True
    )

    # Voice aloud if auto_voice requested
    if req.auto_voice:
        lang_to_speak = req.lang or "english"
        text_to_speak = smoothed["translations"].get(lang_to_speak, en_sentence)
        speak(text_to_speak, lang_to_speak, interrupt=True)

    # Word-by-word sign reference matching
    words = en_sentence.lower().replace(".", "").replace("?", "").replace(",", "").replace("!", "").split()
    references = []
    for w in words:
        clean_w = w.replace("_", " ")
        v_dir = os.path.join(WORD_VIDEOS_DIR, w)
        if os.path.isdir(v_dir):
            clips = [f for f in os.listdir(v_dir) if f.endswith(".mp4")]
            if clips:
                references.append({
                    "word": clean_w,
                    "video_url": f"/word_videos/{w}/{clips[0]}",
                    "type": "video"
                })
                continue
        references.append({
            "word": clean_w,
            "type": "fingerspelling",
            "letters": [c.upper() for c in clean_w if c.isalpha()]
        })

    return {
        "status": "ok",
        "raw": raw_text,
        "smoothed": en_sentence,
        "translations": smoothed["translations"],
        "references": references
    }


# Transcript management & export
@app.get("/api/transcripts")
def get_user_transcripts(request: Request):
    user = require_user(request)
    rows = auth.get_transcripts(user["id"])
    return {"transcripts": rows, "total": len(rows)}


@app.post("/api/transcripts")
def add_user_transcript(req: TranscriptItemRequest, request: Request):
    user = require_user(request)
    row = auth.add_transcript(
        user_id=user["id"],
        speaker=req.speaker or "Signer",
        english=req.english,
        hindi=req.hindi or "",
        kannada=req.kannada or "",
        raw_text=req.raw_text or "",
        is_smoothed=bool(req.is_smoothed)
    )
    return {"status": "ok", "transcript": row}


@app.delete("/api/transcripts")
def clear_user_transcripts(request: Request):
    user = require_user(request)
    auth.clear_transcripts(user["id"])
    return {"status": "ok", "message": "Transcripts cleared successfully"}


@app.get("/api/transcripts/export")
def export_user_transcripts(format: str, request: Request):
    user = require_user(request)
    fmt = (format or "pdf").lower()
    rows = auth.get_transcripts(user["id"], limit=500)
    username = user.get("username", "user")
    date_str = time.strftime("%Y%m%d_%H%M")

    if fmt == "pdf":
        pdf_bytes = generate_transcript_pdf(rows, username)
        filename = f"SilentVoice_Transcript_{username}_{date_str}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    elif fmt == "csv":
        csv_bytes = generate_transcript_csv(rows, username)
        filename = f"SilentVoice_Transcript_{username}_{date_str}.csv"
        return Response(
            content=csv_bytes,
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    elif fmt == "txt":
        txt_str = generate_transcript_txt(rows, username)
        filename = f"SilentVoice_Transcript_{username}_{date_str}.txt"
        return Response(
            content=txt_str,
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'}
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid export format. Choose 'pdf', 'csv', or 'txt'.")



CATEGORY_MAP = {
    "hello": "Greetings", "goodbye": "Greetings", "thank you": "Greetings",
    "sorry": "Greetings", "please": "Greetings", "good morning": "Greetings",
    "mother": "Family", "father": "Family", "brother": "Family", "sister": "Family",
    "friend": "Family", "teacher": "Family", "student": "Family", "me": "Family",
    "you": "Family", "he": "Family", "she": "Family",
    "eat": "Food & Drink", "drink": "Food & Drink", "water": "Food & Drink",
    "food": "Food & Drink", "tea": "Food & Drink",
    "home": "Places", "school": "Places", "hospital": "Places", "market": "Places",
    "come": "Actions", "go": "Actions", "sit": "Actions", "stand": "Actions",
    "read": "Actions", "write": "Actions", "help": "Actions", "stop": "Actions", "okay": "Actions",
    "what": "Questions", "where": "Questions", "when": "Questions", "how": "Questions",
    "today": "Questions", "yes": "Questions", "no": "Questions", "how you": "Questions",
    "how are you": "Questions"
}


@app.get("/api/vocabulary")
def get_vocabulary():
    csv_path = os.path.join(BASE_DIR, "translations.csv")
    words = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                en = (row.get("english") or "").strip().lower()
                if en:
                    words.append({
                        "english": en,
                        "hindi": (row.get("hindi") or "").strip(),
                        "kannada": (row.get("kannada") or "").strip(),
                        "category": CATEGORY_MAP.get(en, "General")
                    })
    return {"words": words, "total": len(words)}


@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(AUDIO_CACHE_DIR, safe_name)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Audio file not found")


# Administration endpoints
def _get_audio_cache_stats():
    count = 0
    total_size = 0
    if os.path.exists(AUDIO_CACHE_DIR):
        for fname in os.listdir(AUDIO_CACHE_DIR):
            fpath = os.path.join(AUDIO_CACHE_DIR, fname)
            if os.path.isfile(fpath) and fname.endswith(".mp3"):
                count += 1
                total_size += os.path.getsize(fpath)
    return {
        "file_count": count,
        "size_mb": round(total_size / (1024 * 1024), 2)
    }


def _reload_translations():
    global camera_worker
    csv_path = os.path.join(BASE_DIR, "translations.csv")
    if camera_worker and os.path.exists(csv_path):
        camera_worker.table = load_translations(csv_path)


@app.get("/api/admin/stats")
def get_admin_stats(request: Request):
    require_admin(request)
    users_info = auth.get_users_count()
    vocab = get_vocabulary()
    cache_stats = _get_audio_cache_stats()

    letters_exists = os.path.exists(os.path.join(BASE_DIR, "model.pkl"))
    words_exists = os.path.exists(os.path.join(BASE_DIR, "words_model.pkl"))

    return {
        "users": users_info,
        "vocabulary_count": vocab["total"],
        "audio_cache": cache_stats,
        "models": {
            "letters_model": letters_exists,
            "words_model": words_exists
        }
    }


@app.get("/api/admin/users")
def get_all_users(request: Request):
    require_admin(request)
    return auth.list_users()


@app.post("/api/admin/users")
def create_user_by_admin(req: CreateUserRequest, request: Request):
    require_admin(request)
    try:
        user = auth.register_user(req.username, req.password, role=req.role)
        return {"status": "ok", "user": user}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/admin/users/{user_id}/role")
def update_role(user_id: int, req: UpdateRoleRequest, request: Request):
    require_admin(request)
    try:
        success = auth.update_user_role(user_id, req.role)
        if not success:
            raise HTTPException(status_code=400, detail="Could not update role")
        return {"status": "ok", "user_id": user_id, "role": req.role}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/admin/users/{user_id}")
def delete_user_route(user_id: int, request: Request):
    curr = require_admin(request)
    if curr["id"] == user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own admin account")
    try:
        success = auth.delete_user(user_id)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")
        return {"status": "ok", "deleted_id": user_id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/admin/vocabulary")
def add_vocabulary_word(req: VocabularyItemRequest, request: Request):
    require_admin(request)
    en = (req.english or "").strip().lower()
    if not en:
        raise HTTPException(status_code=400, detail="English word key is required")

    csv_path = os.path.join(BASE_DIR, "translations.csv")
    rows = []
    found = False
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if (r.get("english") or "").strip().lower() == en:
                    rows.append({"english": en, "hindi": req.hindi.strip(), "kannada": req.kannada.strip()})
                    found = True
                else:
                    rows.append(r)
    if not found:
        rows.append({"english": en, "hindi": req.hindi.strip(), "kannada": req.kannada.strip()})

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["english", "hindi", "kannada"])
        writer.writeheader()
        writer.writerows(rows)

    _reload_translations()
    return {"status": "ok", "word": en}


@app.put("/api/admin/vocabulary/{english}")
def update_vocabulary_word(english: str, req: VocabularyItemRequest, request: Request):
    require_admin(request)
    en = english.strip().lower()
    csv_path = os.path.join(BASE_DIR, "translations.csv")
    rows = []
    found = False
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if (r.get("english") or "").strip().lower() == en:
                    rows.append({"english": en, "hindi": req.hindi.strip(), "kannada": req.kannada.strip()})
                    found = True
                else:
                    rows.append(r)

    if not found:
        raise HTTPException(status_code=404, detail="Word not found in vocabulary")

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["english", "hindi", "kannada"])
        writer.writeheader()
        writer.writerows(rows)

    _reload_translations()
    return {"status": "ok", "word": en}


@app.delete("/api/admin/vocabulary/{english}")
def delete_vocabulary_word(english: str, request: Request):
    require_admin(request)
    en = english.strip().lower()
    csv_path = os.path.join(BASE_DIR, "translations.csv")
    rows = []
    found = False
    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if (r.get("english") or "").strip().lower() == en:
                    found = True
                else:
                    rows.append(r)

    if not found:
        raise HTTPException(status_code=404, detail="Word not found in vocabulary")

    with open(csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["english", "hindi", "kannada"])
        writer.writeheader()
        writer.writerows(rows)

    _reload_translations()
    return {"status": "ok", "deleted": en}


@app.post("/api/admin/cache/precache")
def precache_all_audio(request: Request):
    require_admin(request)

    def _worker():
        try:
            tts = get_tts()
            csv_path = os.path.join(BASE_DIR, "translations.csv")
            tts.precache_vocabulary(csv_path)
            print("[Admin] Pre-cache audio generation completed.")
        except Exception as e:
            print(f"[Admin] Pre-cache error: {e}")

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "ok", "message": "Background audio pre-caching initiated."}


@app.post("/api/admin/cache/clear")
def clear_audio_cache_endpoint(request: Request):
    require_admin(request)
    cleared = 0
    if os.path.exists(AUDIO_CACHE_DIR):
        for fname in os.listdir(AUDIO_CACHE_DIR):
            fpath = os.path.join(AUDIO_CACHE_DIR, fname)
            if os.path.isfile(fpath) and fname.endswith(".mp3"):
                try:
                    os.remove(fpath)
                    cleared += 1
                except Exception:
                    pass
    return {"status": "ok", "cleared_files": cleared}


# Dataset collection and model training
@app.get("/api/admin/collector/stats")
def get_collector_stats(request: Request):
    require_admin(request)
    root = os.path.join(BASE_DIR, "sequences")
    words_data = []
    if os.path.isdir(root):
        for w in sorted(os.listdir(root)):
            wdir = os.path.join(root, w)
            if os.path.isdir(wdir):
                samples = len([f for f in os.listdir(wdir) if f.endswith(".npy")])
                words_data.append({
                    "word": w,
                    "samples": samples,
                    "ready": samples >= 6
                })
    return {"words": words_data, "total": len(words_data)}


@app.post("/api/admin/collector/record")
def trigger_record_gesture(req: RecordGestureRequest, request: Request):
    require_admin(request)
    if not camera_worker:
        raise HTTPException(status_code=503, detail="Camera worker not ready")

    word = req.word.strip().lower().replace(" ", "_")
    if not word:
        raise HTTPException(status_code=400, detail="Word key is required")

    # If new translations were provided, auto-register in vocabulary
    if req.hindi or req.kannada:
        add_vocabulary_word(
            VocabularyItemRequest(
                english=word.replace("_", " "),
                hindi=req.hindi or "",
                kannada=req.kannada or "",
                category=req.category or "General"
            ),
            request
        )

    res = camera_worker.start_recording_gesture(word)
    return res


@app.get("/api/admin/collector/status")
def get_collector_status(request: Request):
    require_admin(request)
    if not camera_worker:
        return {"state": "idle", "message": "Camera not active"}
    return camera_worker.get_recording_status()


@app.post("/api/admin/retrain")
def trigger_model_retrain(request: Request):
    require_admin(request)
    curr = get_training_status()
    if curr.get("state") == "running":
        return {"status": "already_running", "message": "Model retraining already in progress."}

    def _on_complete(new_model):
        if camera_worker:
            camera_worker.reload_words_model(new_model)

    run_training_pipeline(on_complete_callback=_on_complete)
    return {"status": "ok", "message": "Model retraining started."}


@app.get("/api/admin/retrain/status")
def get_retrain_status(request: Request):
    require_admin(request)
    return get_training_status()



def main():
    import sys
    import uvicorn

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    print("\n" + "=" * 60)
    print("  [+] Silent Voice Web Dashboard launching on http://127.0.0.1:8000")
    print("  Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
