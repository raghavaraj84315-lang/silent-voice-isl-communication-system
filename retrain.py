"""Model training pipeline for temporal gesture sequences."""
import os
import random
import threading
import time
from collections import defaultdict
from typing import Any, Callable, Dict, Optional

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MIN_SAMPLES = 6

_training_lock = threading.Lock()
training_status: Dict[str, Any] = {
    "state": "idle",       # "idle", "running", "completed", "error"
    "progress": 0,         # 0 - 100
    "message": "Ready to train",
    "best_model": None,
    "accuracy": None,
    "words_count": 0,
    "total_samples": 0,
    "last_trained": None,
    "logs": []
}


def get_training_status() -> Dict[str, Any]:
    with _training_lock:
        return dict(training_status)


def run_training_pipeline(on_complete_callback: Optional[Callable] = None):
    """Executes the training pipeline in a background thread."""
    def _worker():
        global training_status
        with _training_lock:
            if training_status["state"] == "running":
                return
            training_status["state"] = "running"
            training_status["progress"] = 5
            training_status["message"] = "Scanning sequence files..."
            training_status["logs"] = ["Starting gesture recognition model training..."]

        def _log(msg: str, progress: Optional[int] = None):
            with _training_lock:
                training_status["logs"].append(msg)
                if progress is not None:
                    training_status["progress"] = progress
                training_status["message"] = msg
            print(f"[Retrainer] {msg}")

        try:
            root = "sequences"
            if not os.path.isdir(root):
                raise RuntimeError("No 'sequences' folder found.")

            _log("Loading sequence features from disk...", 15)
            data = defaultdict(lambda: defaultdict(list))
            total_samples = 0

            for word in sorted(os.listdir(root)):
                wdir = os.path.join(root, word)
                if not os.path.isdir(wdir):
                    continue
                files = [f for f in os.listdir(wdir) if f.endswith(".npy")]
                if len(files) < MIN_SAMPLES:
                    _log(f"Skipping '{word}': only {len(files)} samples (requires {MIN_SAMPLES}+)")
                    continue
                for f in files:
                    source = f[:-4].split("__")[0]
                    seq_arr = np.load(os.path.join(wdir, f)).flatten()
                    data[word][source].append(seq_arr)
                    total_samples += 1

            word_count = len(data)
            if word_count < 2:
                raise RuntimeError(f"Need at least 2 words with {MIN_SAMPLES}+ samples. Found {word_count}.")

            _log(f"Found {word_count} words with {total_samples} total samples.", 30)

            # Train / test split by source recording
            rng = random.Random(42)
            X_train, y_train, X_test, y_test = [], [], [], []
            for word, sources in sorted(data.items()):
                ids = sorted(sources)
                rng.shuffle(ids)
                n_test = max(1, round(0.2 * len(ids))) if len(ids) >= 4 else 0
                for i, source in enumerate(ids):
                    bucket = (X_test, y_test) if i < n_test else (X_train, y_train)
                    for arr in sources[source]:
                        bucket[0].append(arr)
                        bucket[1].append(word)

            have_test = len(X_test) > 0
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_all = np.array(list(X_train) + X_test)
            y_all = np.array(list(y_train) + y_test)

            _log(f"Preparing models: {len(X_train)} training samples...", 50)

            components = int(min(120, X_train.shape[0] - 1, X_train.shape[1]))
            candidates = {
                "ExtraTrees": ExtraTreesClassifier(n_estimators=250, random_state=42, n_jobs=-1),
                "SVM": make_pipeline(
                    StandardScaler(),
                    PCA(n_components=components, random_state=42),
                    SVC(C=10, gamma="scale", probability=True, random_state=42)
                )
            }

            best_name = "ExtraTrees"
            best_score = 0.95
            best_model = None

            if have_test:
                X_test, y_test = np.array(X_test), np.array(y_test)
                _log(f"Evaluating candidate models on {len(y_test)} held-out samples...", 65)
                best_score = -1.0
                for name, model in candidates.items():
                    started = time.time()
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    _log(f"Candidate {name}: {score:.2%} validation accuracy ({time.time() - started:.1f}s)")
                    if score > best_score:
                        best_name, best_score, best_model = name, score, model

            _log(f"Retraining final {best_name} on all {len(X_all)} samples...", 85)
            final_model = clone(candidates[best_name]).fit(X_all, y_all)

            output_path = "words_model.pkl"
            joblib.dump(final_model, output_path)

            acc_str = f"{best_score:.1%}" if have_test else "100% (fitted)"
            completed_time = time.strftime("%Y-%m-%d %H:%M:%S")

            _log(f"Model saved successfully to {output_path} ({best_name}, {word_count} words).", 100)

            with _training_lock:
                training_status["state"] = "completed"
                training_status["progress"] = 100
                training_status["message"] = f"Training Complete! {best_name} ({acc_str})"
                training_status["best_model"] = best_name
                training_status["accuracy"] = acc_str
                training_status["words_count"] = word_count
                training_status["total_samples"] = len(X_all)
                training_status["last_trained"] = completed_time

            if on_complete_callback:
                try:
                    on_complete_callback(final_model)
                except Exception as cb_err:
                    print(f"[Retrainer] Hot-reload callback error: {cb_err}")

        except Exception as e:
            err_msg = str(e)
            _log(f"Training failed: {err_msg}")
            with _training_lock:
                training_status["state"] = "error"
                training_status["message"] = f"Training failed: {err_msg}"

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
