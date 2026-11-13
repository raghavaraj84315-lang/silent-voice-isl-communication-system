"""Train the word/phrase recognizer from the recordings in the 'sequences' folder.

Works with your own recordings (collect_sequences.py) and clips converted from videos
(videos_to_sequences.py). Copies made from the same clip are always kept together, so the
accuracy shown is measured on clips the model has never seen.

It tries three kinds of model, keeps the best one on those unseen clips, then retrains it
on everything and saves words_model.pkl.
"""
import os
import random
import time
from collections import defaultdict

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MIN_SAMPLES = 8
root = "sequences"
if not os.path.isdir(root):
    raise SystemExit("No 'sequences' folder. Run collect_sequences.py or videos_to_sequences.py first.")

# data[word][source clip] = list of sequences (original + augmented copies)
data = defaultdict(lambda: defaultdict(list))
for word in sorted(os.listdir(root)):
    wdir = os.path.join(root, word)
    if not os.path.isdir(wdir):
        continue
    files = [f for f in os.listdir(wdir) if f.endswith(".npy")]
    if len(files) < MIN_SAMPLES:
        print(f"Skipping '{word}': only {len(files)} samples (need {MIN_SAMPLES}+)")
        continue
    for f in files:
        source = f[:-4].split("__")[0]
        data[word][source].append(np.load(os.path.join(wdir, f)).flatten())

if len(data) < 2:
    raise SystemExit("Need at least 2 different words with enough samples.")

rng = random.Random(42)
X_train, y_train, X_test, y_test = [], [], [], []
too_few = []
for word, sources in sorted(data.items()):
    n = sum(len(v) for v in sources.values())
    print(f"{word}: {n} samples from {len(sources)} separate recording(s)/clip(s)")
    ids = sorted(sources)
    rng.shuffle(ids)
    n_test = max(1, round(0.2 * len(ids))) if len(ids) >= 4 else 0
    if n_test == 0:
        too_few.append(word)
    for i, source in enumerate(ids):
        bucket = (X_test, y_test) if i < n_test else (X_train, y_train)
        for arr in sources[source]:
            bucket[0].append(arr)
            bucket[1].append(word)

have_test = len(X_test) > 0
X_train, y_train = np.array(X_train), np.array(y_train)
X_all = np.array(list(X_train) + X_test)
y_all = np.array(list(y_train) + y_test)

components = int(min(120, X_train.shape[0] - 1, X_train.shape[1]))
candidates = {
    "ExtraTrees": ExtraTreesClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "SVM": make_pipeline(StandardScaler(), PCA(n_components=components, random_state=42),
                         SVC(C=10, gamma="scale", probability=True, random_state=42)),
    "NeuralNet": make_pipeline(StandardScaler(), PCA(n_components=components, random_state=42),
                               MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300,
                                             random_state=42)),
}

best_name = "ExtraTrees"
if have_test:
    X_test, y_test = np.array(X_test), np.array(y_test)
    print(f"\nComparing models on {len(y_test)} samples from clips held out from training:")
    best_score, best_model = -1.0, None
    for name, model in candidates.items():
        started = time.time()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"  {name:13s} {score:6.2%}   ({time.time() - started:.0f}s)", flush=True)
        if score > best_score:
            best_name, best_score, best_model = name, score, model
    print(f"\nBest model: {best_name}  ({best_score:.2%} on unseen clips)\n")
    print(classification_report(y_test, best_model.predict(X_test), zero_division=0))
if too_few:
    print("Not enough separate clips to test these words fairly (need 4+ different clips/"
          f"recordings): {', '.join(too_few)}")
if not have_test:
    print("\nNo held-out test was possible, so there is no honest accuracy number yet.")

final = clone(candidates[best_name]).fit(X_all, y_all)   # final model learns from everything
joblib.dump(final, "words_model.pkl")
print(f"\nSaved words_model.pkl  ({best_name}, {len(set(y_all))} words)")
