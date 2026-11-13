"""Step 2: train a classifier on the collected samples."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import os
if not os.path.exists("data.csv") or os.path.getsize("data.csv") == 0:
    raise SystemExit("data.csv is empty. Run collect_data.py first and save some samples "
                     "(click the camera window, show a sign, press its key).")
df = pd.read_csv("data.csv", header=None)

MIN_SAMPLES = 20   # signs with fewer samples than this are skipped
counts = df[0].astype(str).value_counts().sort_index()
print("Samples per sign:", ", ".join(f"{k}:{v}" for k, v in counts.items()))
too_few = counts[counts < MIN_SAMPLES]
if len(too_few):
    print(f"\nSkipping signs with fewer than {MIN_SAMPLES} samples: {', '.join(too_few.index)}")
    df = df[df[0].astype(str).isin(counts[counts >= MIN_SAMPLES].index)]
if df[0].nunique() < 2:
    raise SystemExit("\nNot enough data. Collect at least 2 signs with 100+ samples each "
                     "(run collect_data.py again; the new samples are added to data.csv).")
y = df[0].astype(str)
X = df.drop(columns=[0]).values
print(f"{len(df)} samples, {y.nunique()} classes: {sorted(y.unique())}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print(f"\nTest accuracy: {model.score(X_test, y_test):.2%}\n")
print(classification_report(y_test, model.predict(X_test)))

joblib.dump(model, "model.pkl")
print("Saved model.pkl")