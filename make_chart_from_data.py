"""Draw a sign chart from YOUR training data (data.csv).

For every letter it picks the most typical sample in your data and draws the hand skeleton(s)
that MediaPipe saw. This is an abstract picture of the hand shape - not a photo.
For real photos use make_letter_sheet.py.

Usage:
    python make_chart_from_data.py                 (reads data.csv)
    python make_chart_from_data.py other_data.csv
Creates sign_chart_from_data.png
"""
import sys

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
df = pd.read_csv(path, header=None)
df[0] = df[0].astype(str)

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
TIPS = [4, 8, 12, 16, 20]
COLORS = ["#1f77b4", "#e8590c"]

letters = sorted(df[0].unique())
info = {}
for letter in letters:
    X = df[df[0] == letter].drop(columns=[0]).values.astype(float)
    two = np.any(X[:, 63:] != 0, axis=1)
    share = float(two.mean())
    pool = X[two] if share >= 0.5 else X[~two]
    typical = pool[np.argmin(np.linalg.norm(pool - pool.mean(axis=0), axis=1))]
    info[letter] = {"sample": typical.reshape(2, 21, 3), "two_share": share, "mean": X.mean(axis=0)}

cols = 6
rows = int(np.ceil(len(letters) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
axes = np.atleast_1d(axes).ravel()
for ax, letter in zip(axes, letters):
    sample, share = info[letter]["sample"], info[letter]["two_share"]
    n_hands = 2 if share >= 0.5 else 1
    xs, ys = [], []
    for h in range(n_hands):
        pts = sample[h]
        xs += list(pts[:, 0])
        ys += list(pts[:, 1])
        for a, b in BONES:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=COLORS[h], lw=2)
        ax.scatter(pts[:, 0], pts[:, 1], s=10, color=COLORS[h], zorder=3)
        ax.scatter(pts[TIPS, 0], pts[TIPS, 1], s=34, color="black", zorder=4)
    pad = 0.15
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(max(ys) + pad, min(ys) - pad)       # image coordinates: y grows downwards
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    note = f"{n_hands} hand{'s' if n_hands == 2 else ''}"
    if 0.2 < share < 0.8:
        note += " (often loses one)"
    ax.set_title(f"{letter.upper()}   {note}", fontsize=11, fontweight="bold")
for ax in axes[len(letters):]:
    ax.axis("off")
fig.suptitle("Sign chart drawn from your data.csv  (black dots = fingertips, as seen in the camera's mirror view)",
             fontsize=12)
fig.text(0.5, 0.005, "Skeletons come from MediaPipe landmarks, not photos. Compare with the reference photos "
         "(make_letter_sheet.py).", ha="center", fontsize=9)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig("sign_chart_from_data.png", dpi=150)

one = [l.upper() for l in letters if info[l]["two_share"] < 0.2]
two = [l.upper() for l in letters if info[l]["two_share"] >= 0.8]
mixed = [f"{l.upper()} ({info[l]['two_share']:.0%})" for l in letters if 0.2 <= info[l]["two_share"] < 0.8]
print("One-hand letters :", ", ".join(one))
print("Two-hand letters :", ", ".join(two))
print("Mixed (second hand often missed):", ", ".join(mixed) or "-")

means = np.array([info[l]["mean"] for l in letters])
pairs = []
for i in range(len(letters)):
    for j in range(i + 1, len(letters)):
        pairs.append((np.linalg.norm(means[i] - means[j]), letters[i], letters[j]))
pairs.sort()
print("Letters that look most alike in your data:",
      ", ".join(f"{a.upper()}-{b.upper()}" for _, a, b in pairs[:6]))
print("Saved sign_chart_from_data.png")
