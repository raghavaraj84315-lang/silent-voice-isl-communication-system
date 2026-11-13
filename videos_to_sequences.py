"""Turn a folder of sign VIDEO clips into training data for the word model.

Folder layout (folder name = the word):
    videos/
        hello/          clip1.mp4  clip2.mp4 ...
        good_morning/   clip1.mp4 ...
        idle/           clip1.mp4 ...     <- optional: hands up but NOT signing

Usage:
    python videos_to_sequences.py videos
    python videos_to_sequences.py videos --augment 15
    python videos_to_sequences.py videos --only hello,thank_you --max-per-word 12 --frame-step 2
    python videos_to_sequences.py videos --no-flip     (only if the videos are already mirrored/selfie-style)

Only the hand/body coordinates are kept (in the 'sequences' folder), never the video itself.
Each clip should contain ONE sign, with the hands starting and ending lowered.
Only use videos you own or have permission to use (see the licensing notes).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from common import holistic_features, prepare_sequence

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def augment(seq, rng):
    """Make a slightly different copy of a sequence (timing, position, size, jitter)."""
    steps, dims = seq.shape
    present = seq != 0                                   # zeros mean 'hand not visible'
    keep = rng.uniform(0.8, 1.0)                         # use 80-100% of the clip
    start = rng.uniform(0.0, 1.0 - keep)
    src_t = np.linspace(start, start + keep, steps)
    old_t = np.linspace(0.0, 1.0, steps)
    out = np.stack([np.interp(src_t, old_t, seq[:, j]) for j in range(dims)], axis=1)
    present = np.stack([np.interp(src_t, old_t, present[:, j].astype(float)) for j in range(dims)],
                       axis=1) > 0.5
    out = out * rng.uniform(0.9, 1.1)                    # slightly bigger / smaller
    out[:, 0::2] += rng.uniform(-0.05, 0.05) * present[:, 0::2]   # shift left/right
    out[:, 1::2] += rng.uniform(-0.05, 0.05) * present[:, 1::2]   # shift up/down
    out += rng.normal(0.0, 0.01, out.shape) * present    # tiny jitter
    return (out * present).astype(np.float32)


def process_video(path, holistic, flip=True, frame_step=1):
    """Video file -> one (30, 98) sequence, or None if no usable hands/body were found.

    frame_step=2 analyses every 2nd frame (about twice as fast, slightly less detail).
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0
    times, feats, index = [], [], 0
    while True:
        if index % frame_step:             # skipped frame: just move on
            if not cap.grab():
                break
            index += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if flip:
            frame = cv2.flip(frame, 1)   # match the mirrored webcam view used in live mode
        result = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        f = holistic_features(result)
        if f is not None:
            times.append(index / fps)
            feats.append(f)
        index += 1
    cap.release()
    return prepare_sequence(times, feats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="folder containing one sub-folder of clips per word")
    parser.add_argument("--augment", type=int, default=10, help="extra varied copies per clip")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", default="", help="comma-separated words to process, e.g. hello,thank_you")
    parser.add_argument("--max-per-word", type=int, default=0, help="use at most this many clips per word")
    parser.add_argument("--frame-step", type=int, default=1, help="2 = analyse every 2nd frame (faster)")
    args = parser.parse_args()

    import mediapipe as mp
    rng = np.random.default_rng(args.seed)

    wanted = {w.strip().lower() for w in args.only.split(",") if w.strip()}
    for wdir in sorted(p for p in Path(args.root).iterdir() if p.is_dir()):
        word = wdir.name.lower()
        if wanted and word not in wanted:
            continue
        out_dir = Path("sequences") / word
        out_dir.mkdir(parents=True, exist_ok=True)
        done = skipped = 0
        for video in sorted(wdir.iterdir()):
            if video.suffix.lower() not in VIDEO_EXTS:
                continue
            if args.max_per_word and done + skipped >= args.max_per_word:
                break
            # a fresh tracker per video so tracking doesn't carry over between clips
            with mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                                min_tracking_confidence=0.5) as holistic:
                seq = process_video(video, holistic, flip=not args.no_flip,
                                    frame_step=max(1, args.frame_step))
            if seq is None:
                print(f"  skipped {video.name}: hands/upper body not found")
                skipped += 1
                continue
            stem = video.stem.replace("__", "_")
            np.save(out_dir / f"{stem}__orig.npy", seq)
            for k in range(args.augment):
                np.save(out_dir / f"{stem}__aug{k}.npy", augment(seq, rng))
            done += 1
        print(f"{word}: {done} clip(s) processed, {skipped} skipped", flush=True)
    print("Done. Next: python train_words.py")


if __name__ == "__main__":
    main()
