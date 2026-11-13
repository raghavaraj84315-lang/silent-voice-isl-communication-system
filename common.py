"""Shared helpers: turn MediaPipe hand landmarks into a normalized feature vector.

Indian Sign Language uses TWO hands for many signs, so we track up to two hands
and keep their positions relative to each other.
"""
import numpy as np
import warnings

# MediaPipe + newer protobuf print this harmless warning for every video frame; hide it
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")


def open_camera(index=None):
    """Open the camera used by all the live scripts.

    Which camera: the number in camera.txt (create it with list_cameras.py), or the CAMERA
    environment variable, otherwise camera 0 (usually the laptop's built-in webcam).
    """
    import os

    import cv2

    if index is None:
        env = os.environ.get("CAMERA", "").strip()
        if env.isdigit():
            index = int(env)
        elif os.path.exists("camera.txt"):
            try:
                index = int(open("camera.txt").read().strip())
            except ValueError:
                index = 0
        else:
            index = 0

    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY   # DirectShow suits USB cams on Windows
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}.\n"
                           "Close any other app using the camera (Zoom, Teams, Browser, Windows Camera).")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)     # 640x480 keeps things fast on a CPU
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print(f"Using camera {index}")
    return cap


def list_available_cameras(current_index=0, max_to_test=4):
    """Enumerate accessible camera indices on the system."""
    import os
    import cv2

    devices = [{"index": current_index, "name": f"Camera {current_index} (Current)"}]
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    for i in range(max_to_test):
        if i == current_index:
            continue
        try:
            temp = cv2.VideoCapture(i, backend)
            if temp.isOpened():
                devices.append({"index": i, "name": f"Camera {i}"})
                temp.release()
        except Exception:
            pass
    devices.sort(key=lambda d: d["index"])
    return devices



def hands_to_features(multi_hand_landmarks):
    """Up to 2 hands x 21 landmarks x (x, y, z) -> 126 numbers.

    Returns None if no hand is visible. Position- and size-independent, but the
    two hands stay in the right place relative to each other. If only one hand
    is visible, the second slot is filled with zeros.
    """
    if not multi_hand_landmarks:
        return None

    hands = [
        np.array([[lm.x, lm.y, lm.z] for lm in h.landmark], dtype=np.float32)
        for h in multi_hand_landmarks[:2]
    ]
    hands.sort(key=lambda pts: pts[0][0])      # left-most wrist first -> stable order

    all_pts = np.concatenate(hands)
    center = all_pts.mean(axis=0)
    scale = float(np.max(np.abs(all_pts - center)))
    if scale == 0:
        scale = 1.0

    feats = [((pts - center) / scale).flatten() for pts in hands]
    while len(feats) < 2:
        feats.append(np.zeros(63, dtype=np.float32))
    return np.concatenate(feats)                # 126 values


# Word recognition with MediaPipe Holistic
SEQ_LEN = 30                       # every sign is resampled to 30 time steps
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]   # nose, shoulders, elbows, wrists
FEATURES_PER_FRAME = 14 + 42 + 42  # pose xy + left-hand xy + right-hand xy = 98


def holistic_features(results):
    """One video frame -> 98 numbers, relative to the signer's shoulders.

    Positions are measured from the middle of the shoulders and divided by the
    shoulder width, so it doesn't matter how far you stand from the camera.
    Returns None if the upper body is not visible.
    """
    if results.pose_landmarks is None:
        return None
    pose = np.array([[lm.x, lm.y] for lm in results.pose_landmarks.landmark], dtype=np.float32)
    left_sh, right_sh = pose[11], pose[12]
    center = (left_sh + right_sh) / 2
    width = float(np.linalg.norm(left_sh - right_sh))
    if width < 1e-3:
        return None

    def norm(points):
        return ((points - center) / width).flatten()

    parts = [norm(pose[POSE_IDX])]
    for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
        if hand is None:
            parts.append(np.zeros(42, dtype=np.float32))
        else:
            pts = np.array([[lm.x, lm.y] for lm in hand.landmark], dtype=np.float32)
            parts.append(norm(pts))
    return np.concatenate(parts)


def hands_visible(features):
    """True if at least one hand was detected in this frame."""
    return bool(np.any(features[14:]))


def prepare_sequence(times, feats, length=SEQ_LEN, min_frames=8):
    """Cut a recording down to the frames where hands are visible, then resample it to
    exactly `length` time steps. Returns an array (length, 98) or None if unusable."""
    keep = [i for i, f in enumerate(feats) if hands_visible(f)]
    if len(keep) < min_frames:
        return None
    a, b = keep[0], keep[-1]
    t = np.asarray(times[a:b + 1], dtype=np.float64)
    frames = np.asarray(feats[a:b + 1], dtype=np.float32)
    span = t[-1] - t[0]
    if span < 0.3:
        return None
    t = (t - t[0]) / span
    target = np.linspace(0.0, 1.0, length)
    out = np.stack([np.interp(target, t, frames[:, j]) for j in range(frames.shape[1])], axis=1)
    return out.astype(np.float32)


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
