"""Download chosen words from the 'ISL Isolated Word Dataset (40 words)' on Hugging Face.

Dataset: vidit031/isl-isolated-40words  (research / education use; an aggregate of INCLUDE
(CC-BY-4.0), CISLR (AFL-3.0), ISLRTC dictionary clips (MIT) and ISL500 - check each source's
license and cite them in your report).

No extra package is needed (it downloads with plain Python).

Usage:
    python download_isl40.py --list
    python download_isl40.py hello thank_you yes --max 8
    python download_isl40.py hello thank_you please sorry yes no help you me water --max 12
    python download_isl40.py --all                   (all 40 words, about 640 clips / 200 MB)

Clips are saved as word_videos/<word>/<clip>.mp4 and every download is recorded in
word_videos/SOURCES.csv (source dataset, license, link) for your report.
Next step:  python videos_to_sequences.py word_videos --augment 10 --frame-step 2
"""
import argparse
import csv
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO = "vidit031/isl-isolated-40words"
BASE = os.environ.get("ISL40_BASE", f"https://huggingface.co/datasets/{REPO}/resolve/main")


def fetch(rel_path, dest, tries=3):
    """Download one file of the dataset to `dest`."""
    url = f"{BASE}/{urllib.parse.quote(rel_path, safe='/')}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, tries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "silent-voice-student-project"})
            with urllib.request.urlopen(request, timeout=60) as response, open(dest, "wb") as out:
                shutil.copyfileobj(response, out)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == tries:
                raise SystemExit(f"Could not download {rel_path}\n  Reason: {error}\n  Address: {url}\n"
                                 "Check your internet connection, and try opening the address in a browser.")
            time.sleep(2 * attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("words", nargs="*", help="words to download, e.g. hello thank_you")
    parser.add_argument("--list", action="store_true", help="show the 40 words and clip counts")
    parser.add_argument("--all", action="store_true", help="download all 40 words")
    parser.add_argument("--max", type=int, default=0, help="maximum clips per word")
    parser.add_argument("--include-unreviewed", action="store_true",
                        help="also download clips marked 'Needs Manual Review'")
    parser.add_argument("--out", default="word_videos")
    args = parser.parse_args()

    out_root = Path(args.out)
    meta_path = out_root / "metadata.csv"
    print("Downloading the dataset list ...", flush=True)
    fetch("metadata.csv", meta_path)
    df = pd.read_csv(meta_path)
    df["slug"] = df["video_path"].str.split("/").str[0]
    if not args.include_unreviewed and "review_status" in df.columns:
        df = df[df["review_status"].astype(str).str.lower() == "accepted"]

    if args.list or (not args.words and not args.all):
        counts = df["slug"].value_counts().sort_index()
        print(f"{len(df)} usable clips, {len(counts)} words:\n")
        for slug, n in counts.items():
            print(f"  {slug:12s} {n} clips")
        print("\nExample:  python download_isl40.py hello thank_you yes --max 8")
        return

    wanted = sorted(df["slug"].unique()) if args.all else [w.strip().lower().replace(" ", "_") for w in args.words]
    known = set(df["slug"])
    for w in wanted:
        if w not in known:
            print(f"Warning: '{w}' is not in the dataset (use --list to see the words).")

    sources = []
    for word in wanted:
        sub = df[df["slug"] == word]
        if sub.empty:
            continue
        if args.max:
            sub = sub.sample(frac=1, random_state=0).head(args.max)   # a mix of sources, not just the first ones
        count = 0
        for _, row in sub.iterrows():
            target = out_root / word / Path(row["video_path"]).name
            if not (target.exists() and target.stat().st_size > 0):
                fetch(row["video_path"], target)
            sources.append([word, target.name, row.get("dataset", ""), row.get("license", ""),
                            row.get("repository", "")])
            count += 1
            print(f"  {word}: {count}/{len(sub)}", end="\r", flush=True)
        print(f"{word}: {count} clips ready          ", flush=True)

    if sources:
        log = out_root / "SOURCES.csv"
        known_rows = set()
        if log.exists():
            with open(log, newline="", encoding="utf-8") as f:
                known_rows = {(r[0], r[1]) for r in csv.reader(f) if len(r) >= 2}
        with open(log, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not known_rows:
                writer.writerow(["word", "file", "source_dataset", "license", "repository"])
            writer.writerows([r for r in sources if (r[0], r[1]) not in known_rows])
        print(f"\nDone. Source and license list saved in {log}")
        print("Next:  python videos_to_sequences.py word_videos --augment 10 --frame-step 2")


if __name__ == "__main__":
    main()
