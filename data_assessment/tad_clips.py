"""Length, resolution and active area of every TAD clip (the TAD input of Table 2).

Same measurement as for SO-TAD (`active_area.py`), so the two datasets can be compared: the clip length comes from
OpenCV's CAP_PROP_FRAME_COUNT and the active area from `sotad.measure_content_box` over frames at 20, 35, 50, 65 and
80% of the clip. Unlike SO-TAD (always 640x480), TAD resolutions vary, so the raw resolution is stored too.
CAP_PROP_FRAME_COUNT is wrong for some containers, so it is checked against a full decode of 25 random clips
(random.Random(7)); the check is stored in the output.

Inputs : config.TAD_ROOT/{train,test}/<class folder>/<video>.mp4 (label = accident if the path has an
         "accident" folder)
Output : results/tad_clips.json  {"frame_count_check": {...}, "rows": [{video, split, accident, n_frames, w, h, fps,
         box: [y0, y1, x0, x1]}]}

Usage (from the repository root):
    python data_assessment/tad_clips.py [--out results/tad_clips.json]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import sotad as S  # noqa: E402

ROOT = Path(config.TAD_ROOT)
RESULTS = HERE / "results"


def probe(p: Path):
    """Metadata and active area, opening the file once. None if the clip has no frame count."""
    cap = cv2.VideoCapture(str(p))
    try:
        nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if nf <= 0:
            return None
        want = {max(0, int(nf * f)) for f in (.2, .35, .5, .65, .8)}
        frames, i, stop = [], 0, max(want)
        while i <= stop:
            ok, fr = cap.read()
            if not ok:
                break
            if i in want:
                frames.append(fr)
            i += 1
        box = S.measure_content_box(frames).as_tuple() if frames else None
        return nf, w, h, fps, box
    finally:
        cap.release()


def validate_counts(rows, k=25):
    """Check CAP_PROP_FRAME_COUNT against a full decode on a random sample of clips."""
    sample = random.Random(7).sample(rows, min(k, len(rows)))
    diffs = []
    for r in sample:
        cap = cv2.VideoCapture(str(ROOT / r["video"]))
        n = 0
        while cap.grab():
            n += 1
        cap.release()
        diffs.append(n - r["n_frames"])
    d = np.array(diffs)
    return {"n_sample": int(d.size), "exact": int((d == 0).sum()),
            "max_abs_diff": int(np.abs(d).max()) if d.size else 0,
            "mean_diff": float(d.mean()) if d.size else 0.0}


def main():
    ap = argparse.ArgumentParser(description="Length, resolution and active area of every TAD clip.")
    ap.add_argument("--out", type=Path, default=RESULTS / "tad_clips.json")
    args = ap.parse_args()
    vids = sorted(p for p in ROOT.rglob("*.mp4") if p.is_file())
    rows = []
    print(f"reading {len(vids)} videos...", flush=True)
    for k, p in enumerate(vids):
        m = probe(p)
        if m is None:
            print(f"  SKIPPED (no metadata): {p.name}")
            continue
        nf, w, h, fps, box = m
        rel = str(p.relative_to(ROOT))
        rows.append({"video": rel, "split": rel.lower().split("/")[0],
                     "accident": int("/accident" in "/" + rel.lower()),
                     "n_frames": nf, "w": w, "h": h, "fps": fps, "box": box})
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(vids)}", flush=True)
    print(f"read {len(rows)} of {len(vids)}")
    v = validate_counts(rows)
    print(f"frame-count check: {v['exact']}/{v['n_sample']} exact, largest difference {v['max_abs_diff']} frames")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"frame_count_check": v, "rows": rows}, fh, indent=1)
    return rows


if __name__ == "__main__":
    main()
