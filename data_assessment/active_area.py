"""Active area (non-black content rectangle inside the letterbox) of every SO-TAD clip, all 2,186 clips.

For each clip, frames n/4, n/2 and 3n/4 are decoded (solid-green corrupt frames dropped) and the active area is
the rectangle whose rows/columns of the per-pixel max over those frames are brighter than a small threshold
(`sotad.measure_content_box`). The clip length n (frames) is stored too. This CSV is the SO-TAD input of the
length and active-area rules of Table 2 (`length_and_area_rules.py`, `active_area_null.py`) and of the Sec. 5.1
clip statistics (`sotad_clip_lengths.py`, `sotad_area_and_length_checks.py`).

The printed analysis checks how well "active area is not ~640x360 => accident" separates the classes for several
pixel tolerances (the boundary is fuzzy by +-2 px because compression leaves near-black rows at the edge).

Results are streamed to the CSV per clip, so an interrupted run resumes where it stopped.

Inputs : config.SOTAD_ROOT/{train,test}/*.mp4
Output : results/active_area.csv (split, video, accident, n_frames, y0, y1, x0, x1, height, width)

Usage (from the repository root):
    python data_assessment/active_area.py                       # measure (resumes) + analyse
    python data_assessment/active_area.py --analyze-only
    python data_assessment/active_area.py --out /tmp/active_area.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sotad as S  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "active_area.csv"
COLS = ["split", "video", "accident", "n_frames", "y0", "y1", "x0", "x1",
        "height", "width"]


def measure_all(out: Path) -> None:
    done = set()
    if out.exists():
        with open(out) as fh:
            for r in csv.DictReader(fh):
                done.add((r["split"], int(r["video"])))
        print(f"resuming: {len(done)} clips already measured")

    todo = [(sp, v) for sp in ("train", "test") for v in S.list_split(sp)
            if (sp, v) not in done]
    print(f"{len(todo)} clips to measure (probing n/4, n/2, 3n/4)")

    new = not out.exists()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        if new:
            w.writeheader()
        for i, (split, vid) in enumerate(todo, 1):
            path = S.video_path(split, vid)
            n = S.frame_count(path)
            if n < 8:
                continue
            fr = S.read_frames_sparse(path, {n // 4, n // 2, (3 * n) // 4})
            good = [f for f in fr.values() if not S.is_green_corrupt(f)]
            if not good:
                continue
            b = S.measure_content_box(good)
            w.writerow({
                "split": split, "video": vid,
                "accident": int(S.is_accident(vid)), "n_frames": n,
                "y0": b.y0, "y1": b.y1, "x0": b.x0, "x1": b.x1,
                "height": b.height, "width": b.width,
            })
            if i % 100 == 0:
                fh.flush()
                print(f"  {i}/{len(todo)} measured", flush=True)
    print(f"wrote {out}")


def analyze(out: Path) -> None:
    with open(out) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("video", "accident", "n_frames", "y0", "y1", "x0", "x1",
                  "height", "width"):
            r[k] = int(r[k])
    print(f"\nanalysing {len(rows)} clips\n")

    for split in ("train", "test", "both"):
        sub = rows if split == "both" else [r for r in rows if r["split"] == split]
        a = np.array([r["height"] for r in sub if r["accident"]])
        n = np.array([r["height"] for r in sub if not r["accident"]])
        wa = np.array([r["width"] for r in sub if r["accident"]])
        wn = np.array([r["width"] for r in sub if not r["accident"]])
        print(f"=== {split}: {len(a)} accident vs {len(n)} normal ===")
        print(f"  accident height: min={a.min()} max={a.max()} "
              f"n_distinct={len(set(a.tolist()))}")
        print(f"  normal   height: min={n.min()} max={n.max()} "
              f"n_distinct={len(set(n.tolist()))}")
        print(f"  accident width : min={wa.min()} max={wa.max()} "
              f"n_distinct={len(set(wa.tolist()))}")
        print(f"  normal   width : min={wn.min()} max={wn.max()} "
              f"n_distinct={len(set(wn.tolist()))}")
        print(f"  normal clips that are exactly 640x360: "
              f"{int(((wn == 640) & (n == 360)).sum())}/{len(n)} "
              f"({100*((wn == 640) & (n == 360)).mean():.1f}%)")

        # tolerance sensitivity: rule "content is NOT ~640x360 => accident"
        print(f"  {'tol':>4} {'tp':>5} {'fn':>4} {'tn':>5} {'fp':>4} "
              f"{'accuracy':>9} {'precision':>10} {'recall':>8}")
        for tol in (0, 1, 2, 3, 5):
            is_can_a = (np.abs(a - 360) <= tol) & (np.abs(wa - 640) <= tol)
            is_can_n = (np.abs(n - 360) <= tol) & (np.abs(wn - 640) <= tol)
            tp = int((~is_can_a).sum()); fn = int(is_can_a.sum())
            tn = int(is_can_n.sum());    fp = int((~is_can_n).sum())
            acc = (tp + tn) / (len(a) + len(n))
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            print(f"  {tol:>4} {tp:>5} {fn:>4} {tn:>5} {fp:>4} "
                  f"{100*acc:>8.2f}% {100*prec:>9.2f}% {100*rec:>7.2f}%")
        sep = a.max() < n.min() or n.max() < a.min()
        print(f"  perfectly separable by height alone? {'YES' if sep else 'NO'}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Active area of every SO-TAD clip.")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT, help="CSV to write (default: results/active_area.csv)")
    args = ap.parse_args()
    if not args.analyze_only:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        measure_all(args.out)
    analyze(args.out)
