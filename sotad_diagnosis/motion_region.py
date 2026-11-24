#!/usr/bin/env python3
"""Validation of the motion region against the manual boxes (called T7 in the working notes).

Paper result (recorded run: results/motion_region/20261006-092505/summary.json): the motion region contains the box
centre in 81% [73, 89] of the test accidents vs 22% by chance -> primary.cache.pointing, pointing_ci95, chance.

Does not read the frame cache. Decodes each original video sequentially (as detector/frame_cache.py), keeps 1 frame
in 5 and applies Resize(128) + CenterCrop(128); the motion region comes from the same `motion_region` of
controls.py. Descriptive variant ("full"): the same rule, with the |difference| energy computed at 640x480 inside the
crop and summed over the same 16 cells.

Per-video metrics (manual box clipped to the crop, in 128-px coordinates):
  pointing = cell of the box centre ∈ region
  hit      = some cell touched by the box ∈ region
  coverage = fraction of the box area inside the region cells
  chance   = |region| / 16
Criterion: valid if pointing >= 70% and the 95% CI (bootstrap over videos) of pointing − chance > 0; invalid if < 50%.

Inputs: config.SOTAD_ROOT (test/<id>.mp4, Appendix.txt), annotations/keyframe_boxes.csv.
Usage (from the repo root; about 1 min):  python sotad_diagnosis/motion_region.py
Output: sotad_diagnosis/results/motion_region/<date-time>/ (or --out <dir>/<date-time>): per_video.csv, summary.json.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from controls import CELL, GRID, NCELL, config, load_keyframes, motion_region, new_run_dir

HERE = Path(__file__).resolve().parent
VIDEOS = config.SOTAD_ROOT / "test"
ANNOT = HERE / "annotations" / "keyframe_boxes.csv"
S, STRIDE, SEED, NBOOT = 128, 5, 20261006, 10000
GEOM = transforms.Compose([transforms.Resize(S), transforms.CenterCrop(S)])


def decode(vid):
    """Rows in the cache format (1 frame in 5, 128x128 RGB) and the same frames at full resolution (gray)."""
    cap = cv2.VideoCapture(str(VIDEOS / f"{vid}.mp4"))
    small, full, i = [], [], 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % STRIDE == 0:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            small.append(np.asarray(GEOM(Image.fromarray(rgb)), dtype=np.uint8))
            full.append(rgb.astype(np.float32).mean(-1))
        i += 1
    cap.release()
    return np.stack(small), np.stack(full)


def cell_index(h, w):
    """For each pixel of the original frame, the 4x4 crop cell it falls in (−1 outside the crop)."""
    nw, nh = (int(S * w / h), S) if w >= h else (S, int(S * h / w))
    left, top = int(round((nw - S) / 2.0)), int(round((nh - S) / 2.0))
    xc = (np.arange(w) + 0.5) * nw / w - left
    yc = (np.arange(h) + 0.5) * nh / h - top
    cx = np.where((xc >= 0) & (xc < S), (xc // CELL).astype(int), -1)
    cy = np.where((yc >= 0) & (yc < S), (yc // CELL).astype(int), -1)
    idx = cy[:, None] * GRID + cx[None, :]
    return np.where((cy[:, None] >= 0) & (cx[None, :] >= 0), idx, -1)


def motion_region_full(full, r0, idx):
    """Same rule as `motion_region`, with the energy summed per cell at full resolution."""
    lo, hi = max(1, r0 - 5), min(len(full) - 1, r0 + 10)
    if hi < lo:
        return list(range(NCELL))
    e = np.abs(np.diff(full[lo - 1:hi + 1], axis=0)).sum(0)
    m = idx >= 0
    ce = np.bincount(idx[m], weights=e[m], minlength=NCELL)
    order = np.argsort(-ce)
    cum = np.cumsum(ce[order]) / max(ce.sum(), 1e-9)
    return order[:int(np.searchsorted(cum, 0.5) + 1)].tolist()


def box_metrics(box, region):
    x0, y0, x1, y1 = box
    reg = set(region)
    cx, cy = min(int((x0 + x1) / 2 // CELL), GRID - 1), min(int((y0 + y1) / 2 // CELL), GRID - 1)
    touched, inside = set(), 0.0
    for gy in range(GRID):
        for gx in range(GRID):
            ov = max(0.0, min(x1, (gx + 1) * CELL) - max(x0, gx * CELL)) * \
                 max(0.0, min(y1, (gy + 1) * CELL) - max(y0, gy * CELL))
            if ov > 0:
                touched.add(gy * GRID + gx)
                if gy * GRID + gx in reg:
                    inside += ov
    area = (x1 - x0) * (y1 - y0)
    return {"pointing": int(cy * GRID + cx in reg), "hit": int(bool(touched & reg)),
            "coverage": inside / area if area > 0 else 0.0, "chance": len(reg) / NCELL,
            "n_box_cells": len(touched)}


def summarize(rows, key, rng):
    if not rows:
        return None
    p = np.array([r[f"{key}_pointing"] for r in rows], float)
    a = np.array([r[f"{key}_chance"] for r in rows], float)
    d = p - a
    boots = d[rng.integers(0, len(d), (NBOOT, len(d)))].mean(1)
    bp = p[rng.integers(0, len(p), (NBOOT, len(p)))].mean(1)
    out = {"n": len(rows), "pointing": p.mean(), "pointing_ci95": np.percentile(bp, [2.5, 97.5]).tolist(),
           "chance": a.mean(), "pointing_minus_chance": d.mean(), "diff_ci95": np.percentile(boots, [2.5, 97.5]).tolist(),
           "hit": np.mean([r[f"{key}_hit"] for r in rows]),
           "median_coverage": float(np.median([r[f"{key}_coverage"] for r in rows])),
           "median_region_cells": float(np.median([r[f"{key}_chance"] * NCELL for r in rows]))}
    if out["pointing"] >= 0.70 and out["diff_ci95"][0] > 0:
        out["verdict"] = "valid"
    elif out["pointing"] < 0.50:
        out["verdict"] = "invalid"
    else:
        out["verdict"] = "partial"
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "results" / "motion_region"), help="parent folder of the run folder")
    args = ap.parse_args()
    t0 = time.time()
    out_dir = new_run_dir(args.out)
    kf = load_keyframes()
    with ANNOT.open() as f:
        ann = {int(r["video"]): r for r in csv.DictReader(f)}
    rows = []
    for vid in sorted(ann):
        a = ann[vid]
        r0 = -(-kf[vid] // STRIDE)
        row = {"video": vid, "r0": r0, "flag": a["flag"], "frac_in_crop": float(a["frac_in_crop"] or 0)}
        small, full = decode(vid)
        row["rows"] = len(small)
        if a["cx0"] == "":
            row["excluded"] = "box outside the crop"
        elif r0 < 1:
            row["excluded"] = "r0 = 0"
        else:
            row["excluded"] = ""
        box = tuple(float(a[k]) for k in ("cx0", "cy0", "cx1", "cy1")) if a["cx0"] else None
        regions = {"cache": motion_region(small, r0),
                   "full": motion_region_full(full, r0, cell_index(*full.shape[1:]))}
        for key, reg in regions.items():
            row[f"{key}_region"] = " ".join(map(str, sorted(reg)))
            if box:
                for m, v in box_metrics(box, reg).items():
                    row[f"{key}_{m}"] = v
        rows.append(row)
        print(f"  video {vid}: region {row['cache_region']} | pointing {row.get('cache_pointing', '-')}", flush=True)

    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (out_dir / "per_video.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    rng = np.random.default_rng(SEED)
    prim = [r for r in rows if not r["excluded"]]
    subsets = {"primary": prim,
               "without_39": [r for r in prim if r["video"] != 39],
               "box_fully_in_crop": [r for r in prim if r["frac_in_crop"] >= 1.0]}
    summary = {"excluded": {r["video"]: r["excluded"] for r in rows if r["excluded"]},
               "agreement_cache_full": round(float(np.mean([r["cache_region"] == r["full_region"] for r in prim])), 4),
               "seconds": round(time.time() - t0, 1)}
    for name, sub in subsets.items():
        summary[name] = {k: summarize(sub, k, rng) for k in ("cache", "full")}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
