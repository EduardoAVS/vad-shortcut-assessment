#!/usr/bin/env python3
"""Event necessity redone with the manual boxes (called T3d in the working notes).

Paper result (recorded run: results/box_interventions/20261006-093346/summary.json): event necessity of GRUMobileNet
0.015 [−0.01, 0.04] (annotated box vs random region) -> all.grumobilenet.box_cells.event_minus_random.

Same as the event ablation of interventions.py on GRUMobileNet, changing only the ablated region:
  motion_region = motion region of controls.py / interventions.py (reference; should reproduce interventions.py)
  box_cells     = 4x4 cells touched by the manual box; random control = same number of random cells outside them
  exact_box     = only the pixels of the rectangle; random control = rectangle of the same size drawn without
                  overlapping the box
Ablated rows [kf − 1 s, kf + 2 s] get the background pixels (pre-event median); score = mean P(accident) over the
windows r0+2, r0+6, r0+10. Does not read the frame cache: decodes the videos and applies Resize(128) + CenterCrop(128).

Inputs: config.SOTAD_ROOT (test/<id>.mp4, Appendix.txt), config.DETECTOR_CKPT, annotations/keyframe_boxes.csv.
Usage (from the repo root; about 1 min):  python sotad_diagnosis/box_interventions.py
Output: sotad_diagnosis/results/box_interventions/<date-time>/ (or --out <dir>/<date-time>): per_video.csv,
summary.json.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from controls import (CELL, NCELL, agg, config, load_backbone, load_head, load_keyframes, motion_region, new_run_dir,
                      write_csv)
from interventions import background, paste_cells, pooled, score_windows
from motion_region import ANNOT, VIDEOS, decode

HERE = Path(__file__).resolve().parent
SEED = 20260929  # seed of the cited interventions run (same random draws for the motion-region control)
S = 128
REGIONS = ("motion_region", "box_cells", "exact_box")


def box_cells(b):
    x0, y0, x1, y1 = b
    return [gy * 4 + gx for gy in range(4) for gx in range(4)
            if min(x1, (gx + 1) * CELL) > max(x0, gx * CELL) and min(y1, (gy + 1) * CELL) > max(y0, gy * CELL)]


def pixel_box(b):
    x0, y0, x1, y1 = b
    return int(math.floor(x0)), int(math.floor(y0)), int(math.ceil(x1)), int(math.ceil(y1))


def random_rect(b, rng, tries=2000):
    x0, y0, x1, y1 = b
    w, h = x1 - x0, y1 - y0
    if w >= S or h >= S:
        return None
    for _ in range(tries):
        rx, ry = int(rng.integers(0, S - w + 1)), int(rng.integers(0, S - h + 1))
        if rx + w <= x0 or rx >= x1 or ry + h <= y0 or ry >= y1:
            return rx, ry, rx + w, ry + h
    return None


def ablate_rect(backbone, frames, F, rows, rect, bg):
    G = F.copy()
    if rect is None or not len(rows):
        return None
    x0, y0, x1, y1 = rect
    f = np.array(frames[rows])
    f[:, y0:y1, x0:x1] = bg[y0:y1, x0:x1]
    G[rows] = pooled(backbone, f)
    return G


def ablate_cells(backbone, frames, F, rows, cells, bg):
    G = F.copy()
    if len(cells) and len(rows):
        G[rows] = pooled(backbone, paste_cells(frames, rows, cells, bg)[rows])
    return G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(HERE / "results" / "box_interventions"), help="parent folder of the run folder")
    args = ap.parse_args()
    torch.set_num_threads(2)
    t_start = time.time()
    out = new_run_dir(args.out)
    kf = load_keyframes()
    with ANNOT.open() as f:
        ann = {int(r["video"]): r for r in csv.DictReader(f)}
    nor = sorted(v for v in (int(p.stem) for p in VIDEOS.glob("*.mp4")) if v >= 400)
    backbone = load_backbone(config.DETECTOR_CKPT)
    heads = {"grumobilenet": load_head(config.DETECTOR_CKPT)}

    rows_out = []
    vids = sorted(v for v in ann if ann[v]["cx0"] != "")
    for n_i, vid in enumerate(vids, start=1):
        fr, _ = decode(vid)
        K, r0 = len(fr), -(-kf[vid] // 5)
        if r0 < 1 or r0 >= K:
            continue
        rows_c = np.unique(np.clip([r0 + 2, r0 + 6, r0 + 10], 0, K - 1))
        abl = np.arange(max(0, r0 - 5), min(K - 1, r0 + 10) + 1)
        bg = background(fr, min(10, r0))
        F = pooled(backbone, fr)

        # Same sequence of draws as interventions.py for the random control of the motion region.
        rng_v = np.random.default_rng([SEED, 300000 + vid])
        rng_v.choice(nor)
        rng_v.choice(list(range(NCELL - 1)))
        region = motion_region(fr, r0)
        comp = [c for c in range(NCELL) if c not in region]
        rand_region = sorted(rng_v.choice(comp, size=min(len(region), len(comp)), replace=False).tolist()) if comp else []

        rng_b = np.random.default_rng([SEED, 700000 + vid])
        box = tuple(float(ann[vid][k]) for k in ("cx0", "cy0", "cx1", "cy1"))
        bcells = box_cells(box)
        bcomp = [c for c in range(NCELL) if c not in bcells]
        bcells_rd = sorted(rng_b.choice(bcomp, size=min(len(bcells), len(bcomp)), replace=False).tolist()) if bcomp else []
        prect = pixel_box(box)
        prect_rd = random_rect(prect, rng_b)

        feats = {"motion_region_event": ablate_cells(backbone, fr, F, abl, region, bg),
                 "motion_region_random": ablate_cells(backbone, fr, F, abl, rand_region, bg),
                 "box_cells_event": ablate_cells(backbone, fr, F, abl, bcells, bg),
                 "box_cells_random": ablate_cells(backbone, fr, F, abl, bcells_rd, bg) if bcells_rd else None,
                 "exact_box_event": ablate_rect(backbone, fr, F, abl, prect, bg),
                 "exact_box_random": ablate_rect(backbone, fr, F, abl, prect_rd, bg)}
        for name, head in heads.items():
            r = {"video": vid, "model": name, "flag": ann[vid]["flag"],
                 "n_cells_motion_region": len(region), "n_cells_box": len(bcells),
                 "box_area_frac": round((prect[2] - prect[0]) * (prect[3] - prect[1]) / S ** 2, 4),
                 "motion_region_touches_box": int(bool(set(region) & set(bcells))),
                 "orig": float(score_windows(head, F, rows_c).mean())}
            for k, G in feats.items():
                r[k] = float(score_windows(head, G, rows_c).mean()) if G is not None else None
            rows_out.append(r)
        if n_i % 10 == 0 or n_i == len(vids):
            print(f"  {n_i}/{len(vids)} ({time.time() - t_start:.0f} s)", flush=True)

    write_csv(out / "per_video.csv", rows_out)

    def drops(R, reg):
        ev = [r["orig"] - r[f"{reg}_event"] for r in R if r[f"{reg}_event"] is not None]
        pairs = [(r[f"{reg}_random"] - r[f"{reg}_event"]) for r in R
                 if r[f"{reg}_event"] is not None and r[f"{reg}_random"] is not None]
        ct = [r["orig"] - r[f"{reg}_random"] for r in R if r[f"{reg}_random"] is not None]
        return {"drop_event": agg(ev), "drop_random": agg(ct), "event_minus_random": agg(pairs)}

    subsets = {"all": lambda r: True, "without_39": lambda r: r["video"] != 39,
               "motion_region_missed_box": lambda r: r["motion_region_touches_box"] == 0}
    summary = {"n_videos": len({r["video"] for r in rows_out}), "seconds": None}
    for sname, cond in subsets.items():
        summary[sname] = {m: {reg: drops([r for r in rows_out if r["model"] == m and cond(r)], reg)
                              for reg in REGIONS} for m in heads}
    g = summary["all"]["grumobilenet"]["box_cells"]["event_minus_random"]
    if g["mean"] is not None and g["mean"] <= 0.10 and g["ci95"][1] < 0.3:
        summary["reading_grumobilenet"] = "confirmed: the event contributes little"
    elif g["mean"] is not None and g["mean"] >= 0.3:
        summary["reading_grumobilenet"] = "interventions.py underestimated the event; the conclusion changes"
    else:
        summary["reading_grumobilenet"] = "partial"
    R = [r for r in rows_out if r["model"] == "grumobilenet"]
    summary["descriptive"] = {"median_box_cells": float(np.median([r["n_cells_box"] for r in R])),
                              "median_motion_region_cells": float(np.median([r["n_cells_motion_region"] for r in R])),
                              "median_box_area": float(np.median([r["box_area_frac"] for r in R])),
                              "exact_box_random_unavailable": int(sum(r["exact_box_random"] is None for r in R))}
    summary["seconds"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps({k: summary[k] for k in ("n_videos", "reading_grumobilenet", "descriptive", "seconds")}, indent=1,
                     ensure_ascii=False))
    print(f"output: {out}")


if __name__ == "__main__":
    main()
