#!/usr/bin/env python3
"""Attribution µw (lift) on the manual boxes against the two null models (called T3e in the working notes).

Paper results (recorded run: results/box_attribution/20261006-100109/summary.json; Fig. 7 and Table 3):
  - µw on the annotated box: GRUMobileNet 1.64 [1.46, 1.82]; null 1 (randomized head) 0.67; null 2 (video control)
    0.85                       -> primary.box_cells.lift.{grumobilenet, random_head, video_control}
  - paired differences 0.98 [0.77, 1.16] and 0.80 [0.58, 1.02]
                               -> primary.box_cells.evidence.grumobilenet.paired_difference.{random_head, video_control}

Same as the calibrated null of interventions.py (same models, nulls, windows and methods), changing only the region:
  box_cells     = 4x4 cells touched by the manual box (primary)
  exact_box     = box in pixels: mass = Σ cell mass × fraction of the cell covered, ÷ area fraction of the box
  motion_region = motion region of controls.py / interventions.py (reference; should reproduce interventions.py)
Does not read the frame cache: decodes the videos and applies Resize(128) + CenterCrop(128).

Inputs: config.SOTAD_ROOT (test/<id>.mp4, Appendix.txt), config.DETECTOR_CKPT, annotations/keyframe_boxes.csv and the
video-control head of a controls run (default: the cited run, controls.CONTROLS_RUN; another one with --controls-run).
Usage (from the repo root; about 1 min):  python sotad_diagnosis/box_attribution.py
Output: sotad_diagnosis/results/box_attribution/<date-time>/ (or --out <dir>/<date-time>): per_video.csv,
summary.json.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

import controls
from box_interventions import box_cells
from controls import (CELL, CONTROLS_RUN, GRID, METHODS, NCELL, Head, agg, cams, config, eval_rows, lift,
                      load_backbone, load_head, load_keyframes, mass, motion_region, new_run_dir, write_csv)
from interventions import load_control_heads
from motion_region import ANNOT, decode

HERE = Path(__file__).resolve().parent
SEED = 20260929  # seed of the cited controls run (same random head)
S = 128
REGIONS = ("box_cells", "exact_box", "motion_region")
NULLS = ("random_head", "video_control")


def cell_overlap(b):
    """Fraction of each cell covered by the box (16,) and area fraction of the box."""
    x0, y0, x1, y1 = b
    o = np.zeros(NCELL)
    for gy in range(GRID):
        for gx in range(GRID):
            o[gy * GRID + gx] = max(0.0, min(x1, (gx + 1) * CELL) - max(x0, gx * CELL)) * \
                                max(0.0, min(y1, (gy + 1) * CELL) - max(y0, gy * CELL)) / CELL ** 2
    return o, (x1 - x0) * (y1 - y0) / S ** 2


def lift_exact(M, overlap, area):
    return (mass(M) * overlap[None]).sum(1) / area


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls-run", default=str(CONTROLS_RUN), help="controls.py run folder with the heads")
    ap.add_argument("--out", default=str(HERE / "results" / "box_attribution"), help="parent folder of the run folder")
    args = ap.parse_args()
    torch.set_num_threads(2)
    t_start = time.time()
    out = new_run_dir(args.out)
    kf = load_keyframes()
    with ANNOT.open() as f:
        ann = {int(r["video"]): r for r in csv.DictReader(f)}
    backbone = load_backbone(config.DETECTOR_CKPT)
    reals = {"grumobilenet": load_head(config.DETECTOR_CKPT)}
    video_control = load_control_heads(args.controls_run)["video_control"]
    torch.manual_seed(SEED + 7)  # same random head as in controls.py / interventions.py
    heads = {**reals, "random_head": Head().eval(), "video_control": video_control}

    rows = []
    vids = sorted(v for v in ann if ann[v]["cx0"] != "")
    for n_i, vid in enumerate(vids, start=1):
        fr, _ = decode(vid)
        K, r0 = len(fr), -(-kf[vid] // 5)
        if r0 < 1 or r0 >= K:
            continue
        box = tuple(float(ann[vid][k]) for k in ("cx0", "cy0", "cx1", "cy1"))
        regions = {"box_cells": box_cells(box), "motion_region": motion_region(fr, r0)}
        overlap, area = cell_overlap(box)
        maps = controls.feature_maps(backbone, fr)
        pre, post = eval_rows(K, r0)
        rr = np.unique(np.concatenate([[0], pre, post])).astype(int)
        is_pre, is_short = rr < r0, (rr >= r0) & (rr <= r0 + 10)
        for name, head in heads.items():
            C, _ = cams(head, maps, rr)
            for meth in METHODS:
                L = {reg: lift(C[meth], cells) for reg, cells in regions.items()}
                L["exact_box"] = lift_exact(C[meth], overlap, area)
                for reg, v in L.items():
                    rows.append({"video": vid, "model": name, "method": meth, "region": reg,
                                 "lift_post": float(v[~is_pre].mean()) if (~is_pre).any() else None,
                                 "lift_post_2s": float(v[is_short].mean()) if is_short.any() else None,
                                 "lift_pre": float(v[is_pre].mean()) if is_pre.any() else None})
        if n_i % 10 == 0 or n_i == len(vids):
            print(f"  {n_i}/{len(vids)} ({time.time() - t_start:.0f} s)", flush=True)
    write_csv(out / "per_video.csv", rows)

    def pick(model, reg, key, meth="gradcam", keep=lambda v: True):
        return {r["video"]: r[key] for r in rows if r["model"] == model and r["region"] == reg
                and r["method"] == meth and keep(r["video"]) and r[key] is not None}

    summary = {"n_videos": len({r["video"] for r in rows})}
    variants = {"primary": ("lift_post", lambda v: True), "without_39": ("lift_post", lambda v: v != 39),
                "post_up_to_2s": ("lift_post_2s", lambda v: True)}
    for vname, (key, keep) in variants.items():
        block = {}
        for reg in REGIONS:
            lifts = {m: agg(list(pick(m, reg, key, keep=keep).values())) for m in heads}
            ev = {}
            for m in reals:
                rp = pick(m, reg, key, keep=keep)
                d = {nl: agg([rp[v] - pn[v] for v in rp if v in pn])
                     for nl, pn in ((nl, pick(nl, reg, key, keep=keep)) for nl in NULLS)}
                ev[m] = {"paired_difference": d, "evidence": all((x["ci95"][0] or -1) > 0 for x in d.values())}
            block[reg] = {"lift": lifts, "evidence": ev}
        summary[vname] = block
    summary["other_methods_box_cells"] = {f"{m}/{meth}": agg(list(pick(m, "box_cells", "lift_post", meth).values()))
                                          for m in heads for meth in METHODS if meth != "gradcam"}
    summary["seconds"] = round(time.time() - t_start, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(f"output: {out} ({summary['seconds']} s)")


if __name__ == "__main__":
    main()
