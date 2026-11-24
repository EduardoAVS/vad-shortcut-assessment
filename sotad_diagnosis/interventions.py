#!/usr/bin/env python3
"""Pixel interventions (background sufficiency, first frame, event necessity) on the controls and on GRUMobileNet.

Called T3 in the working notes. Paper results produced here (recorded run:
results/interventions/20260929-233839/summary.json):
  - background sufficiency, GRUMobileNet: retention 94% [90, 98]  -> causal.grumobilenet.retention_background_only
  - first frame repeated, GRUMobileNet: retention 93%             -> causal.grumobilenet.retention_first_frame
  - per-video AUC: detection 0.999, background only 0.999, first frame 0.997
                                                                  -> per_video_auc.grumobilenet.{orig_max,
                                                                     background_only, first_frame}
  - Table 3: event control retention 11%, necessity 0.73; video control retention 99%, necessity 0.02
                                                                  -> causal.event_control.*, causal.video_control.*
                                                                     (retention_background_only,
                                                                      drop_event_minus_random)

Criteria fixed before running (verdict GO iff the interventions classify the two controls correctly, on the
marked test clips):
  event_control: mean score drop under event ablation >= 0.3 with 95% CI excluding 0, and background-only
                 retention <= 0.3;
  video_control: background-only retention >= 0.8 and mean drop under event ablation <= 0.05.

Operationalization:
  - score = P(accident) = softmax[1] of the head on the causal window of 16 rows (as in GRUMobileNet);
  - windows measured in each accident video: rows r0+2, r0+6 and r0+10, i.e. up to 2 s after the keyframe
    (r0 = first row with 5r >= kf), clipped to the end of the clip. The video score is the mean of these windows.
    Videos with r0 = 0 are left out (no pre-event segment for the background);
  - background = per-pixel median of the pre-event rows [0, min(10, r0)), without the mark and the event. For
    normal videos, median of the first 10 rows;
  - background only: the whole window becomes the background image. First frame: the whole window becomes row 0;
  - event ablation: in rows [r0 − 5, r0 + 10] (kf − 1 s to kf + 2 s) the cells of the event region receive the
    background pixels. Region = mark cell (controls) or the motion region of controls.py (GRUMobileNet). Random
    ablation = the same number of random cells outside the region, in the same rows;
  - background swap: in every row, the pixels with |gray − gray of the own background| > 25 (foreground) are pasted
    on the background of a random normal test video;
  - retention = mean(score with intervention) / mean(original score) across videos; CIs by bootstrap of videos;
  - GRUMobileNet (descriptive): on the unmarked accidents, with the same interventions, plus the per-video AUC on
    the 479 test videos. There the original score is the max over windows, as in the original protocol, compared
    with background only and first frame;
  - calibrated null for attribution (added after the controls run, before this one): Grad-CAM, Grad-CAM++ and
    LayerCAM lift on the motion region for the random head (same seed as controls.py) and the video control, on
    the same videos and windows as controls.py. Pre-registered reading: a model has attributional evidence of
    looking at the motion iff its post lift (Grad-CAM) exceeds each of the two nulls with the 95% CI of the paired
    difference > 0.

Inputs: config.SOTAD_FRAME_CACHE, config.SOTAD_ROOT/Appendix.txt, config.DETECTOR_CKPT and the heads + mark cells of
a controls run (default: the cited run, controls.CONTROLS_RUN; another one with --controls-run).

Usage (from the repo root; a few minutes on 2 CPU threads):
  tools/run_guarded.sh 2000 work/logs/interventions.log -- python sotad_diagnosis/interventions.py
  Quick test: add --smoke.
Outputs: sotad_diagnosis/results/interventions/<date-time>/ (or --out <dir>/<date-time>): summary.json,
interventions_per_video.csv, video_scores.csv, attribution_null_per_video.csv. Nothing is overwritten.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import controls
from controls import (CELL, CONTROLS_RUN, GRID, METHODS, NCELL, Head, agg, cache, cams, config, eval_rows, lift,
                      load_backbone, load_head, load_keyframes, motion_region, new_run_dir, plant, win_idx, write_csv)

HERE = Path(__file__).resolve().parent


def pooled(backbone, frames):
    return controls.feature_maps(backbone, frames).mean(dim=(2, 3)).numpy()


@torch.no_grad()
def score_windows(head, F, rows):
    X = torch.from_numpy(np.stack([F[win_idx(r)] for r in rows]))
    return torch.softmax(head(X), 1)[:, 1].numpy()


@torch.no_grad()
def score_static(head, f):
    """Score of a window with the same feature in all 16 steps."""
    X = torch.from_numpy(np.repeat(f[None, None].astype(np.float32), controls.T_WIN, axis=1))
    return float(torch.softmax(head(X), 1)[0, 1])


def background(frames, n):
    return np.median(np.asarray(frames[:max(1, n)]), axis=0).astype(np.uint8)


def paste_cells(frames, rows, cells, src):
    f = np.array(frames)
    for c in cells:
        i, j = divmod(int(c), GRID)
        f[rows, CELL * i:CELL * (i + 1), CELL * j:CELL * (j + 1)] = src[CELL * i:CELL * (i + 1), CELL * j:CELL * (j + 1)]
    return f


def swap_background(frames, bg_own, bg_other, thr=25.0):
    fg = np.abs(frames.astype(np.float32).mean(-1) - bg_own.astype(np.float32).mean(-1)[None]) > thr
    return np.where(fg[..., None], frames, bg_other[None])


def ablate(backbone, frames, F, rows, cells, bg):
    """Features with the cells `cells` replaced by the background in rows `rows` (other rows untouched)."""
    G = F.copy()
    if len(cells) and len(rows):
        G[rows] = pooled(backbone, paste_cells(frames, rows, cells, bg)[rows])
    return G


def ratio_ci(num, den, B=1000, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    if len(num) < 2 or den.mean() <= 0:
        return {"mean": None, "ci95": [None, None], "n": int(len(num))}
    idx = np.random.default_rng(seed).integers(0, len(num), (B, len(num)))
    rs = num[idx].mean(1) / np.maximum(den[idx].mean(1), 1e-12)
    return {"mean": float(num.mean() / den.mean()),
            "ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))], "n": int(len(num))}


def summarize(rows, model):
    R = [r for r in rows if r["model"] == model]
    if not R:
        return None
    g = {k: np.array([r[k] for r in R], float)
         for k in ("orig", "background_only", "first_frame", "ablate_event", "ablate_random", "swap_background")}
    return {"n": len(R), "score_orig": agg(g["orig"]),
            "retention_background_only": ratio_ci(g["background_only"], g["orig"]),
            "retention_first_frame": ratio_ci(g["first_frame"], g["orig"]),
            "drop_event": agg(g["orig"] - g["ablate_event"]),
            "drop_random": agg(g["orig"] - g["ablate_random"]),
            "drop_event_minus_random": agg(g["ablate_random"] - g["ablate_event"]),
            "retention_swap_background": ratio_ci(g["swap_background"], g["orig"])}


def load_control_heads(run):
    """Event-control and video-control heads saved by controls.py in the run folder `run`."""
    def head_from(path):
        h = Head()
        h.load_state_dict(torch.load(path, map_location="cpu"))
        return h.eval()
    return {"event_control": head_from(Path(run) / "event_control_head.pt"),
            "video_control": head_from(Path(run) / "video_control_head.pt")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls-run", default=str(CONTROLS_RUN), help="controls.py run folder with the heads")
    ap.add_argument("--seed", type=int, default=None, help="default: the seed of the chosen controls run")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--controls-only", action="store_true", help="only the controls (skip GRUMobileNet and normals)")
    ap.add_argument("--tag", default="", help="suffix of the output folder")
    ap.add_argument("--out", default=str(HERE / "results" / "interventions"), help="parent folder of the run folder")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    ctrl_dir = Path(args.controls_run)
    cfg_path = ctrl_dir / "mark_config.json"
    mcfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"size": CELL, "alpha": 1.0, "seed": 20260929}
    if args.seed is None:
        args.seed = int(mcfg.get("seed", 20260929))
    out = new_run_dir(args.out, args.tag, args.smoke)

    def log(msg):
        print(msg, flush=True)

    t_start = time.time()
    log(f"interventions | output: {out} | control heads: {ctrl_dir}")
    kf = load_keyframes()
    backbone = load_backbone(config.DETECTOR_CKPT)
    ctrl_heads = load_control_heads(ctrl_dir)
    cells = {int(k): int(v) for k, v in json.loads((ctrl_dir / "event_control_cells_test.json").read_text()).items()}
    reals = {"grumobilenet": load_head(config.DETECTOR_CKPT)}
    torch.manual_seed(args.seed + 7)  # same random head as in controls.py
    attr_heads = {**reals, "random_head": Head().eval(), "video_control": ctrl_heads["video_control"]}
    if args.controls_only:
        reals, attr_heads = {}, {}

    test_ids = sorted(int(p.stem) for p in (controls.CACHE / "test").glob("*.npy"))
    acc = [v for v in test_ids if v < 400 and v in kf and v in cells]
    nor = [v for v in test_ids if v >= 400]
    if args.smoke:
        acc, nor = acc[:4], nor[:8]

    causal_rows, attr_rows, video_scores = [], [], []
    skipped, t0 = 0, time.time()
    for n_i, vid in enumerate(acc, start=1):
        fr = np.asarray(cache("test", vid))
        K, r0 = len(fr), -(-kf[vid] // 5)
        if r0 < 1 or r0 >= K:
            skipped += 1
            continue
        rows_c = np.unique(np.clip([r0 + 2, r0 + 6, r0 + 10], 0, K - 1))
        abl = np.arange(max(0, r0 - 5), min(K - 1, r0 + 10) + 1)
        bg = background(fr, min(10, r0))
        f_bg = pooled(backbone, bg[None])[0]
        rng_v = np.random.default_rng([args.seed, 300000 + vid])
        bg_other = background(cache("test", int(rng_v.choice(nor))), 10)

        # Controls, on the marked clip (the event region is the mark cell).
        cell = cells[vid]
        marked = plant(fr, 5 * np.arange(K) >= kf[vid], cell, int(mcfg["size"]), float(mcfg["alpha"]))
        Fm = pooled(backbone, marked)
        rand_cell = [int(rng_v.choice([c for c in range(NCELL) if c != cell]))]
        Fm_ev = ablate(backbone, marked, Fm, abl, [cell], bg)
        Fm_rd = ablate(backbone, marked, Fm, abl, rand_cell, bg)
        Fm_sw = pooled(backbone, swap_background(marked, bg, bg_other))
        for name, head in ctrl_heads.items():
            causal_rows.append({"video": vid, "model": name, "n_region_cells": 1,
                                "orig": float(score_windows(head, Fm, rows_c).mean()),
                                "background_only": score_static(head, f_bg), "first_frame": score_static(head, Fm[0]),
                                "ablate_event": float(score_windows(head, Fm_ev, rows_c).mean()),
                                "ablate_random": float(score_windows(head, Fm_rd, rows_c).mean()),
                                "swap_background": float(score_windows(head, Fm_sw, rows_c).mean())})

        if args.controls_only:
            continue

        # GRUMobileNet, on the unmarked clip (the region is the motion region of controls.py).
        Fu = pooled(backbone, fr)
        region = motion_region(fr, r0)
        comp = [c for c in range(NCELL) if c not in region]
        rand_region = sorted(rng_v.choice(comp, size=min(len(region), len(comp)), replace=False).tolist()) if comp else []
        Fu_ev = ablate(backbone, fr, Fu, abl, region, bg)
        Fu_rd = ablate(backbone, fr, Fu, abl, rand_region, bg)
        Fu_sw = pooled(backbone, swap_background(fr, bg, bg_other))
        for name, head in reals.items():
            causal_rows.append({"video": vid, "model": name, "n_region_cells": len(region),
                                "orig": float(score_windows(head, Fu, rows_c).mean()),
                                "background_only": score_static(head, f_bg), "first_frame": score_static(head, Fu[0]),
                                "ablate_event": float(score_windows(head, Fu_ev, rows_c).mean()),
                                "ablate_random": float(score_windows(head, Fu_rd, rows_c).mean()),
                                "swap_background": float(score_windows(head, Fu_sw, rows_c).mean())})
            video_scores.append({"video": vid, "label": 1, "model": name,
                                 "orig_max": float(score_windows(head, Fu, np.arange(K)).max()),
                                 "background_only": score_static(head, f_bg), "first_frame": score_static(head, Fu[0])})

        # Calibrated null for attribution: same videos, region and windows as controls.py.
        maps_u = controls.feature_maps(backbone, fr)
        pre, post = eval_rows(K, r0)
        rows_r = np.unique(np.concatenate([[0], pre, post])).astype(int)
        is_pre = rows_r < r0
        for name, head in attr_heads.items():
            C, _ = cams(head, maps_u, rows_r)
            for meth in METHODS:
                L = lift(C[meth], region)
                lp = float(L[~is_pre].mean()) if (~is_pre).any() else None
                lq = float(L[is_pre].mean()) if is_pre.any() else None
                attr_rows.append({"video": vid, "model": name, "method": meth, "lift_pre": lq, "lift_post": lp,
                                  "delta": (lp - lq) if (lp is not None and lq is not None) else None})
        if n_i % 10 == 0 or n_i == len(acc):
            log(f"  accidents: {n_i}/{len(acc)} ({time.time() - t0:.0f} s)")

    for n_i, vid in enumerate([] if args.controls_only else nor, start=1):
        fr = np.asarray(cache("test", vid))
        F = pooled(backbone, fr)
        f_bg = pooled(backbone, background(fr, 10)[None])[0]
        for name, head in reals.items():
            video_scores.append({"video": vid, "label": 0, "model": name,
                                 "orig_max": float(score_windows(head, F, np.arange(len(F))).max()),
                                 "background_only": score_static(head, f_bg), "first_frame": score_static(head, F[0])})
        if n_i % 100 == 0 or n_i == len(nor):
            log(f"  normals: {n_i}/{len(nor)} ({time.time() - t0:.0f} s)")

    write_csv(out / "interventions_per_video.csv", causal_rows)
    write_csv(out / "video_scores.csv", video_scores)
    write_csv(out / "attribution_null_per_video.csv", attr_rows)

    # Verdict (controls) and findings (GRUMobileNet).
    S = {m: summarize(causal_rows, m) for m in list(ctrl_heads) + list(reals)}
    mk, sc = S["event_control"], S["video_control"]

    def val(d, key, i=None):
        v = d[key]["mean"] if i is None else d[key]["ci95"][i]
        return v if v is not None else float("nan")

    conds = {"event_control_drop_event": val(mk, "drop_event") >= 0.3 and val(mk, "drop_event", 0) > 0,
             "event_control_retention_background_only": val(mk, "retention_background_only") <= 0.3,
             "video_control_retention_background_only": val(sc, "retention_background_only") >= 0.8,
             "video_control_drop_event": val(sc, "drop_event") <= 0.05}
    verdict = "GO" if all(conds.values()) else "NO-GO"

    auc = {}
    for name in reals:
        V = [r for r in video_scores if r["model"] == name]
        y = [r["label"] for r in V]
        if 0 < sum(y) < len(y):
            auc[name] = {k: float(roc_auc_score(y, [r[k] for r in V]))
                         for k in ("orig_max", "background_only", "first_frame")}

    def pick(model, meth, key):
        return {r["video"]: r[key] for r in attr_rows if r["model"] == model and r["method"] == meth}

    attr = {f"{m}/{k}": {key: agg(list(pick(m, k, key).values())) for key in ("lift_post", "lift_pre", "delta")}
            for m in attr_heads for k in METHODS}
    evidence = {}
    for m in reals:
        real_p = pick(m, "gradcam", "lift_post")
        diffs = {}
        for null in ("random_head", "video_control"):
            null_p = pick(null, "gradcam", "lift_post")
            diffs[null] = agg([real_p[v] - null_p[v] for v in real_p
                               if real_p[v] is not None and null_p.get(v) is not None])
        evidence[m] = {"paired_difference_lift_post": diffs,
                       "attributional_evidence": all((d["ci95"][0] or -1) > 0 for d in diffs.values())}

    try:
        ctrl_dir_rel = str(ctrl_dir.resolve().relative_to(config.REPO))
    except ValueError:
        ctrl_dir_rel = str(ctrl_dir)
    summary = {"preregistration": "docstring of sotad_diagnosis/interventions.py", "controls_run": ctrl_dir_rel,
               "mark": mcfg,
               "accident_videos": len(acc) - skipped, "skipped_r0": skipped, "causal": S, "conditions": conds,
               "verdict": verdict, "per_video_auc": auc, "attribution_null": attr,
               "attribution_evidence": evidence, "total_time_s": round(time.time() - t_start, 1),
               "config": vars(args)}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    def fmt(a):
        if a is None or a["mean"] is None:
            return "—"
        lo, hi = a["ci95"]
        return f"{a['mean']:.2f}" + (f" [{lo:.2f}; {hi:.2f}]" if lo is not None else "")

    log("\n=== Interventions (score = P(accident); retention = intervened/original) ===")
    for m, s in S.items():
        if s:
            log(f"{m:>13}: orig {fmt(s['score_orig'])} | ret. background only {fmt(s['retention_background_only'])}"
                f" | ret. first frame {fmt(s['retention_first_frame'])} | drop event {fmt(s['drop_event'])}"
                f" | drop random {fmt(s['drop_random'])} | ret. swap background {fmt(s['retention_swap_background'])}")
    log(f"Conditions: {json.dumps(conds)}")
    log(f"VERDICT: {verdict}")
    log("\n=== Per-video AUC (479 test videos): original (max) × background only × first frame ===")
    for m, a in auc.items():
        log(f"{m}: orig {a['orig_max']:.3f} | background only {a['background_only']:.3f}"
            f" | first frame {a['first_frame']:.3f}")
    log("\n=== Attribution on the motion region with calibrated nulls (Grad-CAM) ===")
    for m in attr_heads:
        a = attr[f"{m}/gradcam"]
        log(f"{m}: post {fmt(a['lift_post'])} | pre {fmt(a['lift_pre'])} | Δ {fmt(a['delta'])}")
    for m, e in evidence.items():
        d = e["paired_difference_lift_post"]
        log(f"{m}: model − random head {fmt(d['random_head'])} | model − video control {fmt(d['video_control'])}"
            f" | attributional evidence: {e['attributional_evidence']}")
    log(f"\nTotal time: {summary['total_time_s']} s | summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
