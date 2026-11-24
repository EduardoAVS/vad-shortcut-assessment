#!/usr/bin/env python3
"""TAD, attribution: Grad-CAM µw of GRUMobileNet on TAD, without keyframe, against calibrated null models.

Paper: Table 4, µw rows on the 20% held-out training set (event control 9.7-10.6, video control 1.1-1.2,
GRUMobileNet 1.0-1.2, randomized network 0.9-1.2). µw is the `lift` below (mass in the region / area fraction).

BINDING pre-registration: results/attribution/PREREG.md (written on 2026-10-05, before running). New analysis, requested
after the holdout experiment. Deviations go to the "Deviations" section of results/attribution/RESULT.md.

Implementation of the pre-registration:
  - videos: the 80/20 validation split of the holdout experiment (rng(s+1); seeds 0, 1, 2), reproduced with
    holdout.split. The validation accidents with K >= 21 are measured; the validation normals are only descriptive;
  - models: grumobilenet (the heads of the holdout run, <holdout-run>/seed<s>/grumobilenet_head.pt); null 1 =
    random_head (same architecture, torch.manual_seed(s), untrained); null 2 = video_control
    (<official-run>/seed<s>/video_control_head.pt); instrument validation = event_control
    (<official-run>/seed<s>/event_control_head.pt) on the validation accidents marked as in the official-split
    experiment (cell rng([s, k]), mark on rows >= 10);
  - motion region (keyframe-free version of the SO-TAD motion pseudo-region): energy of |gray(r) - gray(r-1)| at
    128 px, summed over ALL rows of the clip, per cell of the 4x4 grid; region = smallest set of cells with >= 50% of
    the energy;
  - attribution: Grad-CAM (controls.cams; target logit[1] - logit[0], summed over the 16 steps); Grad-CAM++ and
    LayerCAM only descriptive;
  - windows: up to 8 per video, evenly spaced among those ending at rows 15 to K - 1;
  - lift (µw) = mass in the region / area fraction (controls.lift), mean over the windows of a video;
  - criteria: instrument valid iff, in >= 2 of the 3 seeds, event_control has mean lift (Grad-CAM) >= 2.0 on the
    mark cell (windows with the mark); otherwise "no diagnosis". grumobilenet has attribution evidence of looking at
    the motion in a seed iff its Grad-CAM lift exceeds EACH null with the 95% CI of the paired difference > 0 (bootstrap
    over videos, B = 1000; controls.agg); the overall reading requires this in >= 2 of the 3 seeds. Reading combined
    with the holdout experiment (decision = "scene"): looks at the motion -> "dissociation replicated on TAD"; does not
    look -> "no dissociation on TAD: attribution agrees with the cause";
  - descriptive: lifts per seed and model with CI; Grad-CAM++ and LayerCAM; lifts on the validation normals; mean area
    fraction of the motion region.

Consistency checks: (i) the backbone maps, averaged over space, must reproduce the cached features
(TAD_FEATURE_CACHE/clean); (ii) the set of validation videos must be the one recorded by the holdout run; (iii) only in
the full run: the grumobilenet (max over windows) and video_control (row 0) scores on the validation videos must match
the CSVs of the holdout run. Any failure aborts.

Inputs: config.TAD_FRAME_CACHE, config.TAD_FEATURE_CACHE (read only), config.DETECTOR_CKPT, the heads and CSVs of the
official-split and holdout runs (--official-run, --holdout-run; default: the recorded runs).
Outputs: <out-dir>/<timestamp>/ (seed<s>/lifts_per_video.csv, seed<s>/seed_summary.json, summary.json, table.md).
Recorded run: results/attribution/20261005-120205/.

Usage (from the repository root):
  tools/run_guarded.sh 2500 work/logs/tad_attribution.log -- python tad_diagnosis/attribution.py
  Smoke (only videos of the TRAIN side of the split; no validation metric): --smoke
  Threads: keep the default --threads 6 (it overrides the 2 threads set by run_guarded.sh). The recorded run used
  6 threads; another thread count may change the results at float round-off.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import official_split as T  # noqa: E402  (puts the repo root and sotad_diagnosis on sys.path)
import holdout as B  # noqa: E402
import config  # noqa: E402
import controls  # noqa: E402
from controls import CELL, GRID, METHODS, Head, agg, cams, lift, load_backbone, plant, write_csv  # noqa: E402
from interventions import score_static, score_windows  # noqa: E402

OFFICIAL_RUN = B.OFFICIAL_RUN                                 # recorded run: control heads
HOLDOUT_RUN = T.RESULTS / "holdout" / "20261005-110712"       # recorded run: grumobilenet heads and validation CSVs
SEEDS = (0, 1, 2)
LIFT_MIN = 2.0
FIRST_ROW, N_WIN = 15, 8
MODELS = ("grumobilenet", "random_head", "video_control")
NULLS = ("random_head", "video_control")


def log(msg):
    print(msg, flush=True)


def motion_region_all(frames):
    """SO-TAD motion pseudo-region without keyframe: energy of |delta gray| over all rows of the clip."""
    g = np.asarray(frames, dtype=np.float32).mean(-1)
    e = np.abs(np.diff(g, axis=0)).sum(0) if len(g) > 1 else np.zeros(g.shape[1:], np.float32)
    ce = e.reshape(GRID, CELL, GRID, CELL).sum((1, 3)).ravel()
    order = np.argsort(-ce)
    cum = np.cumsum(ce[order]) / max(ce.sum(), 1e-9)
    return order[:int(np.searchsorted(cum, 0.5) + 1)].tolist()


def att_rows(K):
    if K - 1 < FIRST_ROW:
        return np.array([K - 1])
    return np.unique(np.linspace(FIRST_ROW, K - 1, N_WIN).round().astype(int))


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def check_holdout_scores(va, realb, scene, backbone, seed, holdout_run):
    """Check (iii): grumobilenet and video_control scores on the validation videos = those recorded by the holdout run."""
    ref = {r["video"]: r for r in read_csv(holdout_run / f"seed{seed}" / "val_per_video.csv")}
    dmax = 0.0
    for d in va:
        Fc = B.feats_clean(backbone, d)
        o = float(score_windows(realb, Fc, np.arange(d["K"])).max())
        c = score_static(scene, Fc[0])
        dmax = max(dmax, abs(o - float(ref[d["vid"]]["orig_max"])),
                   abs(c - float(ref[d["vid"]]["video_control_first_frame"])))
    return dmax


def run_seed(seed, backbone, train, sdir, smoke, official_run, holdout_run):
    t0 = time.time()
    _, va, tr_real = B.split(seed, train)
    b_val = {r["video"] for r in read_csv(holdout_run / f"seed{seed}" / "val_per_video.csv")}
    if {d["vid"] for d in va} != b_val:
        raise SystemExit(f"[s{seed}] the split does not reproduce the validation set of the holdout run")
    realb = B.load_saved(holdout_run / f"seed{seed}" / "grumobilenet_head.pt")
    scene = B.load_saved(official_run / f"seed{seed}" / "video_control_head.pt")
    marker = B.load_saved(official_run / f"seed{seed}" / "event_control_head.pt")
    torch.manual_seed(seed)
    rand = Head().eval()
    heads = {"grumobilenet": realb, "random_head": rand, "video_control": scene}
    checks = {"split_matches_holdout": True}
    if smoke:
        acc = [d for d in tr_real if d["y"] == 1 and d["K"] >= T.MIN_ROWS_CTRL]
        nor = [d for d in tr_real if d["y"] == 0]
        videos = acc[::max(1, len(acc) // 6)][:6] + nor[::max(1, len(nor) // 3)][:3]
        assert not any(d["vid"] in b_val for d in videos)
        log(f"[smoke] {len(videos)} videos, all from the train side of the split")
    else:
        videos = va
        dmax = check_holdout_scores(va, realb, scene, backbone, seed, holdout_run)
        checks["max_score_diff_vs_holdout"] = dmax
        log(f"[s{seed}] check (iii): max |delta score| vs holdout run = {dmax:.2e}")
        if dmax > 1e-5:
            raise SystemExit(f"[s{seed}] scores do not match the holdout run")

    rows, dfeat = [], 0.0
    for d in videos:
        fr = np.asarray(T.frames(d))
        K = d["K"]
        maps = controls.feature_maps(backbone, fr)
        dfeat = max(dfeat, float(np.abs(maps.mean(dim=(2, 3)).numpy() - B.feats_clean(backbone, d)).max()))
        region = motion_region_all(fr)
        rws = att_rows(K)
        base = {"video": d["vid"], "label": d["y"], "K": K, "n_windows": len(rws), "n_region_cells": len(region),
                "area_fraction": len(region) / (GRID * GRID), "measured": int(d["y"] == 1 and K >= T.MIN_ROWS_CTRL)}
        for name, h in heads.items():
            C, _ = cams(h, maps, rws)
            for m in METHODS:
                rows.append({**base, "model": name, "region": "motion", "method": m,
                             "lift": float(lift(C[m], region).mean())})
        if base["measured"]:
            cell = T.marker_cell(seed, d)
            maps_m = controls.feature_maps(backbone, plant(fr, np.arange(K) >= T.KF_ROW, cell))
            C, _ = cams(marker, maps_m, rws)
            for m in METHODS:
                rows.append({**base, "model": "event_control", "region": f"mark_cell_{cell}", "method": m,
                             "lift": float(lift(C[m], [cell]).mean())})
    checks["max_diff_maps_vs_cached_features"] = dfeat
    log(f"[s{seed}] check (i): max |delta| averaged maps vs cached features = {dfeat:.2e}")
    if dfeat > 1e-4:
        raise SystemExit(f"[s{seed}] the maps do not reproduce the cached features")
    write_csv(sdir / "lifts_per_video.csv", rows)

    def vals(model, meth, measured):
        return {r["video"]: r["lift"] for r in rows
                if r["model"] == model and r["method"] == meth and r["measured"] == measured
                and (measured == 1 or r["label"] == 0)}

    lifts = {f"{mo}/{me}": agg(list(vals(mo, me, 1).values())) for mo in MODELS + ("event_control",) for me in METHODS}
    diffs = {}
    for me in METHODS:
        real = vals("grumobilenet", me, 1)
        for null in NULLS:
            nv = vals(null, me, 1)
            diffs[f"{me}/grumobilenet_minus_{null}"] = agg([real[v] - nv[v] for v in real if v in nv])
    evid = all((diffs[f"gradcam/grumobilenet_minus_{n}"]["ci95"][0] is not None
                and diffs[f"gradcam/grumobilenet_minus_{n}"]["ci95"][0] > 0) for n in NULLS)
    mk = lifts["event_control/gradcam"]["mean"]
    normals = {f"{mo}/{me}": agg(list(vals(mo, me, 0).values())) for mo in MODELS for me in METHODS}
    area = {"measured_accidents": agg([r["area_fraction"] for r in rows if r["measured"] == 1
                                       and r["model"] == "grumobilenet" and r["method"] == "gradcam"]),
            "normals": agg([r["area_fraction"] for r in rows if r["label"] == 0 and r["model"] == "grumobilenet"
                            and r["method"] == "gradcam"])}
    summary = {"seed": seed, "checks": checks,
               "n_measured_accidents": len(vals("grumobilenet", "gradcam", 1)),
               "n_normals": len(vals("grumobilenet", "gradcam", 0)),
               "accident_lift": lifts, "paired_differences": diffs,
               "instrument_ok": mk is not None and mk >= LIFT_MIN, "attribution_evidence": evid,
               "normals_descriptive": normals, "region_area_fraction": area, "time_s": round(time.time() - t0, 1)}
    (sdir / "seed_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    log(f"[s{seed}] done ({time.time() - t0:.0f} s)")
    return summary


def final_summary(out, seeds, smoke):
    per = {s: json.loads((out / f"seed{s}" / "seed_summary.json").read_text()) for s in seeds}
    need = 2 if len(seeds) == 3 else len(seeds)
    n_instr = sum(per[s]["instrument_ok"] for s in seeds)
    n_ev = sum(per[s]["attribution_evidence"] for s in seeds)
    valid = n_instr >= need
    if not valid:
        att_reading, combined = "no diagnosis", "no diagnosis"
    elif n_ev >= need:
        att_reading, combined = "looks at the motion", "dissociation replicated on TAD"
    else:
        att_reading = "does not look at the motion"
        combined = "no dissociation on TAD: attribution agrees with the cause"
    summary = {"preregistration": "tad_diagnosis/results/attribution/PREREG.md (binding) + docstring of"
                                  " tad_diagnosis/attribution.py",
               "smoke": smoke, "seeds": list(seeds), "event_control_lift_threshold": LIFT_MIN,
               "seeds_instrument_ok": n_instr, "instrument_valid": valid,
               "seeds_with_evidence": n_ev, "grumobilenet_attribution_reading": att_reading,
               "holdout_decision": "scene", "combined_reading": combined, "per_seed": per}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    f = T.fmt
    L = ["| Seed | n | event_control (mark cell) | grumobilenet | random_head | video_control"
         " | grumobilenet − random_head | grumobilenet − video_control | Evidence |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        p = per[s]
        lf, df = p["accident_lift"], p["paired_differences"]
        L.append(f"| {s} | {p['n_measured_accidents']} | {f(lf['event_control/gradcam'])} | {f(lf['grumobilenet/gradcam'])}"
                 f" | {f(lf['random_head/gradcam'])} | {f(lf['video_control/gradcam'])}"
                 f" | {f(df['gradcam/grumobilenet_minus_random_head'])} | {f(df['gradcam/grumobilenet_minus_video_control'])}"
                 f" | {'yes' if p['attribution_evidence'] else 'no'} |")
    L += ["", "Descriptive: Grad-CAM++ and LayerCAM (measured accidents; lift on the motion region, event_control on"
          " the mark cell)", "",
          "| Seed | Method | event_control | grumobilenet | random_head | video_control | grumobilenet − random_head"
          " | grumobilenet − video_control |", "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        lf, df = per[s]["accident_lift"], per[s]["paired_differences"]
        for me in ("gradcampp", "layercam"):
            L.append(f"| {s} | {me} | {f(lf[f'event_control/{me}'])} | {f(lf[f'grumobilenet/{me}'])}"
                     f" | {f(lf[f'random_head/{me}'])} | {f(lf[f'video_control/{me}'])}"
                     f" | {f(df[f'{me}/grumobilenet_minus_random_head'])}"
                     f" | {f(df[f'{me}/grumobilenet_minus_video_control'])} |")
    L += ["", "Descriptive: validation normals (Grad-CAM, motion region) and area fraction of the region", "",
          "| Seed | n normals | grumobilenet | random_head | video_control | Area (accidents) | Area (normals) |",
          "|---|---|---|---|---|---|---|"]
    for s in seeds:
        p = per[s]
        nm, ar = p["normals_descriptive"], p["region_area_fraction"]
        L.append(f"| {s} | {p['n_normals']} | {f(nm['grumobilenet/gradcam'])} | {f(nm['random_head/gradcam'])}"
                 f" | {f(nm['video_control/gradcam'])} | {f(ar['measured_accidents'])} | {f(ar['normals'])} |")
    (out / "table.md").write_text("\n".join(L) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threads", type=int, default=6,
                    help="torch threads (6 = the recorded runs; needed for bit-exact reproduction)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true", help="only the train side of the split; seed 0")
    ap.add_argument("--resume", default=None, help="existing run directory to complete")
    ap.add_argument("--out-dir", type=Path, default=config.WORK_DIR / "tad_diagnosis" / "attribution",
                    help="parent of the new run directory (default: %(default)s)")
    ap.add_argument("--official-run", type=Path, default=OFFICIAL_RUN,
                    help="official_split.py run with the control heads (default: the recorded run)")
    ap.add_argument("--holdout-run", type=Path, default=HOLDOUT_RUN,
                    help="holdout.py run with the grumobilenet heads and validation CSVs (default: the recorded run)")
    args = ap.parse_args()
    if args.threads > 6:
        raise SystemExit("at most 6 threads")
    torch.set_num_threads(args.threads)
    if args.smoke:
        args.seeds = [0]
    if args.resume:
        out = Path(args.resume)
        cfg = json.loads((out / "config.json").read_text())
        if bool(cfg["smoke"]) != args.smoke or cfg["seeds"] != args.seeds:
            raise SystemExit("configuration differs from the run being resumed")
    else:
        out = T.new_run_dir(args.out_dir, args, args.smoke)
    t_start = time.time()
    log(f"TAD attribution | output: {out} | seeds {args.seeds} | threads {args.threads}")
    train = T.load_index()["train"]
    backbone = load_backbone(config.DETECTOR_CKPT)
    for s in args.seeds:
        sdir = out / f"seed{s}"
        if (sdir / "seed_summary.json").exists():
            log(f"[s{s}] already done, skipping")
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        run_seed(s, backbone, train, sdir, args.smoke, args.official_run, args.holdout_run)
        log(f"  CPU used so far in this invocation: {T.cpu_seconds():.0f} s")
    summary = final_summary(out, args.seeds, args.smoke)
    cpu = T.cpu_seconds()
    with open(out / "cpu_log.csv", "a") as fh:
        fh.write(f"{time.strftime('%Y%m%d-%H%M%S')},{cpu:.1f},{time.time() - t_start:.1f}\n")
    if not args.smoke:
        log(f"\ninstrument valid: {summary['instrument_valid']} ({summary['seeds_instrument_ok']} seeds)"
            f" | evidence: {summary['seeds_with_evidence']} seeds | combined reading: {summary['combined_reading']}")
    log(f"DONE | wall {time.time() - t_start:.0f} s | CPU {cpu:.0f} s | {out / 'summary.json'}")


if __name__ == "__main__":
    main()
