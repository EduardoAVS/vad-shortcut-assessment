#!/usr/bin/env python3
"""TAD, official split: the intervention module (validated on SO-TAD) applied to the TAD benchmark of Xu et al.

Paper: Table 4, official-test rows (GRUMobileNet AUC 0.54-0.68 and video control AUC 0.62-0.65 over 3 seeds).
Also the shared module of the TAD experiments: holdout.py, attribution.py and cross_validation.py import it.

BINDING pre-registration: results/official_split/PREREG.md (written on 2026-10-05, before running). This script
implements it; deviations are listed in the "Deviations" section of results/official_split/RESULT.md.

Summary of the pre-registration, as implemented:
  - data: official split (train 261 accidents + 111 normals; test 16 + 16), cache from build_cache.py
    (5 rows/s by time, 128 px as in the SO-TAD cache);
  - architecture of GRUMobileNet and of the SO-TAD controls (imported from sotad_diagnosis/controls.py): MobileNetV3-Small
    frozen with the weights of the detector checkpoint, GRU(576->128) + Linear(128->2), causal window of 16 rows padded
    with row 0, decision at the last step; 6 epochs, Adam 1e-3, class-weighted CE, every window of every video
    (controls.train_head); all 111 training normals; seeds 0, 1 and 2;
  - grumobilenet (a GRUMobileNet head trained on TAD): window label = video label; video score = max over windows;
  - video_control (negative control): same head, input = row 0 repeated 16 times, video label;
  - event_control (positive control): the SO-TAD mark (controls.plant: checkerboard of 8-px squares covering one cell
    of the 4x4 grid drawn per video) on rows r >= 10 of the accident videos (train and test); window label = 1 iff
    accident and last row >= 10;
  - control score (as in the SO-TAD interventions): mean of the windows ending at rows 12, 16 and 20 of the marked test
    clip; accident videos with fewer than 21 rows are left out of the control measurements;
  - interventions (those of sotad_diagnosis/interventions.py): background = per-pixel median of rows [0, 10);
    background only = the whole window becomes the background; first frame frozen = the whole window becomes row 0;
    swap background = pixels with |gray - gray of its own background| > 25 pasted over the background of a test normal
    drawn at random (fixed seed); no event ablation (there is no keyframe nor region);
  - retention = mean(score with intervention) / mean(original score) over the test accidents, 95% CI by bootstrap over
    videos (B = 1000; interventions.ratio_ci);
  - C0 (control validity): event_control window AUC >= 0.90 on 20% of the training set held out (stratified, fixed
    seed, as on SO-TAD);
  - module GO iff, in >= 2 of the 3 seeds, event_control has background-only retention <= 0.3 and video_control has
    background-only retention >= 0.8 (here also requiring C0 in the seed; the literal version without C0 is recorded
    too);
  - grumobilenet reading: only if GO and original per-video test AUC >= 0.70 (median over seeds); background-only
    retention (median over seeds) >= 0.8 -> "scene"; <= 0.3 -> "event"; in between -> "inconclusive"; otherwise
    "no diagnosis".

Operationalization decisions (ambiguities of the pre-registration, fixed before running; see RESULT.md):
  - event_control and video_control train on the same 80% of the training set (as the SO-TAD controls); grumobilenet
    trains on 100% of it. As on SO-TAD, accidents without any row >= 10 (K <= 10) are left out of control training;
  - with seed s: mark cell rng([s, k]) in train and rng([s, 100000 + k]) in test (k = position of the video in the
    sorted list of its split); 80/20 split with rng(s + 1); heads event_control s + 2, video_control s + 3,
    grumobilenet s + 4; swap-background normal rng([s, 300000 + k]);
  - on the test set, the controls get the real windows of the marked clip (as on SO-TAD); the descriptive
    video_control AUC uses the input it was trained on (row 0 repeated), on the 32 videos;
  - per-video AUC with a 95% CI by class-stratified bootstrap (B = 1000, seed 0).

Inputs: config.TAD_FRAME_CACHE (build_cache.py), config.DETECTOR_CKPT.
Outputs: <out-dir>/<timestamp>/ (seed<s>/*.csv, seed<s>/seed_summary.json, seed<s>/*_head.pt, summary.json,
table.md). Feature caches: config.TAD_FEATURE_CACHE (deterministic; reused across runs and experiments).
Recorded run cited in the paper: results/official_split/20261005-101615/.

Usage (from the repository root):
  tools/run_guarded.sh 2500 work/logs/tad_official_split.log -- python tad_diagnosis/official_split.py
  Smoke (TRAINING videos only, the test set is not touched): --smoke. Resume a run: --resume <run dir>
  Threads: keep the default --threads 6 (it overrides the 2 threads set by run_guarded.sh). The recorded heads were
  trained with 6 threads; with another thread count the heads differ at float round-off and the numbers drift.
"""
import argparse
import csv
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sotad_diagnosis"))
import config  # noqa: E402
import controls  # noqa: E402
from controls import NCELL, agg, load_backbone, plant, safe_auc, train_head, write_csv  # noqa: E402
from interventions import background, pooled, ratio_ci, score_static, score_windows, swap_background  # noqa: E402

CACHE = config.TAD_FRAME_CACHE
FEATS = config.TAD_FEATURE_CACHE
RESULTS = HERE / "results"
KF_ROW = 10                      # artificial keyframe of event_control: row 10 (2 s)
CTRL_ROWS = np.array([12, 16, 20])
MIN_ROWS_CTRL = 21
SEEDS = (0, 1, 2)
THR = {"c0": 0.90, "event_control_max": 0.3, "video_control_min": 0.8, "grumobilenet_auc_min": 0.70,
       "scene_reading": 0.8, "event_reading": 0.3}


def log(msg):
    print(msg, flush=True)


def rel(path):
    """Path as a string relative to the repository root when it is inside it (for the recorded JSON files)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def new_run_dir(out_dir, args, smoke):
    """Create <out_dir>/<timestamp>[-smoke] and write config.json (the parsed command-line options)."""
    out = Path(out_dir) / (time.strftime("%Y%m%d-%H%M%S") + ("-smoke" if smoke else ""))
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps({k: v for k, v in vars(args).items()
                                                 if k not in ("out_dir", "official_run", "holdout_run")}))
    return out


def save_atomic(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def load_index():
    with open(CACHE / "index.csv") as f:
        rows = list(csv.DictReader(f))
    vids = {"train": [], "test": []}
    for r in rows:
        vids[r["split"]].append({"vid": r["vid"], "split": r["split"], "y": int(r["label"]), "K": int(r["K"]),
                                 "folder": r["folder"]})
    for s in vids:
        vids[s].sort(key=lambda d: d["vid"])
        for k, d in enumerate(vids[s]):
            d["k"] = k
    return vids


def frames(d):
    return np.load(CACHE / d["split"] / f"{d['vid']}.npy", mmap_mode="r")


def clean_feats(backbone, d):
    p = FEATS / "clean" / d["split"] / f"{d['vid']}.npy"
    if p.exists():
        return np.load(p)
    F = pooled(backbone, np.asarray(frames(d))).astype(np.float32)
    save_atomic(p, F)
    return F


def marker_cell(seed, d):
    off = 0 if d["split"] == "train" else 100000
    return int(np.random.default_rng([seed, off + d["k"]]).integers(NCELL))


def marked_train_feats(backbone, d, seed, Fc):
    """Features of the training clip with the mark on rows >= 10 (accidents); normals and rows < 10 = clean."""
    if d["y"] == 0 or d["K"] <= KF_ROW:
        return Fc
    p = FEATS / f"marked_s{seed}" / d["split"] / f"{d['vid']}.npy"
    if p.exists():
        return np.load(p)
    fr = np.asarray(frames(d)[KF_ROW:])
    F = Fc.copy()
    F[KF_ROW:] = pooled(backbone, plant(fr, np.ones(len(fr), bool), marker_cell(seed, d)))
    save_atomic(p, F)
    return F


def marker_labels(d):
    return ((np.arange(d["K"]) >= KF_ROW) & (d["y"] == 1)).astype(np.int64)


def video_labels(d):
    return np.full(d["K"], d["y"], dtype=np.int64)


def auc_ci(y, p, B=1000, seed=0):
    y, p = np.asarray(y), np.asarray(p, float)
    if not 0 < y.sum() < len(y):
        return {"auc": None, "ci95": [None, None], "n": int(len(y))}
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    vals = [roc_auc_score(y[idx], p[idx]) for idx in
            (np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(B))]
    return {"auc": float(roc_auc_score(y, p)), "ci95": [float(np.percentile(vals, 2.5)),
                                                        float(np.percentile(vals, 97.5))], "n": int(len(y))}


def summarize(rows, keys_ret):
    if not rows:
        return None
    g = {k: np.array([r[k] for r in rows], float) for k in ("orig",) + keys_ret}
    out = {"n": len(rows), "score_orig": agg(g["orig"])}
    for k in keys_ret:
        out[f"retention_{k}"] = ratio_ci(g[k], g["orig"])
    return out


def cpu_seconds():
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def fmt(a, d=2):
    """'mean [lo, hi]' (or 'auc [lo, hi]') of an agg/ratio_ci/auc_ci dict; '—' if missing."""
    if a is None or a.get("mean", a.get("auc")) is None:
        return "—"
    v = a.get("mean", a.get("auc"))
    lo, hi = a["ci95"]
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]" if lo is not None else f"{v:.{d}f}"


def reading(ret_value):
    if ret_value is None:
        return "no diagnosis"
    if ret_value >= THR["scene_reading"]:
        return "scene"
    if ret_value <= THR["event_reading"]:
        return "event"
    return "inconclusive"


def run_seed(seed, backbone, train, test, epochs, sdir):
    t0 = time.time()
    feats_c = {d["vid"]: clean_feats(backbone, d) for d in train + test}
    ctrl_vids = [d for d in train if not (d["y"] == 1 and d["K"] <= KF_ROW)]
    n_excl = len(train) - len(ctrl_vids)
    Fm = {d["vid"]: marked_train_feats(backbone, d, seed, feats_c[d["vid"]]) for d in ctrl_vids}
    log(f"[s{seed}] marked features ready ({time.time() - t0:.0f} s); accidents excluded from the controls"
        f" (K <= {KF_ROW}): {n_excl}")

    # Stratified 80/20 split for C0 (as on SO-TAD).
    rng = np.random.default_rng(seed + 1)
    val_ids = set()
    for group in ([d["vid"] for d in ctrl_vids if d["y"] == 1], [d["vid"] for d in ctrl_vids if d["y"] == 0]):
        val_ids |= set(rng.choice(group, size=max(1, len(group) // 5), replace=False).tolist())
    tr = [d for d in ctrl_vids if d["vid"] not in val_ids]
    va = [d for d in ctrl_vids if d["vid"] in val_ids]

    t1 = time.time()
    marker = train_head([{"F": Fm[d["vid"]], "y": marker_labels(d)} for d in tr], False, epochs, seed + 2)
    scene = train_head([{"F": feats_c[d["vid"]], "y": video_labels(d)} for d in tr], True, epochs, seed + 3)
    real = train_head([{"F": feats_c[d["vid"]], "y": video_labels(d)} for d in train], False, epochs, seed + 4)
    for name, h in (("event_control", marker), ("video_control", scene), ("grumobilenet", real)):
        torch.save(h.state_dict(), sdir / f"{name}_head.pt")
    auc_m_val = safe_auc(np.concatenate([marker_labels(d) for d in va]),
                         np.concatenate([controls.predict_F(marker, Fm[d["vid"]]) for d in va]))
    auc_s_val = safe_auc(np.concatenate([video_labels(d) for d in va]),
                         np.concatenate([controls.predict_F(scene, feats_c[d["vid"]], scene=True) for d in va]))
    log(f"[s{seed}] training {time.time() - t1:.0f} s | C0 event_control window AUC (val) = {auc_m_val}"
        f" | video_control (val, video label) = {auc_s_val} | control training {len(tr)} videos, val {len(va)},"
        f" grumobilenet {len(train)}")

    # Evaluation set (official test; in the smoke run, held-out training videos).
    nor_test = [d for d in test if d["y"] == 0]
    real_rows, ctrl_rows = [], []
    for d in test:
        fr = np.asarray(frames(d))
        K, Fc = d["K"], feats_c[d["vid"]]
        bg = background(fr, KF_ROW)
        f_bg = pooled(backbone, bg[None])[0]
        rec = {"video": d["vid"], "label": d["y"], "K": K,
               "orig_max": float(score_windows(real, Fc, np.arange(K)).max()),
               "background_only": score_static(real, f_bg), "first_frame": score_static(real, Fc[0]),
               "video_control_first_frame": score_static(scene, Fc[0])}
        if d["y"] == 1:
            rng_v = np.random.default_rng([seed, 300000 + d["k"]])
            other = nor_test[int(rng_v.integers(len(nor_test)))]
            bg_other = background(np.asarray(frames(other)), KF_ROW)
            F_sw = pooled(backbone, swap_background(fr, bg, bg_other))
            rec.update(swap_background=float(score_windows(real, F_sw, np.arange(K)).max()), swap_normal=other["vid"])
            if K >= MIN_ROWS_CTRL:
                cell = marker_cell(seed, d)
                sub = fr[:MIN_ROWS_CTRL]
                marked = plant(sub, np.arange(len(sub)) >= KF_ROW, cell)
                F_m = pooled(backbone, marked)
                F_msw = pooled(backbone, swap_background(marked, bg, bg_other))
                for name, h in (("event_control", marker), ("video_control", scene)):
                    ctrl_rows.append({"video": d["vid"], "K": K, "cell": cell, "model": name,
                                      "orig": float(score_windows(h, F_m, CTRL_ROWS).mean()),
                                      "background_only": score_static(h, f_bg), "first_frame": score_static(h, F_m[0]),
                                      "swap_background": float(score_windows(h, F_msw, CTRL_ROWS).mean()),
                                      "swap_normal": other["vid"]})
        real_rows.append(rec)
    write_csv(sdir / "grumobilenet_per_video.csv", real_rows)
    write_csv(sdir / "controls_per_video.csv", ctrl_rows)

    keys = ("background_only", "first_frame", "swap_background")
    acc = [r for r in real_rows if r["label"] == 1]
    S = {"event_control": summarize([r for r in ctrl_rows if r["model"] == "event_control"], keys),
         "video_control": summarize([r for r in ctrl_rows if r["model"] == "video_control"], keys),
         "grumobilenet": summarize([{**r, "orig": r["orig_max"]} for r in acc], keys)}
    y = [r["label"] for r in real_rows]
    auc = {"grumobilenet_original": auc_ci(y, [r["orig_max"] for r in real_rows]),
           "grumobilenet_background_only": auc_ci(y, [r["background_only"] for r in real_rows]),
           "grumobilenet_first_frame": auc_ci(y, [r["first_frame"] for r in real_rows]),
           "video_control_first_frame": auc_ci(y, [r["video_control_first_frame"] for r in real_rows])}

    def ret(m):
        s = S[m]
        return None if s is None else s["retention_background_only"]["mean"]

    conds = {"c0_event_control_val_auc_ge_0.90": auc_m_val is not None and auc_m_val >= THR["c0"],
             "event_control_ret_background_only_le_0.3":
                 ret("event_control") is not None and ret("event_control") <= THR["event_control_max"],
             "video_control_ret_background_only_ge_0.8":
                 ret("video_control") is not None and ret("video_control") >= THR["video_control_min"]}
    summary = {"seed": seed, "c0": {"event_control_val_window_auc": auc_m_val, "video_control_val_window_auc": auc_s_val,
                                    "n_val": len(va), "n_train_controls": len(tr)},
               "n_accidents_excluded_from_control_training": n_excl, "causal": S, "per_video_auc": auc,
               "conditions": conds, "seed_passes": all(conds.values()),
               "seed_passes_literal_without_c0": conds["event_control_ret_background_only_le_0.3"]
               and conds["video_control_ret_background_only_ge_0.8"],
               "n_eval_accidents": len(acc), "n_eval_accidents_lt21": sum(r["K"] < MIN_ROWS_CTRL for r in acc),
               "n_eval_videos": len(real_rows), "time_s": round(time.time() - t0, 1)}
    tmp = sdir / "seed_summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, sdir / "seed_summary.json")
    log(f"[s{seed}] evaluation done ({time.time() - t0:.0f} s in total for the seed)")
    return summary


def final_summary(out, seeds, smoke):
    per = {s: json.loads((out / f"seed{s}" / "seed_summary.json").read_text()) for s in seeds}
    n_ok = sum(per[s]["seed_passes"] for s in seeds)
    n_ok_lit = sum(per[s]["seed_passes_literal_without_c0"] for s in seeds)
    need = 2 if len(seeds) == 3 else len(seeds)
    go = n_ok >= need
    aucs = [per[s]["per_video_auc"]["grumobilenet_original"]["auc"] for s in seeds]
    rets = [per[s]["causal"]["grumobilenet"]["retention_background_only"]["mean"] for s in seeds]
    med_auc = float(np.median([a for a in aucs if a is not None])) if all(a is not None for a in aucs) else None
    med_ret = float(np.median(rets)) if all(r is not None for r in rets) else None
    gate = go and med_auc is not None and med_auc >= THR["grumobilenet_auc_min"]
    final_reading = reading(med_ret) if gate else "no diagnosis"
    per_seed_reading = {}
    for s in seeds:
        a, r = aucs[seeds.index(s)], rets[seeds.index(s)]
        per_seed_reading[s] = (reading(r) if (go and a is not None and a >= THR["grumobilenet_auc_min"])
                               else "no diagnosis")
    summary = {"preregistration": "tad_diagnosis/results/official_split/PREREG.md (binding) + docstring of"
                                  " tad_diagnosis/official_split.py",
               "smoke": smoke, "seeds": list(seeds), "thresholds": THR,
               "c0_per_seed": {s: per[s]["c0"]["event_control_val_window_auc"] for s in seeds},
               "seeds_passing": n_ok, "seeds_passing_literal_without_c0": n_ok_lit,
               "module_verdict": "GO" if go else "NO-GO",
               "module_verdict_literal_without_c0": "GO" if n_ok_lit >= need else "NO-GO",
               "grumobilenet": {"original_auc_per_seed": aucs, "original_auc_median": med_auc,
                                "retention_background_only_per_seed": rets,
                                "retention_background_only_median": med_ret,
                                "gate_go_and_auc": gate, "reading": final_reading,
                                "reading_per_seed": per_seed_reading},
               "n_eval_accidents_lt21": per[seeds[0]]["n_eval_accidents_lt21"],
               "n_eval_accidents": per[seeds[0]]["n_eval_accidents"], "per_seed": per}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    lines = ["| seed | C0 | AUC grumobilenet orig | AUC grumobilenet background only | AUC grumobilenet first frame"
             " | AUC video_control (row 0) |", "|---|---|---|---|---|---|"]
    for s in seeds:
        p, A = per[s], per[s]["per_video_auc"]
        c0 = p["c0"]["event_control_val_window_auc"]
        c0 = "—" if c0 is None else f"{c0:.3f}"
        lines.append(f"| {s} | {c0} | {fmt(A['grumobilenet_original'])} | {fmt(A['grumobilenet_background_only'])}"
                     f" | {fmt(A['grumobilenet_first_frame'])} | {fmt(A['video_control_first_frame'])} |")
    lines += ["", "| seed | model | n | orig score | ret. background only | ret. first frame | ret. swap background |",
              "|---|---|---|---|---|---|---|"]
    for s in seeds:
        for m in ("event_control", "video_control", "grumobilenet"):
            c = per[s]["causal"][m]
            if c is None:
                lines.append(f"| {s} | {m} | 0 | — | — | — | — |")
                continue
            lines.append(f"| {s} | {m} | {c['n']} | {fmt(c['score_orig'])} | {fmt(c['retention_background_only'])}"
                         f" | {fmt(c['retention_first_frame'])} | {fmt(c['retention_swap_background'])} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6,
                    help="torch threads (6 = the recorded runs; needed for bit-exact reproduction)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true", help="TRAINING videos only; 1 epoch; seed 0")
    ap.add_argument("--resume", default=None, help="existing run directory to complete")
    ap.add_argument("--out-dir", type=Path, default=config.WORK_DIR / "tad_diagnosis" / "official_split",
                    help="parent of the new run directory (default: %(default)s)")
    args = ap.parse_args()
    if args.threads > 6:
        raise SystemExit("at most 6 threads")
    torch.set_num_threads(args.threads)
    if args.smoke:
        args.epochs, args.seeds = 1, [0]
    if args.resume:
        out = Path(args.resume)
        if not out.is_dir():
            raise SystemExit(f"{out} does not exist")
        cfg = json.loads((out / "config.json").read_text())
        if bool(cfg["smoke"]) != args.smoke or cfg["epochs"] != args.epochs or cfg["seeds"] != args.seeds:
            raise SystemExit("configuration differs from the run being resumed")
    else:
        out = new_run_dir(args.out_dir, args, args.smoke)
    t_start = time.time()
    log(f"TAD official split | output: {out} | seeds {args.seeds} | epochs {args.epochs} | threads {args.threads}")

    vids = load_index()
    train_all = vids["train"]
    if args.smoke:
        acc = [d for d in train_all if d["y"] == 1]
        nor = [d for d in train_all if d["y"] == 0]
        acc = acc[::max(1, len(acc) // 14)][:14]
        nor = nor[::max(1, len(nor) // 10)][:10]
        train, test = acc[:10] + nor[:6], acc[10:] + nor[6:]
        log(f"[smoke] pseudo-train {len(train)} and pseudo-evaluation {len(test)} videos, ALL from the training split")
    else:
        train, test = train_all, vids["test"]
        n = (sum(d["y"] for d in train), sum(1 - d["y"] for d in train), sum(d["y"] for d in test),
             sum(1 - d["y"] for d in test))
        if n != (261, 111, 16, 16):
            raise SystemExit(f"unexpected counts in the cache index: {n}")
    backbone = load_backbone(config.DETECTOR_CKPT)

    t0 = time.time()
    for i, d in enumerate(train + test, start=1):
        clean_feats(backbone, d)
        if i % 50 == 0 or i == len(train) + len(test):
            log(f"  clean features: {i}/{len(train) + len(test)} ({time.time() - t0:.0f} s)")

    for s in args.seeds:
        sdir = out / f"seed{s}"
        if (sdir / "seed_summary.json").exists():
            log(f"[s{s}] already done, skipping")
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        run_seed(s, backbone, train, test, args.epochs, sdir)
        log(f"  CPU used so far in this invocation: {cpu_seconds():.0f} s")

    summary = final_summary(out, args.seeds, args.smoke)
    cpu = cpu_seconds()
    with open(out / "cpu_log.csv", "a") as f:
        f.write(f"{time.strftime('%Y%m%d-%H%M%S')},{cpu:.1f},{time.time() - t_start:.1f}\n")
    log(f"\nmodule verdict: {summary['module_verdict']} (seeds passing: {summary['seeds_passing']})"
        f" | grumobilenet reading: {summary['grumobilenet']['reading']}")
    log(f"DONE | wall {time.time() - t_start:.0f} s | CPU {cpu:.0f} s | {out / 'summary.json'}")


if __name__ == "__main__":
    main()
