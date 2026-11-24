#!/usr/bin/env python3
"""TAD, 80/20 holdout: diagnosis of GRUMobileNet on a validation split of the TAD training set.

Paper: supporting analysis (it motivated attribution.py and cross_validation.py); attribution.py reuses its split and
its grumobilenet heads. Not itself a number of Table 4.

BINDING pre-registration: results/holdout/PREREG.md (written on 2026-10-05, before running). It is a new analysis,
motivated by a descriptive result of the official-split experiment (results/official_split/RESULT.md), and must be
reported as such. Deviations go to the "Deviations" section of results/holdout/RESULT.md.

Implementation of the pre-registration:
  - per-seed 80/20 split identical to that of official_split.py (rng(s+1), stratified; 297 train / 73 validation),
    reproduced with the same code. Check: the C0 of the saved event_control head, recomputed on this validation set,
    must match the value recorded by the official-split run;
  - grumobilenet: architecture and training of the official-split grumobilenet (controls.train_head: 6 epochs,
    Adam 1e-3, class-weighted CE, window label = video label; head seed s + 4, as there), trained only on the train side
    of the split (the 297 videos + the 2 accidents with K <= 10). Video score = max over windows;
  - event_control and video_control: the heads saved by the official-split run in <official-run>/seed<s>/ (not
    retrained);
  - measurements on the validation set:
      background = median of rows [0, 10) (median of [0, K) if K < 10); background only; frozen first frame;
      controls on the validation accidents with K >= 21, marked as in the official-split experiment (the same cell
      rng([s, k])), score = mean of the windows ending at rows 12, 16 and 20; background-only and first-frame
      retentions; grumobilenet: per-video AUC (original, background only, first frame) on all validation videos, and
      background-only and first-frame retentions on the same accidents as the controls (K >= 21);
      swap background: not computed (the pre-registration leaves it optional and not validated);
  - CIs: retention by bootstrap over videos (B = 1000, interventions.ratio_ci); AUC by class-stratified bootstrap
    (official_split.auc_ci, B = 1000);
  - criteria (thresholds of the SO-TAD interventions): module valid on the validation set iff, in >= 2 of the 3 seeds,
    event_control has background-only retention <= 0.3 and video_control has background-only retention >= 0.8;
    grumobilenet gate = module valid and grumobilenet per-video validation AUC >= 0.70 (median over the 3 seeds);
    reading by the median grumobilenet background-only retention: >= 0.8 "scene", <= 0.3 "event", in between
    "inconclusive", failed gate "no diagnosis";
  - descriptive: grumobilenet on the official test set (original, background-only and first-frame AUC); AUC of
    video_control (row 0 repeated) on validation and test; composition of the validation set by batch of normals.

Inputs: config.TAD_FRAME_CACHE, config.TAD_FEATURE_CACHE (read only), config.DETECTOR_CKPT, the control heads of the
official-split run (--official-run, default: the recorded run).
Outputs: <out-dir>/<timestamp>/ (seed<s>/*.csv, seed<s>/grumobilenet_head.pt, seed<s>/seed_summary.json, summary.json,
table.md). Recorded run: results/holdout/20261005-110712/.

Usage (from the repository root):
  tools/run_guarded.sh 2500 work/logs/tad_holdout.log -- python tad_diagnosis/holdout.py
  Smoke (only videos of the TRAIN side of the split; no validation or test metric): --smoke
  Resume: --resume <run dir>
  Threads: keep the default --threads 6 (it overrides the 2 threads set by run_guarded.sh). The recorded heads were
  trained with 6 threads; with another thread count the heads differ at float round-off and the numbers drift.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import official_split as T  # noqa: E402  (puts the repo root and sotad_diagnosis on sys.path)
import config  # noqa: E402
import controls  # noqa: E402
from controls import Head, load_backbone, plant, safe_auc, train_head, write_csv  # noqa: E402
from interventions import background, pooled, score_static, score_windows  # noqa: E402

OFFICIAL_RUN = T.RESULTS / "official_split" / "20261005-101615"   # recorded run whose control heads are used
SEEDS = (0, 1, 2)
THR = {"event_control_max": 0.3, "video_control_min": 0.8, "grumobilenet_auc_min": 0.70, "scene_reading": 0.8,
       "event_reading": 0.3}
BATCHES = ("20220518", "20220620", "20230818")   # name prefixes of the three batches of training normals


def log(msg):
    print(msg, flush=True)


def feats_clean(backbone, d):
    p = T.FEATS / "clean" / d["split"] / f"{d['vid']}.npy"
    if p.exists():
        return np.load(p)
    return pooled(backbone, np.asarray(T.frames(d))).astype(np.float32)  # not written to the feature cache


def feats_marked(backbone, d, seed, Fc):
    """Features of the clip with the official-split mark on rows >= 10 (same cell); read from its feature cache.
    As in official_split.marked_train_feats, normals and accidents with K <= 10 get no mark."""
    if d["y"] == 0 or d["K"] <= T.KF_ROW:
        return Fc
    p = T.FEATS / f"marked_s{seed}" / d["split"] / f"{d['vid']}.npy"
    if p.exists():
        return np.load(p)
    fr = np.asarray(T.frames(d)[T.KF_ROW:])
    F = Fc.copy()
    F[T.KF_ROW:] = pooled(backbone, plant(fr, np.ones(len(fr), bool), T.marker_cell(seed, d)))
    return F


def split(seed, train):
    """Exact reproduction of the 80/20 split of the official-split experiment (same code as official_split.run_seed)."""
    ctrl = [d for d in train if not (d["y"] == 1 and d["K"] <= T.KF_ROW)]
    rng = np.random.default_rng(seed + 1)
    val_ids = set()
    for group in ([d["vid"] for d in ctrl if d["y"] == 1], [d["vid"] for d in ctrl if d["y"] == 0]):
        val_ids |= set(rng.choice(group, size=max(1, len(group) // 5), replace=False).tolist())
    tr_ctrl = [d for d in ctrl if d["vid"] not in val_ids]
    va = [d for d in ctrl if d["vid"] in val_ids]
    tr_real = [d for d in train if d["vid"] not in val_ids]
    return tr_ctrl, va, tr_real


def batch(d):
    if d["y"] == 1:
        return "accident"
    pre = d["vid"].split("_")[0]
    return pre if pre in BATCHES else "other"


def load_saved(path):
    h = Head()
    h.load_state_dict(torch.load(path, map_location="cpu"))
    return h.eval()


def measure(videos, realb, scene, marker, backbone, seed, feats_c, with_controls):
    rows, ctrl = [], []
    for d in videos:
        fr = np.asarray(T.frames(d))
        K, Fc = d["K"], feats_c[d["vid"]]
        f_bg = pooled(backbone, background(fr, T.KF_ROW)[None])[0]
        rows.append({"video": d["vid"], "label": d["y"], "K": K, "batch": batch(d), "folder": d["folder"],
                     "orig_max": float(score_windows(realb, Fc, np.arange(K)).max()),
                     "background_only": score_static(realb, f_bg), "first_frame": score_static(realb, Fc[0]),
                     "video_control_first_frame": score_static(scene, Fc[0])})
        if with_controls and d["y"] == 1 and K >= T.MIN_ROWS_CTRL:
            Fm = feats_marked(backbone, d, seed, Fc)
            for name, h in (("event_control", marker), ("video_control", scene)):
                ctrl.append({"video": d["vid"], "K": K, "cell": T.marker_cell(seed, d), "model": name,
                             "orig": float(score_windows(h, Fm, T.CTRL_ROWS).mean()),
                             "background_only": score_static(h, f_bg), "first_frame": score_static(h, Fm[0])})
    return rows, ctrl


def aucs(rows):
    y = [r["label"] for r in rows]
    return {"grumobilenet_original": T.auc_ci(y, [r["orig_max"] for r in rows]),
            "grumobilenet_background_only": T.auc_ci(y, [r["background_only"] for r in rows]),
            "grumobilenet_first_frame": T.auc_ci(y, [r["first_frame"] for r in rows]),
            "video_control_first_frame": T.auc_ci(y, [r["video_control_first_frame"] for r in rows])}


def run_seed(seed, backbone, train, test, epochs, sdir, smoke, official_run):
    t0 = time.time()
    tr_ctrl, va, tr_real = split(seed, train)
    if not smoke and (len(tr_ctrl), len(va), len(tr_real)) != (297, 73, 299):
        raise SystemExit(f"unexpected split: {len(tr_ctrl)}/{len(va)}/{len(tr_real)}")
    marker = load_saved(official_run / f"seed{seed}" / "event_control_head.pt")
    scene = load_saved(official_run / f"seed{seed}" / "video_control_head.pt")
    if smoke:
        # Only videos of the TRAIN side of the split: disjoint pseudo-train, pseudo-validation and pseudo-test.
        acc = [d for d in tr_real if d["y"] == 1 and d["K"] >= T.MIN_ROWS_CTRL]
        nor = [d for d in tr_real if d["y"] == 0]
        acc, nor = acc[::max(1, len(acc) // 16)][:16], nor[::max(1, len(nor) // 10)][:10]
        tr_real, va, test = acc[:10] + nor[:6], acc[10:14] + nor[6:8], acc[14:16] + nor[8:10]
        val_side = {d["vid"] for d in split(seed, train)[1]}
        assert not any(d["vid"] in val_side or d["split"] != "train" for d in tr_real + va + test)
        log(f"[smoke] pseudo-train {len(tr_real)}, pseudo-val {len(va)}, pseudo-test {len(test)}; all from the train"
            f" side of the split")
    feats_c = {d["vid"]: feats_clean(backbone, d) for d in tr_real + va + test}

    t1 = time.time()
    realb = train_head([{"F": feats_c[d["vid"]], "y": T.video_labels(d)} for d in tr_real], False, epochs, seed + 4)
    torch.save(realb.state_dict(), sdir / "grumobilenet_head.pt")
    log(f"[s{seed}] grumobilenet trained on {len(tr_real)} videos ({time.time() - t1:.0f} s)")

    c0_check = None
    if not smoke:  # split consistency: the C0 of the saved event_control head must match the official-split one
        Fm_va = {d["vid"]: feats_marked(backbone, d, seed, feats_c[d["vid"]]) for d in va}
        c0 = safe_auc(np.concatenate([T.marker_labels(d) for d in va]),
                      np.concatenate([controls.predict_F(marker, Fm_va[d["vid"]]) for d in va]))
        ref = json.loads((official_run / f"seed{seed}" / "seed_summary.json").read_text())["c0"][
            "event_control_val_window_auc"]
        c0_check = {"c0_recomputed": c0, "c0_official_split": ref, "match": c0 is not None and abs(c0 - ref) < 1e-6}
        log(f"[s{seed}] split check: recomputed C0 {c0:.6f} vs official split {ref:.6f} -> match={c0_check['match']}")
        if not c0_check["match"]:
            raise SystemExit("the split does not reproduce the official-split one")

    val_rows, ctrl_rows = measure(va, realb, scene, marker, backbone, seed, feats_c, with_controls=True)
    test_rows, _ = measure(test, realb, scene, marker, backbone, seed, feats_c, with_controls=False)
    write_csv(sdir / "val_per_video.csv", val_rows)
    write_csv(sdir / "controls_val_per_video.csv", ctrl_rows)
    write_csv(sdir / "test_per_video.csv", test_rows)

    keys = ("background_only", "first_frame")
    acc21 = [r for r in val_rows if r["label"] == 1 and r["K"] >= T.MIN_ROWS_CTRL]
    S = {"event_control": T.summarize([r for r in ctrl_rows if r["model"] == "event_control"], keys),
         "video_control": T.summarize([r for r in ctrl_rows if r["model"] == "video_control"], keys),
         "grumobilenet": T.summarize([{**r, "orig": r["orig_max"]} for r in acc21], keys)}

    def ret(m):
        return None if S[m] is None else S[m]["retention_background_only"]["mean"]

    conds = {"event_control_ret_background_only_le_0.3":
                 ret("event_control") is not None and ret("event_control") <= THR["event_control_max"],
             "video_control_ret_background_only_ge_0.8":
                 ret("video_control") is not None and ret("video_control") >= THR["video_control_min"]}
    acc_val = [r for r in val_rows if r["label"] == 1]
    comp = {"accidents": len(acc_val), **{f"normals_{b}": sum(r["batch"] == b for r in val_rows) for b in BATCHES},
            "normals_other": sum(r["batch"] == "other" for r in val_rows)}
    summary = {"seed": seed, "n_train_grumobilenet": len(tr_real), "n_val": len(val_rows), "val_composition": comp,
               "n_val_accidents_lt21": sum(r["K"] < T.MIN_ROWS_CTRL for r in acc_val),
               "n_val_accidents_ge21": len(acc21), "split_check": c0_check, "val_causal": S,
               "val_auc": aucs(val_rows), "test_auc": aucs(test_rows), "conditions": conds,
               "seed_valid": all(conds.values()), "time_s": round(time.time() - t0, 1)}
    tmp = sdir / "seed_summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, sdir / "seed_summary.json")
    log(f"[s{seed}] done ({time.time() - t0:.0f} s)")
    return summary


def reading(v):
    if v is None:
        return "no diagnosis"
    if v >= THR["scene_reading"]:
        return "scene"
    if v <= THR["event_reading"]:
        return "event"
    return "inconclusive"


def final_summary(out, seeds, smoke, official_run):
    per = {s: json.loads((out / f"seed{s}" / "seed_summary.json").read_text()) for s in seeds}
    need = 2 if len(seeds) == 3 else len(seeds)
    n_ok = sum(per[s]["seed_valid"] for s in seeds)
    valid = n_ok >= need
    aucs_v = [per[s]["val_auc"]["grumobilenet_original"]["auc"] for s in seeds]
    rets = [None if per[s]["val_causal"]["grumobilenet"] is None
            else per[s]["val_causal"]["grumobilenet"]["retention_background_only"]["mean"] for s in seeds]
    med_auc = float(np.median(aucs_v)) if all(a is not None for a in aucs_v) else None
    med_ret = float(np.median(rets)) if all(r is not None for r in rets) else None
    gate = valid and med_auc is not None and med_auc >= THR["grumobilenet_auc_min"]
    per_seed = {s: (reading(r) if (valid and a is not None and a >= THR["grumobilenet_auc_min"]) else "no diagnosis")
                for s, a, r in zip(seeds, aucs_v, rets)}
    summary = {"preregistration": "tad_diagnosis/results/holdout/PREREG.md (binding) + docstring of"
                                  " tad_diagnosis/holdout.py",
               "smoke": smoke, "seeds": list(seeds), "thresholds": THR, "control_heads": T.rel(official_run),
               "valid_seeds": n_ok, "module_valid_on_validation": valid,
               "grumobilenet": {"val_original_auc_per_seed": aucs_v, "val_original_auc_median": med_auc,
                                "retention_background_only_per_seed": rets,
                                "retention_background_only_median": med_ret,
                                "gate": gate, "reading": reading(med_ret) if gate else "no diagnosis",
                                "reading_per_seed": per_seed},
               "per_seed": per}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    L = ["| Seed | n val (acc./norm.) | AUC grumobilenet original | AUC grumobilenet background only"
         " | AUC grumobilenet first frame | AUC video_control (row 0) |", "|---|---|---|---|---|---|"]
    for s in seeds:
        p, A = per[s], per[s]["val_auc"]
        L.append(f"| {s} | {p['n_val']} ({p['val_composition']['accidents']}/{p['n_val'] - p['val_composition']['accidents']})"
                 f" | {T.fmt(A['grumobilenet_original'])} | {T.fmt(A['grumobilenet_background_only'])}"
                 f" | {T.fmt(A['grumobilenet_first_frame'])} | {T.fmt(A['video_control_first_frame'])} |")
    L += ["", "| Seed | Model | n | Original score | Ret. background only | Ret. first frame |", "|---|---|---|---|---|---|"]
    for s in seeds:
        for m in ("event_control", "video_control", "grumobilenet"):
            x = per[s]["val_causal"][m]
            if x is None:
                L.append(f"| {s} | {m} | 0 | — | — | — |")
                continue
            L.append(f"| {s} | {m} | {x['n']} | {T.fmt(x['score_orig'])}"
                     f" | {T.fmt(x['retention_background_only'])} | {T.fmt(x['retention_first_frame'])} |")
    L += ["", "| Seed | n test | AUC grumobilenet original | AUC grumobilenet background only"
          " | AUC grumobilenet first frame | AUC video_control (row 0) |", "|---|---|---|---|---|---|"]
    for s in seeds:
        A = per[s]["test_auc"]
        L.append(f"| {s} | {A['grumobilenet_original']['n']} | {T.fmt(A['grumobilenet_original'])}"
                 f" | {T.fmt(A['grumobilenet_background_only'])} | {T.fmt(A['grumobilenet_first_frame'])}"
                 f" | {T.fmt(A['video_control_first_frame'])} |")
    (out / "table.md").write_text("\n".join(L) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6,
                    help="torch threads (6 = the recorded runs; needed for bit-exact reproduction)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--smoke", action="store_true", help="only the train side of the split; 1 epoch; seed 0")
    ap.add_argument("--resume", default=None, help="existing run directory to complete")
    ap.add_argument("--out-dir", type=Path, default=config.WORK_DIR / "tad_diagnosis" / "holdout",
                    help="parent of the new run directory (default: %(default)s)")
    ap.add_argument("--official-run", type=Path, default=OFFICIAL_RUN,
                    help="official_split.py run whose control heads are used (default: the recorded run)")
    args = ap.parse_args()
    if args.threads > 6:
        raise SystemExit("at most 6 threads")
    torch.set_num_threads(args.threads)
    if args.smoke:
        args.epochs, args.seeds = 1, [0]
    if args.resume:
        out = Path(args.resume)
        cfg = json.loads((out / "config.json").read_text())
        if bool(cfg["smoke"]) != args.smoke or cfg["epochs"] != args.epochs or cfg["seeds"] != args.seeds:
            raise SystemExit("configuration differs from the run being resumed")
    else:
        out = T.new_run_dir(args.out_dir, args, args.smoke)
    t_start = time.time()
    log(f"TAD holdout | output: {out} | seeds {args.seeds} | epochs {args.epochs} | threads {args.threads}")
    vids = T.load_index()
    train, test = vids["train"], vids["test"]
    backbone = load_backbone(config.DETECTOR_CKPT)
    for s in args.seeds:
        sdir = out / f"seed{s}"
        if (sdir / "seed_summary.json").exists():
            log(f"[s{s}] already done, skipping")
            continue
        sdir.mkdir(parents=True, exist_ok=True)
        run_seed(s, backbone, train, [] if args.smoke else test, args.epochs, sdir, args.smoke, args.official_run)
        log(f"  CPU used so far in this invocation: {T.cpu_seconds():.0f} s")
    summary = final_summary(out, args.seeds, args.smoke, args.official_run)
    cpu = T.cpu_seconds()
    with open(out / "cpu_log.csv", "a") as fh:
        fh.write(f"{time.strftime('%Y%m%d-%H%M%S')},{cpu:.1f},{time.time() - t_start:.1f}\n")
    if not args.smoke:
        log(f"\nmodule valid on validation: {summary['module_valid_on_validation']}"
            f" ({summary['valid_seeds']} seeds) | grumobilenet reading: {summary['grumobilenet']['reading']}")
    log(f"DONE | wall {time.time() - t_start:.0f} s | CPU {cpu:.0f} s | {out / 'summary.json'}")


if __name__ == "__main__":
    main()
