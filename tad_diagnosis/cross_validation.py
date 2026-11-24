#!/usr/bin/env python3
"""TAD, 5-fold cross-validation of the GRUMobileNet diagnosis on the TAD training set.

Paper: Table 4, cross-validation rows (AUC 0.92 [0.89, 0.95]; video control AUC 0.88; background-only retention
GRUMobileNet 90% [87, 94], first frame 93% [90, 96]; event control 11% [8, 15]; video control 95% [91, 98]).

BINDING pre-registration: results/cross_validation/PREREG.md (written on 2026-10-05, before running). Refinement of the
holdout experiment on the same data (not independent of it). Deviations go to the "Deviations" section of
results/cross_validation/RESULT.md.

Implementation of the pre-registration:
  - folds: StratifiedKFold(5, shuffle=True, random_state=0) by class over the 370 training videos that have some
    row >= 10. The 2 accidents with K <= 10 always stay in the grumobilenet training set and are never diagnosed;
  - per fold, trained on the 4 training folds with the architecture, training, mark and labels of the official-split
    and holdout experiments (controls.train_head, 6 epochs, Adam 1e-3, class-weighted CE; official-split mark with
    seed 0, cell rng([0, k]), rows >= 10): event_control (label = accident and row >= 10; head seed 0 + 2),
    video_control (row 0 repeated, video label; 0 + 3) and grumobilenet (video label, + the 2 accidents with K <= 10;
    0 + 4). Seed 0 everywhere: a single repetition;
  - measurements on the held-out fold (holdout.measure): background = median of rows [0, 10); background only; frozen
    first frame; controls on the accidents with K >= 21 (marked clip; mean of the windows ending at 12, 16 and 20);
    grumobilenet = max over windows; no swap background;
  - pooled out-of-fold aggregation: retentions on the accidents with K >= 21 (interventions.ratio_ci, bootstrap over
    videos, B = 1000); AUC on the 370 videos (class-stratified bootstrap, official_split.auc_ci, B = 1000); C0 = pooled
    event_control window AUC;
  - criteria: C0 >= 0.90; module valid iff event_control has background-only retention <= 0.3 and video_control has
    background-only retention >= 0.8 (pooled; C0 is also required here, and the literal version without C0 is
    recorded); gate = pooled grumobilenet per-video AUC >= 0.70; reading by the pooled grumobilenet background-only
    retention: >= 0.8 "scene", <= 0.3 "event", in between "inconclusive", failed gate or module "no diagnosis";
    descriptive: whether the whole 95% CI lies above 0.8;
  - descriptive: per-fold values; pooled background-only and first-frame AUC; retention and AUC per batch of normals
    (AUC = all accidents against the normals of the batch; retention = background only / original on the normals of
    the batch); AUC on the official test set (32 videos) of the mean of the 5 grumobilenet heads (mean of the
    per-video scores).

Consistency check (in the smoke run, with no out-of-fold data): with the training code of this script, retrain the
seed-0 heads of the official-split run (event_control, video_control) and of the holdout run (grumobilenet) on the
seed-0 80/20 split and compare the weights with the saved ones. If they do not match, abort. The full run reads the
check from the latest smoke run in the same --out-dir, so run --smoke first.

Inputs: config.TAD_FRAME_CACHE, config.TAD_FEATURE_CACHE (read only), config.DETECTOR_CKPT; for the smoke check, the
heads of the official-split and holdout runs (--official-run, --holdout-run; default: the recorded runs).
Outputs: <out-dir>/<timestamp>/ (fold<f>/*.csv, fold<f>/*_head.pt, fold<f>/fold_summary.json, summary.json, table.md,
test_mean_of_models.csv). Recorded run: results/cross_validation/20261005-120959/.

Usage (from the repository root):
  tools/run_guarded.sh 2500 work/logs/tad_cross_validation_smoke.log -- python tad_diagnosis/cross_validation.py --smoke
  tools/run_guarded.sh 2500 work/logs/tad_cross_validation.log -- python tad_diagnosis/cross_validation.py
  Smoke (only the training folds of fold 0; no out-of-fold metric; includes the check): --smoke
  Resume: --resume <run dir>
  Threads: keep the default --threads 6 (it overrides the 2 threads set by run_guarded.sh). The recorded heads were
  trained with 6 threads; with another thread count the heads differ at float round-off and the numbers drift.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import official_split as T  # noqa: E402  (puts the repo root and sotad_diagnosis on sys.path)
import holdout as B  # noqa: E402
import config  # noqa: E402
import controls  # noqa: E402
from controls import load_backbone, safe_auc, train_head, write_csv  # noqa: E402
from interventions import ratio_ci  # noqa: E402

OFFICIAL_RUN = B.OFFICIAL_RUN                                 # recorded runs used by the smoke consistency check
HOLDOUT_RUN = T.RESULTS / "holdout" / "20261005-110712"
SEED, N_FOLDS = 0, 5
THR = {"c0": 0.90, "event_control_max": 0.3, "video_control_min": 0.8, "grumobilenet_auc_min": 0.70,
       "scene_reading": 0.8, "event_reading": 0.3}
BATCHES = B.BATCHES


def log(msg):
    print(msg, flush=True)


class Feats:
    """Clean and marked features (official-split mark with seed 0), read from TAD_FEATURE_CACHE, memoized per video."""

    def __init__(self, backbone):
        self.bb, self.c, self.m = backbone, {}, {}

    def clean(self, d):
        if d["vid"] not in self.c:
            self.c[d["vid"]] = B.feats_clean(self.bb, d)
        return self.c[d["vid"]]

    def marked(self, d):
        if d["vid"] not in self.m:
            self.m[d["vid"]] = B.feats_marked(self.bb, d, SEED, self.clean(d))
        return self.m[d["vid"]]


def train_models(tr_ctrl, tr_real, FS, epochs):
    marker = train_head([{"F": FS.marked(d), "y": T.marker_labels(d)} for d in tr_ctrl], False, epochs, SEED + 2)
    scene = train_head([{"F": FS.clean(d), "y": T.video_labels(d)} for d in tr_ctrl], True, epochs, SEED + 3)
    real = train_head([{"F": FS.clean(d), "y": T.video_labels(d)} for d in tr_real], False, epochs, SEED + 4)
    return marker, scene, real


def make_folds(train):
    ctrl = [d for d in train if not (d["y"] == 1 and d["K"] <= T.KF_ROW)]
    excl = [d for d in train if d["y"] == 1 and d["K"] <= T.KF_ROW]
    y = np.array([d["y"] for d in ctrl])
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of = {}
    for f, (_, te) in enumerate(skf.split(np.zeros((len(y), 1)), y)):
        for i in te:
            fold_of[ctrl[i]["vid"]] = f
    return ctrl, excl, fold_of


def max_param_diff(h, path):
    ref = torch.load(path, map_location="cpu")
    return max(float((v - ref[k]).abs().max()) for k, v in h.state_dict().items())


def consistency_check(train, FS, official_run, holdout_run):
    """Retrain the seed-0 heads of the official-split and holdout runs with this script's code; compare with the saved."""
    tr_ctrl, _, tr_real = B.split(0, train)
    t0 = time.time()
    marker, scene, real = train_models(tr_ctrl, tr_real, FS, 6)
    diffs = {"event_control_vs_official_split_s0": max_param_diff(marker, official_run / "seed0" / "event_control_head.pt"),
             "video_control_vs_official_split_s0": max_param_diff(scene, official_run / "seed0" / "video_control_head.pt"),
             "grumobilenet_vs_holdout_s0": max_param_diff(real, holdout_run / "seed0" / "grumobilenet_head.pt")}
    ok = all(v <= 1e-5 for v in diffs.values())
    log(f"[check] max |delta weight|: {json.dumps({k: f'{v:.1e}' for k, v in diffs.items()})} -> match={ok}"
        f" ({time.time() - t0:.0f} s)")
    return {"max_weight_diffs": diffs, "match": ok}


def aucs(rows):
    y = [r["label"] for r in rows]
    return {"grumobilenet_original": T.auc_ci(y, [r["orig_max"] for r in rows]),
            "grumobilenet_background_only": T.auc_ci(y, [r["background_only"] for r in rows]),
            "grumobilenet_first_frame": T.auc_ci(y, [r["first_frame"] for r in rows]),
            "video_control_first_frame": T.auc_ci(y, [r["video_control_first_frame"] for r in rows])}


def causal(rows, ctrl_rows):
    keys = ("background_only", "first_frame")
    acc21 = [{**r, "orig": r["orig_max"]} for r in rows if r["label"] == 1 and r["K"] >= T.MIN_ROWS_CTRL]
    return {"event_control": T.summarize([r for r in ctrl_rows if r["model"] == "event_control"], keys),
            "video_control": T.summarize([r for r in ctrl_rows if r["model"] == "video_control"], keys),
            "grumobilenet": T.summarize(acc21, keys)}


def run_fold(f, held, tr_ctrl, tr_real, test, FS, backbone, epochs, fdir):
    t0 = time.time()
    marker, scene, real = train_models(tr_ctrl, tr_real, FS, epochs)
    for name, h in (("event_control", marker), ("video_control", scene), ("grumobilenet", real)):
        torch.save(h.state_dict(), fdir / f"{name}_head.pt")
    t_train = time.time() - t0
    y = np.concatenate([T.marker_labels(d) for d in held])
    p = np.concatenate([controls.predict_F(marker, FS.marked(d)) for d in held])
    np.savez(fdir / "c0_windows.npz", y=y, p=p)
    rows, ctrl_rows = B.measure(held, real, scene, marker, backbone, SEED, {d["vid"]: FS.clean(d) for d in held}, True)
    test_rows, _ = B.measure(test, real, scene, marker, backbone, SEED, {d["vid"]: FS.clean(d) for d in test}, False)
    for r in rows + ctrl_rows + test_rows:
        r["fold"] = f
    write_csv(fdir / "heldout_per_video.csv", rows)
    write_csv(fdir / "controls_heldout.csv", ctrl_rows)
    write_csv(fdir / "test_per_video.csv", test_rows)
    summ = {"fold": f, "n_train_controls": len(tr_ctrl), "n_train_grumobilenet": len(tr_real), "n_heldout": len(held),
            "n_heldout_accidents": sum(d["y"] for d in held),
            "n_heldout_accidents_ge21": sum(d["y"] == 1 and d["K"] >= T.MIN_ROWS_CTRL for d in held),
            "c0_fold": safe_auc(y, p), "causal": causal(rows, ctrl_rows), "auc": aucs(rows),
            "train_time_s": round(t_train, 1), "time_s": round(time.time() - t0, 1)}
    tmp = fdir / "fold_summary.json.tmp"
    tmp.write_text(json.dumps(summ, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, fdir / "fold_summary.json")
    log(f"[fold {f}] training {t_train:.0f} s | held out: {len(held)} videos | done ({time.time() - t0:.0f} s)")
    return summ


def read_csv(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("label", "K", "fold", "cell"):
            if k in r:
                r[k] = int(r[k])
        for k in ("orig_max", "background_only", "first_frame", "video_control_first_frame", "orig"):
            if k in r:
                r[k] = float(r[k])
    return rows


def reading(v):
    if v is None:
        return "no diagnosis"
    if v >= THR["scene_reading"]:
        return "scene"
    if v <= THR["event_reading"]:
        return "event"
    return "inconclusive"


def final_summary(out, folds, smoke, check):
    rows, ctrl_rows, test_rows, ys, ps, per = [], [], [], [], [], {}
    for f in folds:
        fdir = out / f"fold{f}"
        rows += read_csv(fdir / "heldout_per_video.csv")
        ctrl_rows += read_csv(fdir / "controls_heldout.csv")
        test_rows += read_csv(fdir / "test_per_video.csv")
        z = np.load(fdir / "c0_windows.npz")
        ys.append(z["y"])
        ps.append(z["p"])
        per[f] = json.loads((fdir / "fold_summary.json").read_text())
    c0 = safe_auc(np.concatenate(ys), np.concatenate(ps))
    S = causal(rows, ctrl_rows)
    A = aucs(rows)

    def ret(m):
        return None if S[m] is None else S[m]["retention_background_only"]["mean"]

    conds = {"c0_ge_0.90": c0 is not None and c0 >= THR["c0"],
             "event_control_ret_background_only_le_0.3":
                 ret("event_control") is not None and ret("event_control") <= THR["event_control_max"],
             "video_control_ret_background_only_ge_0.8":
                 ret("video_control") is not None and ret("video_control") >= THR["video_control_min"]}
    valid = all(conds.values())
    valid_lit = conds["event_control_ret_background_only_le_0.3"] and conds["video_control_ret_background_only_ge_0.8"]
    auc_o = A["grumobilenet_original"]["auc"]
    gate = valid and auc_o is not None and auc_o >= THR["grumobilenet_auc_min"]
    r_real = ret("grumobilenet")
    lo = None if S["grumobilenet"] is None else S["grumobilenet"]["retention_background_only"]["ci95"][0]
    # Descriptive, per batch of normals.
    acc = [r for r in rows if r["label"] == 1]
    batches = {}
    for b in BATCHES:
        nl = [r for r in rows if r["batch"] == b]
        if not nl:
            continue
        batches[b] = {"n_normals": len(nl), "auc": aucs(acc + nl),
                      "normals_retention_background_only": ratio_ci([r["background_only"] for r in nl],
                                                                    [r["orig_max"] for r in nl]),
                      "normals_retention_first_frame": ratio_ci([r["first_frame"] for r in nl],
                                                                [r["orig_max"] for r in nl])}
    # Descriptive: official test set with the mean of the grumobilenet heads (mean of the per-video scores).
    by_vid = {}
    for r in test_rows:
        by_vid.setdefault(r["video"], []).append(r)
    tmean = [{"video": v, "label": R[0]["label"], "n_models": len(R),
              **{k: float(np.mean([x[k] for x in R]))
                 for k in ("orig_max", "background_only", "first_frame", "video_control_first_frame")}}
             for v, R in sorted(by_vid.items())]
    write_csv(out / "test_mean_of_models.csv", tmean)
    acc_all = [r for r in rows if r["label"] == 1]
    summary = {"preregistration": "tad_diagnosis/results/cross_validation/PREREG.md (binding) + docstring of"
                                  " tad_diagnosis/cross_validation.py",
               "smoke": smoke, "folds": list(folds), "thresholds": THR, "consistency_check": check,
               "n_diagnosed_videos": len(rows), "n_accidents": len(acc_all),
               "n_accidents_lt21": sum(r["K"] < T.MIN_ROWS_CTRL for r in acc_all),
               "n_accidents_ge21": sum(r["K"] >= T.MIN_ROWS_CTRL for r in acc_all),
               "pooled_c0": c0, "pooled_causal": S, "pooled_auc": A, "conditions": conds,
               "module_valid": valid, "module_valid_literal_without_c0": valid_lit,
               "gate": gate, "grumobilenet_reading": reading(r_real) if gate else "no diagnosis",
               "grumobilenet_ret_background_only": r_real, "ci95_entirely_above_0.8": lo is not None and lo > 0.8,
               "per_normal_batch": batches, "test_mean_of_models": aucs(tmean) if tmean else None,
               "per_fold": per}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))

    fm = T.fmt
    L = ["| Scope | n videos | C0 | AUC grumobilenet original | AUC grumobilenet background only"
         " | AUC grumobilenet first frame | AUC video_control (row 0) |",
         "|---|---|---|---|---|---|---|",
         f"| pooled | {len(rows)} | {'—' if c0 is None else f'{c0:.3f}'} | {fm(A['grumobilenet_original'])}"
         f" | {fm(A['grumobilenet_background_only'])} | {fm(A['grumobilenet_first_frame'])}"
         f" | {fm(A['video_control_first_frame'])} |"]
    for f in folds:
        p, a = per[f], per[f]["auc"]
        c0f = "—" if p["c0_fold"] is None else f"{p['c0_fold']:.3f}"
        L.append(f"| fold {f} | {p['n_heldout']} | {c0f} | {fm(a['grumobilenet_original'])}"
                 f" | {fm(a['grumobilenet_background_only'])} | {fm(a['grumobilenet_first_frame'])}"
                 f" | {fm(a['video_control_first_frame'])} |")
    L += ["", "| Scope | Model | n | Original score | Ret. background only | Ret. first frame |", "|---|---|---|---|---|---|"]
    for scope, SS in [("pooled", S)] + [(f"fold {f}", per[f]["causal"]) for f in folds]:
        for m in ("event_control", "video_control", "grumobilenet"):
            x = SS[m]
            if x is None:
                L.append(f"| {scope} | {m} | 0 | — | — | — |")
                continue
            L.append(f"| {scope} | {m} | {x['n']} | {fm(x['score_orig'])}"
                     f" | {fm(x['retention_background_only'])} | {fm(x['retention_first_frame'])} |")
    L += ["", "| Batch of normals | n normals | AUC original (all accidents × batch) | AUC background only"
          " | AUC first frame | Ret. background only on normals | Ret. first frame on normals |",
          "|---|---|---|---|---|---|---|"]
    for k, v in batches.items():
        L.append(f"| {k} | {v['n_normals']} | {fm(v['auc']['grumobilenet_original'])}"
                 f" | {fm(v['auc']['grumobilenet_background_only'])} | {fm(v['auc']['grumobilenet_first_frame'])}"
                 f" | {fm(v['normals_retention_background_only'])} | {fm(v['normals_retention_first_frame'])} |")
    if tmean:
        t = summary["test_mean_of_models"]
        L += ["", "| Official test (mean of the grumobilenet heads) | n | AUC original | AUC background only"
              " | AUC first frame | AUC video_control (mean) |", "|---|---|---|---|---|---|",
              f"| {len(tmean)} videos | {len(tmean)} | {fm(t['grumobilenet_original'])}"
              f" | {fm(t['grumobilenet_background_only'])} | {fm(t['grumobilenet_first_frame'])}"
              f" | {fm(t['video_control_first_frame'])} |"]
    (out / "table.md").write_text("\n".join(L) + "\n")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=6,
                    help="torch threads (6 = the recorded runs; needed for bit-exact reproduction)")
    ap.add_argument("--smoke", action="store_true", help="only the training folds of fold 0; includes the check")
    ap.add_argument("--resume", default=None, help="existing run directory to complete")
    ap.add_argument("--out-dir", type=Path, default=config.WORK_DIR / "tad_diagnosis" / "cross_validation",
                    help="parent of the new run directory; the full run reads the check of the latest smoke run"
                         " here (default: %(default)s)")
    ap.add_argument("--official-run", type=Path, default=OFFICIAL_RUN,
                    help="official_split.py run used by the smoke check (default: the recorded run)")
    ap.add_argument("--holdout-run", type=Path, default=HOLDOUT_RUN,
                    help="holdout.py run used by the smoke check (default: the recorded run)")
    args = ap.parse_args()
    if args.threads > 6:
        raise SystemExit("at most 6 threads")
    torch.set_num_threads(args.threads)
    if args.resume:
        out = Path(args.resume)
        cfg = json.loads((out / "config.json").read_text())
        if bool(cfg["smoke"]) != args.smoke or cfg["epochs"] != args.epochs:
            raise SystemExit("configuration differs from the run being resumed")
    else:
        out = T.new_run_dir(args.out_dir, args, args.smoke)
    t_start = time.time()
    log(f"TAD cross-validation | output: {out} | epochs {args.epochs} | threads {args.threads} | smoke={args.smoke}")
    vids = T.load_index()
    train, test = vids["train"], vids["test"]
    backbone = load_backbone(config.DETECTOR_CKPT)
    FS = Feats(backbone)
    ctrl, excl, fold_of = make_folds(train)
    sizes = [sum(fold_of[d["vid"]] == f for d in ctrl) for f in range(N_FOLDS)]
    log(f"folds: {len(ctrl)} diagnosable videos, {len(excl)} always in training; sizes {sizes};"
        f" accidents per fold {[sum(fold_of[d['vid']] == f and d['y'] == 1 for d in ctrl) for f in range(N_FOLDS)]}")
    if not args.smoke and (len(ctrl), len(excl)) != (370, 2):
        raise SystemExit("unexpected counts")

    check_path = out / "check.json"
    if args.smoke:
        check = consistency_check(train, FS, args.official_run, args.holdout_run)
        check_path.write_text(json.dumps(check, indent=2))
        if not check["match"]:
            raise SystemExit("consistency check failed")
        side = [d for d in ctrl if fold_of[d["vid"]] != 0]  # only the TRAINING folds of fold 0
        acc = [d for d in side if d["y"] == 1 and d["K"] >= T.MIN_ROWS_CTRL]
        nor = [d for d in side if d["y"] == 0]
        acc, nor = acc[::max(1, len(acc) // 16)][:16], nor[::max(1, len(nor) // 10)][:10]
        tr_ctrl, held, ptest = acc[:10] + nor[:6], acc[10:14] + nor[6:8], acc[14:16] + nor[8:10]
        assert not any(fold_of[d["vid"]] == 0 for d in tr_ctrl + held + ptest)
        fdir = out / "fold0"
        fdir.mkdir(exist_ok=True)
        log(f"[smoke] pseudo-train {len(tr_ctrl)} (+{len(excl)} for grumobilenet), pseudo-held-out {len(held)},"
            f" pseudo-test {len(ptest)}; all from the training folds of fold 0; 1 epoch")
        run_fold(0, held, tr_ctrl, tr_ctrl + excl, ptest, FS, backbone, 1, fdir)
        folds = [0]
    else:
        smokes = sorted(p for p in out.parent.glob("*-smoke") if (p / "check.json").exists())
        if not smokes:
            raise SystemExit(f"run --smoke first (with the same --out-dir): it performs the consistency check")
        check = json.loads((smokes[-1] / "check.json").read_text())
        check["source"] = T.rel(smokes[-1])
        if not check["match"]:
            raise SystemExit("the consistency check of the smoke run did not match")
        log(f"consistency check (from smoke run {smokes[-1].name}): match={check['match']}")
        folds = list(range(N_FOLDS))
        for f in folds:
            fdir = out / f"fold{f}"
            if (fdir / "fold_summary.json").exists():
                log(f"[fold {f}] already done, skipping")
                continue
            fdir.mkdir(exist_ok=True)
            held = [d for d in ctrl if fold_of[d["vid"]] == f]
            held_ids = {d["vid"] for d in held}
            tr_ctrl = [d for d in ctrl if d["vid"] not in held_ids]
            tr_real = [d for d in train if d["vid"] not in held_ids]
            run_fold(f, held, tr_ctrl, tr_real, test, FS, backbone, args.epochs, fdir)
            log(f"  CPU used so far in this invocation: {T.cpu_seconds():.0f} s")
    summary = final_summary(out, folds, args.smoke, check)
    cpu = T.cpu_seconds()
    with open(out / "cpu_log.csv", "a") as fh:
        fh.write(f"{time.strftime('%Y%m%d-%H%M%S')},{cpu:.1f},{time.time() - t_start:.1f}\n")
    if not args.smoke:
        log(f"\npooled C0 {summary['pooled_c0']:.3f} | module valid: {summary['module_valid']} | gate:"
            f" {summary['gate']} | grumobilenet reading: {summary['grumobilenet_reading']}")
    log(f"DONE | wall {time.time() - t_start:.0f} s | CPU {cpu:.0f} s | {out / 'summary.json'}")


if __name__ == "__main__":
    main()
