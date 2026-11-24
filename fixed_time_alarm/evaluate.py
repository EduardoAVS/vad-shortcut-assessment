"""Fixed-time alarm vs GRUMobileNet on the SO-TAD test accidents (temporal-position check of the data assessment).

Paper result: an alarm fixed at 3.18 s into each clip (the median keyframe time of the training accidents), which
does not look at the image, reaches a within-video AUC of 0.822, above the 0.711 of GRUMobileNet
(results/reference_predictors.csv row "(ii)" column `auc`; results/variants.csv row grumobilenet/raw or warmup5).

What it computes (all on the 90 test accident videos, score grid of config.SOTAD_TEST_SCORES):
- Reference predictors that never look at the image, each an alarm at a fixed time T:
    (i) the first frame (T = 0); (ii) the median training keyframe (T = 79.5 frames = 3.18 s);
    (iii) a clock T* chosen on the training accidents to maximize S (T* = 90 frames = 3.60 s).
  As a score, a fixed-time alarm is the step function 1[t >= T]; its within-video AUC is computed like the
  detector's. The strongest reference (largest S) is the "strongest null".
- GRUMobileNet with 5 causal post-processing variants (raw, warmup5, smooth5, runmin, cusum; see metrics.py).
- Detection score S: mean of hit(k), k = 0..4 s in steps of 0.2 s, window [tau - 0.5 s, tau + k]; an alarm before
  tau - 0.5 s is a false alarm. Alarm thresholds (three calibrations):
    train999 (primary): 99.9% quantile of the scores of 200 training regular videos (config.SOTAD_TRAIN_SCORES);
    train99 (sensitivity): 99% quantile of the same; test999 (sensitivity): 99.9% of the 389 test regular videos.
- Within-video AUC (macro over videos, neg = [0, tau), pos = [tau, tau + 250)); placebo AUC with tau' = tau - 3 s
  on [0, tau) for the 9 videos with tau >= 4 s.
- Null: 1000 circular shifts of each accident video's series (same shifts for all variants and calibrations),
  thresholds fixed; p_single = (1 + #{shift: S_null >= S}) / 1001.
- 95% CIs: bootstrap over videos, 1000 resamples, percentile.

Inputs: config.SOTAD_TEST_SCORES (detector/score_test_videos.py), config.SOTAD_TRAIN_SCORES (train_scores.py),
        config.SOTAD_ROOT/Appendix.txt.
Outputs (--out, default fixed_time_alarm/results/): variants.csv (primary calibration, wide),
        variants_all_calibrations.csv, reference_predictors.csv, summary.json.
Usage (from the repo root):
    tools/run_guarded.sh 2000 work/logs/evaluate.log -- python fixed_time_alarm/evaluate.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import metrics as C  # noqa: E402

MODEL = "grumobilenet"
N_PERM, SEED, CHUNK = 1000, 20260929, 100
SCHEMES = {"train999": ("train", C.FPR_MAIN), "train99": ("train", C.FPR_SENS),
           "test999": ("test", C.FPR_MAIN)}
PRIMARY = "train999"
REF_NAMES = ["(i) first frame", "(ii) median train keyframe", "(iii) clock T* (train)"]


def pad_pair(series):
    X = C.pad_series(series["S"])
    F = C.pad_series(series["F"], fill=-1, dtype=int)
    n = np.array([len(s) for s in series["S"]])
    return X, F, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "results")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    kfd = C.keyframes()
    test = pd.read_csv(config.SOTAD_TEST_SCORES)
    acc = C.series_from_df(test, True)
    nor = C.series_from_df(test, False)
    Xacc, Facc, n = pad_pair(acc)
    kf = acc["kf"].astype(float)
    V, L = Facc.shape
    assert all(kfd[int(v)] == int(k) for v, k in zip(acc["vids"], acc["kf"]))

    # ---- calibration regular videos
    train = pd.read_csv(config.SOTAD_TRAIN_SCORES)
    Xnor_test = pad_pair(nor)[0]
    sub = train[train.is_accident_video == 0].sort_values(["video_id", "frame_index"])
    Xnor_train = C.pad_series([g.score.to_numpy(float) for _, g in sub.groupby("video_id")])
    calib_src = {sc: (f"train_normals({Xnor_train.shape[0]})" if src == "train"
                      else f"test_normals({Xnor_test.shape[0]})") for sc, (src, _) in SCHEMES.items()}

    # ---- vectorized variants vs reference loops
    vec_check = max(float(np.nanmax(np.abs(C.VARIANTS[vn](Xacc)[i, :n[i]] - C.VARIANTS_REF[vn](acc["S"][i]))))
                    for vn in C.VARIANT_NAMES for i in range(V))
    assert vec_check < 1e-9, vec_check

    # ---- thresholds
    thr = {}
    for vn in C.VARIANT_NAMES:
        for sc, (src, fpr) in SCHEMES.items():
            Xc = Xnor_train if src == "train" else Xnor_test
            thr[(sc, vn)] = C.threshold(C.VARIANTS[vn](Xc), fpr)

    neg, pos = C.auc_masks(Facc, kf)
    plc = kf >= 4 * C.FPS
    kfp = kf - 3 * C.FPS
    neg_p, pos_p = C.auc_masks(Facc, kf, lo=np.zeros(V), split=kfp, hi=kf)
    neg_p[~plc] = False; pos_p[~plc] = False

    # ---- observed variants
    long_rows, curves = [], {}
    for vn in C.VARIANT_NAMES:
        Z = C.VARIANTS[vn](Xacc)[None]
        auc_v = C.auc_per_video(Z, neg, pos)[0]
        aucp_v = C.auc_per_video(Z, neg_p, pos_p)[0][plc]
        aucs = {"auc": float(np.nanmean(auc_v))}
        lo, hi = C.boot_ci(auc_v, seed=1); aucs["auc_lo"], aucs["auc_hi"] = float(lo), float(hi)
        aucs["auc_placebo"] = float(np.nanmean(aucp_v))
        lo, hi = C.boot_ci(aucp_v, seed=2)
        aucs["auc_placebo_lo"], aucs["auc_placebo_hi"] = float(lo), float(hi)
        for sc in SCHEMES:
            t = thr[(sc, vn)]
            al = C.first_alarm(Z, Facc, t)[0]
            hit = C.hits_from_alarm(al, kf)
            summ, _ = C.summarize_hits(hit, al, kf)
            comp = C.per_video_components(hit, al, kf)
            lo, hi = C.boot_ci(comp, seed=3)
            r = {"scheme": sc, "model": MODEL, "variant": vn, "eligible": sc != "test999",
                 "calib": calib_src[sc], "thr": t, **summ}
            for j, nm in enumerate(["S", "hit1", "hit2", "hit4", "early"]):
                r[f"{nm}_lo"], r[f"{nm}_hi"] = float(lo[j]), float(hi[j])
            Zn = C.VARIANTS[vn](Xnor_test)
            valid = ~np.isnan(Zn)
            r["test_frame_fpr"] = float((Zn[valid] >= t).mean())
            r["test_video_fa"] = float((np.nan_to_num(Zn, nan=-np.inf) >= t).any(-1).mean())
            r.update(aucs)
            long_rows.append(r)
            if sc == PRIMARY:
                curves[f"{MODEL}/{vn}"] = hit.mean(0).tolist()
    TL = pd.DataFrame(long_rows)

    # ---- reference predictors (fixed-time alarms)
    tra = C.series_from_df(train, True)
    Ftr = C.pad_series(tra["F"], fill=-1, dtype=int); kftr = tra["kf"].astype(float)
    assert all(kfd[int(v)] == int(k) for v, k in zip(tra["vids"], tra["kf"]))
    grid = np.arange(0, 501, 5)
    S_train = np.array([C.hits_from_alarm(C.alarm_at(Ftr, T_), kftr).mean() for T_ in grid])
    T_star = float(grid[int(np.argmax(S_train))])
    T_med = float(np.median(kftr))
    ref_rows = []
    for nm, T_ in zip(REF_NAMES, (0.0, T_med, T_star)):
        al = C.alarm_at(Facc, T_)
        hit = C.hits_from_alarm(al, kf)
        summ, _ = C.summarize_hits(hit, al, kf)
        comp = C.per_video_components(hit, al, kf)
        lo, hi = C.boot_ci(comp, seed=3)
        step = np.where(Facc >= 0, (Facc >= T_).astype(float), np.nan)[None]
        auc_v = C.auc_per_video(step, neg, pos)[0]
        alo, ahi = C.boot_ci(auc_v, seed=1)
        aucp = C.auc_per_video(step, neg_p, pos_p)[0][plc]
        plo, phi = C.boot_ci(aucp, seed=2)
        r = {"ref": nm, "T_frames": T_, "T_s": T_ / C.FPS, **summ}
        for j, k2 in enumerate(["S", "hit1", "hit2", "hit4", "early"]):
            r[f"{k2}_lo"], r[f"{k2}_hi"] = float(lo[j]), float(hi[j])
        r.update({"auc": float(np.nanmean(auc_v)), "auc_lo": float(alo), "auc_hi": float(ahi),
                  "auc_placebo": float(np.nanmean(aucp)), "auc_placebo_lo": float(plo),
                  "auc_placebo_hi": float(phi),
                  "S_train": float(C.hits_from_alarm(C.alarm_at(Ftr, T_), kftr).mean())})
        curves[nm] = hit.mean(0).tolist()
        ref_rows.append(r)
    R = pd.DataFrame(ref_rows)
    i_strong = int(R.S.idxmax())
    S_null = float(R.S.max()); strongest = R.ref[i_strong]
    clock = np.where(Facc >= 0, Facc.astype(float), np.nan)[None]
    auc_clock = float(np.nanmean(C.auc_per_video(clock, neg, pos)[0]))

    # ---- circular-shift null (same shifts for every variant and calibration)
    rs = np.random.RandomState(SEED)
    shifts = np.floor(rs.uniform(size=(N_PERM, V)) * n[None, :]).astype(int)
    Snull = {sc: np.zeros((N_PERM, len(C.VARIANT_NAMES))) for sc in SCHEMES}
    for s in range(0, N_PERM, CHUNK):
        Rr = C.rotate(Xacc, n, shifts[s:s + CHUNK])
        for vi, vn in enumerate(C.VARIANT_NAMES):
            Z = C.VARIANTS[vn](Rr)
            for sc in SCHEMES:
                al = C.first_alarm(Z, Facc, thr[(sc, vn)])
                Snull[sc][s:s + CHUNK, vi] = C.hits_from_alarm(al, kf).mean((-1, -2))
    print(f"null: {N_PERM} shifts in {time.perf_counter() - t0:.0f}s", flush=True)
    parts = []
    for sc in SCHEMES:
        sub = TL[TL.scheme == sc].set_index("variant").loc[C.VARIANT_NAMES].reset_index()
        Sobs = sub.S.to_numpy()
        sub["p_single"] = (1 + (Snull[sc] >= Sobs[None, :]).sum(0)) / (N_PERM + 1)
        sub["null_mean"] = Snull[sc].mean(0)
        sub["null_q95"] = np.quantile(Snull[sc], 0.95, axis=0)
        sub["excess"] = sub.S - S_null
        parts.append(sub)
    TL = pd.concat(parts, ignore_index=True)

    TL = TL.sort_values(["scheme", "S"], ascending=[True, False], kind="mergesort").reset_index(drop=True)
    cols = ["model", "variant", "scheme"] + [c for c in TL.columns if c not in ("model", "variant", "scheme")]
    TL = TL[cols]
    TL.to_csv(a.out / "variants_all_calibrations.csv", index=False)
    TP = TL[TL.scheme == PRIMARY].drop(columns="scheme").reset_index(drop=True)
    for sc in SCHEMES:
        if sc == PRIMARY:
            continue
        o = TL[TL.scheme == sc].set_index("variant")
        for col in ("thr", "S", "early", "test_frame_fpr"):
            TP[f"{col}_{sc}"] = [o.loc[v, col] for v in TP.variant]
    TP.to_csv(a.out / "variants.csv", index=False)
    R.to_csv(a.out / "reference_predictors.csv", index=False)
    summary = {
        "n_accident_videos": int(V), "n_placebo_videos": int(plc.sum()),
        "keyframe_train_median_frames": T_med, "T_star_frames": T_star,
        "S_train_curve": {int(g): float(s) for g, s in zip(grid, S_train)},
        "strongest_null": {"ref": strongest, "S": S_null},
        "refs": ref_rows, "auc_clock_as_score": auc_clock,
        "vectorized_variant_check_maxdiff": vec_check,
        "schemes": {k: {"src": v[0], "fpr": v[1]} for k, v in SCHEMES.items()}, "primary": PRIMARY,
        "thresholds": {f"{sc}/{MODEL}/{vn}": thr[(sc, vn)] for (sc, vn) in thr},
        "calibration_source": {f"{sc}/{MODEL}": v for sc, v in calib_src.items()},
        "curves_hit_k_primary": curves, "ks_s": C.KS_S.tolist(),
    }
    json.dump(summary, open(a.out / "summary.json", "w"), indent=2)

    pd.set_option("display.width", 250)
    print(R[["ref", "T_frames", "S", "S_lo", "S_hi", "hit1", "hit2", "hit4", "early", "auc",
             "auc_placebo", "S_train"]].round(3).to_string(index=False))
    print(f"\nstrongest null: {strongest} S={S_null:.3f}  | AUC of the clock as a score = {auc_clock:.3f}")
    show = ["model", "variant", "eligible", "thr", "S", "S_lo", "S_hi", "hit1", "hit2", "hit4",
            "early", "auc", "auc_placebo", "p_single", "excess", "test_frame_fpr"]
    for sc in SCHEMES:
        print(f"\n=== {sc} ===")
        print(TL[TL.scheme == sc][show].round(4).to_string(index=False))
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
