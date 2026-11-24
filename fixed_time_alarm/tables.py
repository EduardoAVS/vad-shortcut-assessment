"""Markdown tables of the fixed-time-alarm analysis, with the paired CI of the excess over the strongest null.

Reads the outputs of evaluate.py, recomputes the per-video S of each row to get the paired bootstrap CI of
S - S_strongest_null (1000 resamples of videos), and checks reference (ii) with an independent pandas computation.
The rows are the three reference predictors and GRUMobileNet raw / warmup5 (the paper's 0.822 vs 0.711 are
the "AUC macro" column of rows (ii) and grumobilenet).

Inputs: results/{summary.json, variants_all_calibrations.csv, reference_predictors.csv} (evaluate.py),
        config.SOTAD_TEST_SCORES.
Outputs: results/tables.md, results/table_main_train999.csv.
Usage (from the repo root):
    python fixed_time_alarm/tables.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import config  # noqa: E402
import metrics as C  # noqa: E402

MODEL = "grumobilenet"
SCHEME_NAMES = {"train999": "PRIMARY: threshold at 99.9% of the TRAINING regular videos",
                "test999": "SENSITIVITY: threshold at 99.9% of the TEST regular videos",
                "train99": "pre-registered SENSITIVITY: threshold at 99% of the training regular videos"}


def fmt(x, lo=None, hi=None, nd=3):
    if lo is None:
        return f"{x:.{nd}f}"
    return f"{x:.{nd}f} [{lo:.{nd}f}; {hi:.{nd}f}]"


def to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=HERE / "results", help="folder with the evaluate.py outputs")
    a = ap.parse_args()
    summ = json.load(open(a.out / "summary.json"))
    TL = pd.read_csv(a.out / "variants_all_calibrations.csv")
    R = pd.read_csv(a.out / "reference_predictors.csv")
    S_null = summ["strongest_null"]["S"]; strongest = summ["strongest_null"]["ref"]

    test = pd.read_csv(config.SOTAD_TEST_SCORES)
    acc = C.series_from_df(test, True)
    F = C.pad_series(acc["F"], fill=-1, dtype=int); kf = acc["kf"].astype(float)
    X = C.pad_series(acc["S"])

    def per_video_S(vn, thr):
        Z = C.VARIANTS[vn](X)[None]
        return C.hits_from_alarm(C.first_alarm(Z, F, thr)[0], kf).mean(-1)

    Tref = {r.ref: r.T_frames for r in R.itertuples()}
    s_null_v = C.hits_from_alarm(C.alarm_at(F, Tref[strongest]), kf).mean(-1)
    assert abs(s_null_v.mean() - S_null) < 1e-12

    # independent check of reference (ii): alarm at the first grid frame >= the median training keyframe
    g = test[test.is_accident_video == 1]
    Tm = summ["keyframe_train_median_frames"]; chk = []
    for _, d in g.groupby("video_id"):
        fr = d.frame_index.to_numpy(); k = d.keyframe.iloc[0]
        al = fr[fr >= Tm].min() if (fr >= Tm).any() else np.inf
        chk.append(np.mean([(al >= k - 12.5) and (al <= k + kk) for kk in np.arange(0, 101, 5)]))
    assert abs(np.mean(chk) - R.loc[R.ref.str.startswith("(ii)"), "S"].iloc[0]) < 1e-12

    def paired_ci(s_v):
        d = s_v - s_null_v
        lo, hi = C.boot_ci(d, seed=5)
        return float(d.mean()), float(lo), float(hi)

    def table(sc):
        sub = TL[(TL.scheme == sc) & (TL.model == MODEL) & TL.variant.isin(["raw", "warmup5"])]
        rows = []
        for r in R.itertuples():
            tag = " **(strongest null)**" if r.ref == strongest else ""
            rows.append({"row": f"{r.ref}{tag} (T = {r.T_s:.2f} s)",
                         "S [95% CI]": fmt(r.S, r.S_lo, r.S_hi), "hit@1s": fmt(r.hit1), "hit@2s": fmt(r.hit2),
                         "hit@4s": fmt(r.hit4), "% alarm < tau-0.5s": f"{100 * r.early:.1f}",
                         "AUC macro [95% CI]": fmt(r.auc, r.auc_lo, r.auc_hi),
                         "AUC placebo (n=9)": fmt(r.auc_placebo),
                         "excess S [paired 95% CI]": "-" if r.ref == strongest else
                         "{:.3f} [{:.3f}; {:.3f}]".format(*paired_ci(
                             C.hits_from_alarm(C.alarm_at(F, r.T_frames), kf).mean(-1))),
                         "test frame FPR": "-"})
        for r in sub.sort_values("S", ascending=False, kind="mergesort").itertuples():
            ex, lo, hi = paired_ci(per_video_S(r.variant, r.thr))
            el = "" if r.eligible else " (ineligible)"
            rows.append({"row": f"{r.model}/{r.variant}{el}",
                         "S [95% CI]": fmt(r.S, r.S_lo, r.S_hi), "hit@1s": fmt(r.hit1), "hit@2s": fmt(r.hit2),
                         "hit@4s": fmt(r.hit4), "% alarm < tau-0.5s": f"{100 * r.early:.1f}",
                         "AUC macro [95% CI]": fmt(r.auc, r.auc_lo, r.auc_hi),
                         "AUC placebo (n=9)": fmt(r.auc_placebo, r.auc_placebo_lo, r.auc_placebo_hi),
                         "excess S [paired 95% CI]": f"{ex:.3f} [{lo:.3f}; {hi:.3f}]",
                         "test frame FPR": f"{100 * r.test_frame_fpr:.2f}%"})
        return pd.DataFrame(rows)

    md = []
    for sc in ("train999", "test999", "train99"):
        t = table(sc)
        if sc == "train999":
            t.to_csv(a.out / f"table_main_{sc}.csv", index=False)
        md.append(f"### Fixed-time alarm vs GRUMobileNet ({SCHEME_NAMES[sc]})\n\n" + to_md(t) + "\n")
    fa = TL[TL.scheme == "train999"][["model", "variant", "thr", "test_frame_fpr", "test_video_fa", "calib"]].copy()
    fa["test_frame_fpr"] = (100 * fa.test_frame_fpr).round(2)
    fa["test_video_fa"] = (100 * fa.test_video_fa).round(1)
    md.append("### Realized false alarms on the test regular videos with the primary threshold "
              "(nominal 0.1% per frame)\n\n"
              + to_md(fa.sort_values("test_frame_fpr", ascending=False, kind="mergesort")) + "\n")
    (a.out / "tables.md").write_text("\n".join(md))
    print("\n".join(md[:1]))
    print("ok ->", a.out / "tables.md")


if __name__ == "__main__":
    main()
