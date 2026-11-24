"""GRUMobileNet scores on the SO-TAD training split, used to calibrate the fixed-time alarm and the thresholds.

The fixed-time-alarm analysis (evaluate.py) needs two things from the training split:
- the score grid and keyframes of the training accident videos, from which the fixed alarm times are chosen
  (median training keyframe = 3.18 s; clock time T* that maximizes S on the training accidents);
- GRUMobileNet scores on training regular videos, whose 99.9% / 99% quantiles are the alarm thresholds.

Steps
1. Pooled backbone features. The backbone is frozen, so each cached frame (config.SOTAD_FRAME_CACHE, built by
   detector/extract_frames.py) is passed once through the MobileNetV3-Small backbone of config.DETECTOR_CKPT and
   the 576-d pooled features are stored as config.SOTAD_FEATURE_CACHE/<split>_<video_id>.npz (arrays `f` [N, 576]
   and `green` [N], the solid-green failed-decode flag). Videos: all accident videos of both splits, all 389
   test regular videos and 200 training regular videos sampled with random.Random(7). Existing files are reused.
2. Scores. The GRU head is run on windows of 16 cached rows (sequence stride 5 frames, padded with row 0), the
   same windows as detector/score_test_videos.py; windows containing a green frame are skipped.
   Training accident videos: every row with a label (frame < keyframe + 250). Training regular videos (the 200
   sampled): every row. Written to config.SOTAD_TRAIN_SCORES with the columns of the test scores CSV
   (video_id, frame_index, label, score, is_accident_video, keyframe, distance_to_keyframe).
3. Check (--check): the head over cached test features reproduces config.SOTAD_TEST_SCORES (all 90 accident
   videos + 60 regular videos sampled with random.Random(2026)); with --compare-train <csv>, the head over cached
   training features also reproduces an earlier decode-based training scores file (the one of the original
   working copy, 192 accident + 300 regular videos). Written to results/train_scores_check.json.

Inputs: config.SOTAD_ROOT (video list, Appendix.txt), config.SOTAD_FRAME_CACHE, config.DETECTOR_CKPT
        (and config.SOTAD_TEST_SCORES for --check).
Outputs: config.SOTAD_FEATURE_CACHE/*.npz, config.SOTAD_TRAIN_SCORES, results/train_scores_check.json (--check).
Usage (from the repo root):
    tools/run_guarded.sh 2000 work/logs/train_scores.log -- python fixed_time_alarm/train_scores.py --check
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "detector"))
import config  # noqa: E402
from model import AccidentMobileNetV3SmallGRU  # noqa: E402

torch.set_num_threads(2)

SEQ, STRIDE, POS_WINDOW = 16, 5, 250
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
N_TRAIN_NORMAL, TRAIN_NORMAL_SEED = 200, 7
N_TEST_NORMAL = 389
CHECK_NORMALS, CHECK_SEED, TOL = 60, 2026, 1e-3


def keyframes():
    kf = {}
    for line in open(config.SOTAD_ROOT / "Appendix.txt", encoding="utf-8"):
        line = line.strip()
        if line:
            n, k = line.split("\t")
            kf[int(n.split(".")[0])] = int(k)
    return kf


def label_of(v, frame, kf):
    """0 = before the keyframe (or regular video), 1 = [keyframe, keyframe + 250), None = ignored."""
    if v >= 400:
        return 0
    if v not in kf:
        return None
    s = kf[v]
    return 0 if frame < s else (1 if frame < s + POS_WINDOW else None)


def is_green(a):
    """Solid-green failed decode."""
    f = a.reshape(-1, 3)
    return bool(f[:, 0].mean() < 5 and f[:, 1].mean() > 20 and f[:, 2].mean() < 5)


def load_detector():
    ck = torch.load(config.DETECTOR_CKPT, map_location="cpu", weights_only=False)
    c = ck["config"]
    m = AccidentMobileNetV3SmallGRU(2, c["image_size"], c["freeze_backbone"], False,
                                    c["gru_hidden_size"], c["gru_num_layers"], c["gru_dropout"])
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def video_sets(kf):
    """The videos whose features are cached (same selection as the original feature extraction)."""
    tr_ids = sorted(int(p.stem) for p in (config.SOTAD_ROOT / "train").glob("*.mp4"))
    te_ids = sorted(int(p.stem) for p in (config.SOTAD_ROOT / "test").glob("*.mp4"))
    tr_acc = [v for v in tr_ids if v < 400 and v in kf]
    te_acc = [v for v in te_ids if v < 400 and v in kf]
    tr_nor = sorted(random.Random(TRAIN_NORMAL_SEED).sample([v for v in tr_ids if v >= 400], N_TRAIN_NORMAL))
    te_nor = sorted(v for v in te_ids if v >= 400)[:N_TEST_NORMAL]
    return {"train": (tr_acc, tr_nor), "test": (te_acc, te_nor)}


@torch.no_grad()
def features(m, split, vid):
    """[N, 576] pooled backbone features + green mask of one cached video; cached as npz."""
    p = config.SOTAD_FEATURE_CACHE / f"{split}_{vid}.npz"
    if p.exists():
        z = np.load(p)
        return z["f"], z["green"]
    src = config.SOTAD_FRAME_CACHE / split / f"{vid}.npy"
    if not src.exists():
        return None, None
    arr = np.load(src)
    green = np.array([is_green(arr[i]) for i in range(arr.shape[0])])
    x = ((arr.astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)
    t = torch.from_numpy(np.ascontiguousarray(x))
    f = torch.cat([m._extract_features_4d(t[i:i + 64]) for i in range(0, len(t), 64)], 0).numpy()
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, f=f, green=green)
    return f, green


@torch.no_grad()
def score_rows(m, f, green, rows):
    """GRU scores at frames 5*i for i in rows (windows with a green frame are skipped). Returns (rows, scores)."""
    wins, ok = [], []
    for i in rows:
        r = [max(0, i - o) for o in reversed(range(SEQ))]
        if any(green[x] for x in r):
            continue
        wins.append(r); ok.append(i)
    if not ok:
        return [], np.array([])
    X = torch.from_numpy(np.stack([f[r] for r in wins]).astype(np.float32))
    out, _ = m.gru(X)
    p = torch.softmax(m.classifier(out[:, -1, :]), dim=1)[:, 1].numpy()
    return ok, p


def compare(m, split, df, vids):
    """Max |head over cached features - recorded score| on the rows of df for the given videos."""
    diffs, miss = [], 0
    for v in vids:
        g = df[df.video_id == v]
        f, gr = features(m, split, v)
        rows = (g.frame_index.to_numpy() // STRIDE).tolist()
        ok, p = score_rows(m, f, gr, [i for i in rows if i < f.shape[0]])
        got = dict(zip(ok, p))
        for fi, s in zip(g.frame_index.to_numpy(), g.score.to_numpy()):
            if fi // STRIDE in got:
                diffs.append(abs(got[fi // STRIDE] - s))
            else:
                miss += 1
    d = np.array(diffs)
    return {"n_videos": len(vids), "n_compared": int(len(d)), "n_missing": int(miss),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="check the head over cached features against the "
                    "test scores CSV and write results/train_scores_check.json")
    ap.add_argument("--compare-train", type=Path, default=None,
                    help="decode-based training scores CSV to compare against (with --check)")
    ap.add_argument("--out", type=Path, default=config.SOTAD_TRAIN_SCORES, help="output CSV")
    a = ap.parse_args()
    t0 = time.perf_counter()
    kf = keyframes()
    m = load_detector()
    sets = video_sets(kf)

    print("pooled backbone features ->", config.SOTAD_FEATURE_CACHE, flush=True)
    for split, (acc, nor) in sets.items():
        for n, v in enumerate(acc + nor, 1):
            features(m, split, v)
            if n % 100 == 0:
                print(f"  {split} {n}/{len(acc) + len(nor)}  {time.perf_counter() - t0:.0f}s", flush=True)

    tr_acc, tr_nor = sets["train"]
    parts = []
    for v in tr_acc + tr_nor:
        f, gr = features(m, "train", v)
        if f is None:
            continue
        rows = [i for i in range(f.shape[0]) if label_of(v, i * STRIDE, kf) is not None]
        ok, p = score_rows(m, f, gr, rows)
        if not ok:
            continue
        fr = np.array(ok, int) * STRIDE
        k = kf.get(v, -1)
        parts.append(pd.DataFrame({
            "video_id": v, "frame_index": fr, "label": [label_of(v, x, kf) for x in fr],
            "score": p.astype(float), "is_accident_video": int(v < 400 and v in kf), "keyframe": k,
            "distance_to_keyframe": (fr - k) if k >= 0 else None}))
    out = pd.concat(parts, ignore_index=True).sort_values(["video_id", "frame_index"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"{out.video_id.nunique()} videos ({out[out.is_accident_video == 1].video_id.nunique()} accident), "
          f"{len(out)} rows -> {a.out}", flush=True)

    if not a.check:
        return
    rep = {}
    test = pd.read_csv(config.SOTAD_TEST_SCORES)
    acc_v = sorted(test[test.is_accident_video == 1].video_id.unique())
    nor_v = sorted(random.Random(CHECK_SEED).sample(
        sorted(test[test.is_accident_video == 0].video_id.unique()), CHECK_NORMALS))
    r = compare(m, "test", test, [int(v) for v in acc_v + nor_v])
    r["pass"] = bool(r["max_abs_diff"] < TOL and r["n_missing"] == 0)
    rep["test_reproduction"] = r
    print("test reproduction:", r, flush=True)
    if a.compare_train is not None:
        ref = pd.read_csv(a.compare_train)
        common = sorted(set(ref.video_id.unique()) & set(tr_acc + tr_nor))
        rep["train_vs_decoded_train_scores"] = compare(m, "train", ref, common)
        print("train vs decoded train scores:", rep["train_vs_decoded_train_scores"], flush=True)
        acc_ref = ref[ref.is_accident_video == 1]
        acc_new = out[out.is_accident_video == 1]
        same = (set(map(tuple, acc_ref[["video_id", "frame_index"]].to_numpy()))
                == set(map(tuple, acc_new[["video_id", "frame_index"]].to_numpy())))
        rep["train_accident_grid_identical"] = bool(same)
        print("train accident grid identical:", same, flush=True)
    nor = out[out.is_accident_video == 0]
    rep["train_normals"] = {"n_videos": int(nor.video_id.nunique()), "n_frames": int(len(nor))}
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(rep, open(HERE / "results" / "train_scores_check.json", "w"), indent=2)
    print(f"done in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
