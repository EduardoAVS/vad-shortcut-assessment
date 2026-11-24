"""SO-TAD clip lengths against what the SO-TAD paper declares (Sec. 5.1).

  A. measured mean lengths, total hours and counts vs the values declared by the SO-TAD authors;
  B. the declared cut rule "clips with durations ranging from 6 to 20 s": accident clips below 6 s (83/282 = 29.4%,
     minimum 2.72 s) and normal clips at the 20 s ceiling (500-501 frames: 1877/1904 = 98.6%);
  C. per-split length ranges of the two classes (test: accident 79-375 frames vs normal 388-514, disjoint);
  D. clip length inside the active areas that hold both classes (raw AUC = P(accident longer); 1 - raw is the
     symmetrized AUC, see sotad_area_and_length_checks.py);
  E. how many active areas hold a single class;
  F. median keyframe time of the 90 test accidents (Appendix.txt keyframe / 25 fps: 2.10 s).

Inputs : results/active_area.csv (active_area.py); config.SOTAD_ROOT/Appendix.txt (for F)
Output : results/sotad_clip_lengths.json

Usage (from the repository root):
    python data_assessment/sotad_clip_lengths.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sotad as S  # noqa: E402

RESULTS = HERE / "results"
FPS = S.FPS


def auc(sc, lb):
    """Raw AUC: P(accident clip longer than normal clip), ties count 1/2."""
    p, q = sc[lb == 1], sc[lb == 0]
    if p.size == 0 or q.size == 0:
        return None
    gt = (p[:, None] > q[None, :]).sum()
    eq = (p[:, None] == q[None, :]).sum()
    return float((gt + 0.5 * eq) / (p.size * q.size))


def main():
    with open(RESULTS / "active_area.csv") as fh:
        rows = list(csv.DictReader(fh))
    n = np.array([int(r["n_frames"]) for r in rows])
    acc = np.array([int(r["accident"]) for r in rows])
    spl = np.array([r["split"] for r in rows])
    geo = [(int(r["y0"]), int(r["y1"]), int(r["x0"]), int(r["x1"])) for r in rows]
    ids = np.array([int(r["video"]) for r in rows])

    print("=" * 78)
    print("A. MEASURED vs DECLARED (SO-TAD paper: accident 197 / normal 500 frames)")
    print("=" * 78)
    for lab, name, declared in ((1, "accident", 197), (0, "non-accident", 500)):
        v = n[acc == lab]
        print(f"  {name:13s} measured mean={v.mean():7.2f}  declared={declared}  delta={v.mean()-declared:+6.2f}"
              f"  (median={np.median(v):.0f}, n={v.size})")
    total_h = n.sum() / FPS / 3600
    print(f"  total duration  measured={total_h:.2f} h        declared=11.2 h")
    print(f"  overall mean    measured={n.mean():.1f}        declared=461")
    print(f"  counts          measured={(acc == 1).sum()} accidents / {(acc == 0).sum()} normal   declared=282 / 1904")

    print()
    print("=" * 78)
    print("B. DECLARED CUT RULE: 'clips with durations ranging from 6 to 20 s'")
    print("=" * 78)
    lo = 6 * FPS
    for lab, name in ((1, "accident"), (0, "normal")):
        v = n[acc == lab]
        below = (v < lo).sum()
        print(f"  {name:9s} min={v.min():4d} ({v.min()/FPS:5.2f}s)  max={v.max():4d} ({v.max()/FPS:5.2f}s)"
              f"   below 6 s: {below:4d}/{v.size} ({100*below/v.size:5.1f}%)")
    count_norm = Counter(n[acc == 0].tolist())
    count_acc = Counter(n[acc == 1].tolist())
    print(f"  normal clips at 500-501 frames (20 s ceiling): {count_norm[500]+count_norm[501]} of {(acc == 0).sum()} "
          f"({100*(count_norm[500]+count_norm[501])/(acc == 0).sum():.1f}%)")
    print(f"  accident clips at 500-501 frames:             {count_acc[500]+count_acc[501]}")

    print()
    print("=" * 78)
    print("C. PER-SPLIT LENGTH RANGES (frames)")
    print("=" * 78)
    ranges = {}
    for s in ("test", "train"):
        m = spl == s
        a, nn = n[m & (acc == 1)], n[m & (acc == 0)]
        gap = nn.min() - a.max()
        print(f"  [{s:5s}] accident [{a.min():3d},{a.max():3d}]  normal [{nn.min():3d},{nn.max():3d}]"
              f"  gap={gap:+5d}  disjoint={'YES' if gap > 0 else 'no'}")
        ranges[s] = {"accident": [int(a.min()), int(a.max())], "normal": [int(nn.min()), int(nn.max())],
                     "gap": int(gap)}

    print()
    print("=" * 78)
    print("D. ARE THE TWO LEAKS THE SAME OR INDEPENDENT?")
    print("=" * 78)
    by_box = defaultdict(list)
    for i, g in enumerate(geo):
        by_box[g].append(i)
    mixed = {g: i for g, i in by_box.items() if len(set(acc[i].tolist())) == 2}
    print(f"  distinct active areas:          {len(by_box)}")
    print(f"  areas with BOTH classes:        {len(mixed)}  (the only ones where length is testable)")
    aucs = []
    tp = tq = 0
    for g, i in mixed.items():
        i = np.array(i)
        a = auc(n[i].astype(float), acc[i])
        if a is not None:
            aucs.append(a)
            tp += (acc[i] == 1).sum()
            tq += (acc[i] == 0).sum()
    if aucs:
        print(f"  raw length AUC INSIDE mixed areas: mean={np.mean(aucs):.4f}"
              f" (min={min(aucs):.3f}, max={max(aucs):.3f}) in {len(aucs)} areas, {tp} acc vs {tq} norm")
    print(f"  raw length AUC on the whole dataset:  {auc(n.astype(float), acc):.4f}")

    print()
    print("=" * 78)
    print("E. IS THE ACTIVE AREA A SOURCE SIGNATURE?")
    print("=" * 78)
    pure = [g for g, i in by_box.items() if len(set(acc[i].tolist())) == 1]
    covered = sum(len(by_box[g]) for g in pure)
    print(f"  single-class areas: {len(pure)}/{len(by_box)} = {100*len(pure)/len(by_box):.1f}%")
    print(f"  clips in single-class areas: {covered}/{len(rows)} = {100*covered/len(rows):.1f}%")
    heights = Counter(g[1] - g[0] for g in geo)
    print(f"  distinct active-area heights: {len(heights)}   most common: {heights.most_common(5)}")

    print()
    print("=" * 78)
    print("F. KEYFRAME TIME OF THE TEST ACCIDENTS")
    print("=" * 78)
    keyframes = S.load_keyframes()
    test_acc = ids[(spl == "test") & (acc == 1)]
    kf = np.array([keyframes[v] for v in test_acc if v in keyframes])
    kf_median_s = float(np.median(kf) / FPS)
    print(f"  {kf.size} test accidents with a keyframe; median keyframe {np.median(kf):.1f} frames = {kf_median_s:.2f} s")

    with open(RESULTS / "sotad_clip_lengths.json", "w") as fh:
        json.dump({"accident_mean_frames": float(n[acc == 1].mean()),
                   "normal_mean_frames": float(n[acc == 0].mean()),
                   "overall_mean_frames": float(n.mean()),
                   "total_hours": float(total_h),
                   "accidents_below_6s": int((n[acc == 1] < lo).sum()),
                   "normals_at_500_501_frames": int(count_norm[500] + count_norm[501]),
                   "raw_length_auc_mixed_boxes": float(np.mean(aucs)) if aucs else None,
                   "n_mixed_boxes": len(mixed),
                   "n_boxes": len(by_box),
                   "raw_length_auc_all": auc(n.astype(float), acc),
                   "length_ranges_frames": ranges,
                   "n_test_accidents_with_keyframe": int(kf.size),
                   "median_test_keyframe_s": kf_median_s},
                  fh, indent=2)


if __name__ == "__main__":
    main()
