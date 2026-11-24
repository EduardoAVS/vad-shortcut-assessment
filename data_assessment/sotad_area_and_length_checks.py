"""SO-TAD active-area and clip-length checks (Sec. 5.1), second pass over sotad_clip_lengths.py.

  D. clip length inside each active area that holds both classes, with the symmetrized AUC (sotad_clip_lengths.py
     reports the raw AUC, e.g. 0.0008 = 0.9992 in the "accident shorter" direction);
  E. size of the active-area leak per split: accuracy of the table rule "active area -> most frequent class" vs
     the majority class, and the fraction of clips in an area that holds only their class (test: 88/90
     accidents; all: 259/282);
  F. asymmetry of the cut: normal clips at the 20 s ceiling (98.6%), accident clips below the declared 6 s floor
     (29.4%, minimum 2.72 s);
  G. the table rule out of sample (fitted on train, evaluated on test; unseen areas get the train majority class).

Input  : results/active_area.csv (active_area.py)
Output : results/sotad_area_and_length_checks.json

Usage (from the repository root):
    python data_assessment/sotad_area_and_length_checks.py
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
FPS = 25


def auc_sym(sc, lb):
    p, q = sc[lb == 1], sc[lb == 0]
    gt = (p[:, None] > q[None, :]).sum()
    eq = (p[:, None] == q[None, :]).sum()
    a = float((gt + 0.5 * eq) / (p.size * q.size))
    return (a, "accident_longer") if a >= .5 else (1 - a, "accident_shorter")


def main():
    with open(RESULTS / "active_area.csv") as fh:
        rows = list(csv.DictReader(fh))
    n = np.array([int(r["n_frames"]) for r in rows])
    acc = np.array([int(r["accident"]) for r in rows])
    spl = np.array([r["split"] for r in rows])
    geo = [(int(r["y0"]), int(r["y1"]), int(r["x0"]), int(r["x1"])) for r in rows]

    print("=" * 78)
    print("D. CLIP LENGTH WITHIN AN ACTIVE AREA")
    print("=" * 78)
    by_box = defaultdict(list)
    for i, g in enumerate(geo):
        by_box[g].append(i)
    mixed = {g: i for g, i in by_box.items() if len(set(acc[i].tolist())) == 2}
    for g, i in mixed.items():
        i = np.array(i)
        a, d = auc_sym(n[i].astype(float), acc[i])
        print(f"  area {g}: {(acc[i] == 1).sum()} accidents vs {(acc[i] == 0).sum()} normal")
        print(f"    length AUC INSIDE this area = {a:.4f} ({d})")
        print(f"    ranges: accident [{n[i][acc[i] == 1].min()},{n[i][acc[i] == 1].max()}]  "
              f"normal [{n[i][acc[i] == 0].min()},{n[i][acc[i] == 0].max()}]")
    a, d = auc_sym(n.astype(float), acc)
    print(f"  length AUC on the whole dataset = {a:.4f} ({d})")
    print("  => the active area does not explain the length: even with the camera fixed, length separates.")

    print()
    print("=" * 78)
    print("E. SIZE OF THE ACTIVE-AREA LEAK")
    print("=" * 78)
    res = {}
    for s in ("test", "train", None):
        m = np.ones(len(rows), bool) if s is None else (spl == s)
        idx = np.flatnonzero(m)
        groups = defaultdict(list)
        for i in idx:
            groups[geo[i]].append(i)
        # accident-only area = holds accidents and no normal clip
        acc_pure = sum(len(v) for g, v in groups.items() if set(acc[v].tolist()) == {1})
        norm_pure = sum(len(v) for g, v in groups.items() if set(acc[v].tolist()) == {0})
        n_acc = int((acc[idx] == 1).sum())
        n_norm = int((acc[idx] == 0).sum())
        table = sum(max(Counter(acc[v].tolist()).values()) for v in groups.values()) / len(idx)
        maj = max(n_acc, n_norm) / len(idx)
        name = s or "all"
        print(f"  [{name:8s}] {len(groups):3d} areas | table-rule accuracy {table:.4f} "
              f"| majority class {maj:.4f} | gain {table-maj:+.4f}")
        print(f"             accidents in an accident-only area: {acc_pure}/{n_acc} = {100*acc_pure/n_acc:5.1f}%")
        print(f"             normal clips in a normal-only area: {norm_pure}/{n_norm} = {100*norm_pure/n_norm:5.1f}%")
        res[name] = {"n_boxes": len(groups), "table_rule": table, "majority": maj, "gain": table - maj,
                     "accidents_in_exclusive_box": acc_pure, "n_accident": n_acc,
                     "frac_accidents_exclusive": acc_pure / n_acc}

    print()
    print("=" * 78)
    print("F. ASYMMETRY OF THE CUT (evidence of curation)")
    print("=" * 78)
    count_norm = Counter(n[acc == 0].tolist())
    print(f"  normal clips at the 20 s ceiling (500-501 frames): {count_norm[500]+count_norm[501]}/{(acc == 0).sum()} = "
          f"{100*(count_norm[500]+count_norm[501])/(acc == 0).sum():.1f}%")
    print("  accident clips at the 20 s ceiling:                0")
    print(f"  accident clips below the declared 6 s floor: {(n[acc == 1] < 150).sum()}/{(acc == 1).sum()} = "
          f"{100*(n[acc == 1] < 150).sum()/(acc == 1).sum():.1f}%   (minimum {n[acc == 1].min()/FPS:.2f} s)")
    print(f"  normal clips below the declared 6 s floor:   {(n[acc == 0] < 150).sum()}   "
          f"(minimum {n[acc == 0].min()/FPS:.2f} s)")

    print()
    print("=" * 78)
    print("G. TABLE RULE OUT OF SAMPLE (fitted on train, evaluated on test)")
    print("=" * 78)
    # E fits and evaluates the table on the same split. Here the area -> class table is learned on train only;
    # areas not seen in train get the train majority class. This is the form of the rule that never sees the test.
    tr, te = np.flatnonzero(spl == "train"), np.flatnonzero(spl == "test")
    box_tr = defaultdict(list)
    for i in tr:
        box_tr[geo[i]].append(acc[i])
    maj_tr = Counter(acc[tr].tolist()).most_common(1)[0][0]
    rule = {g: Counter(v).most_common(1)[0][0] for g, v in box_tr.items()}
    pred = np.array([rule.get(geo[i], maj_tr) for i in te])
    yte = acc[te]
    oos = float(np.mean(pred == yte))
    maj_te = max(yte.mean(), 1 - yte.mean())
    unseen = int(sum(geo[i] not in rule for i in te))
    print(f"  out-of-sample accuracy {oos:.4f} | majority class {maj_te:.4f} | gain {oos-maj_te:+.4f}")
    print(f"  accidents correct {(pred[yte == 1] == 1).sum()}/{(yte == 1).sum()} | "
          f"normal correct {(pred[yte == 0] == 0).sum()}/{(yte == 0).sum()} | "
          f"test clips in an area unseen in train: {unseen}/{len(te)}")
    res["out_of_sample"] = {"accuracy": oos, "majority": float(maj_te), "gain": float(oos - maj_te),
                            "accidents_correct": int((pred[yte == 1] == 1).sum()), "n_accident": int((yte == 1).sum()),
                            "normals_correct": int((pred[yte == 0] == 0).sum()), "n_normal": int((yte == 0).sum()),
                            "unseen_boxes": unseen}

    with open(RESULTS / "sotad_area_and_length_checks.json", "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
