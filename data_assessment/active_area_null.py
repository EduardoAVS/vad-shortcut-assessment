"""Active-area rule against a label-shuffle null (Table 2, active-area columns; Sec. 5.1 exclusive areas).

With many distinct active areas and few clips per area (TAD train: 114 areas for 372 clips), almost every area
holds one class by chance, so the raw accuracy of the table rule "active area -> most frequent class" overstates
the leak. This script shuffles the labels 2,000 times, keeping the partition into areas fixed, and reports the
observed table-rule accuracy and fraction of accidents in an accident-only area next to their null means and 95th
percentiles. Table 2 reports `table_rule_observed` vs `table_rule_null_mean` for SO-TAD (test), SO-TAD (all) and
TAD (all) (99.58 vs 82.07; 98.95 vs 87.48; 74.50 vs 76.75).

Random stream: numpy Generator (seed 20260915), consumed in the order TAD (test), TAD (train), TAD (all),
SO-TAD (test), SO-TAD (all).

Inputs : results/tad_clips.json (tad_clips.py), results/active_area.csv (active_area.py)
Output : results/active_area_null.json

Usage (from the repository root):
    python data_assessment/active_area_null.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
RNG = np.random.default_rng(20260915)
N_PERM = 2000


def metrics(groups, lb):
    table = sum(max(Counter(lb[i] for i in v).values()) for v in groups) / len(lb)
    n_acc = int((lb == 1).sum())
    acc_excl = sum(len(v) for v in groups if {lb[i] for i in v} == {1})
    return table, (acc_excl / n_acc if n_acc else np.nan)


def null(groups, lb):
    t, e = [], []
    for _ in range(N_PERM):
        p = RNG.permutation(lb)
        a, b = metrics(groups, p)
        t.append(a)
        e.append(b)
    return (float(np.mean(t)), float(np.quantile(t, .95)),
            float(np.mean(e)), float(np.quantile(e, .95)))


def evaluate(name, boxes, lb):
    by_box = defaultdict(list)
    for i, b in enumerate(boxes):
        by_box[tuple(b) if b else None].append(i)
    groups = list(by_box.values())
    table, excl = metrics(groups, lb)
    maj = max((lb == 1).mean(), (lb == 0).mean())
    tn, t95, en, e95 = null(groups, lb)
    print(f"\n[{name}]  n={len(lb)}  areas={len(groups)}  mean {len(lb)/len(groups):.1f} clips/area")
    print(f"  table rule        : observed {100*table:5.2f}%  "
          f"null {100*tn:5.2f}% (p95 {100*t95:5.2f}%)  "
          f"EXCESS {100*(table-tn):+6.2f} pp   [majority {100*maj:.2f}%]")
    print(f"  exclusive accid.  : observed {100*excl:5.1f}%  "
          f"null {100*en:5.1f}% (p95 {100*e95:5.1f}%)  "
          f"EXCESS {100*(excl-en):+6.1f} pp")
    verdict = "LEAK" if table > t95 else "within the null -> NOT a leak"
    print(f"  -> {verdict}")
    return {"name": name, "n": len(lb), "n_boxes": len(groups),
            "table_rule_observed": table, "table_rule_null_mean": tn, "table_rule_null_p95": t95,
            "table_rule_excess": table - tn, "majority": float(maj),
            "exclusive_observed": excl, "exclusive_null_mean": en,
            "exclusive_excess": excl - en, "above_null": bool(table > t95)}


def main():
    res = []
    with open(RESULTS / "tad_clips.json") as fh:
        tad = json.load(fh)["rows"]
    for split in ("test", "train", None):
        sub = [r for r in tad if split is None or r["split"] == split]
        lb = np.array([r["accident"] for r in sub])
        if len(np.unique(lb)) < 2:
            continue
        res.append(evaluate(f"TAD ({split or 'all'})", [r["box"] for r in sub], lb))

    with open(RESULTS / "active_area.csv") as fh:
        rows = list(csv.DictReader(fh))
    for split in ("test", None):
        sub = [r for r in rows if split is None or r["split"] == split]
        lb = np.array([int(r["accident"]) for r in sub])
        bx = [(int(r["y0"]), int(r["y1"]), int(r["x0"]), int(r["x1"])) for r in sub]
        res.append(evaluate(f"SO-TAD ({split or 'all'})", bx, lb))

    with open(RESULTS / "active_area_null.json", "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
