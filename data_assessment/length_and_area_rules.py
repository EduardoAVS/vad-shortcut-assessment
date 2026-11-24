"""Clip-length and active-area rules on SO-TAD and TAD, measured the same way (Table 2, length columns).

For each set (TAD test / train / all; SO-TAD test / all):
  * clip length: symmetrized AUC of the length as a score for "accident" (with its direction), bootstrap 95% CI
    (2,000 resamples), permutation null (2,000 label shuffles), class ranges, and the accuracy of the best
    single-threshold length rule against the majority-class baseline;
  * active area: accuracy of the table rule "active area -> most frequent class of that area" against the
    majority baseline, and the number of clips whose active area holds only their class;
  * TAD only: the same table rule on the raw resolution (SO-TAD is always 640x480).
Table 2 reports `auc` and `separability.best_single_threshold_accuracy` vs `separability.majority_class` for
SO-TAD (test), SO-TAD (all) and TAD (all). The active-area columns of Table 2 come from `active_area_null.py`.

Random streams: one numpy Generator (seed 20260915) shared by the bootstrap and the permutation null, consumed in
the order TAD (test), TAD (train), TAD (all), SO-TAD (test), SO-TAD (all).

Inputs : results/tad_clips.json (tad_clips.py), results/active_area.csv (active_area.py)
Output : results/length_and_area_rules.json; a Markdown summary table on stdout

Usage (from the repository root):
    python data_assessment/length_and_area_rules.py
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
RNG = np.random.default_rng(20260915)


# ------------------------------------------------------------------ length metrics
def auc_raw(sc, lb):
    p, q = sc[lb == 1], sc[lb == 0]
    if p.size == 0 or q.size == 0:
        return float("nan")
    gt = (p[:, None] > q[None, :]).sum()
    eq = (p[:, None] == q[None, :]).sum()
    return float((gt + 0.5 * eq) / (p.size * q.size))


def auc_sym(sc, lb):
    a = auc_raw(sc, lb)
    return (a, "accident_longer") if a >= .5 else (1 - a, "accident_shorter")


def boot(sc, lb, n=2000):
    v, idx = [], np.arange(sc.size)
    for _ in range(n):
        s = RNG.choice(idx, idx.size, True)
        if len(np.unique(lb[s])) < 2:
            continue
        v.append(auc_sym(sc[s], lb[s])[0])
    return (float(np.quantile(v, .025)), float(np.quantile(v, .975))) if v else (np.nan, np.nan)


def null_perm(sc, lb, n=2000):
    v = [auc_sym(sc, RNG.permutation(lb))[0] for _ in range(n)]
    return {"mean": float(np.mean(v)), "p95": float(np.quantile(v, .95)), "max": float(max(v))}


def separability(sc, lb):
    p, q = sc[lb == 1], sc[lb == 0]
    gap = max(q.min() - p.max(), p.min() - q.max())
    best, bt = 0., None
    for t in np.unique(sc):
        for sg in (1, -1):
            acc = float(((sg * sc >= sg * t).astype(int) == lb).mean())
            if acc > best:
                best, bt = acc, float(t)
    maj = max((lb == 1).mean(), (lb == 0).mean())
    return {"accident_range": [float(p.min()), float(p.max())],
            "normal_range": [float(q.min()), float(q.max())],
            "disjoint": bool(gap > 0), "gap": float(gap),
            "best_single_threshold_accuracy": best, "threshold": bt,
            "majority_class": float(maj), "gain": float(best - maj)}


def evaluate_length(name, sc, lb):
    sc = np.asarray(sc, float)
    lb = np.asarray(lb, int)
    a, d = auc_sym(sc, lb)
    return {"set": name, "n": int(sc.size),
            "n_accident": int((lb == 1).sum()), "n_normal": int((lb == 0).sum()),
            "auc": a, "direction": d, "ci95": list(boot(sc, lb)),
            "null": null_perm(sc, lb), "separability": separability(sc, lb),
            "median_accident": float(np.median(sc[lb == 1])),
            "median_normal": float(np.median(sc[lb == 0]))}


# ------------------------------------------------------------------ table rules
def active_area_rule(boxes, labels):
    """Table rule on the active area, with the majority-class baseline."""
    groups = defaultdict(list)
    for i, b in enumerate(boxes):
        groups[tuple(b) if b else None].append(i)
    n = len(labels)
    table = sum(max(Counter(labels[i] for i in v).values()) for v in groups.values()) / n
    maj = max((labels == 1).mean(), (labels == 0).mean())
    acc_excl = sum(len(v) for v in groups.values() if {labels[i] for i in v} == {1})
    norm_excl = sum(len(v) for v in groups.values() if {labels[i] for i in v} == {0})
    n_acc = int((labels == 1).sum())
    n_norm = int((labels == 0).sum())
    return {"n_boxes": len(groups), "table_rule": table, "majority": float(maj),
            "gain": table - float(maj),
            "accidents_in_exclusive_box": acc_excl, "n_accident": n_acc,
            "frac_accidents_exclusive": acc_excl / n_acc if n_acc else float("nan"),
            "normals_in_exclusive_box": norm_excl, "n_normal": n_norm,
            "frac_normals_exclusive": norm_excl / n_norm if n_norm else float("nan")}


def resolution_rule(ws, hs, labels):
    """Table rule on the raw resolution (constant in SO-TAD, variable in TAD)."""
    key = [f"{w}x{h}" for w, h in zip(ws, hs)]
    groups = defaultdict(list)
    for i, k in enumerate(key):
        groups[k].append(i)
    n = len(labels)
    table = sum(max(Counter(labels[i] for i in v).values()) for v in groups.values()) / n
    maj = max((labels == 1).mean(), (labels == 0).mean())
    return {"n_resolutions": len(groups), "table_rule": table, "majority": float(maj),
            "gain": table - float(maj),
            "most_common": Counter(key).most_common(5)}


def sotad_rows(split=None):
    with open(RESULTS / "active_area.csv") as fh:
        rows = list(csv.DictReader(fh))
    if split:
        rows = [r for r in rows if r["split"] == split]
    n = np.array([int(r["n_frames"]) for r in rows])
    a = np.array([int(r["accident"]) for r in rows])
    b = [(int(r["y0"]), int(r["y1"]), int(r["x0"]), int(r["x1"])) for r in rows]
    return n, a, b


def table_line(name, res, geo):
    s = res["separability"]
    return (f"| {name} | {res['n']} ({res['n_accident']}/{res['n_normal']}) | "
            f"{res['auc']:.4f} | [{res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}] | "
            f"{'yes' if s['disjoint'] else 'no'} | "
            f"{100*s['best_single_threshold_accuracy']:.2f}% | {100*s['majority_class']:.2f}% | "
            f"**{100*s['gain']:+.2f} pp** | {100*geo['gain']:+.2f} pp |")


def main():
    with open(RESULTS / "tad_clips.json") as fh:
        raw = json.load(fh)
    check, rows = raw["frame_count_check"], raw["rows"]
    print(f"TAD frame-count check: {check['exact']}/{check['n_sample']} exact, "
          f"largest difference {check['max_abs_diff']}\n")

    lines, out = [], {}
    for split in ("test", "train", None):
        sub = [r for r in rows if split is None or r["split"] == split]
        if not sub:
            continue
        lb = np.array([r["accident"] for r in sub])
        if len(np.unique(lb)) < 2:
            continue
        nf = np.array([r["n_frames"] for r in sub], float)
        bx = [r["box"] for r in sub]
        ws = [r["w"] for r in sub]
        hs = [r["h"] for r in sub]
        name = f"TAD ({split or 'all'})"
        res = evaluate_length(name, nf, lb)
        geo = active_area_rule(bx, lb)
        res["active_area"] = geo
        res["resolution"] = resolution_rule(ws, hs, lb)
        out[name] = res
        lines.append(table_line(f"**{name}**", res, geo))

    for split in ("test", None):
        n, a, b = sotad_rows(split)
        name = f"SO-TAD ({split or 'all'})"
        res = evaluate_length(name, n.astype(float), a)
        geo = active_area_rule(b, a)
        res["active_area"] = geo
        out[name] = res
        lines.append(table_line(name, res, geo))

    print("| set | n (acc/norm) | length AUC | 95% CI | disjoint | 1-threshold acc. | majority | "
          "length gain | active-area gain |")
    print("|---|---|---|---|:---:|---|---|---|---|")
    for line in lines:
        print(line)

    print("\n--- TAD detail ---")
    for k, v in out.items():
        if not k.startswith("TAD"):
            continue
        s, g, r = v["separability"], v["active_area"], v["resolution"]
        print(f"\n[{k}]  n={v['n']}  AUC={v['auc']:.4f} ({v['direction']})  null p95={v['null']['p95']:.3f}")
        print(f"  ranges: accident {s['accident_range']}  normal {s['normal_range']}  gap={s['gap']:+.0f}")
        print(f"  medians: accident {v['median_accident']:.0f}  normal {v['median_normal']:.0f} frames")
        print(f"  active area: {g['n_boxes']} boxes | table rule {100*g['table_rule']:.2f}% vs majority "
              f"{100*g['majority']:.2f}% | gain {100*g['gain']:+.2f} pp")
        print(f"    accidents in an exclusive box: {g['accidents_in_exclusive_box']}/{g['n_accident']} = "
              f"{100*g['frac_accidents_exclusive']:.1f}%")
        print(f"  resolution: {r['n_resolutions']} distinct | gain {100*r['gain']:+.2f} pp | {r['most_common'][:3]}")

    with open(RESULTS / "length_and_area_rules.json", "w") as fh:
        json.dump(out, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
