#!/usr/bin/env python3
"""Control models with a planted ground truth, Grad-CAM µw and the motion region on SO-TAD (called T2 in the notes).

Paper results produced here (recorded run: results/controls/20260929-223547/summary.json):
  - video control, window-level AUC 0.945            -> controls_validation.window_auc_video_control
  - Table 3, event control µw 5.9 on the mark        -> controls.event_control/gradcam.lift_post
  - Sec. 5.4, on the mark region: video control µw 1.31, randomized network 1.92
                                                     -> controls.video_control/gradcam.lift_post,
                                                        controls.random_head/gradcam.lift_post
The control heads saved here (event_control_head.pt, video_control_head.pt) are loaded by interventions.py,
box_attribution.py, the TAD diagnosis and the figure scripts.

Criteria fixed before running (verdict GO iff C0 and C1-C4 hold):
  C0 valid control: window-level AUC of the event control >= 0.90 on 20% of the training videos held out.
  C1 event control: mean lift (Grad-CAM) on the mark cell in the post-keyframe windows >= 2.0 and the 95% CI of
     Δ(post − pre) with lower bound > 0.
  C2 false alarm: video control on the same marked clips, lift on the mark cell <= 1.2 and 95% CI containing 1.
  C3 randomization: with the head (GRU + linear) re-initialized, post lift / lift of the trained event control <= 0.5.
  C4 agreement: per-video median Spearman (16 cells, mean maps of the post windows, event control) >= 0.5 between
     Grad-CAM and occlusion and >= 0.5 between Grad-CAM and LayerCAM.
(The recorded verdict is NO-GO because C2 and C4 fail; the paper uses the descriptive numbers above.)

Operationalization:
  - GRUMobileNet architecture: frozen ImageNet MobileNetV3-Small (backbone weights of the detector checkpoint),
    128-px input from the frame cache, GRU(576->128) + Linear(128->2), causal window of 16 cache rows (5-frame
    step) padded with frame 0, decision at the last step; attribution target = logit[1] − logit[0];
  - mark: black/white checkerboard (8-px squares) covering one 4x4 cell (32x32 px), cell drawn per video, present
    in the rows with 5r >= keyframe, only in accident videos (train and test);
  - event control: window label = 1 iff accident and 5r >= keyframe. Video control: same head and labels, with
    input = frame 0 repeated 16 times (it can only learn the scene). At test time both get the real marked windows;
  - training: training accidents + 300 random normals, 6 epochs, Adam 1e-3, class-weighted cross-entropy;
  - attribution per window: Grad-CAM, Grad-CAM++ and LayerCAM summed over the 16 steps; occlusion = drop of the
    target when the cell is painted with the ImageNet mean colour in the 16 frames; lift of a region = mass / area
    fraction (lift is the paper's µw);
  - windows per test video: up to 3 before the keyframe and up to 5 after, spaced in [kf, kf + 250 frames);
  - cascading randomization: (a) re-initialized head (random_head, null 1 of the paper); (b) (a) + features[11]
    and features[12] re-initialized (random_head_backbone);
  - descriptive (outside the verdict): GRUMobileNet on the unmarked test accidents. Lift on the motion region
    (smallest set of cells with >= 50% of the energy of |difference| between rows in [kf − 1 s, kf + 2 s]),
    Δ post − pre and stationarity (correlation with the map of the frame-0 window).

Inputs: config.SOTAD_FRAME_CACHE (detector/extract_frames.py), config.SOTAD_ROOT/Appendix.txt (keyframes),
config.DETECTOR_CKPT; optionally config.SOTAD_TEST_SCORES (pipeline check: the GRUMobileNet scores of test video 1
recomputed here are compared with that file, logged as pipeline.delta_score_max; informational only).

Usage (from the repo root; about 4 min on 2 CPU threads):
  tools/run_guarded.sh 2000 work/logs/controls.log -- python sotad_diagnosis/controls.py
  Quick test (~2 min): add --smoke.
Outputs: sotad_diagnosis/results/controls/<date-time>/ (or --out <dir>/<date-time>): summary.json,
controls_per_video.csv, agreement_per_video.csv, detector_per_video.csv, event_control_cells_test.json,
mark_config.json and the trained heads event_control_head.pt, video_control_head.pt.
"""
import argparse
import copy
import csv
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402

CACHE = config.SOTAD_FRAME_CACHE
APPENDIX = config.SOTAD_ROOT / "Appendix.txt"
# Cited controls run: its heads and mark cells are used by interventions.py, box_attribution.py, the TAD diagnosis
# and the figures. After re-running this script, pass the new run folder with --controls-run where available.
CONTROLS_RUN = HERE / "results" / "controls" / "20260929-223547"
# Test accidents whose motion energy at the keyframe is < 50% inside the 128-px centre crop (data assessment).
FLAGGED = {32, 70, 93}

T_WIN, CELL, GRID = 16, 32, 4
NCELL = GRID * GRID
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
MEAN_U8 = np.array([124, 116, 104], dtype=np.uint8)
METHODS = ("gradcam", "gradcampp", "layercam")
CONTROL_MODELS = ("event_control", "video_control", "random_head", "random_head_backbone")
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*constant.*")


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(576, 128, batch_first=True)
        self.classifier = nn.Linear(128, 2)

    def forward(self, seq):
        out, _ = self.gru(seq)
        return self.classifier(out[:, -1])


def load_backbone(path):
    m = torchvision.models.mobilenet_v3_small(weights=None)
    m.classifier = nn.Identity()
    sd = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    missing, _ = m.load_state_dict(bb, strict=False)
    if not bb or any(k.startswith("features.") for k in missing):
        raise RuntimeError(f"incomplete backbone in checkpoint ({len(bb)} keys; missing {missing[:3]})")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_head(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and "model_state_dict" in ck:
        ck = ck["model_state_dict"]
    head = Head()
    head.load_state_dict({k: v for k, v in ck.items() if k.startswith(("gru.", "classifier."))})
    return head.eval()


def to_tensor(frames):
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN) / STD


@torch.no_grad()
def feature_maps(backbone, frames, bs=64):
    return torch.cat([backbone.features(to_tensor(frames[i:i + bs])) for i in range(0, len(frames), bs)])


def win_idx(r):
    return np.clip(np.arange(r - T_WIN + 1, r + 1), 0, None)


def window(F, r, scene):
    return np.repeat(F[:1], T_WIN, axis=0) if scene else F[win_idx(r)]


def make_checker(size):
    sq = max(1, size // 4)
    yy, xx = np.mgrid[0:size, 0:size]
    return np.repeat(((((yy // sq) + (xx // sq)) % 2) * 255.0)[..., None], 3, axis=-1)


def plant(frames, rows_mask, cell, size=CELL, alpha=1.0):
    """Checkerboard of `size` px centred on the cell, with opacity `alpha` (32 px and 1.0 = the paper's mark)."""
    f = np.array(frames)
    i, j = divmod(int(cell), GRID)
    o = (CELL - size) // 2
    ys, xs = slice(CELL * i + o, CELL * i + o + size), slice(CELL * j + o, CELL * j + o + size)
    patch = f[rows_mask, ys, xs].astype(np.float32)
    f[rows_mask, ys, xs] = np.clip((1 - alpha) * patch + alpha * make_checker(size), 0, 255).round().astype(np.uint8)
    return f


def load_keyframes():
    kf = {}
    for line in APPENDIX.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            kf[int(Path(parts[0]).stem)] = int(float(parts[1]))
    return kf


def cache(split, vid):
    return np.load(CACHE / split / f"{vid}.npy", mmap_mode="r")


def labels(K, vid, kf):
    if vid >= 400:
        return np.zeros(K, dtype=np.int64)
    return (5 * np.arange(K) >= kf[vid]).astype(np.int64)


def boot_ci(x, B=1000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return None, None
    m = x[np.random.default_rng(seed).integers(0, len(x), (B, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def agg(vals):
    """Mean and 95% bootstrap CI (B = 1000, seed 0) of the finite values."""
    x = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if len(x) == 0:
        return {"mean": None, "ci95": [None, None], "n": 0}
    lo, hi = boot_ci(x)
    return {"mean": float(x.mean()), "ci95": [lo, hi], "n": int(len(x))}


def mass(M):
    m = np.asarray(M, np.float64).reshape(len(M), -1)
    s = m.sum(1, keepdims=True)
    return np.where(s > 0, m / np.where(s > 0, s, 1.0), 1.0 / m.shape[1])


def lift(M, cells):
    """µw of the paper: attribution mass in `cells` divided by their area fraction (1 = chance)."""
    return mass(M)[:, cells].sum(1) / (len(cells) / NCELL)


def cams(head, maps, rows):
    """4x4 maps per window (summed over the 16 steps) for each method, and the target y = logit1 − logit0."""
    idx = torch.from_numpy(np.stack([win_idx(r) for r in rows])).long()
    A = maps[idx].clone().requires_grad_(True)  # (W,16,576,4,4)
    lg = head(A.mean(dim=(3, 4)))
    y = lg[:, 1] - lg[:, 0]
    (g,) = torch.autograd.grad(y.sum(), A)
    A = A.detach()
    gc = torch.relu((g.mean(dim=(3, 4), keepdim=True) * A).sum(2)).sum(1)
    g2 = g * g
    aij = g2 / (2 * g2 + A.sum(dim=(3, 4), keepdim=True) * g2 * g + 1e-7)
    aij = torch.where(g != 0, aij, torch.zeros_like(aij))
    gpp = torch.relu(((torch.relu(g) * aij).sum(dim=(3, 4), keepdim=True) * A).sum(2)).sum(1)
    lc = torch.relu((torch.relu(g) * A).sum(2)).sum(1)
    return {"gradcam": gc.numpy(), "gradcampp": gpp.numpy(), "layercam": lc.numpy()}, y.detach().numpy()


@torch.no_grad()
def occlusion_maps(backbone, head, frames, rows, y0):
    """Drop of the target when each cell is painted with the mean colour in the 16 frames of each window."""
    idx = np.stack([win_idx(r) for r in rows])
    need = np.unique(idx)
    loc = np.searchsorted(need, idx)
    base = frames[need]
    out = np.empty((len(rows), NCELL), np.float32)
    for c in range(NCELL):
        f = base.copy()
        i, j = divmod(c, GRID)
        f[:, CELL * i:CELL * (i + 1), CELL * j:CELL * (j + 1)] = MEAN_U8
        P = feature_maps(backbone, f).mean(dim=(2, 3)).numpy()
        lg = head(torch.from_numpy(P[loc]))
        out[:, c] = (lg[:, 1] - lg[:, 0]).numpy()
    return np.maximum(y0[:, None] - out, 0.0).reshape(len(rows), GRID, GRID)


@torch.no_grad()
def predict_F(head, F, scene=False):
    X = torch.from_numpy(np.stack([window(F, r, scene) for r in range(len(F))]))
    return torch.softmax(head(X), 1)[:, 1].numpy()


def safe_auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None


def spear(a, b):
    r = spearmanr(np.ravel(a), np.ravel(b)).correlation
    return float(r) if np.isfinite(r) else None


def motion_region(frames, r0):
    """Motion region: smallest set of cells holding >= 50% of the |row difference| energy in [r0 − 5, r0 + 10]."""
    lo, hi = max(1, r0 - 5), min(len(frames) - 1, r0 + 10)
    if hi < lo:
        return list(range(NCELL))
    g = frames[lo - 1:hi + 1].astype(np.float32).mean(-1)
    e = np.abs(np.diff(g, axis=0)).sum(0)
    ce = e.reshape(GRID, CELL, GRID, CELL).sum((1, 3)).ravel()
    order = np.argsort(-ce)
    cum = np.cumsum(ce[order]) / max(ce.sum(), 1e-9)
    return order[:int(np.searchsorted(cum, 0.5) + 1)].tolist()


def eval_rows(K, r0):
    pre = np.unique(np.linspace(0, r0 - 1, 3).round().astype(int)) if r0 >= 1 else np.array([], int)
    hi = min(K - 1, r0 + 49)
    post = np.unique(np.linspace(r0, hi, 5).round().astype(int)) if hi >= r0 else np.array([], int)
    return pre, post


def check_pipeline(backbone, detector, log, vid=1):
    """Max |Δ| between the GRUMobileNet scores of test video `vid` computed here and config.SOTAD_TEST_SCORES.

    Informational only (None if that file does not exist); the recorded run gave 8.3e-07."""
    path = config.SOTAD_TEST_SCORES
    if not path.exists():
        log(f"[pipeline] {path} not found; check skipped")
        return None
    p = predict_F(detector, feature_maps(backbone, np.asarray(cache("test", vid))).mean(dim=(2, 3)).numpy())
    scores = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["video_id"] == str(vid):
                scores[int(float(row["frame_index"]))] = float(row["score"])
    d_score = max((abs(float(p[r]) - scores[5 * r]) for r in range(len(p)) if 5 * r in scores), default=None)
    log(f"[pipeline] video {vid}: max |Δscore GRUMobileNet| = {d_score}")
    return d_score


def build_train(backbone, kf, args, log):
    rng = np.random.default_rng(args.seed)
    ids = sorted(int(p.stem) for p in (CACHE / "train").glob("*.npy"))
    acc = [v for v in ids if v < 400 and v in kf]
    nor = [v for v in ids if v >= 400]
    if args.smoke:
        acc, nor = acc[:10], nor[:10]
    else:
        nor = sorted(rng.choice(nor, size=min(args.n_normals, len(nor)), replace=False).tolist())
    videos, t0 = [], time.time()
    for k, v in enumerate(acc + nor):
        fr = cache("train", v)
        y = labels(len(fr), v, kf)
        if v < 400 and y.sum() == 0:
            continue  # keyframe beyond the end of the clip (e.g. train/261)
        frm = plant(fr, y.astype(bool), np.random.default_rng([args.seed, v]).integers(NCELL),
                    args.mark_size, args.mark_alpha) if v < 400 else np.asarray(fr)
        videos.append({"vid": v, "F": feature_maps(backbone, frm).mean(dim=(2, 3)).numpy(), "y": y})
        if (k + 1) % 100 == 0:
            log(f"  training features: {k + 1}/{len(acc) + len(nor)} ({time.time() - t0:.0f} s)")
    log(f"[train] {sum(v['vid'] < 400 for v in videos)} accidents + {sum(v['vid'] >= 400 for v in videos)}"
        f" normals ({time.time() - t0:.0f} s)")
    return videos


def train_head(videos, scene, epochs, seed):
    torch.manual_seed(seed)
    head = Head()
    samples = [(i, r) for i, d in enumerate(videos) for r in range(len(d["y"]))]
    ys = np.array([videos[i]["y"][r] for i, r in samples])
    w1 = float((ys == 0).sum()) / max(float((ys == 1).sum()), 1.0)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, w1]))
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(samples))
        for b0 in range(0, len(order), 256):
            bi = order[b0:b0 + 256]
            X = torch.from_numpy(np.stack([window(videos[samples[k][0]]["F"], samples[k][1], scene) for k in bi]))
            opt.zero_grad()
            loss_fn(head(X), torch.from_numpy(ys[bi])).backward()
            opt.step()
    return head.eval()


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def new_run_dir(parent, tag="", smoke=False):
    """Creates <parent>/<date-time>[-tag][-smoke] (never overwrites)."""
    d = Path(parent) / (time.strftime("%Y%m%d-%H%M%S") + (f"-{tag}" if tag else "") + ("-smoke" if smoke else ""))
    d.mkdir(parents=True, exist_ok=False)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--n-normals", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-occlusion", action="store_true", help="skip occlusion (C4 is not evaluated)")
    ap.add_argument("--smoke", action="store_true", help="minimal run to test the script")
    ap.add_argument("--mark-size", type=int, default=CELL, help="side of the mark in px (<= 32)")
    ap.add_argument("--mark-alpha", type=float, default=1.0, help="opacity of the mark (1.0 = opaque)")
    ap.add_argument("--tag", default="", help="suffix of the output folder")
    ap.add_argument("--out", default=str(HERE / "results" / "controls"), help="parent folder of the run folder")
    args = ap.parse_args()
    if args.smoke:
        args.epochs = 1
    torch.set_num_threads(args.threads)
    out = new_run_dir(args.out, args.tag, args.smoke)
    (out / "mark_config.json").write_text(json.dumps(
        {"size": args.mark_size, "alpha": args.mark_alpha, "seed": args.seed}))

    def log(msg):
        print(msg, flush=True)

    t_start = time.time()
    log(f"controls | output: {out}")
    kf = load_keyframes()
    backbone = load_backbone(config.DETECTOR_CKPT)
    detector = load_head(config.DETECTOR_CKPT)
    d_score = check_pipeline(backbone, detector, log)

    # 1) Controls with a planted ground truth: trained on 80% of the videos, C0 on the other 20%.
    videos = build_train(backbone, kf, args, log)
    rng = np.random.default_rng(args.seed + 1)
    val_ids = set()
    for group in ([v["vid"] for v in videos if v["vid"] < 400], [v["vid"] for v in videos if v["vid"] >= 400]):
        val_ids |= set(rng.choice(group, size=max(1, len(group) // 5), replace=False).tolist())
    tr = [v for v in videos if v["vid"] not in val_ids]
    va = [v for v in videos if v["vid"] in val_ids]
    t0 = time.time()
    event_head = train_head(tr, scene=False, epochs=args.epochs, seed=args.seed + 2)
    video_head = train_head(tr, scene=True, epochs=args.epochs, seed=args.seed + 3)
    y_va = np.concatenate([v["y"] for v in va])
    auc_event_val = safe_auc(y_va, np.concatenate([predict_F(event_head, v["F"]) for v in va]))
    auc_video_val = safe_auc(y_va, np.concatenate([predict_F(video_head, v["F"], scene=True) for v in va]))
    log(f"[controls] training {time.time() - t0:.0f} s | window-level AUC (validation): event_control="
        f"{auc_event_val} video_control={auc_video_val}")
    torch.save(event_head.state_dict(), out / "event_control_head.pt")
    torch.save(video_head.state_dict(), out / "video_control_head.pt")

    torch.manual_seed(args.seed + 7)
    rand_head = Head().eval()
    rand_backbone = copy.deepcopy(backbone)
    torch.manual_seed(args.seed + 8)
    for blk in (rand_backbone.features[11], rand_backbone.features[12]):
        for m in blk.modules():
            if m is not blk and hasattr(m, "reset_parameters"):
                m.reset_parameters()
    real_heads = {"grumobilenet": detector}

    # 2) Evaluation on the test accidents.
    test_ids = sorted(v for v in (int(p.stem) for p in (CACHE / "test").glob("*.npy")) if v < 400 and v in kf)
    if args.smoke:
        test_ids = test_ids[:4]
    ctrl_rows, agree_rows, real_rows, cells = [], [], [], {}
    t0 = time.time()
    for n_i, vid in enumerate(test_ids, start=1):
        fr = np.asarray(cache("test", vid))
        K, r0 = len(fr), -(-kf[vid] // 5)
        if r0 >= K:
            continue
        cell = int(np.random.default_rng([args.seed, 100000 + vid]).integers(NCELL))
        cells[vid] = cell
        post_mask = 5 * np.arange(K) >= kf[vid]
        frm = plant(fr, post_mask, cell, args.mark_size, args.mark_alpha)
        pre, post = eval_rows(K, r0)
        rows = np.concatenate([pre, post]).astype(int)
        npre = len(pre)
        maps_m = feature_maps(backbone, frm)
        Fm = maps_m.mean(dim=(2, 3)).numpy()
        res = {"event_control": cams(event_head, maps_m, rows), "video_control": cams(video_head, maps_m, rows),
               "random_head": cams(rand_head, maps_m, rows),
               "random_head_backbone": cams(rand_head, feature_maps(rand_backbone, frm), rows)}
        within = {"event_control": safe_auc(post_mask, predict_F(event_head, Fm)),
                  "video_control": safe_auc(post_mask, predict_F(video_head, Fm))}
        for model, (C, _) in res.items():
            for meth in METHODS:
                L = lift(C[meth], [cell])
                lp = float(L[npre:].mean()) if len(post) else None
                lq = float(L[:npre].mean()) if npre else None
                ctrl_rows.append({"video": vid, "cell": cell, "model": model, "method": meth,
                                  "lift_pre": lq, "lift_post": lp,
                                  "delta": (lp - lq) if (lp is not None and lq is not None) else None,
                                  "within_video_auc": within.get(model)})
        if len(post):
            gc_post = mass(res["event_control"][0]["gradcam"][npre:]).mean(0)
            lc_post = mass(res["event_control"][0]["layercam"][npre:]).mean(0)
            rec = {"video": vid, "cell": cell, "spearman_gc_lc": spear(gc_post, lc_post),
                   "top1_gc": int(np.argmax(gc_post)) == cell}
            if not args.no_occlusion:
                occ = occlusion_maps(backbone, event_head, frm, post, res["event_control"][1][npre:])
                oc_post = mass(occ).mean(0)
                L = lift(occ, [cell])
                ctrl_rows.append({"video": vid, "cell": cell, "model": "event_control", "method": "occlusion",
                                  "lift_pre": None, "lift_post": float(L.mean()), "delta": None,
                                  "within_video_auc": within["event_control"]})
                rec.update(spearman_gc_occ=spear(gc_post, oc_post), top1_occ=int(np.argmax(oc_post)) == cell)
            agree_rows.append(rec)

        maps_u = feature_maps(backbone, fr)
        region = motion_region(fr, r0)
        rows_r = np.unique(np.concatenate([[0], rows])).astype(int)
        is_pre = rows_r < r0
        for model, head in real_heads.items():
            C, _ = cams(head, maps_u, rows_r)
            for meth in METHODS:
                L = lift(C[meth], region)
                ms = mass(C[meth])
                stat = [float(np.corrcoef(ms[0], ms[k])[0, 1]) for k in range(1, len(ms))
                        if ms[0].std() > 0 and ms[k].std() > 0]
                lp = float(L[~is_pre].mean()) if (~is_pre).any() else None
                lq = float(L[is_pre].mean()) if is_pre.any() else None
                real_rows.append({"video": vid, "model": model, "method": meth, "n_region_cells": len(region),
                                  "lift_pre": lq, "lift_post": lp,
                                  "delta": (lp - lq) if (lp is not None and lq is not None) else None,
                                  "stationarity": float(np.mean(stat)) if stat else None,
                                  "motion_outside_crop": vid in FLAGGED})
        if n_i % 10 == 0 or n_i == len(test_ids):
            log(f"  test: {n_i}/{len(test_ids)} videos ({time.time() - t0:.0f} s)")

    write_csv(out / "controls_per_video.csv", ctrl_rows)
    write_csv(out / "agreement_per_video.csv", agree_rows)
    write_csv(out / "detector_per_video.csv", real_rows)
    (out / "event_control_cells_test.json").write_text(json.dumps(cells))

    # 3) Aggregation and verdict.
    def pick(rows, model, meth, key):
        return [r[key] for r in rows if r["model"] == model and r["method"] == meth]

    ctrl = {f"{m}/{k}": {"lift_post": agg(pick(ctrl_rows, m, k, "lift_post")),
                         "lift_pre": agg(pick(ctrl_rows, m, k, "lift_pre")),
                         "delta": agg(pick(ctrl_rows, m, k, "delta"))}
            for m in CONTROL_MODELS
            for k in METHODS + (("occlusion",) if m == "event_control" else ())}
    mk, sc = ctrl["event_control/gradcam"], ctrl["video_control/gradcam"]
    rh, rb = ctrl["random_head/gradcam"], ctrl["random_head_backbone/gradcam"]
    med = lambda key: (float(np.median([r[key] for r in agree_rows if r.get(key) is not None]))  # noqa: E731
                       if any(r.get(key) is not None for r in agree_rows) else None)
    sp_occ, sp_lc = med("spearman_gc_occ"), med("spearman_gc_lc")
    ratio = (rh["lift_post"]["mean"] / mk["lift_post"]["mean"]) if mk["lift_post"]["mean"] else None
    conds = {
        "C0_valid_control": auc_event_val is not None and auc_event_val >= 0.90,
        "C1_event_control": (mk["lift_post"]["mean"] or 0) >= 2.0 and (mk["delta"]["ci95"][0] or -1) > 0,
        "C2_false_alarm": sc["lift_post"]["mean"] is not None and sc["lift_post"]["mean"] <= 1.2
                          and sc["lift_post"]["ci95"][0] is not None
                          and sc["lift_post"]["ci95"][0] <= 1.0 <= sc["lift_post"]["ci95"][1],
        "C3_randomization": ratio is not None and ratio <= 0.5,
        "C4_agreement": None if sp_occ is None else (sp_occ >= 0.5 and (sp_lc or 0) >= 0.5),
    }
    if not conds["C0_valid_control"]:
        verdict = "INVALID (the control did not learn the mark)"
    elif conds["C4_agreement"] is None:
        verdict = "INCOMPLETE (no occlusion)"
    else:
        verdict = "GO" if all(conds[k] for k in conds if k != "C0_valid_control") else "NO-GO"

    real = {f"{m}/{k}": {"lift_post": agg(pick(real_rows, m, k, "lift_post")),
                         "lift_pre": agg(pick(real_rows, m, k, "lift_pre")),
                         "delta": agg(pick(real_rows, m, k, "delta")),
                         "stationarity": agg(pick(real_rows, m, k, "stationarity"))}
            for m in real_heads for k in METHODS}
    summary = {"preregistration": "docstring of sotad_diagnosis/controls.py",
               "pipeline": {"delta_score_max": d_score},
               "controls_validation": {"window_auc_event_control": auc_event_val,
                                       "window_auc_video_control": auc_video_val},
               "controls": ctrl, "lift_ratio_random_head": ratio,
               "lift_ratio_random_head_backbone": (rb["lift_post"]["mean"] / mk["lift_post"]["mean"])
               if mk["lift_post"]["mean"] else None,
               "median_spearman": {"gradcam_vs_occlusion": sp_occ, "gradcam_vs_layercam": sp_lc},
               "top1_on_mark": {"gradcam": float(np.mean([r["top1_gc"] for r in agree_rows])) if agree_rows else None,
                                "occlusion": float(np.mean([r["top1_occ"] for r in agree_rows if "top1_occ" in r]))
                                if any("top1_occ" in r for r in agree_rows) else None},
               "conditions": conds, "verdict": verdict, "detector": real,
               "n_test_videos": len(cells), "total_time_s": round(time.time() - t_start, 1),
               "config": vars(args)}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    def fmt(a):
        if a["mean"] is None:
            return "—"
        lo, hi = a["ci95"]
        return f"{a['mean']:.2f}" + (f" [{lo:.2f}; {hi:.2f}]" if lo is not None else "")

    log("\n=== Controls (Grad-CAM unless stated) ===")
    log(f"C0 window-level AUC of the event control (validation) = {auc_event_val} | video control = {auc_video_val}")
    log(f"C1 event control: post lift = {fmt(mk['lift_post'])} | Δ post−pre = {fmt(mk['delta'])}")
    log(f"C2 video control (false alarm): post lift = {fmt(sc['lift_post'])}")
    log(f"C3 random-head ratio = {ratio} | random head + backbone: post lift = {fmt(rb['lift_post'])}")
    log(f"C4 median Spearman: Grad-CAM×occlusion = {sp_occ} | Grad-CAM×LayerCAM = {sp_lc}")
    for k in ("event_control/gradcampp", "event_control/layercam", "event_control/occlusion"):
        if k in ctrl:
            log(f"   {k}: post lift = {fmt(ctrl[k]['lift_post'])}")
    log(f"Conditions: {json.dumps(conds)}")
    log(f"VERDICT: {verdict}")
    log("\n=== GRUMobileNet (descriptive; motion region; lift 1 = chance) ===")
    for k, v in real.items():
        log(f"{k}: post {fmt(v['lift_post'])} | pre {fmt(v['lift_pre'])} | Δ {fmt(v['delta'])}"
            f" | stationarity {fmt(v['stationarity'])}")
    log(f"\nTotal time: {summary['total_time_s']} s | summary: {out / 'summary.json'}")


if __name__ == "__main__":
    main()
