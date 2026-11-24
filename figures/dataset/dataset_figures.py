#!/usr/bin/env python3
"""Figures of the paper with real SO-TAD frames: Figs. 5, 6 and 8.

  fig5_first_frames.png   first frames of two accident and two normal test videos, with the active area (content box)
  fig6_motion_region.png  motion region vs annotated box: 2 hits and 2 misses (regions from the recorded
                          motion_region run, boxes from sotad_diagnosis/annotations/keyframe_boxes.csv)
  fig8_gradcam.png        Grad-CAM (top) and background only (bottom) for the event control, the video control and
                          GRUMobileNet, test video 90 (control heads of the cited controls run)

The scores, background and Grad-CAM of Fig. 8 follow the protocol of sotad_diagnosis/interventions.py: score = mean of
the windows r0+2, r0+6, r0+10; background only = median of the pre-event rows [0, min(10, r0)) repeated in the window;
Grad-CAM on the same windows, normalized per window and then averaged. The controls get the marked clip; GRUMobileNet
the unmarked clip.

Style: IBM Plex Sans, one font size and a common scale (font of 34 units for 1600 = \\textwidth, as in
figures/schematic/schematic_figures.py). Each figure is drawn at the size it has in the paper (\\textwidth), without a
"tight" bounding box, so the scale does not change.

Inputs: config.SOTAD_ROOT (test/<id>.mp4, Appendix.txt), config.SOTAD_FRAME_CACHE, config.DETECTOR_CKPT and the
recorded runs above.
Usage (from the repo root; about 1 min):  python figures/dataset/dataset_figures.py [--out DIR] [--fonts DIR]
  defaults: --out figures/output, --fonts figures/fonts.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as P  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib import font_manager  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "sotad_diagnosis"))
import controls  # noqa: E402
from controls import (CELL, CONTROLS_RUN, GRID, cache, cams, config, load_backbone, load_head,  # noqa: E402
                      load_keyframes, plant)
from interventions import background, load_control_heads, pooled, score_static, score_windows  # noqa: E402

SOTAD = config.SOTAD_ROOT
ANNOT = REPO / "sotad_diagnosis" / "annotations" / "keyframe_boxes.csv"
MOTION_REGION_RUN = REPO / "sotad_diagnosis" / "results" / "motion_region" / "20261006-092505"
OUT_DIR = HERE.parent / "output"
FONTS_DIR = HERE.parent / "fonts"
FPS, S = 25, 128

GREEN, BLUE, ORANGE = "#1a9850", "#2c7bb6", "#fdae61"
TW = 6.3                       # \textwidth in inches (only the ratio matters)
FS = 34 / 1600 * TW * 72       # 34 units in 1600 = \textwidth, in pt
INK = "#1C2633"


def read_csv(path):
    with open(path, newline="") as f:
        return {int(r["video"]): r for r in csv.DictReader(f)}


def read_frames(path, idxs):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = {}
    for i in sorted(set(idxs)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(i, n - 1)))
        ok, f = cap.read()
        if ok:
            out[i] = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()
    return out, n


def first_frame(path, at=10):
    fr, n = read_frames(path, [at])
    return fr.get(at), n


def content_box(img, thr=12):
    """Active area: rows and columns whose mean gray level is above `thr` (excludes black bars)."""
    g = img.mean(-1)
    rows = np.where(g.mean(1) > thr)[0]
    cols = np.where(g.mean(0) > thr)[0]
    if len(rows) == 0 or len(cols) == 0:
        return 0, 0, img.shape[1], img.shape[0]
    return cols[0], rows[0], cols[-1] + 1, rows[-1] + 1


def crop_geom(w, h):
    """Origin of the Resize(128) + CenterCrop(128) square in the original frame and the size of a 4x4 cell in px."""
    sc = S / min(w, h)
    nw, nh = w * sc, h * sc
    left, top = round((nw - S) / 2), round((nh - S) / 2)
    return left / sc, top / sc, CELL / sc


def draw_grid(ax, w, h, region=(), crop=True):
    x0, y0, c = crop_geom(w, h)
    if crop:
        ax.add_patch(P.Rectangle((x0, y0), 4 * c, 4 * c, fill=False, ec="white", lw=1.2, ls="--"))
    for cell in region:
        i, j = divmod(int(cell), GRID)
        ax.add_patch(P.Rectangle((x0 + j * c, y0 + i * c), c, c, fc=BLUE, alpha=0.35, ec=BLUE, lw=1.0))
    for k in range(1, GRID):
        ax.plot([x0 + k * c] * 2, [y0, y0 + 4 * c], color="white", lw=0.4, alpha=0.6)
        ax.plot([x0, x0 + 4 * c], [y0 + k * c] * 2, color="white", lw=0.4, alpha=0.6)


def draw_box(ax, a, color=GREEN):
    x0, y0, x1, y1 = (float(a[k]) for k in ("x0", "y0", "x1", "y1"))
    ax.add_patch(P.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=2.0))


def box_crop(ax, a):
    x0, y0, x1, y1 = (float(a[k]) for k in ("cx0", "cy0", "cx1", "cy1"))
    ax.add_patch(P.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=GREEN, lw=2.0))


def cell_box(ax, cell):
    i, j = divmod(cell, GRID)
    ax.add_patch(P.Rectangle((j * CELL, i * CELL), CELL, CELL, fill=False, ec=ORANGE, lw=2.0))


def off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def show_cam(ax, img, cam, fs):
    ax.imshow(img, extent=(0, 128, 128, 0))
    m = cam / max(cam.max(), 1e-12)
    rgba = plt.cm.inferno(m)
    rgba[..., 3] = 0.15 + 0.55 * m
    ax.imshow(rgba, extent=(0, 128, 128, 0), interpolation="nearest")
    for k in range(1, GRID):
        ax.axhline(k * CELL, color="white", lw=0.4, alpha=0.6)
        ax.axvline(k * CELL, color="white", lw=0.4, alpha=0.6)
    for i in range(GRID):
        for j in range(GRID):
            ax.text(j * CELL + CELL / 2, i * CELL + CELL / 2, f"{round(100 * cam[i, j])}",
                    ha="center", va="center", fontsize=fs, color="white",
                    bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.35, lw=0))
    ax.set_xlim(0, 128)
    ax.set_ylim(128, 0)


class GradCamRun:
    """Score, background-only score and Grad-CAM of one model on one test video (protocol of interventions.py)."""

    def __init__(self, controls_run=CONTROLS_RUN):
        self.kf = load_keyframes()
        self.backbone = load_backbone(config.DETECTOR_CKPT)
        self.cells = {int(k): int(v) for k, v in
                      json.loads((Path(controls_run) / "event_control_cells_test.json").read_text()).items()}
        self.heads = {**load_control_heads(controls_run), "grumobilenet": load_head(config.DETECTOR_CKPT)}

    def __call__(self, vid, model):
        fr = np.asarray(cache("test", vid))
        K, r0 = len(fr), -(-self.kf[vid] // 5)
        rows = np.unique(np.clip([r0 + 2, r0 + 6, r0 + 10], 0, K - 1))
        clip = plant(fr, 5 * np.arange(K) >= self.kf[vid], self.cells[vid]) if model != "grumobilenet" else fr
        head = self.heads[model]
        F = pooled(self.backbone, clip)
        bg = background(fr, min(10, r0))
        C, _ = cams(head, controls.feature_maps(self.backbone, clip), rows)
        g = C["gradcam"].reshape(len(rows), -1)
        g = g / np.maximum(g.sum(1, keepdims=True), 1e-12)
        return {"img": clip[rows[len(rows) // 2]], "bg": bg, "cam": g.mean(0).reshape(GRID, GRID),
                "orig": float(score_windows(head, F, rows).mean()),
                "background": score_static(head, pooled(self.backbone, bg[None])[0]),
                "first_frame": score_static(head, F[0])}


def save(fig, out, name):
    fig.savefig(out / name, dpi=300)
    plt.close(fig)
    print("ok", out / name)


def fig_first_frames(out, acc, normal, name, frac=1.0):
    fig, axs = plt.subplots(1, 4, figsize=(frac * TW, frac * TW * 0.25), layout="constrained")
    for ax, (vid, lab) in zip(axs, [(v, "accident") for v in acc] + [(v, "normal") for v in normal]):
        img, n = first_frame(SOTAD / "test" / f"{vid}.mp4")
        x0, y0, x1, y1 = content_box(img)
        ax.imshow(img)
        ax.add_patch(P.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=ORANGE, lw=1.2))
        ax.set_title(f"{lab}\n{x1 - x0}×{y1 - y0}, {n / FPS:.1f} s")
        off(ax)
    save(fig, out, name)


def fig_motion_region(out, ann, regions, hits, misses, name, frac=1.0):
    fig, axs = plt.subplots(1, 4, figsize=(frac * TW, frac * TW * 0.22), layout="constrained")
    for ax, vid in zip(axs, hits + misses):
        q = int(ann[vid]["frame"])
        img = read_frames(SOTAD / "test" / f"{vid}.mp4", [q])[0][q]
        h, w = img.shape[:2]
        ax.imshow(img)
        draw_grid(ax, w, h, regions[vid]["cache_region"].split())
        draw_box(ax, ann[vid])
        ax.set_title(f"video {vid}: {'hit' if vid in hits else 'miss'}")
        off(ax)
    save(fig, out, name)


def fig_gradcam(out, ann, run, vid, name, frac=1.0):
    lab = {"event_control": "Event control", "video_control": "Video control", "grumobilenet": "GRUMobileNet"}
    fig, axs = plt.subplots(2, 3, figsize=(frac * TW, frac * TW * 0.74), layout="constrained")
    for col, m in enumerate(("event_control", "video_control", "grumobilenet")):
        r = run(vid, m)
        top, bot = axs[0, col], axs[1, col]
        show_cam(top, r["img"], r["cam"], fs=FS)
        if m == "grumobilenet":
            box_crop(top, ann[vid])
        else:
            cell_box(top, run.cells[vid])
        top.set_title(f"{lab[m]}\nscore {r['orig']:.2f}")
        bot.imshow(r["bg"])
        bot.set_title(f"background only\nscore {r['background']:.2f} ({round(100 * r['background'] / r['orig'])}%)")
        off(top)
        off(bot)
        print(vid, m, {k: round(v, 3) for k, v in r.items() if isinstance(v, float)})
    save(fig, out, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT_DIR), help="output folder")
    ap.add_argument("--fonts", default=str(FONTS_DIR), help="folder with IBMPlexSans-*.ttf")
    ap.add_argument("--controls-run", default=str(CONTROLS_RUN), help="controls.py run folder with the heads")
    args = ap.parse_args()
    torch.set_num_threads(2)
    for f in Path(args.fonts).glob("IBMPlexSans-*.ttf"):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "IBM Plex Sans", "font.size": FS, "text.color": INK,
                         "axes.titlesize": FS, "axes.titlepad": 4})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ann = read_csv(ANNOT)
    regions = read_csv(MOTION_REGION_RUN / "per_video.csv")
    fig_first_frames(out, [10, 206], [1578, 1995], "fig5_first_frames.png")
    fig_motion_region(out, ann, regions, [90, 191], [29, 110], "fig6_motion_region.png")
    fig_gradcam(out, ann, GradCamRun(args.controls_run), 90, "fig8_gradcam.png")


if __name__ == "__main__":
    main()
