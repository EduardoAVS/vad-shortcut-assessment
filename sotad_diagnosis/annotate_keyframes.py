#!/usr/bin/env python3
"""Manual annotation of the accident area in the 90 test accident videos (validates the motion region).

This is the tool used to draw the 90 boxes of annotations/keyframe_boxes.csv (protocol in annotations/NOTES.md).
Shows the ORIGINAL video (640x480, 25 fps, every frame) from kf − 1 s to kf + 6 s. The tests (motion region,
ablation, model windows) only use up to kf + 2 s; after that the top bar turns orange. Opens at the keyframe. One box
per video, on the frame where the collision is clearest. The dashed vertical lines mark the crop the model sees (the
128-px cache cuts the sides). The box may fall outside them; the fraction inside the crop is computed and saved. The
motion region and Grad-CAM are not shown, so as not to bias the annotation.

Cache geometry (checked: cache row r = frame 5r transformed, difference 0): Resize(128) maps WxH to
round(128·W/H)x128 and CenterCrop(128) cuts (W' − 128)/2 px from each side. For 640x480: x_c = x·170/640 − 21,
y_c = y·128/480.

Keys (click the window first):
  drag the mouse    draws the box (dragging again replaces it)
  a / d             previous / next frame (1 frame = 0.04 s)
  z / c             5 frames back / forward (0.2 s)
  k                 back to the keyframe
  p                 plays the whole window once
  ENTER or space    saves the box and goes to the next video
  i                 saves the box with the flag "uncertain"
  n                 accident not visible in the window (no box)
  u                 back to the previous video (its annotation is redone)
  q                 quits; progress is saved and the next run continues where it stopped

Output: annotations/keyframe_boxes.csv with, per video: annotated frame, box in the original (x0..y1), box in the
128-px cache clipped to the frame (cx0..cy1; empty if the box is entirely outside the crop), fraction of the box
area inside the crop and the flag (ok / uncertain / not_visible). Needs a display (cv2.imshow).

Inputs: config.SOTAD_ROOT (test/<id>.mp4, Appendix.txt).
Usage (from the repo root):  python sotad_diagnosis/annotate_keyframes.py
"""
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import config  # noqa: E402

VIDEOS = config.SOTAD_ROOT / "test"
APPENDIX = config.SOTAD_ROOT / "Appendix.txt"
OUT = HERE / "annotations" / "keyframe_boxes.csv"
FIELDS = ["video", "frame", "dt_s", "width", "height", "x0", "y0", "x1", "y1",
          "cx0", "cy0", "cx1", "cy1", "frac_in_crop", "flag"]
S, FPS = 128, 25
TEST_END_S, MAX_S = 2, 6  # end of the test window and navigation limit, in seconds after the keyframe
WIN = "annotate keyframe"


def load_keyframes():
    kf = {}
    for line in APPENDIX.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            kf[int(Path(parts[0]).stem)] = int(float(parts[1]))
    return kf


def load_done():
    if not OUT.exists():
        return {}
    with OUT.open() as f:
        return {int(r["video"]): r for r in csv.DictReader(f)}


def save(done):
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for v in sorted(done):
            w.writerow(done[v])


def geometry(w, h):
    """Scale and offset of Resize(S) + CenterCrop(S), as in torchvision."""
    if w >= h:
        nw, nh = int(S * w / h), S
    else:
        nw, nh = S, int(S * h / w)
    left, top = int(round((nw - S) / 2.0)), int(round((nh - S) / 2.0))
    return nw / w, nh / h, left, top


def to_cache(box, w, h):
    sx, sy, left, top = geometry(w, h)
    x0, y0, x1, y1 = box[0] * sx - left, box[1] * sy - top, box[2] * sx - left, box[3] * sy - top
    area = (x1 - x0) * (y1 - y0)
    c = (max(0.0, x0), max(0.0, y0), min(S, x1), min(S, y1))
    inside = max(0.0, c[2] - c[0]) * max(0.0, c[3] - c[1])
    frac = inside / area if area > 0 else 0.0
    return ([round(v, 2) for v in c] if inside > 0 else ["", "", "", ""]), round(frac, 3)


def read_window(path, lo, hi):
    """Decodes sequentially (as the cache) and keeps frames lo..hi."""
    cap = cv2.VideoCapture(str(path))
    out, i = {}, 0
    while i <= hi:
        ok, f = cap.read()
        if not ok:
            break
        if i >= lo:
            out[i] = f
        i += 1
    cap.release()
    return out


class Box:
    def __init__(self):
        self.p0 = self.p1 = None
        self.drag = False

    def on_mouse(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_LBUTTONDOWN:
            self.p0, self.p1, self.drag = (x, y), (x, y), True
        elif ev == cv2.EVENT_MOUSEMOVE and self.drag:
            self.p1 = (x, y)
        elif ev == cv2.EVENT_LBUTTONUP and self.drag:
            self.p1, self.drag = (x, y), False

    def coords(self, w, h):
        if self.p0 is None:
            return None
        x0, x1 = sorted(int(np.clip(v, 0, w)) for v in (self.p0[0], self.p1[0]))
        y0, y1 = sorted(int(np.clip(v, 0, h)) for v in (self.p0[1], self.p1[1]))
        return (x0, y0, x1, y1) if x1 - x0 >= 3 and y1 - y0 >= 3 else None


def dashed_vline(img, x, color):
    for y in range(0, img.shape[0], 12):
        cv2.line(img, (x, y), (x, min(img.shape[0] - 1, y + 6)), color, 1)


def render(frames, q, kf, vid, i, n, box):
    img = frames[q].copy()
    h, w = img.shape[:2]
    sx, _, left, _ = geometry(w, h)
    for xc in (left, left + S):
        dashed_vline(img, int(round(xc / sx)), (0, 255, 255))
    if box.p0 is not None:
        cv2.rectangle(img, box.p0, box.p1, (0, 0, 255), 2)
    late = q - kf > TEST_END_S * FPS
    txt = f"video {vid} ({i + 1}/{n})  t = kf {(q - kf) / FPS:+.2f} s" + ("  [KEYFRAME]" if q == kf else "") \
        + (f"  (after +{TEST_END_S} s: outside the test window)" if late else "")
    cv2.rectangle(img, (0, 0), (w, 22), (0, 110, 230) if late else (0, 0, 0), -1)
    cv2.putText(img, txt, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(WIN, img)


def main():
    kf = load_keyframes()
    vids = sorted(v for v in (int(p.stem) for p in VIDEOS.glob("*.mp4")) if v < 400 and v in kf)
    done = load_done()
    print(f"{len(vids)} accident videos in the test split; {len(done)} already annotated in {OUT.name}")
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    box = Box()
    cv2.setMouseCallback(WIN, box.on_mouse)
    i = next((j for j, v in enumerate(vids) if v not in done), len(vids))
    while i < len(vids):
        vid, k0 = vids[i], kf[vids[i]]
        frames = read_window(VIDEOS / f"{vid}.mp4", max(0, k0 - FPS), k0 + MAX_S * FPS)
        if not frames:
            print(f"  video {vid}: could not decode; skipping")
            i += 1
            continue
        lo, hi = min(frames), max(frames)
        q = min(max(k0, lo), hi)
        kref = min(max(k0, lo), hi)
        h, w = frames[q].shape[:2]
        box.p0 = box.p1 = None
        nxt = None
        while nxt is None:
            render(frames, q, kref, vid, i, len(vids), box)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("a"), ord("d"), ord("z"), ord("c")):
                q = min(hi, max(lo, q + {"a": -1, "d": 1, "z": -5, "c": 5}[chr(key)]))
            elif key == ord("k"):
                q = kref
            elif key == ord("p"):
                for qq in range(lo, hi + 1):
                    render(frames, qq, kref, vid, i, len(vids), box)
                    cv2.waitKey(1000 // FPS)
            elif key in (13, 32, ord("i")):
                b = box.coords(w, h)
                if b is None:
                    print("  draw a box first (or use n)")
                    continue
                c, frac = to_cache(b, w, h)
                done[vid] = dict(zip(FIELDS, [vid, q, round((q - k0) / FPS, 2), w, h, *b, *c, frac,
                                              "uncertain" if key == ord("i") else "ok"]))
                if frac < 1:
                    print(f"  video {vid}: {frac:.0%} of the box inside the crop")
                nxt = i + 1
            elif key == ord("n"):
                done[vid] = dict(zip(FIELDS, [vid, q, round((q - k0) / FPS, 2), w, h] + [""] * 9 + ["not_visible"]))
                nxt = i + 1
            elif key == ord("u") and i > 0:
                done.pop(vids[i - 1], None)
                nxt = i - 1
            elif key == ord("q") or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                save(done)
                print(f"saved: {len(done)}/{len(vids)}")
                return
        save(done)
        i = nxt
    cv2.destroyAllWindows()
    print(f"done: {len(done)}/{len(vids)} in {OUT}")


if __name__ == "__main__":
    main()
