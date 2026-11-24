#!/usr/bin/env python3
"""Frame cache of the TAD benchmark (Xu et al.): 128-px rows, one uint8 .npy per video.

Used by every TAD experiment (Table 4 of the paper). Pre-registered in results/official_split/PREREG.md, section
"Cache" (binding):
  - 5 rows per second, by time: row r is the frame closest to t = r/5 s;
  - resizing identical to the SO-TAD frame cache of the detector (128 px), uint8, one file per video;
  - no cap on the number of rows.

Implementation:
  - the geometry comes from the detector itself (`detector/frame_cache.build_geometry_transform(128)`: Resize(128) on
    the shorter side + CenterCrop(128)), with the same conversion (cv2 BGR -> RGB -> PIL -> geometry -> uint8);
  - sequential decoding with cv2 (no seek). The time of each frame is its own CAP_PROP_POS_MSEC.
    Row r (target r/5 s) gets the frame with the smallest |t - r/5| (tie: the earlier one). Only the candidate frames
    are converted to RGB (the last one before each target and the first one after it); if a candidate is missing (a
    time jump larger than expected), the available candidate is used and this is counted in `n_fallback`;
  - K = number of targets with r/5 <= t of the last frame (targets within the video duration);
  - output: <TAD_FRAME_CACHE>/<split>/<file without .mp4>.npy (uint8 [K,128,128,3]) + a .json with metadata
    (class folder, fps, frames, K, chosen frames, maximum deviation |t - i/fps|). Atomic writes; videos already done
    are skipped (resumable). At the end, writes <TAD_FRAME_CACHE>/index.csv from the .json files.
  - `*Zone.Identifier` files are ignored.

Inputs: the TAD videos under config.TAD_ROOT ({train,test}/<class folder>/<video>.mp4).
Outputs: config.TAD_FRAME_CACHE (default work/tad_frame_cache).

Usage (from the repository root):
  tools/run_guarded.sh 2500 work/logs/tad_build_cache.log -- python tad_diagnosis/build_cache.py --workers 6
  Smoke: --limit 4 --splits train (training videos only).
"""
import argparse
import csv
import json
import os
import resource
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
import config  # noqa: E402

DATA = config.TAD_ROOT
OUT = config.TAD_FRAME_CACHE
ROWS_PER_S = 5
IMAGE_SIZE = 128


def list_videos(splits):
    vids = []
    for split in splits:
        for folder in sorted((DATA / split).iterdir()):
            if not folder.is_dir():
                continue
            label = 0 if folder.name == "normal" else 1
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() == ".mp4" and "Zone.Identifier" not in p.name:
                    vids.append({"split": split, "folder": folder.name, "file": p.name, "vid": p.stem,
                                 "label": label, "path": str(p)})
    return vids


def extract(item):
    import cv2
    sys.path.insert(0, str(REPO / "detector"))
    import frame_cache  # same geometry as the SO-TAD frame cache
    from PIL import Image

    cv2.setNumThreads(1)
    geometry = frame_cache.build_geometry_transform(IMAGE_SIZE)
    out_npy = OUT / item["split"] / f"{item['vid']}.npy"
    out_json = out_npy.with_suffix(".json")
    if out_npy.exists() and out_json.exists():
        return item["vid"], "skipped", 0.0
    t0 = time.time()
    try:
        cap = cv2.VideoCapture(item["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, item["dec_threads"]])
    except Exception:  # noqa: BLE001
        cap = cv2.VideoCapture(item["path"])
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dt_nom = 1.0 / fps if fps > 0 else 0.04

    def to_row(bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.asarray(geometry(Image.fromarray(rgb)), dtype=np.uint8)

    rows, chosen = [], []
    r, i = 0, 0
    prev_t, prev_img, prev_i = None, None, -1
    max_dev, n_fallback, n_nonmono, last_t = 0.0, 0, 0, None
    while cap.grab():
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if last_t is not None and t < last_t:
            n_nonmono += 1
        last_t = t
        if fps > 0:
            max_dev = max(max_dev, abs(t - i / fps))
        cur_img = None
        # Resolve every target already reached by this frame (t >= target): the closest is the previous or this one.
        while t >= r / ROWS_PER_S - 1e-9:
            target = r / ROWS_PER_S
            use_prev = prev_t is not None and (target - prev_t) <= (t - target)
            if use_prev and prev_img is None:
                n_fallback += 1
                use_prev = False
            if use_prev:
                rows.append(prev_img)
                chosen.append(prev_i)
            else:
                if cur_img is None:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    cur_img = to_row(frame)
                rows.append(cur_img)
                chosen.append(i)
            r += 1
        # Keep this frame as the "previous" candidate if the next target may fall before the next frame.
        next_target = r / ROWS_PER_S
        if t + 1.5 * dt_nom >= next_target - 1e-9:
            if cur_img is None:
                ok, frame = cap.retrieve()
                cur_img = to_row(frame) if ok else None
            prev_img = cur_img
        else:
            prev_img = None
        prev_t, prev_i = t, i
        i += 1
    cap.release()
    K = len(rows)
    arr = np.stack(rows) if K else np.zeros((0, IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    out_npy.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_npy.with_suffix(".npy.tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, out_npy)
    meta = {k: item[k] for k in ("split", "folder", "file", "vid", "label")}
    meta.update({"fps": fps, "width": width, "height": height, "n_frames": i, "t_last": last_t,
                 "K": K, "chosen_frames": chosen, "max_dev_t_vs_i_fps": max_dev,
                 "n_fallback": n_fallback, "n_nonmonotonic_time": n_nonmono,
                 "seconds": round(time.time() - t0, 2)})
    tmpj = out_json.with_suffix(".json.tmp")
    tmpj.write_text(json.dumps(meta, ensure_ascii=False))
    os.replace(tmpj, out_json)
    return item["vid"], f"K={K}", time.time() - t0


def write_index():
    metas = [json.loads(p.read_text()) for p in sorted(OUT.glob("*/*.json"))]
    keys = ["split", "folder", "file", "vid", "label", "fps", "width", "height", "n_frames", "t_last", "K",
            "max_dev_t_vs_i_fps", "n_fallback", "n_nonmonotonic_time", "seconds"]
    tmp = OUT / "index.csv.tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(metas)
    os.replace(tmp, OUT / "index.csv")
    return metas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dec-threads", type=int, default=1, help="decoder threads per process")
    ap.add_argument("--splits", nargs="+", default=["train", "test"])
    ap.add_argument("--limit", type=int, default=0, help="only N videos, evenly spaced (smoke)")
    args = ap.parse_args()
    if args.workers * args.dec_threads > 6:
        raise SystemExit("at most 6 threads/processes")
    vids = list_videos(args.splits)
    if args.limit:
        vids = vids[::max(1, len(vids) // args.limit)][:args.limit]
    for v in vids:
        v["dec_threads"] = args.dec_threads
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"TAD cache | {len(vids)} videos | {args.workers} processes x {args.dec_threads} thread(s) | {OUT}",
          flush=True)
    done = 0
    with Pool(args.workers, maxtasksperchild=20) as pool:
        for vid, status, sec in pool.imap_unordered(extract, vids, chunksize=1):
            done += 1
            if done % 25 == 0 or done == len(vids):
                print(f"  {done}/{len(vids)} ({time.time() - t0:.0f} s) last: {vid} {status} {sec:.1f} s", flush=True)
    metas = write_index()
    ru_s, ru_c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = ru_s.ru_utime + ru_s.ru_stime + ru_c.ru_utime + ru_c.ru_stime
    rows = sum(m["K"] for m in metas)
    print(f"DONE: {len(metas)} videos in the index, {rows} rows | wall {time.time() - t0:.0f} s | CPU {cpu:.0f} s",
          flush=True)
    with open(OUT / "cpu_log.csv", "a") as f:
        f.write(f"{time.strftime('%Y%m%d-%H%M%S')},{len(vids)},{cpu:.1f},{time.time() - t0:.1f}\n")


if __name__ == "__main__":
    main()
