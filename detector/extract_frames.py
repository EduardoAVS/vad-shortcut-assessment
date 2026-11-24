"""Build the on-disk SO-TAD frame cache read by frame_cache.py (training of GRUMobileNet).

One sequential pass over every video, keeping every CACHE_STRIDE-th frame (5), already resized and centre-cropped
to the model input geometry (128 px). Run once; every training epoch then reads memmaps instead of decoding video.

Inputs : config.SOTAD_ROOT/{train,test}/<id>.mp4
Outputs: config.SOTAD_FRAME_CACHE/{train,test}/<id>.npy (uint8 [K, 128, 128, 3]) + manifest.json  (~9.4 GB)

Usage (from the repository root):
    python detector/extract_frames.py
    python detector/extract_frames.py --out /tmp/cache --limit 3    # smoke test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import config  # noqa: E402
import frame_cache  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(config.SOTAD_FRAME_CACHE))
    parser.add_argument("--dataset-root", default=str(config.SOTAD_ROOT))
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N videos per split (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="re-extract videos already present")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    geometry = frame_cache.build_geometry_transform(args.image_size)
    splits = [s for s in args.splits.split(",") if s]

    print(f"cache dir   : {out}")
    print(f"image size  : {args.image_size}  (Resize -> CenterCrop, same as the eval preset)")
    print(f"cache stride: {frame_cache.CACHE_STRIDE}")

    total_frames, total_bytes, started = 0, 0, time.time()
    for split in splits:
        source = os.path.join(args.dataset_root, split)
        target = os.path.join(out, split)
        os.makedirs(target, exist_ok=True)

        videos = sorted(f for f in os.listdir(source) if f.lower().endswith(".mp4"))
        if args.limit:
            videos = videos[:args.limit]
        print(f"\n{split}: {len(videos)} videos", flush=True)

        for position, name in enumerate(videos, start=1):
            video_id = os.path.splitext(name)[0]
            destination = os.path.join(target, f"{video_id}.npy")
            if os.path.exists(destination) and not args.force:
                continue

            frames = frame_cache.extract_video(
                os.path.join(source, name), geometry, args.image_size)
            if frames.shape[0] == 0:
                print(f"  WARNING {name}: no frames decoded, skipped", flush=True)
                continue

            # Write via a temp file so an interrupted run never leaves a half-written .npy
            # that would later be memmapped as if it were complete.
            temporary = destination + ".tmp.npy"
            np.save(temporary, frames)
            os.replace(temporary, destination)

            total_frames += frames.shape[0]
            total_bytes += frames.nbytes
            if position % 100 == 0:
                elapsed = time.time() - started
                print(f"  {position}/{len(videos)}  {total_frames} frames  "
                      f"{total_bytes / 1e9:.2f} GB  {elapsed / 60:.1f} min", flush=True)

    with open(frame_cache.manifest_path(out), "w") as handle:
        json.dump({
            "image_size": args.image_size,
            "cache_stride": frame_cache.CACHE_STRIDE,
            "splits": splits,
            "geometry": "Resize(image_size) -> CenterCrop(image_size), matching "
                        "weights.transforms(resize_size=S, crop_size=S)",
        }, handle, indent=1)

    print(f"\nDONE {total_frames} frames, {total_bytes / 1e9:.2f} GB, "
          f"{(time.time() - started) / 60:.1f} min")
    print(f"manifest: {frame_cache.manifest_path(out)}")


if __name__ == "__main__":
    main()
