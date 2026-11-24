"""On-disk frame cache of the SO-TAD videos, to stop re-decoding the same frames every epoch.

The training datasets open a video file and seek for *every* sample, on *every* epoch. Measured
cost: 106 ms for one frame, 1025 ms for a 16-frame GRU window. That makes a GRU epoch ~8.7 h and
8 epochs ~3 days, almost all of it spent decoding the same frames repeatedly.

This module stores each needed frame once, already resized and cropped to the model's input
geometry, as a per-video uint8 .npy memmap.

Key property: frames are stored at exactly `image_size` x `image_size`, produced by the same
Resize + CenterCrop that `weights.transforms(resize_size=S, crop_size=S)` applies. Feeding such an
image back through that transform is a **no-op** on geometry (resizing a 128px-shorter-side image
to 128 changes nothing; centre-cropping 128 from 128 changes nothing). So the existing transform
pipelines work unchanged and no separate code path is needed.

Layout:
    <cache>/<split>/<video_id>.npy    uint8 [K, S, S, 3]; row i holds frame index i * CACHE_STRIDE
    <cache>/manifest.json             settings, so a mismatched cache is detected rather than used

Location: the FRAME_CACHE_DIR environment variable if set (an empty value disables the cache and falls back to
video decoding), otherwise `config.SOTAD_FRAME_CACHE`. Build it with `python detector/extract_frames.py`.
"""
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

# Every 5th frame. Both samplers land on multiples of 5: the temporal windows step by
# SEQUENCE_STRIDE=5 from a last frame that is a multiple of PREDICTION_STRIDE=25, and the
# frame-level dataset uses FRAME_STRIDE=25. A request that is not a multiple of this falls
# back to decoding the video.
CACHE_STRIDE = 5

# An open memmap costs one file descriptor (CPython's mmap dup()s the fd), so an unbounded
# pool exhausts RLIMIT_NOFILE -- 1024 by default on WSL. Shuffled training touches a new
# video almost every sample, so the pool has to be evicted, not just filled. Reopening is a
# single mmap syscall, far cheaper than the video decode this cache exists to avoid.
MAX_OPEN_MEMMAPS = 128

_ARRAYS: "OrderedDict[str, Optional[np.ndarray]]" = OrderedDict()
_MANIFEST: Optional[dict] = None
_WARNED = set()


def cache_dir() -> Optional[str]:
    """FRAME_CACHE_DIR if set (empty = cache disabled), else config.SOTAD_FRAME_CACHE."""
    value = os.environ.get('FRAME_CACHE_DIR')
    if value is None:
        value = str(config.SOTAD_FRAME_CACHE)
    return value or None


def build_geometry_transform(image_size: int):
    """Resize + CenterCrop only -- the geometric half of the eval preset, no tensor conversion."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ])


def manifest_path(root: str) -> str:
    return os.path.join(root, 'manifest.json')


def load_manifest(root: str) -> Optional[dict]:
    global _MANIFEST
    if _MANIFEST is None:
        path = manifest_path(root)
        if not os.path.isfile(path):
            return None
        with open(path) as handle:
            _MANIFEST = json.load(handle)
    return _MANIFEST


def check_manifest(root: str, image_size: int) -> None:
    """Refuse a cache built for different settings rather than silently training on it."""
    manifest = load_manifest(root)
    if manifest is None:
        raise FileNotFoundError(
            f'frame cache {root} has no manifest.json. '
            f'Build it with:  python detector/extract_frames.py --out {root}  '
            f'(or set FRAME_CACHE_DIR= to decode the videos instead, which is very slow)')
    if int(manifest['image_size']) != int(image_size):
        raise ValueError(
            f'frame cache was built for image_size={manifest["image_size"]} but this run uses '
            f'{image_size}. Rebuild the cache or change IMAGE_SIZE.')
    if int(manifest['cache_stride']) != CACHE_STRIDE:
        raise ValueError(
            f'frame cache stride {manifest["cache_stride"]} != expected {CACHE_STRIDE}.')


def split_of(video_path: str) -> str:
    return os.path.basename(os.path.dirname(video_path))


def video_id_of(video_path: str) -> str:
    return os.path.splitext(os.path.basename(video_path))[0]


def _close(array: Optional[np.ndarray]) -> None:
    """Release the memmap's file descriptor. Dropping the reference alone may not, promptly."""
    handle = getattr(array, '_mmap', None)
    if handle is not None:
        handle.close()


def _array_for(root: str, video_path: str) -> Optional[np.ndarray]:
    """Memmap for one video, from a bounded most-recently-used pool."""
    key = f'{split_of(video_path)}/{video_id_of(video_path)}'
    if key in _ARRAYS:
        _ARRAYS.move_to_end(key)
        return _ARRAYS[key]

    path = os.path.join(root, f'{key}.npy')
    try:
        _ARRAYS[key] = np.load(path, mmap_mode='r')
    except (FileNotFoundError, ValueError):
        _ARRAYS[key] = None

    while len(_ARRAYS) > MAX_OPEN_MEMMAPS:
        _close(_ARRAYS.popitem(last=False)[1])
    return _ARRAYS[key]


def get_frame(video_path: str, frame_index: int) -> Optional[Image.Image]:
    """Cached frame as a PIL image, or None if unavailable (caller should decode)."""
    root = cache_dir()
    if root is None:
        return None
    if int(frame_index) % CACHE_STRIDE != 0:
        if 'stride' not in _WARNED:
            _WARNED.add('stride')
            print(f'frame_cache: frame index {frame_index} is not a multiple of {CACHE_STRIDE}; '
                  f'falling back to video decoding for such frames')
        return None

    array = _array_for(root, video_path)
    if array is None:
        return None
    row = int(frame_index) // CACHE_STRIDE
    if row >= array.shape[0]:
        return None
    # Copy, don't view: `np.asarray` would alias the mapping, and eviction closing that
    # mapping under a still-referenced image is a segfault, not an exception. One frame is
    # 48 KB.
    return Image.fromarray(np.array(array[row]))


def extract_video(video_path: str, geometry, image_size: int) -> np.ndarray:
    """Sequentially decode a video, keeping every CACHE_STRIDE-th frame.

    Sequential decoding, not seeking -- seeking is ~5x slower and this is the one pass where
    that matters.
    """
    capture = cv2.VideoCapture(video_path)
    frames, index = [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % CACHE_STRIDE == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(np.asarray(geometry(Image.fromarray(rgb)), dtype=np.uint8))
        index += 1
    capture.release()
    if not frames:
        return np.zeros((0, image_size, image_size, 3), dtype=np.uint8)
    return np.stack(frames)
