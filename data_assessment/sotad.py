"""SO-TAD dataset access helpers for the data assessment (read-only).

Facts this module relies on:
  * root/{train,test}/<video_id>.mp4, 640x480 @ 25 fps
  * Appendix.txt is TAB-separated "<video_id>.mp4\t<keyframe>", accident videos only
  * video_id < 400 -> accident, >= 400 -> normal
  * frames are letterboxed with solid black bars; the non-black rectangle is the "active area"
  * a small number of frames are corrupt solid-green failed decodes
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

DATASET_ROOT = Path(config.SOTAD_ROOT)
FPS = 25.0
ACCIDENT_ID_MAX = 400  # video_id < 400 => accident


# --------------------------------------------------------------------------- #
# labels
# --------------------------------------------------------------------------- #
def is_accident(video_id: int) -> bool:
    return video_id < ACCIDENT_ID_MAX


def load_keyframes() -> dict[int, int]:
    """video_id -> first frame of the accident. Accident videos only."""
    out: dict[int, int] = {}
    with open(DATASET_ROOT / "Appendix.txt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            name, kf = line.split("\t")
            out[int(name.removesuffix(".mp4"))] = int(kf)
    return out


def list_split(split: str) -> list[int]:
    return sorted(
        int(p.stem) for p in (DATASET_ROOT / split).glob("*.mp4")
    )


def video_path(split: str, video_id: int) -> Path:
    return DATASET_ROOT / split / f"{video_id}.mp4"


# --------------------------------------------------------------------------- #
# corrupt (solid green) frame detection
# --------------------------------------------------------------------------- #
def is_green_corrupt(frame: np.ndarray) -> bool:
    """Solid-green failed decode: very few unique colours, R~0, G>20, B~0.

    Measured on the known examples test/1869.mp4:0, test/1875.mp4:150,
    test/1877.mp4:105. Uses the whole frame (letterbox bars are black, which
    only pulls the means down, so the G>20 test stays conservative).
    """
    small = frame[::4, ::4]  # subsample: solid frames are solid everywhere
    b, g, r = (small[..., i] for i in range(3))
    if r.mean() > 8 or b.mean() > 8:
        return False
    if g.mean() <= 20:
        return False
    # few unique colours -> quantise to 5 bits and count
    q = (small >> 3).reshape(-1, 3)
    n_unique = len(np.unique(q, axis=0))
    return n_unique <= 64


# --------------------------------------------------------------------------- #
# letterbox geometry (active area)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContentBox:
    """Active area: inclusive-exclusive content rectangle inside the frame."""

    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def area(self) -> int:
        return self.height * self.width

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y0 : self.y1, self.x0 : self.x1]

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.y0, self.y1, self.x0, self.x1)


def measure_content_box(
    frames: list[np.ndarray], row_thresh: float = 6.0
) -> ContentBox:
    """Find the non-black content rectangle from a stack of frames.

    Bars are solid black and static, so we take the per-pixel max across the
    given frames and then find rows/cols whose mean brightness clears a small
    threshold. Using max-across-frames avoids mistaking a momentarily dark
    scene row for a letterbox bar.
    """
    stack = np.max(np.stack([f.max(axis=2) for f in frames]), axis=0).astype(
        np.float32
    )
    h, w = stack.shape
    rows = stack.mean(axis=1)
    cols = stack.mean(axis=0)

    def span(profile: np.ndarray, n: int) -> tuple[int, int]:
        live = np.flatnonzero(profile > row_thresh)
        if live.size == 0:
            return 0, n
        return int(live[0]), int(live[-1]) + 1

    y0, y1 = span(rows, h)
    x0, x1 = span(cols, w)
    return ContentBox(y0, y1, x0, x1)


# --------------------------------------------------------------------------- #
# sequential decode
# --------------------------------------------------------------------------- #
def read_frames_sparse(path: Path, wanted: set[int]) -> dict[int, np.ndarray]:
    """Decode sequentially and keep only the wanted indices: {frame_index: BGR frame}.

    Never seeks (CAP_PROP_POS_FRAMES is ~5x slower on this dataset). cap.grab() advances the decoder without
    converting the frame; retrieve() converts only the wanted frames. Missing keys mean the video was shorter.
    """
    if not wanted:
        return {}
    stop = max(wanted)
    cap = cv2.VideoCapture(str(path))
    out: dict[int, np.ndarray] = {}
    idx = 0
    try:
        while idx <= stop:
            if not cap.grab():
                break
            if idx in wanted:
                ok, frame = cap.retrieve()
                if ok:
                    out[idx] = frame
            idx += 1
    finally:
        cap.release()
    return out


def frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
