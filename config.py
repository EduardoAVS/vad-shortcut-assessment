"""Paths shared by every script in the repository.

Datasets and regenerable caches live outside version control. Every path can be overridden with an environment
variable of the same name, e.g.

    export SOTAD_ROOT=/data/so-tad TAD_ROOT=/data/TAD-benchmark

Scripts add the repository root to ``sys.path`` and ``import config``.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _path(name, default):
    return Path(os.environ.get(name, default)).expanduser()


# Datasets (not redistributed; see README).
DATA_DIR = _path("DATA_DIR", REPO / "data")
SOTAD_ROOT = _path("SOTAD_ROOT", DATA_DIR / "so-tad")            # {train,test}/<id>.mp4, Appendix.txt
TAD_ROOT = _path("TAD_ROOT", DATA_DIR / "TAD-benchmark")         # {train,test}/<class>/<video>.mp4

# Regenerable caches and intermediate files (git-ignored).
WORK_DIR = _path("WORK_DIR", REPO / "work")
SOTAD_FRAME_CACHE = _path("SOTAD_FRAME_CACHE", WORK_DIR / "sotad_frame_cache")   # detector/extract_frames.py
TAD_FRAME_CACHE = _path("TAD_FRAME_CACHE", WORK_DIR / "tad_frame_cache")         # tad_diagnosis build_cache
TAD_FEATURE_CACHE = _path("TAD_FEATURE_CACHE", WORK_DIR / "tad_features")        # tad_diagnosis
SOTAD_FEATURE_CACHE = _path("SOTAD_FEATURE_CACHE", WORK_DIR / "sotad_features")  # pooled backbone features
SOTAD_TEST_SCORES = _path("SOTAD_TEST_SCORES", WORK_DIR / "grumobilenet_test_scores.csv")    # per-window scores
SOTAD_TRAIN_SCORES = _path("SOTAD_TRAIN_SCORES", WORK_DIR / "grumobilenet_train_scores.csv")

# Trained detector (GRUMobileNet), included in the repository.
DETECTOR_CKPT = _path("DETECTOR_CKPT", REPO / "detector" / "weights" / "grumobilenet.pt")
