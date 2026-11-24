"""Per-window GRUMobileNet scores on the SO-TAD test videos (within-video protocol).

Every test video is decoded sequentially; a causal window of 16 frames, 5 frames apart, ends at every `--stride`-th
frame (5 by default, i.e. one score per cached frame). Frames after keyframe + 250 are ignored; windows that contain
the solid-green failed-decode frame are dropped. The resulting CSV is the input of fixed_time_alarm (the
GRUMobileNet within-video AUC of 0.711 in Sec. 5.1 is computed there from these scores).

The printed report gives the within-video AUC (pre-keyframe = 0 vs post-keyframe = 1 windows of the accident
videos), false alarms on the normal videos, mean score vs distance to the keyframe, a few sanity checks, and, as a
control, the video-level AUC with the max-score rule.

Inputs : config.SOTAD_ROOT/test/*.mp4 and Appendix.txt; checkpoint (default config.DETECTOR_CKPT)
Output : CSV (default config.SOTAD_TEST_SCORES), one row per scored window, columns
         video_id, frame_index (last frame of the window), label (0/1), score (softmax p(accident)),
         is_accident_video, keyframe (-1 for normal videos), distance_to_keyframe (frame_index - keyframe; empty
         for normal videos)

Usage (from the repository root; ~479 videos, CPU):
    tools/run_guarded.sh 2000 work/logs/score_test_videos.log -- python detector/score_test_videos.py
    python detector/score_test_videos.py --limit 3 --out /tmp/scores_smoke.csv     # smoke test
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402

import config  # noqa: E402
from model import (  # noqa: E402
    AccidentMobileNetV3Small,
    AccidentMobileNetV3SmallGRU,
    get_mobilenet_v3_small_weights,
)

cv2.setNumThreads(2)

DATASET_ROOT = str(config.SOTAD_ROOT)
POSITIVE_WINDOW_FRAMES = 250
DEFAULT_STRIDE = 5
BLANK_FRAME_VIDEO, BLANK_FRAME_INDEX = 1869, 0
DEVICE = torch.device("cpu")


# --------------------------------------------------------------------------- data

def load_key_frames(root=DATASET_ROOT):
    key_frames = {}
    path = os.path.join(root, "Appendix.txt")
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                name, key_frame = line.split("\t")
                key_frames[int(name.split(".")[0])] = int(key_frame)
    return key_frames


def is_positive_video(video_index):
    return video_index < 400


def get_frame_label(video_index, frame_index, key_frames, window=POSITIVE_WINDOW_FRAMES):
    """Same rule as train.py:get_frame_label. None means 'ignore'."""
    if not is_positive_video(video_index):
        return 0
    if video_index not in key_frames:
        return None
    start = key_frames[video_index]
    if frame_index < start:
        return 0
    if frame_index < start + window:
        return 1
    return None


def is_green(frame_rgb):
    """Solid-green failed decode: Y=U=V=0 renders as roughly (0, 135, 0)."""
    flat = frame_rgb.reshape(-1, 3)
    return bool(flat[:, 0].mean() < 5 and flat[:, 1].mean() > 20 and flat[:, 2].mean() < 5)


def decode_frames(video_path, wanted):
    """Sequential decode (seeking is ~5x slower here). Returns {index: RGB array}.

    Frames are kept at NATIVE resolution: training reads the full-size frame and lets
    weights.transforms(resize_size=128, crop_size=128) resize the shorter side and
    centre-crop. Pre-resizing to a square here would change the aspect handling and
    therefore the scores.
    """
    capture = cv2.VideoCapture(video_path)
    frames, index, last = {}, 0, max(wanted)
    while index <= last:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames[index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    return frames


# --------------------------------------------------------------------------- model

def load_any_checkpoint(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    config = checkpoint.get("config", {}) or {}
    image_size = int(config.get("image_size") or 128)
    kind = "gru" if config.get("model_name") == "mobilenet_gru" else "frame"

    if kind == "gru":
        model = AccidentMobileNetV3SmallGRU(
            input_size=image_size,
            freeze_backbone=False,
            pretrained=False,
            gru_hidden_size=int(config.get("gru_hidden_size") or 128),
            gru_num_layers=int(config.get("gru_num_layers") or 1),
            gru_dropout=float(config.get("gru_dropout") or 0.0),
        )
    else:
        model = AccidentMobileNetV3Small(
            input_size=image_size, freeze_backbone=False, pretrained=False
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    settings = {
        "kind": kind,
        "image_size": image_size,
        "sequence_length": int(config.get("sequence_length") or 16),
        "sequence_stride": int(config.get("sequence_stride") or 5),
        "stride": int(config.get("prediction_stride") or config.get("frame_stride") or
                      (25 if kind == "gru" else 15)),
    }
    return model, settings, checkpoint


def build_transform(image_size):
    weights = get_mobilenet_v3_small_weights()
    try:
        return weights.transforms(crop_size=image_size, resize_size=image_size)
    except TypeError:
        return weights.transforms()


@torch.no_grad()
def backbone_features(model, frames_rgb, transform, batch_size=32):
    """Per-frame backbone features. Computed once per frame and reused across
    overlapping windows -- this is what makes a full pass affordable on CPU."""
    tensors = torch.stack([transform(Image.fromarray(f)) for f in frames_rgb])
    chunks = [
        model._extract_features_4d(tensors[i:i + batch_size])
        for i in range(0, len(tensors), batch_size)
    ]
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def score_from_features(model, kind, features):
    """features: [N, D] for 'frame', [N, T, D] for 'gru'. Returns p1 per sample."""
    if kind == "gru":
        output, _ = model.gru(features)
        logits = model.classifier(output[:, -1, :])
    else:
        logits = model.classifier(features)
    return torch.softmax(logits, dim=1)[:, 1].numpy()


# --------------------------------------------------------------------------- pass

def window_indices(last_frame, sequence_length, sequence_stride):
    return [max(0, last_frame - offset * sequence_stride)
            for offset in reversed(range(sequence_length))]


def evaluate(model, settings, stride, key_frames, window=POSITIVE_WINDOW_FRAMES,
             limit=None, verbose=True):
    kind = settings["kind"]
    transform = build_transform(settings["image_size"])
    seq_len, seq_stride = settings["sequence_length"], settings["sequence_stride"]

    videos = sorted(
        int(f.split(".")[0]) for f in os.listdir(f"{DATASET_ROOT}/test") if f.endswith(".mp4")
    )
    if limit:
        accident = [v for v in videos if v < 400][:limit]
        normal = [v for v in videos if v >= 400][:limit]
        videos = sorted(accident + normal)

    rows = []
    for position, video in enumerate(videos, start=1):
        path = f"{DATASET_ROOT}/test/{video}.mp4"
        capture = cv2.VideoCapture(path)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()

        targets = [f for f in range(0, total, stride)
                   if get_frame_label(video, f, key_frames, window) is not None]
        if not targets:
            continue

        if kind == "gru":
            windows = {t: window_indices(t, seq_len, seq_stride) for t in targets}
            wanted = {i for w in windows.values() for i in w}
        else:
            windows = {t: [t] for t in targets}
            wanted = set(targets)

        frames = decode_frames(path, wanted)
        targets = [t for t in targets if all(i in frames for i in windows[t])]
        if not targets:
            continue

        order = sorted(frames)
        green = {i for i in order if is_green(frames[i])}
        targets = [t for t in targets if not any(i in green for i in windows[t])]
        if not targets:
            continue

        feats = backbone_features(model, [frames[i] for i in order], transform)
        lookup = {index: row for row, index in enumerate(order)}

        if kind == "gru":
            batch = torch.stack(
                [feats[[lookup[i] for i in windows[t]]] for t in targets]
            )
        else:
            batch = feats[[lookup[t] for t in targets]]

        scores = score_from_features(model, kind, batch)
        keyframe = key_frames.get(video, -1)
        for target, score in zip(targets, scores):
            rows.append({
                "video_id": video,
                "frame_index": target,
                "label": get_frame_label(video, target, key_frames, window),
                "score": float(score),
                "is_accident_video": int(is_positive_video(video) and video in key_frames),
                "keyframe": keyframe,
                "distance_to_keyframe": target - keyframe if keyframe >= 0 else None,
            })

        if verbose and position % 50 == 0:
            print(f"  {position}/{len(videos)} videos, {len(rows)} samples", flush=True)

    return pd.DataFrame(rows)


@torch.no_grad()
def blank_frame_score(model, settings):
    """Score the solid-green failed-decode frame. A real detector must not fire on it."""
    path = f"{DATASET_ROOT}/test/{BLANK_FRAME_VIDEO}.mp4"
    frames = decode_frames(path, {BLANK_FRAME_INDEX})
    if BLANK_FRAME_INDEX not in frames:
        return None
    transform = build_transform(settings["image_size"])
    feats = backbone_features(model, [frames[BLANK_FRAME_INDEX]], transform)
    if settings["kind"] == "gru":
        feats = feats.unsqueeze(0).repeat(1, settings["sequence_length"], 1)
    return float(score_from_features(model, settings["kind"], feats)[0])


# --------------------------------------------------------------------------- metrics

def safe_auc(y_true, y_score):
    return roc_auc_score(y_true, y_score) if len(set(y_true)) == 2 else float("nan")


def binary_block(y_true, y_score, threshold=0.5):
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    return {
        "n": len(y_true),
        "auc": safe_auc(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def best_f1(y_true, y_score):
    grid = np.unique(np.round(np.asarray(y_score), 3))
    if len(grid) == 0:
        return float("nan"), float("nan")
    scored = [(f1_score(y_true, (np.asarray(y_score) >= t).astype(int), zero_division=0), t)
              for t in grid]
    value, threshold = max(scored)
    return value, threshold


def report(frames_df, model, settings, checkpoint_path, threshold=0.5):
    name = os.path.basename(checkpoint_path)
    line = "=" * 76
    print(f"\n{line}\n{name}  ({settings['kind']})\n{line}")

    accident = frames_df[frames_df.is_accident_video == 1]
    normal = frames_df[frames_df.is_accident_video == 0]

    # ---- headline: within accident videos --------------------------------------
    print("\n[1] WITHIN ACCIDENT VIDEOS  (pre-keyframe = 0 vs post-keyframe = 1)")
    print("    same camera, scene and encoding on both sides -- corpus confound cancels")
    micro = binary_block(accident.label.tolist(), accident.score.tolist(), threshold)
    f1_at_best, best_threshold = best_f1(accident.label.tolist(), accident.score.tolist())
    print(f"    micro  n={micro['n']:5d}  AUC={micro['auc']:.3f}  F1={micro['f1']:.3f}  "
          f"P={micro['precision']:.3f}  R={micro['recall']:.3f}")
    print(f"           best F1={f1_at_best:.3f} at threshold {best_threshold:.3f}")

    per_video = []
    for _, group in accident.groupby("video_id"):
        if group.label.nunique() == 2:
            per_video.append(safe_auc(group.label.tolist(), group.score.tolist()))
    if per_video:
        print(f"    macro  {len(per_video)} videos with frames on both sides, "
              f"mean per-video AUC={np.nanmean(per_video):.3f} "
              f"(median {np.nanmedian(per_video):.3f})")

    # ---- operational false alarms ----------------------------------------------
    print("\n[2] FALSE ALARMS ON NORMAL VIDEOS")
    frame_fpr = float((normal.score >= threshold).mean())
    per_video_any = normal.groupby("video_id").score.max() >= threshold
    print(f"    frame-level FPR : {frame_fpr:.4f}  ({int((normal.score >= threshold).sum())}"
          f"/{len(normal)} frames)")
    print(f"    video-level FPR : {per_video_any.mean():.4f}  "
          f"({int(per_video_any.sum())}/{len(per_video_any)} videos flagged)")

    # ---- score vs time ----------------------------------------------------------
    print("\n[3] MEAN SCORE vs DISTANCE TO KEYFRAME  (accident videos)")
    edges = [(-10**9, -150), (-150, -100), (-100, -50), (-50, -25), (-25, 0),
             (0, 25), (25, 50), (50, 100), (100, 150), (150, 10**9)]
    for low, high in edges:
        chunk = accident[(accident.distance_to_keyframe >= low)
                         & (accident.distance_to_keyframe < high)]
        if len(chunk):
            marker = " <- accident" if low == 0 else ""
            label = f"[{low if low > -10**9 else '-inf':>6},{high if high < 10**9 else 'inf':>5})"
            print(f"    {label}  n={len(chunk):4d}  mean p1={chunk.score.mean():.3f}{marker}")

    # ---- guardrails --------------------------------------------------------------
    print("\n[4] GUARDRAILS")
    normal_baseline = float(normal.score.mean())
    at_zero = accident[accident.frame_index == 0].score
    pre = accident[accident.label == 0].score
    post = accident[accident.label == 1].score
    blank = blank_frame_score(model, settings)

    def verdict(ok):
        return "PASS" if ok else "FAIL"

    print(f"    mean p1, normal videos (baseline)   : {normal_baseline:.3f}")
    print(f"    mean p1 at FRAME 0 of accident vids : {at_zero.mean():.3f}   "
          f"[{verdict(at_zero.mean() < normal_baseline + 0.15)}] should sit near baseline")
    print(f"    mean p1 pre-accident                : {pre.mean():.3f}")
    print(f"    mean p1 post-accident               : {post.mean():.3f}")
    print(f"    lift attributable to the accident   : {post.mean() - pre.mean():+.3f}")
    if blank is not None:
        # Judge against the normal-video baseline, not a fixed 0.5, and on the same terms as
        # the frame-0 check above. An empty image should score like a non-accident frame; a
        # model sitting at 0.498 is a coin flip on a solid colour, which cleared the old bar
        # by 0.002 and read as a pass.
        print(f"    blank green frame (video 1869 f0)   : {blank:.3f}   "
              f"[{verdict(blank < normal_baseline + 0.15)}] must not fire on an empty image")

    # ---- control -------------------------------------------------------------------
    print("\n[5] CONTROL -- video-level metric (max score over the windows of each video)")
    video_df = frames_df.groupby("video_id").agg(
        video_label=("is_accident_video", "max"), max_score=("score", "max")
    ).reset_index()
    control = binary_block(video_df.video_label.tolist(), video_df.max_score.tolist(), threshold)
    print(f"    rule=max_score_any_frame  n={control['n']}  AUC={control['auc']:.4f}  "
          f"F1={control['f1']:.4f}  P={control['precision']:.4f}  R={control['recall']:.4f}")
    print(f"\n    -> gap: corpus-level AUC {control['auc']:.3f} vs "
          f"within-video AUC {micro['auc']:.3f}")
    return {"within_auc": micro["auc"], "control_auc": control["auc"]}


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--checkpoint", default=str(config.DETECTOR_CKPT))
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                        help="frame sampling stride (last frame of each window); 5 in the paper")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None,
                        help="use only the first N accident and N normal videos (smoke test)")
    parser.add_argument("--window", type=int, default=POSITIVE_WINDOW_FRAMES,
                        help="positive window in frames after the keyframe (250, as in training)")
    parser.add_argument("--out", default=str(config.SOTAD_TEST_SCORES),
                        help="output CSV (default: config.SOTAD_TEST_SCORES)")
    args = parser.parse_args()

    checkpoint_path = os.path.abspath(args.checkpoint)
    model, settings, checkpoint = load_any_checkpoint(checkpoint_path)
    stride = args.stride
    key_frames = load_key_frames()

    print(f"checkpoint : {checkpoint_path}")
    print(f"kind       : {settings['kind']}   epoch: {checkpoint.get('epoch')}")
    print(f"image size : {settings['image_size']}   stride: {stride}", end="")
    if settings["kind"] == "gru":
        print(f"   sequence: {settings['sequence_length']}x{settings['sequence_stride']}")
    else:
        print()

    trained_window = (checkpoint.get("config") or {}).get("positive_window_frames")
    note = ""
    if trained_window is not None and int(trained_window) != args.window:
        note = f"   (model trained with {trained_window})"
    print(f"eval window: {args.window} frames{note}")

    frames_df = evaluate(model, settings, stride, key_frames,
                         window=args.window, limit=args.limit)
    report(frames_df, model, settings, checkpoint_path, threshold=args.threshold)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames_df.to_csv(out, index=False)
    print(f"\nper-window scores written to {out}")


if __name__ == "__main__":
    main()
