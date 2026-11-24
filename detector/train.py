"""Train GRUMobileNet, the accident detector evaluated in the paper, on SO-TAD.

Model: ImageNet MobileNetV3-Small (frozen) per frame -> GRU (128 hidden units) -> linear head (2 classes),
128-px input. A sample is a causal window of 16 frames, 5 frames apart (5 fps), ending at every 25th frame of a
video; frames missing at the start of the video are padded with frame 0. Labels follow the SO-TAD keyframes
(Appendix.txt): windows ending before the keyframe are negative, windows ending in [keyframe, keyframe + 250) are
positive, later windows are ignored; every window of a normal video (id >= 400) is negative. Loss: cross-entropy
with class weights [1, min(neg/pos, 5)] = [1, 5]; Adam, lr 1e-3, batch 16, 6 epochs; the checkpoint with the best
test F1 is kept (epoch 5 in the paper run, `detector/weights/grumobilenet.pt`, whose `config` dict records these
settings).

Every setting can be overridden by an environment variable of the same name (see "settings" below); the defaults
reproduce the paper run. Frames are read from the frame cache built by `detector/extract_frames.py`
(config.SOTAD_FRAME_CACHE, or FRAME_CACHE_DIR).

Inputs : config.SOTAD_ROOT/{train,test}/*.mp4, Appendix.txt; the frame cache
Outputs: OUTPUT_DIR (default config.WORK_DIR/runs/grumobilenet): best.pt, last.pt (resumable), metrics_table.csv,
         loss_curve.png, confusion_matrix.png, analysis/*.csv (test predictions of the best checkpoint, video-level
         metrics, false positives, recall by distance to the keyframe)

Usage (from the repository root; CPU, several hours per epoch):
    tools/run_guarded.sh 6000 work/logs/train.log -- python detector/train.py
    # smoke test:
    EPOCHS=1 TRAIN_VIDEO_LIMIT=3 TEST_VIDEO_LIMIT=3 NUM_WORKERS=0 OUTPUT_DIR=/tmp/train_smoke python detector/train.py

Bit-exact reproduction of the checkpoint is not guaranteed (data-loader worker scheduling, library versions).
"""
import csv
import os
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

# Figures are only written to disk. Must precede the pyplot import: with DISPLAY set (WSLg) matplotlib would pick
# TkAgg, and Tcl is not fork-safe, which kills the DataLoader workers forked afterwards.
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

import config  # noqa: E402
import frame_cache  # noqa: E402
from model import AccidentMobileNetV3SmallGRU, get_mobilenet_v3_small_weights  # noqa: E402

MODEL_NAME = 'mobilenet_gru'  # stored in the checkpoint config; score_test_videos.py keys off it


# --------------------------------------------------------------------------- settings

def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def env_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    value = os.environ.get(name)
    if value is None or value.strip() == '':
        return default
    return int(value)


DATASET_ROOT = str(config.SOTAD_ROOT)
SEED = int(os.environ.get('SEED', 42))
EPOCHS = int(os.environ.get('EPOCHS', 6))
TRAIN_VIDEO_LIMIT = env_optional_int('TRAIN_VIDEO_LIMIT', None)   # first N videos per split (smoke tests)
TEST_VIDEO_LIMIT = env_optional_int('TEST_VIDEO_LIMIT', None)
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 16))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 4))
LR = float(os.environ.get('LR', 1e-3))
IMAGE_SIZE = int(os.environ.get('IMAGE_SIZE', 128))
FREEZE_BACKBONE = env_bool('FREEZE_BACKBONE', True)
PRETRAINED_BACKBONE = env_bool('PRETRAINED_BACKBONE', True)
LOG_INTERVAL = int(os.environ.get('LOG_INTERVAL', 100))
SEQUENCE_LENGTH = int(os.environ.get('SEQUENCE_LENGTH', 16))
SEQUENCE_STRIDE = int(os.environ.get('SEQUENCE_STRIDE', 5))
PREDICTION_STRIDE = int(os.environ.get('PREDICTION_STRIDE', 25))
GRU_HIDDEN_SIZE = int(os.environ.get('GRU_HIDDEN_SIZE', 128))
GRU_NUM_LAYERS = int(os.environ.get('GRU_NUM_LAYERS', 1))
GRU_DROPOUT = float(os.environ.get('GRU_DROPOUT', 0.0))
MAX_FRAMES_PER_VIDEO = env_optional_int('MAX_FRAMES_PER_VIDEO', None)
DECISION_THRESHOLD = float(os.environ.get('DECISION_THRESHOLD', 0.5))
SELECTION_METRIC = os.environ.get('SELECTION_METRIC', 'f1_score')
DISABLE_CLASS_WEIGHTS = env_bool('DISABLE_CLASS_WEIGHTS', False)
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', str(config.WORK_DIR / 'runs' / 'grumobilenet'))

# Frames after the keyframe that count as accident; later frames are ignored (label None), not negative.
# At 250 this almost never truncates: in 98% of accident clips keyframe + 250 lies past the end of the video
# (train 189/192, test 88/90), so "positive" effectively means "keyframe -> end of clip".
POSITIVE_WINDOW_FRAMES = int(os.environ.get('POSITIVE_WINDOW_FRAMES', 250))
POS_CLASS_WEIGHT_CAP = 5.0

assert SELECTION_METRIC in {'accuracy', 'f1_score', 'recall', 'auc'}


# --------------------------------------------------------------------------- data

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_key_frames(root: str) -> Dict[int, int]:
    appendix_path = os.path.join(root, 'Appendix.txt')
    if not os.path.isfile(appendix_path):
        return {}

    key_frames: Dict[int, int] = {}
    with open(appendix_path, 'r', encoding='utf-8') as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            video_name, key_frame = line.split('\t')
            key_frames[int(video_name.split('.')[0])] = int(key_frame)
    return key_frames


def is_positive_video(video_index: int) -> bool:
    return video_index < 400


def list_video_files(root: str, mode: str, video_limit: Optional[int] = None) -> List[str]:
    mode_dir = os.path.join(root, mode)
    if not os.path.isdir(mode_dir):
        raise ValueError(f'dataset directory not found: {mode_dir}')

    files = sorted(file_name for file_name in os.listdir(mode_dir) if file_name.lower().endswith('.mp4'))
    if video_limit is not None:
        files = files[:video_limit]
    return files


def sample_frame_indices(
    video_path: str,
    frame_stride: int,
    max_frames_per_video: Optional[int] = None,
) -> List[int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f'could not open video: {video_path}')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if frame_count <= 0:
        return []

    frame_indices = list(range(0, frame_count, frame_stride))

    if max_frames_per_video is not None and len(frame_indices) > max_frames_per_video:
        positions = torch.linspace(0, len(frame_indices) - 1, steps=max_frames_per_video)
        frame_indices = [frame_indices[int(position.item())] for position in positions]

    return frame_indices


def get_frame_label(
    video_index: int,
    frame_index: int,
    key_frames: Dict[int, int],
    positive_window_frames: int,
) -> Optional[int]:
    """0 = no accident, 1 = accident, None = ignore."""
    if not is_positive_video(video_index):
        return 0                 # normal video: every frame is negative
    if video_index not in key_frames:
        return None              # accident video without a keyframe: ignore rather than guess
    accident_start = key_frames[video_index]
    if frame_index < accident_start:
        return 0                 # before the accident
    if frame_index < accident_start + positive_window_frames:
        return 1                 # inside the positive window
    return None                  # after the positive window: ignore


class TemporalWindowDataset(Dataset):
    """One sample per window: (frames [T, 3, S, S], label, video_id, last_frame, keyframe, distance)."""

    def __init__(
        self,
        root: str,
        mode: str,
        transform=None,
        sequence_length: int = 16,
        sequence_stride: int = 5,
        prediction_stride: int = 25,
        max_sequences_per_video: Optional[int] = None,
        video_limit: Optional[int] = None,
    ):
        self.root = root
        self.mode = mode
        self.transform = transform
        self.sequence_length = sequence_length
        self.sequence_stride = sequence_stride
        self.prediction_stride = prediction_stride
        self.max_sequences_per_video = max_sequences_per_video
        self.video_limit = video_limit
        self.key_frames = load_key_frames(root)
        self.samples: List[Tuple[str, List[int], int, int, int, int, int]] = []
        self._build_index()

    def _build_index(self):
        for file_name in list_video_files(self.root, self.mode, self.video_limit):
            video_index = int(Path(file_name).stem)
            video_path = os.path.join(self.root, self.mode, file_name)

            last_frame_indices = sample_frame_indices(
                video_path=video_path,
                frame_stride=self.prediction_stride,
                max_frames_per_video=self.max_sequences_per_video,
            )

            for last_frame_index in last_frame_indices:
                label = get_frame_label(
                    video_index=video_index,
                    frame_index=last_frame_index,
                    key_frames=self.key_frames,
                    positive_window_frames=POSITIVE_WINDOW_FRAMES,
                )
                if label is None:
                    continue

                # Causal window ending at t. Missing frames at the start are padded by repeating frame 0.
                frame_indices = [
                    max(0, last_frame_index - offset * self.sequence_stride)
                    for offset in reversed(range(self.sequence_length))
                ]
                keyframe = self.key_frames.get(video_index, -1)
                distance_to_keyframe = last_frame_index - keyframe if keyframe >= 0 else -10**9

                self.samples.append((
                    video_path,
                    frame_indices,
                    last_frame_index,
                    label,
                    video_index,
                    keyframe,
                    distance_to_keyframe,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        video_path, frame_indices, label_frame_index, label, video_index, keyframe, distance = self.samples[index]
        images = self._read_frames(video_path, frame_indices)

        if self.transform is not None:
            # Same random augmentation (flip, colour jitter) for every frame of the window.
            transform_seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
            images = [self._apply_transform_with_seed(image, transform_seed) for image in images]
        else:
            images = [transforms.ToTensor()(image) for image in images]

        frames = torch.stack(images, dim=0)
        return (
            frames,
            torch.tensor(label, dtype=torch.long),
            video_index,
            label_frame_index,
            keyframe,
            distance,
        )

    def _apply_transform_with_seed(self, image: Image.Image, seed: int) -> torch.Tensor:
        torch_state = torch.get_rng_state()
        random_state = random.getstate()
        torch.manual_seed(seed)
        random.seed(seed)
        try:
            return self.transform(image)
        finally:
            torch.set_rng_state(torch_state)
            random.setstate(random_state)

    @staticmethod
    def _read_frames(video_path: str, frame_indices: List[int]) -> List[Image.Image]:
        # Cached frames are already at the model's input geometry, so the transform's Resize/CenterCrop are a
        # no-op on them. A 16-frame window costs ~1 s to decode by seeking, against a memmap read from the cache.
        cached = [frame_cache.get_frame(video_path, index) for index in frame_indices]
        if all(image is not None for image in cached):
            return cached

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"can't open: {video_path}")

        images = []
        try:
            for position, frame_index in enumerate(frame_indices):
                if cached[position] is not None:
                    images.append(cached[position])
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f"can't read frame {frame_index} from {video_path}")
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        finally:
            cap.release()

        return images


# Sample tuple fields: (video_path, frame_indices, last_frame, label, video_id, keyframe, distance).
def get_sample_label(sample) -> int:
    return int(sample[3])


def get_sample_frame_index(sample) -> int:
    return int(sample[2])


def get_sample_video_index(sample) -> int:
    return int(sample[4])


def get_class_distribution(dataset: Dataset) -> Dict[int, int]:
    distribution = {0: 0, 1: 0}
    for sample in dataset.samples:
        label = get_sample_label(sample)
        distribution[label] = distribution.get(label, 0) + 1
    return distribution


def build_class_weights(dataset, device):
    distribution = get_class_distribution(dataset)
    neg_count = distribution.get(0, 0)
    pos_count = distribution.get(1, 0)
    if neg_count == 0 or pos_count == 0:
        return None

    pos_weight = min(neg_count / pos_count, POS_CLASS_WEIGHT_CAP)
    weights = torch.tensor([1.0, pos_weight], dtype=torch.float32, device=device)
    print('class weights:', weights.tolist())
    return weights


def inspect_dataset_labels(dataset, name):
    print(f'\n{name}')
    print('total samples:', len(dataset))
    print('class distribution:', get_class_distribution(dataset))

    by_video = defaultdict(Counter)
    for sample in dataset.samples:
        by_video[get_sample_video_index(sample)][get_sample_label(sample)] += 1

    positive_videos = [v for v in by_video if is_positive_video(v)]
    negative_videos = [v for v in by_video if not is_positive_video(v)]
    print('positive videos used:', len(positive_videos))
    print('negative videos used:', len(negative_videos))

    print('\nexample positive videos:')
    for video_index in positive_videos[:5]:
        frames = [get_sample_frame_index(s) for s in dataset.samples if get_sample_video_index(s) == video_index]
        print('video:', video_index, 'keyframe:', dataset.key_frames.get(video_index),
              'counts:', dict(by_video[video_index]), 'first frame:', min(frames), 'last frame used:', max(frames))


# --------------------------------------------------------------------------- metrics and epochs

@dataclass
class Metrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc: float
    fpr: float


@dataclass
class EpochResult:
    metrics: Metrics
    labels: List[int]
    preds: List[int]
    scores: List[float]
    negative_scores: List[float]
    video_indices: List[int]
    frame_indices: List[int]
    splits: List[str]
    keyframes: List[int]
    distance_to_keyframes: List[int]


def build_mobilenet_preprocess(image_size: int):
    weights = get_mobilenet_v3_small_weights()
    try:
        return weights.transforms(crop_size=image_size, resize_size=image_size)
    except TypeError:
        return weights.transforms()


def build_transforms(image_size: int):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
        build_mobilenet_preprocess(image_size),
    ])


def build_eval_transforms(image_size: int):
    return build_mobilenet_preprocess(image_size)


def compute_metrics(total_loss, total_samples, labels, preds, positive_scores):
    loss = total_loss / total_samples if total_samples else 0.0
    accuracy = float(accuracy_score(labels, preds))
    precision = float(precision_score(labels, preds, zero_division=0))
    recall = float(recall_score(labels, preds, zero_division=0))
    f1 = float(f1_score(labels, preds, zero_division=0))
    try:
        auc = float(roc_auc_score(labels, positive_scores))
    except ValueError:
        auc = float('nan')

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return Metrics(loss=loss, accuracy=accuracy, precision=precision, recall=recall, f1_score=f1, auc=auc, fpr=fpr)


def get_current_lr(optimizer):
    if optimizer is None or not optimizer.param_groups:
        return None
    return optimizer.param_groups[0]['lr']


def to_int_list(values) -> List[int]:
    if torch.is_tensor(values):
        return [int(value) for value in values.detach().cpu().tolist()]
    return [int(value) for value in values]


def run_epoch(
    model,
    loader,
    criterion,
    device,
    threshold=0.5,
    optimizer=None,
    epoch: Optional[int] = None,
    total_epochs: Optional[int] = None,
    split: str = 'train',
    log_interval: int = 100,
):
    """One pass over `loader`; trains when an optimizer is given."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_samples = 0
    recent_loss = 0.0
    recent_samples = 0
    started_at = time.time()
    total_batches = len(loader)

    labels_list, preds_list, positive_scores, negative_scores = [], [], [], []
    video_indices_list, frame_indices_list, split_list, keyframe_list, distance_list = [], [], [], [], []

    for batch_idx, batch in enumerate(loader, start=1):
        images, labels, video_indices, frame_indices, keyframes, distances = batch
        images = images.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        probs_all = torch.softmax(logits, dim=1)
        probs0 = probs_all[:, 0]
        probs = probs_all[:, 1]
        preds = (probs >= threshold).long()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        recent_loss += loss.item() * batch_size
        recent_samples += batch_size

        labels_list.extend(labels.detach().cpu().tolist())
        preds_list.extend(preds.detach().cpu().tolist())
        negative_scores.extend(probs0.detach().cpu().tolist())
        positive_scores.extend(probs.detach().cpu().tolist())
        video_indices_list.extend(to_int_list(video_indices))
        frame_indices_list.extend(to_int_list(frame_indices))
        split_list.extend([split] * batch_size)
        keyframe_list.extend(to_int_list(keyframes))
        distance_list.extend(to_int_list(distances))

        if log_interval and batch_idx % log_interval == 0:
            seconds_per_batch = (time.time() - started_at) / batch_idx
            recent_avg_loss = recent_loss / recent_samples if recent_samples else 0.0
            lr = get_current_lr(optimizer)
            lr_text = f' lr {lr:.2e} |' if lr is not None else ''
            epoch_text = f'epoch {epoch}/{total_epochs}' if epoch is not None and total_epochs is not None else 'epoch'
            print(f'{epoch_text} {split} batch {batch_idx}/{total_batches} | '
                  f'recent loss {recent_avg_loss:.4f} |{lr_text} {seconds_per_batch:.2f}s/batch', flush=True)
            recent_loss = 0.0
            recent_samples = 0

    metrics = compute_metrics(total_loss, total_samples, labels_list, preds_list, positive_scores)
    return EpochResult(
        metrics=metrics,
        labels=labels_list,
        preds=preds_list,
        scores=positive_scores,
        negative_scores=negative_scores,
        video_indices=video_indices_list,
        frame_indices=frame_indices_list,
        splits=split_list,
        keyframes=keyframe_list,
        distance_to_keyframes=distance_list,
    )


def metrics_row(epoch: int, split: str, metrics: Metrics) -> Dict[str, float]:
    return {
        'epoch': epoch,
        'split': split,
        'loss': metrics.loss,
        'accuracy': metrics.accuracy,
        'precision': metrics.precision,
        'recall': metrics.recall,
        'f1_score': metrics.f1_score,
        'auc': metrics.auc,
        'fpr': metrics.fpr,
    }


# --------------------------------------------------------------------------- outputs

def write_metrics_table(output_path: str, rows: List[Dict[str, float]]):
    fieldnames = ['epoch', 'split', 'loss', 'accuracy', 'precision', 'recall', 'f1_score', 'auc', 'fpr']
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_loss_curve(output_path: str, train_losses: List[float], test_losses: List[float]):
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, marker='o', label='train loss')
    plt.plot(epochs, test_losses, marker='o', label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Test Loss Curve')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(output_path: str, labels: List[int], preds: List[int]):
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['negative', 'positive'])
    plt.yticks([0, 1], ['negative', 'positive'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    threshold = cm.max() / 2 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > threshold else 'black'
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_prediction_rows(output_path: str, eval_result: EpochResult):
    fieldnames = ['split', 'video_id', 'frame_idx', 'keyframe', 'distance_to_keyframe', 'label', 'pred', 'p0', 'p1']
    with open(output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(eval_result.labels)):
            keyframe = eval_result.keyframes[index]
            distance = eval_result.distance_to_keyframes[index]
            has_keyframe = keyframe >= 0
            writer.writerow({
                'split': eval_result.splits[index],
                'video_id': eval_result.video_indices[index],
                'frame_idx': eval_result.frame_indices[index],
                'keyframe': keyframe if has_keyframe else '',
                'distance_to_keyframe': distance if has_keyframe else '',
                'label': eval_result.labels[index],
                'pred': eval_result.preds[index],
                'p0': eval_result.negative_scores[index],
                'p1': eval_result.scores[index],
            })
    print('saved:', output_path)


def save_dataframe(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print('saved:', path)


# --------------------------------------------------------------------------- post-training analysis

def safe_auc(y_true, y_score):
    try:
        if len(set(y_true)) < 2:
            return float('nan')
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float('nan')


def binary_metrics_from_predictions(y_true, y_pred, y_score=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'auc': safe_auc(y_true, y_score) if y_score is not None else float('nan'),
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }


def has_k_consecutive_positive(scores, threshold, k):
    count = 0
    for score in scores:
        if score >= threshold:
            count += 1
            if count >= k:
                return 1
        else:
            count = 0
    return 0


def frame_dataframe(eval_result):
    return pd.DataFrame({
        'video_id': eval_result.video_indices,
        'frame_index': eval_result.frame_indices,
        'label': eval_result.labels,
        'pred': eval_result.preds,
        'score': eval_result.scores,
    })


def build_video_level_analysis(eval_result, key_frames, threshold=0.5, min_k_values=(1, 3, 5),
                               consecutive_k_values=(3, 5), top_k=3):
    """Video-level predictions with several aggregation rules (any window, >= k windows, k consecutive)."""
    frame_df = frame_dataframe(eval_result)
    video_rows = []
    for video_id, group in frame_df.groupby('video_id'):
        group = group.sort_values('frame_index')
        scores = group['score'].tolist()
        positive_frame_count = int((group['score'] >= threshold).sum())
        row = {
            'video_id': int(video_id),
            'video_label': 1 if (is_positive_video(video_id) and video_id in key_frames) else 0,
            'num_frames_used': int(len(group)),
            'max_score': float(group['score'].max()),
            f'top_{top_k}_mean_score': float(group['score'].nlargest(min(top_k, len(group))).mean()),
            'positive_frame_count': positive_frame_count,
        }
        for k in min_k_values:
            row[f'pred_min_{k}_frames'] = 1 if positive_frame_count >= k else 0
        for k in consecutive_k_values:
            row[f'pred_{k}_consecutive_frames'] = has_k_consecutive_positive(scores, threshold, k)
        video_rows.append(row)

    video_df = pd.DataFrame(video_rows)
    y_true = video_df['video_label'].tolist()
    y_score = video_df['max_score'].tolist()

    rules = [('max_score_any_frame', (video_df['max_score'] >= threshold).astype(int).tolist())]
    rules += [(f'min_{k}_positive_frames', video_df[f'pred_min_{k}_frames'].astype(int).tolist())
              for k in min_k_values]
    rules += [(f'{k}_consecutive_positive_frames', video_df[f'pred_{k}_consecutive_frames'].astype(int).tolist())
              for k in consecutive_k_values]
    metric_rows = []
    for rule, y_pred in rules:
        row = {'rule': rule, 'threshold': threshold}
        row.update(binary_metrics_from_predictions(y_true, y_pred, y_score))
        metric_rows.append(row)
    return video_df, pd.DataFrame(metric_rows)


def build_error_and_window_analysis(eval_result, key_frames, positive_window_frames=250, bin_size=50):
    """False positives (normal video vs before the accident) and recall by distance to the keyframe."""
    frame_df = frame_dataframe(eval_result)

    rows = []
    for _, row in frame_df.iterrows():
        video_id, frame_index = int(row['video_id']), int(row['frame_index'])
        label, pred, score = int(row['label']), int(row['pred']), float(row['score'])
        keyframe = key_frames.get(video_id, None)
        video_is_positive = is_positive_video(video_id) and keyframe is not None
        if label == 0 and pred == 1:
            rows.append({
                'video_id': video_id,
                'frame_index': frame_index,
                'label': label,
                'pred': pred,
                'score': score,
                'video_is_positive': video_is_positive,
                'keyframe': keyframe,
                'distance_to_keyframe': frame_index - keyframe if video_is_positive else None,
                'error_context': ('false_positive_pre_accident_positive_video' if video_is_positive
                                  else 'false_positive_negative_video'),
            })
    false_positives_df = pd.DataFrame(rows)

    if len(false_positives_df) > 0:
        fp_by_video_df = (
            false_positives_df
            .groupby(['video_id', 'error_context'])
            .agg(
                false_positive_frames=('frame_index', 'count'),
                first_fp_frame=('frame_index', 'min'),
                last_fp_frame=('frame_index', 'max'),
                max_fp_score=('score', 'max'),
                mean_fp_score=('score', 'mean'),
                keyframe=('keyframe', 'first'),
                min_distance_to_keyframe=('distance_to_keyframe', 'min'),
                max_distance_to_keyframe=('distance_to_keyframe', 'max'),
            )
            .reset_index()
            .sort_values('false_positive_frames', ascending=False)
        )
    else:
        fp_by_video_df = pd.DataFrame()

    positive_window_rows = []
    for _, row in frame_df.iterrows():
        video_id, frame_index = int(row['video_id']), int(row['frame_index'])
        label, pred, score = int(row['label']), int(row['pred']), float(row['score'])
        keyframe = key_frames.get(video_id, None)
        if keyframe is None or label != 1:
            continue
        offset = frame_index - keyframe
        if offset < 0 or offset >= positive_window_frames:
            continue
        bin_start = int(offset // bin_size) * bin_size
        bin_end = bin_start + bin_size
        positive_window_rows.append({
            'video_id': video_id,
            'frame_index': frame_index,
            'keyframe': keyframe,
            'offset_from_keyframe': offset,
            'bin_start': bin_start,
            'bin_end': bin_end,
            'bin': f'{bin_start}-{bin_end}',
            'pred': pred,
            'score': score,
        })
    positive_window_df = pd.DataFrame(positive_window_rows)

    if len(positive_window_df) > 0:
        window_summary_df = (
            positive_window_df
            .groupby(['bin_start', 'bin_end', 'bin'])
            .agg(
                frames=('pred', 'count'),
                videos=('video_id', 'nunique'),
                recall_in_bin=('pred', 'mean'),
                mean_score=('score', 'mean'),
                min_score=('score', 'min'),
                max_score=('score', 'max'),
            )
            .reset_index()
            .sort_values('bin_start')
        )
    else:
        window_summary_df = pd.DataFrame()

    return false_positives_df, fp_by_video_df, positive_window_df, window_summary_df


# --------------------------------------------------------------------------- main

def build_run_config():
    """Stored in every checkpoint as `config` (score_test_videos.py reads it)."""
    return {
        'dataset_root': DATASET_ROOT,
        'model_name': MODEL_NAME,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'num_workers': NUM_WORKERS,
        'lr': LR,
        'image_size': IMAGE_SIZE,
        'backbone': 'mobilenet_v3_small',
        'pretrained_backbone': PRETRAINED_BACKBONE,
        'freeze_backbone': FREEZE_BACKBONE,
        'log_interval': LOG_INTERVAL,
        'sequence_length': SEQUENCE_LENGTH,
        'sequence_stride': SEQUENCE_STRIDE,
        'prediction_stride': PREDICTION_STRIDE,
        'gru_hidden_size': GRU_HIDDEN_SIZE,
        'gru_num_layers': GRU_NUM_LAYERS,
        'gru_dropout': GRU_DROPOUT,
        'max_frames_per_video': MAX_FRAMES_PER_VIDEO,
        'train_video_limit': TRAIN_VIDEO_LIMIT,
        'test_video_limit': TEST_VIDEO_LIMIT,
        'decision_threshold': DECISION_THRESHOLD,
        'selection_metric': SELECTION_METRIC,
        'disable_class_weights': DISABLE_CLASS_WEIGHTS,
        'positive_window_frames': POSITIVE_WINDOW_FRAMES,
        'seed': SEED,
        'label_strategy': f'frame_level_window_{POSITIVE_WINDOW_FRAMES}',
    }


def main():
    print('torch:', torch.__version__, '| CUDA available:', torch.cuda.is_available())
    print('dataset root:', DATASET_ROOT)
    for name in ['Appendix.txt', 'train', 'test']:
        path = os.path.join(DATASET_ROOT, name)
        print(' ', name, '->', 'ok' if os.path.exists(path) else 'missing', path)

    set_seed(SEED)

    # Fail loudly on a missing cache or one built for other settings.
    cache_root = frame_cache.cache_dir()
    if cache_root:
        frame_cache.check_manifest(cache_root, IMAGE_SIZE)
        print('frame cache:', cache_root)
    else:
        print('frame cache: disabled (FRAME_CACHE_DIR is empty); decoding the videos')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    print('output dir:', OUTPUT_DIR)
    for key, value in build_run_config().items():
        print(f'  {key}: {value}')

    # ---- data
    train_dataset = TemporalWindowDataset(
        root=DATASET_ROOT,
        mode='train',
        transform=build_transforms(IMAGE_SIZE),
        sequence_length=SEQUENCE_LENGTH,
        sequence_stride=SEQUENCE_STRIDE,
        prediction_stride=PREDICTION_STRIDE,
        max_sequences_per_video=MAX_FRAMES_PER_VIDEO,
        video_limit=TRAIN_VIDEO_LIMIT,
    )
    test_dataset = TemporalWindowDataset(
        root=DATASET_ROOT,
        mode='test',
        transform=build_eval_transforms(IMAGE_SIZE),
        sequence_length=SEQUENCE_LENGTH,
        sequence_stride=SEQUENCE_STRIDE,
        prediction_stride=PREDICTION_STRIDE,
        max_sequences_per_video=MAX_FRAMES_PER_VIDEO,
        video_limit=TEST_VIDEO_LIMIT,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    print('train samples:', len(train_dataset))
    print('test samples:', len(test_dataset))
    print('train class distribution:', get_class_distribution(train_dataset))
    print('test class distribution:', get_class_distribution(test_dataset))

    # ---- model (created after set_seed: its random initialisation draws from the torch RNG)
    model = AccidentMobileNetV3SmallGRU(
        input_size=IMAGE_SIZE,
        freeze_backbone=FREEZE_BACKBONE,
        pretrained=PRETRAINED_BACKBONE,
        gru_hidden_size=GRU_HIDDEN_SIZE,
        gru_num_layers=GRU_NUM_LAYERS,
        gru_dropout=GRU_DROPOUT,
    ).to(device)

    class_weights = None if DISABLE_CLASS_WEIGHTS else build_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=LR)

    inspect_dataset_labels(train_dataset, 'train')
    inspect_dataset_labels(test_dataset, 'test')

    # ---- training state (resumed from last.pt if present)
    best_score = float('-inf')
    metrics_rows = []
    train_losses = []
    test_losses = []

    metrics_path = os.path.join(OUTPUT_DIR, 'metrics_table.csv')
    loss_curve_path = os.path.join(OUTPUT_DIR, 'loss_curve.png')
    confusion_matrix_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    best_checkpoint_path = os.path.join(OUTPUT_DIR, 'best.pt')
    last_checkpoint_path = os.path.join(OUTPUT_DIR, 'last.pt')

    def save_checkpoint(path, epoch, best_score):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': best_score,
            'metrics_rows': metrics_rows,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'config': build_run_config(),
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        tmp_path = path + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    start_epoch = 1
    if os.path.exists(last_checkpoint_path):
        print('resuming from checkpoint:', last_checkpoint_path)
        try:
            checkpoint = torch.load(last_checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_score = checkpoint.get('best_score', float('-inf'))
            metrics_rows = checkpoint.get('metrics_rows', [])
            train_losses = checkpoint.get('train_losses', [])
            test_losses = checkpoint.get('test_losses', [])
            start_epoch = checkpoint['epoch'] + 1

            rng_state = checkpoint.get('rng_state')
            if rng_state is not None:
                try:
                    if 'python' in rng_state:
                        random.setstate(rng_state['python'])
                    if 'numpy' in rng_state:
                        np.random.set_state(rng_state['numpy'])
                    if 'torch' in rng_state:
                        torch.set_rng_state(rng_state['torch'])
                    if torch.cuda.is_available() and rng_state.get('cuda') is not None:
                        torch.cuda.set_rng_state_all(rng_state['cuda'])
                    print('RNG state restored.')
                except Exception as error:
                    print('warning: could not restore RNG state:', error)

            print(f'continuing from epoch {start_epoch}/{EPOCHS}')
        except (RuntimeError, ValueError, KeyError):
            print('warning: incompatible checkpoint; starting from scratch.')
    else:
        print('no checkpoint found; starting from scratch.')

    # ---- training
    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_started_at = time.time()
        current_lr = get_current_lr(optimizer)
        print(f'epoch {epoch}/{EPOCHS} started | lr {current_lr:.2e}' if current_lr is not None
              else f'epoch {epoch}/{EPOCHS} started')

        train_result = run_epoch(model, train_loader, criterion, device, threshold=DECISION_THRESHOLD,
                                 optimizer=optimizer, epoch=epoch, total_epochs=EPOCHS, split='train',
                                 log_interval=LOG_INTERVAL)
        eval_result = run_epoch(model, test_loader, criterion, device, threshold=DECISION_THRESHOLD,
                                epoch=epoch, total_epochs=EPOCHS, split='test', log_interval=LOG_INTERVAL)

        train_metrics = train_result.metrics
        eval_metrics = eval_result.metrics
        train_losses.append(train_metrics.loss)
        test_losses.append(eval_metrics.loss)
        metrics_rows.extend([metrics_row(epoch, 'train', train_metrics), metrics_row(epoch, 'test', eval_metrics)])

        write_metrics_table(metrics_path, metrics_rows)
        save_loss_curve(loss_curve_path, train_losses, test_losses)
        save_confusion_matrix(confusion_matrix_path, eval_result.labels, eval_result.preds)

        def fmt(m):
            return (f'loss {m.loss:.4f} acc {m.accuracy:.4f} prec {m.precision:.4f} rec {m.recall:.4f} '
                    f'f1 {m.f1_score:.4f} auc {m.auc:.4f} fpr {m.fpr:.4f}')
        print(f'epoch {epoch}/{EPOCHS} | train {fmt(train_metrics)} | test {fmt(eval_metrics)} | '
              f'elapsed {time.time() - epoch_started_at:.1f}s', flush=True)

        selection_score = getattr(eval_metrics, SELECTION_METRIC)
        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_score': best_score,
                    'config': build_run_config(),
                    f'best_{SELECTION_METRIC}_score': best_score,
                },
                best_checkpoint_path,
            )
            print('new best checkpoint saved to:', best_checkpoint_path)

        # Always saved at the end of the epoch, best or not (for resuming).
        save_checkpoint(last_checkpoint_path, epoch, best_score)
        print('last checkpoint saved to:', last_checkpoint_path)

    print('metrics table:', metrics_path)
    print('loss curve:', loss_curve_path)
    print('confusion matrix:', confusion_matrix_path)
    print('best checkpoint:', best_checkpoint_path)
    print('last checkpoint:', last_checkpoint_path)

    with open(metrics_path, 'r', encoding='utf-8') as csv_file:
        rows = list(csv.DictReader(csv_file))
    print('last metric rows:')
    for row in rows[-4:]:
        print(row)

    # ---- test-set analysis of the best checkpoint
    analysis_dir = os.path.join(OUTPUT_DIR, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    if os.path.isfile(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print('loaded best checkpoint, epoch', checkpoint.get('epoch'))
    else:
        print('best checkpoint not found; using the current model state.')

    eval_result = run_epoch(model=model, loader=test_loader, criterion=criterion, device=device,
                            threshold=DECISION_THRESHOLD, optimizer=None)
    print('test metrics:', eval_result.metrics)

    save_prediction_rows(os.path.join(analysis_dir, 'temporal_predictions.csv'), eval_result)

    video_df, video_metrics_df = build_video_level_analysis(
        eval_result=eval_result, key_frames=test_dataset.key_frames, threshold=DECISION_THRESHOLD)
    save_dataframe(video_df, os.path.join(analysis_dir, 'video_level_predictions.csv'))
    save_dataframe(video_metrics_df, os.path.join(analysis_dir, 'video_level_metrics.csv'))
    print('\nvideo-level metrics:')
    print(video_metrics_df.to_string(index=False))

    false_positives_df, fp_by_video_df, positive_window_df, window_summary_df = build_error_and_window_analysis(
        eval_result=eval_result, key_frames=test_dataset.key_frames,
        positive_window_frames=POSITIVE_WINDOW_FRAMES, bin_size=50)
    save_dataframe(false_positives_df, os.path.join(analysis_dir, 'frame_false_positives.csv'))
    save_dataframe(fp_by_video_df, os.path.join(analysis_dir, 'false_positives_by_video.csv'))
    save_dataframe(positive_window_df, os.path.join(analysis_dir, 'positive_window_frames.csv'))
    save_dataframe(window_summary_df, os.path.join(analysis_dir, 'positive_window_summary_by_bin.csv'))

    print('\nfalse positives by video:')
    print(fp_by_video_df.head(20).to_string(index=False))
    print('\npositive window summary:')
    print(window_summary_df.to_string(index=False))


if __name__ == '__main__':
    main()
