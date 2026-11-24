"""Shared helpers of the fixed-time-alarm analysis (vectorized numpy harness).

Conventions
- SO-TAD at 25 fps; score grid with step 5 frames (5 samples/s).
- The series of a video = the samples of the per-window scores CSV, in frame_index order (accident
  video: [0, keyframe + 250); regular video: the whole clip).
- First alarm = first sample with value >= threshold. hit(k) = 1 if the first alarm falls in
  [tau - 0.5 s, tau + k]. An alarm before tau - 0.5 s is a false alarm (miss); no alarm is a miss.
- S = mean of hit(k) over k = 0, 0.2, ..., 4.0 s (21 points).
- Post-processing variants: raw, warmup5, smooth5, runmin, cusum. They are implemented here twice:
  vectorized over padded arrays (VARIANTS) and as plain per-series loops (VARIANTS_REF), and the
  evaluation checks that both agree.
- Within-video AUC: per accident video, Mann-Whitney AUC of the samples after the keyframe against
  those before it; reported as the mean over videos (macro).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FPS = 25
STRIDE = 5
PRE_TOL = 0.5 * FPS                  # 12.5 frames: an alarm >= tau - 12.5 counts
KS_S = np.round(np.arange(0, 4.0 + 1e-9, 0.2), 1)   # 21 values of k in seconds
KS_F = KS_S * FPS                                    # 0, 5, ..., 100 frames
VARIANT_NAMES = ["raw", "warmup5", "smooth5", "runmin", "cusum"]
FPR_MAIN, FPR_SENS = 0.001, 0.01


def keyframes():
    """{video_id: keyframe} from SO-TAD's Appendix.txt."""
    kf = {}
    for line in open(config.SOTAD_ROOT / "Appendix.txt", encoding="utf-8"):
        line = line.strip()
        if line:
            n, k = line.split("\t")
            kf[int(n.split(".")[0])] = int(k)
    return kf


# ------------------------------------------------------------------ series <-> padded arrays
def pad_series(list_of_arrays, fill=np.nan, dtype=float):
    L = max(len(a) for a in list_of_arrays)
    out = np.full((len(list_of_arrays), L), fill, dtype=dtype)
    for i, a in enumerate(list_of_arrays):
        out[i, :len(a)] = a
    return out


def series_from_df(df, acc):
    """Per-video series (in video_id order) with frames, scores, keyframe and label."""
    sub = df[df.is_accident_video == (1 if acc else 0)].sort_values(["video_id", "frame_index"])
    vids, F, S, KF, LAB = [], [], [], [], []
    for v, g in sub.groupby("video_id", sort=True):
        vids.append(int(v)); F.append(g.frame_index.to_numpy(int)); S.append(g.score.to_numpy(float))
        KF.append(int(g.keyframe.iloc[0])); LAB.append(g.label.to_numpy(int))
    return {"vids": np.array(vids), "F": F, "S": S, "kf": np.array(KF), "lab": LAB}


# ------------------------------------------------------------------ variants (vectorized)
def _nanmedian_first(X, k=5):
    return np.nanmedian(X[..., :k], axis=-1, keepdims=True)


def v_raw(X):
    return X.copy()


def v_warmup5(X):
    return X - _nanmedian_first(X, 5)


def v_smooth5(X, w=5):
    Z = np.nan_to_num(X, nan=0.0)
    c = np.cumsum(Z, axis=-1)
    c = np.concatenate([np.zeros(c.shape[:-1] + (1,)), c], axis=-1)
    L = X.shape[-1]
    i = np.arange(L)
    a = np.maximum(0, i - w + 1)
    out = (c[..., i + 1] - c[..., a]) / (i + 1 - a)
    out[np.isnan(X)] = np.nan
    return out


def v_runmin(X):
    return X - np.minimum.accumulate(X, axis=-1)


def v_cusum(X, k_frac=0.5, ref_n=5):
    ref = X[..., :ref_n]
    mu = np.nanmedian(ref, axis=-1)
    sd = np.nanstd(ref, axis=-1) + 1e-6
    k = k_frac * sd
    out = np.empty_like(X)
    acc = np.zeros(X.shape[:-1])
    for t in range(X.shape[-1]):
        acc = np.maximum(0.0, acc + (X[..., t] - mu - k))
        out[..., t] = acc
    out[np.isnan(X)] = np.nan
    return out


VARIANTS = {"raw": v_raw, "warmup5": v_warmup5, "smooth5": v_smooth5,
            "runmin": v_runmin, "cusum": v_cusum}


# ------------------------------------------------------------------ variants (reference loops, one series)
def ref_warmup(s, k):
    k = max(1, min(int(k), len(s)))
    return s - float(np.median(s[:k]))


def ref_smooth(s, w):
    """Causal moving average over the last w samples (expanding at the start)."""
    out = np.empty_like(s)
    c = np.cumsum(np.insert(s, 0, 0.0))
    for i in range(len(s)):
        a = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[a]) / (i + 1 - a)
    return out


def ref_runmin(s):
    """Score minus its causal running minimum."""
    return s - np.minimum.accumulate(s)


def ref_cusum(s, k_frac=0.5, ref_n=5):
    """One-sided CUSUM on the score (reference = median/std of the first ref_n samples)."""
    ref = s[:max(1, min(ref_n, len(s)))]
    mu, sd = float(np.median(ref)), float(np.std(ref) + 1e-6)
    k = k_frac * sd
    out = np.empty_like(s); acc = 0.0
    for i, x in enumerate(s):
        acc = max(0.0, acc + (x - mu - k))
        out[i] = acc
    return out


VARIANTS_REF = {"raw": lambda s: s.copy(), "warmup5": lambda s: ref_warmup(s, 5),
                "smooth5": lambda s: ref_smooth(s, 5), "runmin": ref_runmin,
                "cusum": lambda s: ref_cusum(s, 0.5)}


# ------------------------------------------------------------------ harness
def threshold(normal_padded_transformed, fpr):
    v = normal_padded_transformed[~np.isnan(normal_padded_transformed)]
    return float(np.quantile(v, 1.0 - fpr))


def first_alarm(Z, Fpad, thr):
    """Z [..., V, L] (NaN = padding), Fpad [V, L]. Returns the frame of the first alarm (inf = none)."""
    fired = Z >= thr                       # NaN -> False
    anyf = fired.any(axis=-1)
    j = fired.argmax(axis=-1)
    a = np.take_along_axis(np.broadcast_to(Fpad, Z.shape), j[..., None], axis=-1)[..., 0]
    return np.where(anyf, a.astype(float), np.inf)


def alarm_at(Fp, T_):
    """Fixed-time alarm: first grid frame >= T_ in each video (inf if none). Fp [V, L], padding -1."""
    ok = Fp >= T_
    anyf = ok.any(-1)
    a = np.take_along_axis(Fp, ok.argmax(-1)[:, None], -1)[:, 0].astype(float)
    return np.where(anyf, a, np.inf)


def hits_from_alarm(a, kf):
    """a [..., V] frame of the first alarm; kf [V]. Returns hit [..., V, 21] (bool)."""
    a = a[..., None]
    kfb = kf[:, None].astype(float)
    return (a >= kfb - PRE_TOL) & (a <= kfb + KS_F[None, :])


def summarize_hits(hit, a, kf):
    """Harness statistics from hit [V, 21] and alarm a [V]."""
    s_v = hit.mean(-1)
    ik = {1: int(np.where(KS_S == 1.0)[0][0]), 2: int(np.where(KS_S == 2.0)[0][0]),
          4: int(np.where(KS_S == 4.0)[0][0])}
    early = a < kf - PRE_TOL
    return {"S": float(s_v.mean()), "hit1": float(hit[:, ik[1]].mean()),
            "hit2": float(hit[:, ik[2]].mean()), "hit4": float(hit[:, ik[4]].mean()),
            "early": float(early.mean()), "noalarm": float(np.isinf(a).mean())}, s_v


def per_video_components(hit, a, kf):
    """[V, 5] matrix of per-video components (S_v, hit1, hit2, hit4, early), for the bootstrap."""
    ik = [int(np.where(KS_S == x)[0][0]) for x in (1.0, 2.0, 4.0)]
    return np.column_stack([hit.mean(-1), hit[:, ik[0]], hit[:, ik[1]], hit[:, ik[2]],
                            (a < kf - PRE_TOL).astype(float)])


# ------------------------------------------------------------------ within-video AUC (vectorized)
def auc_masks(Fpad, kf, lo=None, hi=None, split=None):
    """[V, L] masks of negatives and positives.

    Default (protocol): neg = [0, kf), pos = [kf, kf+250) (the series already ends at kf+250).
    Placebo: lo=0, split=kf', hi=kf -> neg = [0, kf'), pos = [kf', kf).
    """
    Fv = Fpad.astype(float)
    valid = Fv >= 0
    if split is None:
        neg = valid & (Fv < kf[:, None])
        pos = valid & (Fv >= kf[:, None])
    else:
        neg = valid & (Fv >= lo[:, None]) & (Fv < split[:, None])
        pos = valid & (Fv >= split[:, None]) & (Fv < hi[:, None])
    return neg, pos


def auc_per_video(Z, neg, pos, chunk=16):
    """Per-video Mann-Whitney AUC (ties = 1/2). Z [B, V, L]; neg/pos [V, L].

    Returns [B, V] (NaN where one of the classes is missing)."""
    B, V, L = Z.shape
    M2 = (pos[:, :, None] & neg[:, None, :])            # [V, L(pos), L(neg)]
    n1 = pos.sum(-1); n0 = neg.sum(-1)
    den = (n1 * n0).astype(float)
    out = np.full((B, V), np.nan)
    ok = den > 0
    Zf = np.nan_to_num(Z, nan=0.0)
    for s in range(0, B, chunk):
        z = Zf[s:s + chunk]
        D = z[:, :, :, None] - z[:, :, None, :]          # [b, V, L, L]
        gt = ((D > 0) & M2).sum((-1, -2))
        eq = ((D == 0) & M2).sum((-1, -2))
        u = gt + 0.5 * eq
        out[s:s + chunk][:, ok] = u[:, ok] / den[ok]
    return out


# ------------------------------------------------------------------ bootstrap
def boot_ci(per_video, n_boot=1000, seed=0, stat=np.nanmean):
    """95% percentile bootstrap CI over videos. per_video: [V] or [V, m]."""
    x = np.asarray(per_video, float)
    rs = np.random.RandomState(seed)
    V = x.shape[0]
    idx = rs.randint(0, V, size=(n_boot, V))
    bs = stat(x[idx], axis=1)
    return np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


# ------------------------------------------------------------------ circular shift
def rotate(Xpad, n, shifts):
    """Xpad [V, L], n [V], shifts [B, V] integers in [0, n). Returns [B, V, L].

    The value at position i moves to (i + r) mod n; padding stays NaN."""
    V, L = Xpad.shape
    j = np.arange(L)[None, None, :]
    src = (j - shifts[:, :, None]) % n[None, :, None]
    src = np.where(j < n[None, :, None], src, j)
    return Xpad[np.arange(V)[None, :, None], src]
