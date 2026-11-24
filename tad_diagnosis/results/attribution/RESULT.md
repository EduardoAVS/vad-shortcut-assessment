# Attribution experiment: result (Grad-CAM attribution on TAD, without keyframe)

(Called T3-TAD-c in the working notes.) Run on 2026-10-05, following `PREREG.md` (in this folder), which is binding. No
criterion or threshold was changed.

**It is a new analysis**, requested after the holdout experiment, and must be reported as such.

- **Run:** `tad_diagnosis/results/attribution/20261005-120205/` (`summary.json`, `table.md`, `seed{0,1,2}/`).
- **Script:** `tad_diagnosis/attribution.py`. It imports `official_split.py`, `holdout.py` and the functions of the
  SO-TAD controls (`sotad_diagnosis/controls.py`), without editing anything.
- **Smoke test:** smoke run 20261005-120138-smoke (not included), only with videos of the train side, with no validation
  metric.

## Verdict by the pre-registration

- **Instrument valid: yes, 3 of 3 seeds.** event_control has a Grad-CAM lift of 9.71, 9.94 and 10.62 on the mark cell;
  the threshold is ≥ 2.0.
- **Attribution evidence that grumobilenet looks at the motion: no, 0 of 3 seeds** (2 required). In no seed does the
  grumobilenet lift exceed both nulls with the 95% CI of the paired difference > 0.
- **Attribution reading of grumobilenet:** "does not look at the motion".
- **Reading combined with the holdout experiment** (decision = "scene"): **"no dissociation on TAD: attribution agrees
  with the cause".**

On SO-TAD, the Grad-CAM of the real models concentrated mass on the motion regions (lift ~1.9 against a null of ~1.0),
while the decision came from the scene. This dissociation **does not repeat on TAD**:
- grumobilenet has a lift of ~1.0–1.2 on the motion region;
- the two nulls, which do not look at the event, are at the same level.

## Consistency checks (all exact)

In the 3 seeds:
- The set of validation videos is the same as in the holdout experiment.
- The grumobilenet scores (max over windows) and the video_control scores (row 0) reproduce the CSVs of the holdout
  experiment, with a maximum |Δ| of 0.0.
- The backbone maps, averaged over space, reproduce the cached features, with a maximum |Δ| of 0.0.

## Table (Grad-CAM; validation accidents with K ≥ 21)

Mean lift per video, with 95% CI by bootstrap over videos (B = 1000).
- grumobilenet and nulls: lift on the motion region.
- event_control: lift on the mark cell, in the marked clip.
- Differences: paired by video.

| Seed | n | event_control (mark cell) | grumobilenet | random_head | video_control | grumobilenet − random_head | grumobilenet − video_control | Evidence |
|---|---|---|---|---|---|---|---|---|
| 0 | 51 | 9.71 [9.26, 10.15] | 1.06 [0.90, 1.24] | 1.18 [1.06, 1.31] | 1.12 [0.95, 1.31] | −0.12 [−0.32, 0.06] | −0.05 [−0.15, 0.04] | no |
| 1 | 51 | 9.94 [9.59, 10.32] | 1.18 [1.06, 1.30] | 1.06 [0.94, 1.18] | 1.24 [1.12, 1.39] | 0.12 [−0.04, 0.29] | −0.07 [−0.21, 0.07] | no |
| 2 | 50 | 10.62 [10.17, 11.05] | 0.98 [0.85, 1.11] | 0.92 [0.83, 1.02] | 1.12 [1.00, 1.25] | 0.06 [−0.07, 0.20] | −0.15 [−0.24, −0.05] | no |

In seed 2, grumobilenet puts **less** mass on the motion region than video_control (whole CI below 0). In the other
seeds, the differences contain 0.

## Descriptive (outside the verdict)

**Grad-CAM++ and LayerCAM** (same accidents and same regions):

| Seed | Method | event_control | grumobilenet | random_head | video_control | grumobilenet − random_head | grumobilenet − video_control |
|---|---|---|---|---|---|---|---|
| 0 | Grad-CAM++ | 5.55 [5.08, 6.05] | 1.16 [1.03, 1.30] | 1.21 [1.11, 1.32] | 1.22 [1.09, 1.36] | −0.05 [−0.10, 0.00] | −0.06 [−0.08, −0.03] |
| 0 | LayerCAM | 5.05 [4.61, 5.50] | 1.17 [1.05, 1.30] | 1.20 [1.10, 1.31] | 1.21 [1.08, 1.34] | −0.03 [−0.08, 0.01] | −0.04 [−0.06, −0.02] |
| 1 | Grad-CAM++ | 5.69 [5.35, 6.06] | 1.19 [1.09, 1.30] | 1.17 [1.08, 1.28] | 1.22 [1.13, 1.33] | 0.02 [−0.02, 0.06] | −0.03 [−0.06, 0.00] |
| 1 | LayerCAM | 5.31 [4.99, 5.65] | 1.19 [1.09, 1.30] | 1.17 [1.08, 1.27] | 1.22 [1.12, 1.32] | 0.02 [−0.01, 0.05] | −0.03 [−0.06, −0.00] |
| 2 | Grad-CAM++ | 5.28 [4.79, 5.81] | 1.15 [1.04, 1.27] | 1.14 [1.04, 1.24] | 1.17 [1.06, 1.30] | 0.01 [−0.03, 0.06] | −0.02 [−0.06, 0.01] |
| 2 | LayerCAM | 5.02 [4.56, 5.52] | 1.15 [1.04, 1.27] | 1.13 [1.04, 1.24] | 1.17 [1.06, 1.28] | 0.02 [−0.02, 0.05] | −0.02 [−0.05, 0.01] |

The three methods agree:
- event_control is detected by a wide margin (lift of ~5 to ~10);
- grumobilenet stays at the level of the nulls (~1.15–1.2);
- when the difference to video_control has a CI that excludes 0, it is **negative**.

**Validation normals** (Grad-CAM, motion region) and **mean area fraction of the region**:

| Seed | n normals | grumobilenet | random_head | video_control | Region area (accidents) | Region area (normals) |
|---|---|---|---|---|---|---|
| 0 | 22 | 1.56 [1.27, 1.88] | 1.28 [1.05, 1.50] | 1.62 [1.24, 2.06] | 0.34 [0.31, 0.36] | 0.29 [0.25, 0.33] |
| 1 | 22 | 1.22 [0.99, 1.49] | 1.24 [0.96, 1.56] | 1.52 [1.25, 1.84] | 0.32 [0.30, 0.35] | 0.28 [0.25, 0.32] |
| 2 | 22 | 0.92 [0.79, 1.05] | 0.92 [0.75, 1.10] | 1.05 [0.88, 1.21] | 0.32 [0.30, 0.34] | 0.30 [0.28, 0.33] |

- The motion region covers about 1/3 of the image (5 of 16 cells, on average).
- On the normals, grumobilenet does not stand out from the nulls in any seed.

## Reading and limits

- On TAD, attribution and the causal test say the same thing: grumobilenet does not rely on what moves.
  - The causal test of the holdout experiment found "scene".
  - Grad-CAM does not concentrate mass on the motion above the nulls.
- This contrasts with SO-TAD, where attribution alone would have misled (lift ~1.9). The attribution × cause
  dissociation seen there **is not universal**: it depends on the case.
- Limitations declared in the pre-registration:
  - without a keyframe, the motion region includes normal traffic, so "looking at the motion" here is not "looking at
    the accident". The test is looser than the SO-TAD one, and even so grumobilenet does not pass;
  - the CAM has only 4×4 cells;
  - the validation set is small (~51 accidents per seed).
- The nulls are slightly above 1 (~0.9–1.25). With 4×4 cells, even a random head does not spread mass uniformly. That
  is why the comparison is with the nulls and not with 1.

## CPU time

| Step | CPU (s) | Wall (s) |
|---|---|---|
| Smoke (9 videos of the train side) | 18 | 3 |
| Full run (3 seeds) | 389 | 69 |
| **Total of the attribution experiment** | **~407 (~7 min, of the 3 h cap)** | |

Configuration: 6 threads and `tools/run_guarded.sh` with a 2500 MB limit.

## Deviations

**No criterion or threshold was changed, and there was no implementation deviation.** Below, the decisions on points
the pre-registration does not fix:

1. **Windows:** `unique(round(linspace(15, K−1, 8)))`. Short videos have fewer than 8 windows (K = 21 gives 6). All end
   at row 15 or later.
2. **"Windows with the mark" of event_control:** the same windows as grumobilenet, in the clip marked on rows ≥ 10 (cell
   `rng([s, k])`, as in the official-split experiment). All contain marked rows, because they end at r ≥ 15.
3. **random_head:** `torch.manual_seed(s)` followed by `Head()`, untrained.
4. **Motion region:** computed on the clean clip of the cache (128 px), with gray = mean of the RGB channels, as in the
   SO-TAD controls.
5. **CI of the paired difference:** `controls.agg`, percentile bootstrap over videos with B = 1000 and seed 0, as in the
   SO-TAD controls and interventions.
6. **Check (iii)** (scores = CSVs of the holdout experiment): done only in the full run, so that the smoke run computed
   nothing on the validation set.
7. **Bytecode:** the script disabled writing `__pycache__` so as not to create files outside its folder. The existing
   `__pycache__` of the holdout folder was from 11:06, from the holdout run. (The current script no longer does this;
   `__pycache__` is git-ignored.)
