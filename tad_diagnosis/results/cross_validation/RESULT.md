# Cross-validation experiment: result (5-fold cross-validation of the diagnosis on TAD)

(Called T3-TAD-d in the working notes.) Run on 2026-10-05, following `PREREG.md` (in this folder), which is binding. No
criterion or threshold was changed.

**It is a refinement of the holdout experiment on the same data**, so it is not independent of it. It was requested by
the user.

- **Run:** `tad_diagnosis/results/cross_validation/20261005-120959/` (`summary.json`, `table.md`,
  `test_mean_of_models.csv`, `fold{0..4}/`).
- **Script:** `tad_diagnosis/cross_validation.py`. It imports `official_split.py` and `holdout.py` without editing
  anything.
- **Smoke test and check:** smoke run 20261005-120738-smoke (not included), only on the training folds of fold 0, with no
  out-of-fold metric.

## Verdict by the pre-registration

Out-of-fold aggregation: 370 videos, each diagnosed once; retentions measured on 253 accidents with K ≥ 21.

| Criterion | Value | Threshold | Passes |
|---|---|---|---|
| C0 (pooled event_control window AUC) | 0.994 | ≥ 0.90 | yes |
| event_control: background-only retention | 0.11 [0.08, 0.15] | ≤ 0.3 | yes |
| video_control: background-only retention | 0.95 [0.91, 0.98] | ≥ 0.8 | yes |
| Gate: grumobilenet per-video AUC | 0.92 [0.89, 0.95] | ≥ 0.70 | yes |
| **Reading: grumobilenet background-only retention** | **0.90 [0.87, 0.94]** | ≥ 0.8 → "scene" | **"scene"** |

- **Module valid** (without C0, the literal version gives the same). **Gate passes.**
- **grumobilenet reading: "scene".**
- As descriptive: **the whole 95% CI lies above 0.8** (lower bound 0.87).

The result confirms the holdout experiment with more precision:
- in the holdout experiment, the median over seeds was AUC 0.938 and retention 0.94, with retention CIs of width 0.06
  to 0.16 per seed;
- here, with every video diagnosed, the retention is 0.90 with a CI of width 0.07.

Within the distribution of the TAD training set, one static background image per video keeps 90% of the grumobilenet
score on the accidents. With it, the per-video AUC is 0.88, against 0.92 for the original.

## Consistency check (in the smoke run)

With the training code of this script, I retrained the seed-0 heads on the 80/20 split of the official-split and
holdout experiments. The weights were reproduced **bit for bit** (maximum |Δ| = 0.0):
- the event_control and video_control saved by the official-split experiment;
- the grumobilenet saved by the holdout experiment.

The full run only runs if this check passed. It confirms that mark, labels, features and training are identical to those
of the official-split and holdout experiments.

## Table

Per-video AUC, with 95% CI by class-stratified bootstrap (B = 1000). C0 is the window AUC.

| Scope | n videos | C0 | AUC grumobilenet original | AUC grumobilenet background only | AUC grumobilenet first frame | AUC video_control (row 0) |
|---|---|---|---|---|---|---|
| **pooled** | 370 | 0.994 | 0.92 [0.89, 0.95] | 0.88 [0.84, 0.92] | 0.89 [0.85, 0.93] | 0.88 [0.84, 0.92] |
| fold 0 | 74 | 0.996 | 0.91 [0.83, 0.97] | 0.86 [0.77, 0.95] | 0.89 [0.81, 0.96] | 0.87 [0.79, 0.94] |
| fold 1 | 74 | 0.996 | 0.94 [0.86, 0.99] | 0.90 [0.79, 0.98] | 0.91 [0.83, 0.97] | 0.93 [0.85, 0.97] |
| fold 2 | 74 | 0.997 | 0.90 [0.82, 0.97] | 0.83 [0.71, 0.94] | 0.85 [0.73, 0.94] | 0.87 [0.78, 0.94] |
| fold 3 | 74 | 0.997 | 0.94 [0.86, 0.99] | 0.93 [0.83, 0.99] | 0.92 [0.82, 0.98] | 0.89 [0.77, 0.97] |
| fold 4 | 74 | 0.995 | 0.93 [0.85, 0.99] | 0.89 [0.78, 0.98] | 0.88 [0.77, 0.96] | 0.85 [0.74, 0.94] |

Retentions on the accidents with K ≥ 21: mean with intervention / original mean, with 95% CI by bootstrap over videos
(B = 1000).

| Scope | Model | n | Original score | Ret. background only | Ret. first frame |
|---|---|---|---|---|---|
| **pooled** | event_control | 253 | 0.98 [0.96, 0.99] | 0.11 [0.08, 0.15] | 0.11 [0.07, 0.14] |
| **pooled** | video_control | 253 | 0.90 [0.87, 0.94] | 0.95 [0.91, 0.98] | 1.00 [0.98, 1.01] |
| **pooled** | grumobilenet | 253 | 0.98 [0.96, 0.99] | 0.90 [0.87, 0.94] | 0.93 [0.90, 0.96] |
| fold 0 | event_control | 49 | 0.99 [0.99, 1.00] | 0.10 [0.02, 0.21] | 0.06 [0.00, 0.14] |
| fold 0 | video_control | 49 | 0.86 [0.77, 0.95] | 0.95 [0.87, 1.04] | 1.00 [0.95, 1.05] |
| fold 0 | grumobilenet | 49 | 0.96 [0.90, 1.00] | 0.81 [0.70, 0.92] | 0.90 [0.81, 0.98] |
| fold 1 | event_control | 51 | 0.96 [0.91, 0.99] | 0.06 [0.02, 0.12] | 0.15 [0.07, 0.24] |
| fold 1 | video_control | 51 | 0.93 [0.85, 0.99] | 0.93 [0.84, 1.01] | 1.01 [1.00, 1.05] |
| fold 1 | grumobilenet | 51 | 0.98 [0.94, 1.00] | 0.94 [0.86, 1.00] | 0.97 [0.93, 1.00] |
| fold 2 | event_control | 52 | 0.98 [0.95, 1.00] | 0.10 [0.02, 0.20] | 0.12 [0.04, 0.22] |
| fold 2 | video_control | 52 | 0.88 [0.79, 0.96] | 0.96 [0.88, 1.02] | 0.98 [0.95, 1.00] |
| fold 2 | grumobilenet | 52 | 0.98 [0.94, 1.00] | 0.88 [0.79, 0.96] | 0.86 [0.76, 0.94] |
| fold 3 | event_control | 50 | 0.98 [0.95, 1.00] | 0.11 [0.04, 0.21] | 0.05 [0.00, 0.11] |
| fold 3 | video_control | 50 | 0.92 [0.85, 0.98] | 0.96 [0.88, 1.03] | 1.02 [0.99, 1.07] |
| fold 3 | grumobilenet | 50 | 1.00 [1.00, 1.00] | 0.92 [0.84, 0.98] | 0.98 [0.93, 1.00] |
| fold 4 | event_control | 51 | 0.96 [0.92, 1.00] | 0.16 [0.07, 0.25] | 0.15 [0.06, 0.25] |
| fold 4 | video_control | 51 | 0.92 [0.84, 0.98] | 0.94 [0.87, 1.00] | 0.98 [0.94, 1.00] |
| fold 4 | grumobilenet | 51 | 0.96 [0.91, 1.00] | 0.95 [0.88, 1.00] | 0.93 [0.86, 0.99] |

The criteria also hold fold by fold:
- event_control: 0.06–0.16;
- video_control: 0.93–0.96;
- grumobilenet: 0.81–0.95, always in the "scene" range.

## Descriptive (outside the verdict)

**Per batch of normals.** The AUC compares all 259 diagnosed accidents with the normals of the batch. The retention is
computed on the normals of the batch: background only / original and first frame / original.

| Batch | n normals | AUC original | AUC background only | AUC first frame | Ret. background only on normals | Ret. first frame on normals |
|---|---|---|---|---|---|---|
| 20220518 | 31 | 0.87 [0.79, 0.93] | 0.83 [0.73, 0.91] | 0.83 [0.75, 0.91] | 0.67 [0.42, 0.88] | 0.74 [0.51, 0.93] |
| 20220620 | 21 | 0.92 [0.85, 0.97] | 0.89 [0.77, 0.97] | 0.91 [0.85, 0.96] | 0.56 [0.20, 0.89] | 0.67 [0.33, 1.00] |
| 20230818 | 59 | 0.95 [0.91, 0.98] | 0.91 [0.86, 0.95] | 0.91 [0.86, 0.95] | 0.53 [0.28, 0.80] | 0.77 [0.57, 0.94] |

- All three batches are separated from the accidents by a static image (background-only AUC of 0.83 to 0.91). The
  shortcut does not depend on a single batch.
- On the normals, the background **lowers** the score (retention of 0.53 to 0.67). The original score of the normals is
  the max over windows, and some windows fire on what moves. The static background removes these peaks, which makes the
  normals even more separated from the accidents.

**Official test set (32 videos), mean of the 5 grumobilenet models** (mean of the per-video scores):

| AUC original | AUC background only | AUC first frame | AUC video_control (mean of the 5) |
|---|---|---|---|
| 0.67 [0.46, 0.87] | 0.66 [0.45, 0.84] | 0.60 [0.39, 0.80] | 0.65 [0.44, 0.84] |

The pattern of the official-split and holdout experiments repeats: AUC ~0.9 within the training distribution, ~0.65 on
the official test set, where the scene (video_control 0.65) does not carry the label in the same way.

**Diagnosed accidents with fewer than 21 rows:** 6 of 259. They are left out of the retentions, but enter the AUC.

## Limitations (declared in the pre-registration)

- It uses the same data as the holdout experiment and is not independent of it. The folds do not separate cameras, on
  purpose: the test measures what the model uses within the training distribution.
- "Scene" means that the performance within the distribution comes from the scene. It does not mean that the official
  test set would be solved by the scene.
- It has a single seed. The variation across seeds was seen in the holdout experiment, with reading "scene" in all 3.
- Without a keyframe, the caveats about the background of [0, 10) containing part of the accident apply, a bias in favor
  of "scene".

## CPU time

| Step | CPU (s) | Wall (s) |
|---|---|---|
| Smoke + consistency check (3 retrainings of 6 epochs) | 655 | 123 |
| Full run (5 folds × 3 heads + measurements) | 3,315 | 584 |
| **Total of the cross-validation experiment** | **~3,970 (~66 min, of the 3 h cap)** | |

Configuration: 6 threads and `tools/run_guarded.sh` with a 2500 MB limit.

## Deviations

**No criterion or threshold was changed, and there was no implementation deviation.** Below, the decisions on points
the pre-registration does not fix:

1. **Folds:** scikit-learn `StratifiedKFold(5, shuffle=True, random_state=0)`, over the 370 videos sorted by name. There
   are 5 folds of 74 videos, with 51, 52, 52, 52 and 52 accidents.
2. **"Seed 0 for the heads":** uses the scheme of the official-split experiment with s = 0, the same in all folds:
   event_control 0 + 2, video_control 0 + 3, grumobilenet 0 + 4. The mark is the seed-0 one (cell `rng([0, k])`,
   rows ≥ 10), read from `TAD_FEATURE_CACHE/marked_s0/`.
3. **C0 in the module validity.** It was required, as in the official-split experiment. The literal version, with only
   the two retentions, gives the same result.
4. **Descriptive per batch.**
   - AUC: all diagnosed accidents against the normals of the batch.
   - Retention: computed on the normals of the batch, because the pre-registration does not define retention for
     normals.
5. **Test with the mean of the 5 grumobilenet models:** first the score of each model (max over windows), then the mean
   of the 5. The background-only, first-frame and mean-video_control AUCs are descriptive extras.
6. **Pooled video_control AUC (row 0):** descriptive extra, not requested.
7. **Consistency check in the smoke run** (bit-for-bit retraining of the seed-0 heads). The full run requires it to have
   passed. In the `table.md` of the smoke run, the "Official test" row is a pseudo-test with videos of the training
   folds.
8. **Bytecode:** the script disabled writing `__pycache__` so as not to create files outside its folder. (The current
   script no longer does this; `__pycache__` is git-ignored.)
