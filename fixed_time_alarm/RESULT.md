# Fixed-time alarm vs GRUMobileNet (revised temporal harness)

Date: 2026-09-29. Criteria fixed before running (called T4a in the working notes; the pre-registration is a
personal working note that is not included). Total cost: about 2 min of CPU (2 threads, under `run_guarded.sh`).

> **Scope of this record.** The original run evaluated a family of 45 variants (9 models x 5 post-processings)
> from an exploratory phase, and also ran an injection power analysis (T1) and a robustness check. Those models,
> their scores and the T1 / robustness scripts are not part of this repository, so the rows, sections and numbers
> that depend on them (the other 8 models, the family-wise null of the maximum and its p_max, the T1 power / MDE
> tables) were removed from this record. Everything below is computed by the scripts in this folder; the numbers
> are those of the original run.

Paper result (Sec. 5.1): an alarm fixed at **3.18 s** (the median keyframe time of the training accidents), which
does not look at the image, reaches a within-video AUC of **0.822**, above the **0.711** of GRUMobileNet (main
table: row "(ii)" and the GRUMobileNet rows, column "within-video AUC").

## Verdict

- **What decides the result is the trivial null (ii):** an alarm fixed at 3.18 s, the median training keyframe,
  has S = 0.558 [0.498; 0.617]. The right tail of the window (up to tau + 4 s) is wide for the spread of the
  SO-TAD keyframes (IQR 1.3-2.8 s).
- **GRUMobileNet does not "detect the event"** under any of the 5 post-processings: all stay below the strongest
  trivial null (GRUMobileNet warmup5: excess -0.324, paired 95% CI [-0.421; -0.230]).
- **Calibrating on the test regular videos changes no verdict.** See the sensitivity section.

## What was done

1. `train_scores.py` (originally `s01_train_scores.py`): check of the sources and training scores.
   - Running only the GRUMobileNet head over the cached features of the test videos reproduces the test scores
     CSV with a maximum difference of 1.2e-7 (sample of 150 videos: 90 accidents + 60 regular).
   - GRUMobileNet on the training split, cached features vs the decode-based training scores: maximum difference
     of 1.2e-7 on 227 videos.
   - Result: scores of the 200 training regular videos that have cached features.
2. `evaluate.py` (originally `s02_t4a.py`): the harness.
   - Window [tau - 0.5 s, tau + k]; S = mean of hit(k) over k = 0...4 s.
   - The 3 trivial references.
   - 1000 circular shifts, with the same shift for all variants.
   - Placebo with tau' = tau - 3 s.
   - Bootstrap with 1000 resamples.
   - Three calibrations.
   - The vectorized variants were checked against the reference loop implementations (maximum difference < 1e-9).
3. `tables.py` (originally `s04_tables.py`): tables and paired CI of the excess.

## Main table (PRIMARY calibration: 99.9% quantile of the TRAINING regular videos)

The "excess" column is S - S_strongest_null, with a 95% CI from a paired bootstrap over videos. The placebo AUC
uses only n = 9 videos (tau >= 4 s).

| row | S [95% CI] | hit@1s | hit@2s | hit@4s | % 1st alarm < tau-0.5 s | within-video AUC macro [95% CI] | placebo AUC | excess [95% CI] | realized FA/frame (test) |
|---|---|---|---|---|---|---|---|---|---|
| (i) first frame | 0.022 [0.000; 0.056] | 0.022 | 0.022 | 0.022 | 97.8 | 0.500 | 0.500 | -0.536 | - |
| **(ii) median training keyframe, 3.18 s = strongest null** | **0.558 [0.498; 0.617]** | 0.289 | 0.678 | 0.844 | 14.4 | 0.822 [0.801; 0.844] | 0.803 | - | - |
| (iii) clock T* = 3.60 s (chosen on the training set) | 0.508 [0.452; 0.566] | 0.178 | 0.544 | 0.856 | 10.0 | 0.780 [0.754; 0.806] | 0.762 | -0.050 [-0.098; 0.004] | - |
| GRUMobileNet warmup5 | 0.234 [0.159; 0.317] | 0.200 | 0.244 | 0.289 | 57.8 | 0.711 [0.647; 0.774] | 0.724 [0.532; 0.889] | -0.324 [-0.421; -0.230] | 2.05% |
| GRUMobileNet raw | 0.020 [0.000; 0.051] | 0.022 | 0.022 | 0.022 | 97.8 | 0.711 [0.647; 0.774] | 0.724 | -0.538 [-0.602; -0.470] | 3.83% |

- Used as a score, the clock (t) has a within-video AUC of 1.000. The AUC remains gameable; S does not.
- Without correction for the number of variants, GRUMobileNet warmup5 is above its own circular null (p = 0.004).
- All values (the 5 post-processings, the three calibrations) are in `results/variants.csv` and
  `results/variants_all_calibrations.csv`.

## Sensitivity A: threshold on the TEST regular videos (99.9%), requested after the first results

Every variant is ineligible under the pre-registration. This is only a sensitivity analysis.

| row | S [95% CI] | hit@1s | hit@2s | hit@4s | % < tau-0.5 s | AUC [95% CI] | excess [95% CI] |
|---|---|---|---|---|---|---|---|
| strongest null (ii) | 0.558 [0.498; 0.617] | 0.289 | 0.678 | 0.844 | 14.4 | 0.822 | - |
| GRUMobileNet warmup5 | 0.201 [0.130; 0.273] | 0.167 | 0.233 | 0.267 | 12.2 | 0.711 | -0.357 [-0.444; -0.265] |
| GRUMobileNet raw | 0.018 | 0.011 | 0.022 | 0.022 | 96.7 | 0.711 | -0.540 |

**The verdict does not change.** What changes is the mechanism: with the test threshold, the early alarms of
GRUMobileNet warmup5 drop from 57.8% to 12.2%. In exchange, the threshold rises (0.064 vs 0.006) and most videos
get no alarm.

## Sensitivity B (pre-registered, report only): threshold at 99% of the training regular videos

- GRUMobileNet warmup5: S = 0.133, with 86.7% early alarms.

## Caveat: "training regular videos seen in training"

- The primary threshold is calibrated on 200 training regular videos, which GRUMobileNet saw during training.
- So the threshold is optimistic, and the realized FA on the test regular videos exceeds the nominal 0.1% per
  frame:
  - GRUMobileNet warmup5: 2.05% per frame (7.2% of the videos);
  - GRUMobileNet raw: 3.83%.
- This is the direct cause of the 56-58% early alarms of the primary calibration. The verdict does not depend on
  it: with the test threshold the conclusion is the same.

## Deviations from the pre-registration

1. **DEVIATION (minor).** The calibration uses 200 of the 1,515 training regular videos, the sample that has
   cached features (random.Random(7)). The pre-registration did not fix the subset.
2. **Addition not pre-registered**, requested after the first results: the sensitivity with the threshold of the
   test regular videos.
3. **Points the pre-registration did not fix**, chosen before running:
   - uniform shift in {0...n-1};
   - S = simple mean of the 21 points of k.
4. **Limitation, not a deviation.** Only 9 of the 90 videos have tau >= 4 s, so the placebo is weak (wide CIs).

## Files (`fixed_time_alarm/`)

- Scripts: `metrics.py`, `train_scores.py`, `evaluate.py`, `tables.py`.
- Intermediate data (not versioned): pooled features in `config.SOTAD_FEATURE_CACHE`, training scores in
  `config.SOTAD_TRAIN_SCORES`.
- `results/train_scores_check.json` (checks of `train_scores.py`).
- `results/variants.csv` (primary calibration, wide format), `results/variants_all_calibrations.csv` (3
  calibrations), `results/reference_predictors.csv`, `results/summary.json`.
- Tables: `results/tables.md` (includes the realized FA of the 5 post-processings) and
  `results/table_main_train999.csv`.
