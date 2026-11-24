# Cross-validation experiment: cross-validation of the diagnosis on TAD — pre-registration

(Called T3-TAD-d in the working notes.) Script: `tad_diagnosis/cross_validation.py`.

Written on 2026-10-05, BEFORE running. Deviations go to the "Deviations" section of RESULT.md, with the reason.

**Motivation (declared):** the holdout experiment read grumobilenet (the GRUMobileNet head trained on TAD) as "scene"
with only 73 validation videos per seed. This test repeats the diagnosis so that every training video is diagnosed once,
to gain precision. It uses the same data and is not independent of the holdout experiment. It is a refinement, requested
by the user.

## Design

- **Folds:** class-stratified cross-validation, with 5 folds over the 370 training videos. The 2 accidents with K ≤ 10
  always stay in training and are never diagnosed. Seed 0 for the split and for the heads: a single repetition, because
  the variation across seeds was already shown in the holdout experiment.
- **Models per fold**, trained on the 4 training folds (for grumobilenet, plus the 2 accidents with K ≤ 10), with
  architecture, training, mark and labels identical to those of the official-split and holdout experiments:
  - event_control;
  - video_control;
  - grumobilenet.
- **Measurements on the held-out fold:** scores and interventions identical to those of the holdout experiment
  (background = median of rows [0, 10); background only; frozen first frame). Swap background is not included.
- **Aggregation:** the measurements are made out of fold and then pooled over all diagnosed videos.
  - retentions: accidents with at least 21 rows;
  - AUC: all 370 videos;
  - CIs by bootstrap over videos (B = 1000); for the AUC, stratified by class.

## Criteria (the same thresholds)

- **C0:** event_control window AUC, pooled out of fold, ≥ 0.90.
- **Module valid** iff, pooling out of fold:
  - event_control has background-only retention ≤ 0.3;
  - video_control has background-only retention ≥ 0.8.
- **Gate:** grumobilenet per-video AUC, pooled out of fold, ≥ 0.70.
- **grumobilenet reading,** by the pooled background-only retention:
  - ≥ 0.8 → "scene";
  - ≤ 0.3 → "event";
  - in between → "inconclusive";
  - failed gate or module → "no diagnosis".

  Also report, only as descriptive, whether the whole 95% CI lies above 0.8.

## Descriptive (outside the verdict)

- Per-fold values.
- Pooled background-only and first-frame AUC.
- Retention and AUC per batch of normals (20220518, 20220620, 20230818).
- AUC on the official test set (32 videos) of the mean of the 5 grumobilenet models.

## Limitations declared in advance

- It uses the same data as the holdout experiment, and the split does not separate cameras (on purpose, as in the
  holdout experiment).
- It has a single seed.

## Execution

- Reuse `official_split.py` and `holdout.py` by import, without editing anything.
- `tools/run_guarded.sh`, at most 6 threads, cap of 3 h (expected: less than 1 h).
- Smoke test only on training folds, with no out-of-fold metric.
- Outputs in `tad_diagnosis/results/cross_validation/<timestamp>/`. Nothing is overwritten.
