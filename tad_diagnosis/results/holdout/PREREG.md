# Holdout experiment: pre-registration

(Called T3-TAD-b in the working notes.) Script: `tad_diagnosis/holdout.py`.

Written on 2026-10-05, BEFORE running. Nothing changes after seeing results; deviations go to the "Deviations" section
of RESULT.md, with the reason.

## Motivation (declared: new analysis, motivated by a descriptive result)

In the official-split experiment (`../official_split/PREREG.md`, `../official_split/RESULT.md`), grumobilenet did not
pass the gate on the official test set: AUC from 0.54 to 0.68. video_control separated the classes on the validation
part of the training set (window AUC from 0.86 to 0.94), but not on the test set (0.62 to 0.65). This test was
motivated by that result and must be reported as a new analysis, with its own pre-registration, and not as part of the
official-split experiment.

**Question:** within the distribution of the TAD training set, where grumobilenet has a chance to work, does its
performance come from the scene or from the event?

## Data and split

- The same per-seed 80/20 split as in the official-split experiment (`rng(s+1)`): 297 training videos and 73
  validation videos, with 22 normals in validation. Seeds 0, 1 and 2; the split changes between seeds.
- The 2 training accidents with K ≤ 10 enter the training of grumobilenet; they do not affect the controls.
- The cache and features of the official-split experiment are reused (`config.TAD_FRAME_CACHE`,
  `config.TAD_FEATURE_CACHE`).

## Models

- **grumobilenet:** same architecture and same training as the grumobilenet of the official-split experiment (6 epochs,
  Adam 1e-3, class-weighted cross-entropy, window label = video label, score = maximum over the windows). Trains only
  on the training side of the split (80%).
- **event_control and video_control:** the heads already trained in the official-split experiment, on the same 80%.
  They are not retrained.

## Measurements on the validation set (20%)

- **Background, background only and frozen first frame:** definitions identical to those of the official-split
  experiment (background = median of rows [0, 10)).
- **Controls:** on the validation accidents with at least 21 rows, marked as in the official-split experiment (cell
  drawn per video, fixed seed). Score = mean of the windows at rows 12, 16 and 20. The background-only and first-frame
  retentions are measured.
- **grumobilenet:**
  - per-video AUC on all validation videos: original, background only and first frame. For videos with fewer than 10
    rows, the background is the median of [0, K).
  - Background-only and first-frame retentions on the same accidents used for the controls (at least 21 rows).
- **Swap background:** does NOT enter the criteria, because the official-split experiment showed that it does not
  separate the controls on TAD. It may appear in the descriptive part, marked as "not validated".
- **CIs:** retention by bootstrap over videos (B = 1000); AUC by class-stratified bootstrap, as in the official-split
  experiment.

## Criteria (thresholds of the SO-TAD interventions, the same as in the official-split experiment)

- **Module valid on the validation set** iff, in at least 2 of the 3 seeds:
  - event_control has background-only retention ≤ 0.3;
  - video_control has background-only retention ≥ 0.8.

  Both measurements are made on the validation accidents. C0 was already measured in the official-split experiment
  (0.993, 0.999 and 0.999).
- **grumobilenet gate:** module valid and per-video validation AUC of grumobilenet ≥ 0.70, by the median of the 3
  seeds.
- **Reading of grumobilenet** (by the median of the 3 seeds, with the values of each seed reported):
  - background-only retention ≥ 0.8 → "scene";
  - ≤ 0.3 → "event";
  - in between → "inconclusive";
  - failed gate → "no diagnosis".

## Descriptive (outside the verdict)

- grumobilenet on the official test set (32 videos): original, background-only and first-frame AUC, to compare with
  the validation set.
- Per-video AUC of video_control (row 0 repeated) on validation and test.
- Composition of the validation set by batch: normals 20220518, 20220620 and 20230818, and accidents.

## Limitations declared in advance

- The motivation came after seeing a result (see above).
- The validation set is not split by camera, and this is on purpose: the test measures what the model uses in the
  distribution it was trained on, which is where a shortcut would operate. "Scene" here means that the
  in-distribution performance comes from the scene. It does not mean that the official test set would be solved by the
  scene.
- The validation set is small: 73 videos per seed (~51 accidents and 22 normals). The CIs will be wide.
- Without a keyframe, the caveats of the official-split experiment about the background containing the accident apply.
  Report how many validation accidents have fewer than 21 rows.

## Execution

- Reuse `official_split.py` by import, without editing it. New script `tad_diagnosis/holdout.py`.
- `tools/run_guarded.sh`, at most 6 threads, cap of 3 h (expected: ~15 min).
- Smoke test only on videos of the training side, without validation or test metrics.
- Outputs in `<out-dir>/<timestamp>/` (recorded run: `tad_diagnosis/results/holdout/<timestamp>/`). Nothing is
  overwritten.
