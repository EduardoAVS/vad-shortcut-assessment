# Holdout experiment: result (diagnosis of grumobilenet on the 80/20 validation split of the TAD training set)

(Called T3-TAD-b in the working notes.) Run on 2026-10-05, following `tad_diagnosis/results/holdout/PREREG.md`, which
is binding. No criterion or threshold was changed.

**This is a new test**, motivated by a descriptive result of the official-split experiment (video_control separates
the classes on the validation set, but not on the test set). It must be reported separately, and not as part of the
official-split experiment.

- **Valid run:** `tad_diagnosis/results/holdout/20261005-110712/` (`summary.json`, `table.md`, `seed{0,1,2}/`).
- **Script:** `tad_diagnosis/holdout.py`. Imports `official_split.py` without editing it. Reads
  `config.TAD_FRAME_CACHE`, `config.TAD_FEATURE_CACHE` and the event_control and video_control heads of
  `tad_diagnosis/results/official_split/20261005-101615/seed<s>/`.
- **Smoke tests:** smoke runs 20261005-110441-smoke and 20261005-110658-smoke (not included), only with videos of the
  training side of the split, without any validation or test metric.
- **Aborted run:** 20261005-110508 (not included; see Deviations, item 1).

## Summary

- **Split reproduced exactly.** In each seed: 297 control training videos, 73 validation videos and 299 grumobilenet
  training videos. The C0 of the saved event_control head, recomputed on this validation set, matches the
  official-split experiment to the 6th decimal place: 0.993197, 0.998794 and 0.998820.
- **Module valid on the validation set: yes, 3 of 3 seeds.**
  - Background-only retention of event_control: 0.08, 0.06 and 0.01.
  - Background-only retention of video_control: 0.99, 0.96 and 0.97.
- **grumobilenet gate: passes.** The per-video validation AUC was 0.89, 0.96 and 0.94 (median 0.938 ≥ 0.70).
- **Reading of grumobilenet: "scene"** by the median and in each of the 3 seeds. The background-only retention was
  0.94, 0.89 and 0.98 (median 0.94 ≥ 0.8).

  Within the distribution of the TAD training set, a single static background image per video keeps 89–98% of the
  model's score on the accidents. With it, the per-video AUC is 0.87–0.92, against 0.89–0.96 for the original.

## 1. Validity of the module on the validation set, per seed

Rule: valid iff, in at least 2 of the 3 seeds, event_control has background-only retention ≤ 0.3 and video_control
has background-only retention ≥ 0.8, measured on the validation accidents with K ≥ 21. C0 had already been measured in
the official-split experiment and was only rechecked here.

| Seed | Measured accidents | event_control: ret. background only ≤ 0.3 | video_control: ret. background only ≥ 0.8 | Seed valid |
|---|---|---|---|---|
| 0 | 51 | 0.08 [0.02, 0.16] ✓ | 0.99 [0.91, 1.07] ✓ | yes |
| 1 | 51 | 0.06 [0.00, 0.14] ✓ | 0.96 [0.89, 1.03] ✓ | yes |
| 2 | 50 | 0.01 [0.00, 0.02] ✓ | 0.97 [0.89, 1.02] ✓ | yes |

**Module valid on the validation set (3/3).**

## 2. Gate and reading of grumobilenet

- **Gate** (module valid and per-video validation AUC of grumobilenet ≥ 0.70, by the median): valid, with AUC of 0.89,
  0.96 and 0.94 and median 0.938. **Passes.**
- **Background-only retention of grumobilenet** on the same accidents as the controls: 0.94 [0.87, 1.00],
  0.89 [0.80, 0.96] and 0.98 [0.94, 1.00]. The median is 0.94.
- **Reading: "scene"** (≥ 0.8), in the median and in each seed.

Scope of the reading, declared in the pre-registration:
- "Scene" means that the performance of grumobilenet **within the distribution of the training set** (validation not
  split by camera, on purpose) comes from the scene.
- It does **not** mean that the official test set would be solved by the scene. On the test set, video_control stays
  at 0.62–0.65 (section 4).

## 3. Table per seed (validation: 73 videos, 51 accidents and 22 normals)

Per-video AUC, with a 95% CI by class-stratified bootstrap (B = 1000):

| Seed | n val (acc./norm.) | AUC grumobilenet original | AUC grumobilenet background only | AUC grumobilenet first frame | AUC video_control (row 0) |
|---|---|---|---|---|---|
| 0 | 73 (51/22) | 0.89 [0.81, 0.97] | 0.87 [0.76, 0.95] | 0.90 [0.81, 0.96] | 0.87 [0.77, 0.96] |
| 1 | 73 (51/22) | 0.96 [0.92, 0.99] | 0.92 [0.85, 0.98] | 0.94 [0.89, 0.98] | 0.93 [0.85, 0.99] |
| 2 | 73 (51/22) | 0.94 [0.86, 0.99] | 0.90 [0.80, 0.97] | 0.91 [0.82, 0.98] | 0.91 [0.82, 0.98] |

Retentions on the validation accidents with K ≥ 21: mean with intervention / original mean, with a 95% CI by bootstrap
over videos (B = 1000).
- Control score: mean of windows 12, 16 and 20 of the marked clip.
- grumobilenet score: maximum over the windows of the clip without the mark.

| Seed | Model | n | Original score | Ret. background only | Ret. first frame |
|---|---|---|---|---|---|
| 0 | event_control | 51 | 0.97 [0.94, 1.00] | 0.08 [0.02, 0.16] | 0.06 [0.00, 0.14] |
| 0 | video_control | 51 | 0.89 [0.81, 0.95] | 0.99 [0.91, 1.07] | 1.02 [0.97, 1.06] |
| 0 | grumobilenet | 51 | 0.96 [0.90, 1.00] | 0.94 [0.87, 1.00] | 0.98 [0.93, 1.00] |
| 1 | event_control | 51 | 0.99 [0.98, 1.00] | 0.06 [0.00, 0.14] | 0.06 [0.00, 0.13] |
| 1 | video_control | 51 | 0.92 [0.85, 0.97] | 0.96 [0.89, 1.03] | 1.01 [0.96, 1.06] |
| 1 | grumobilenet | 51 | 0.97 [0.92, 1.00] | 0.89 [0.80, 0.96] | 0.91 [0.83, 0.97] |
| 2 | event_control | 50 | 0.96 [0.91, 1.00] | 0.01 [0.00, 0.02] | 0.02 [0.00, 0.05] |
| 2 | video_control | 50 | 0.89 [0.81, 0.96] | 0.97 [0.89, 1.02] | 1.01 [0.96, 1.06] |
| 2 | grumobilenet | 50 | 0.98 [0.94, 1.00] | 0.98 [0.94, 1.00] | 0.96 [0.90, 1.00] |

## 4. Descriptive (outside the verdict)

**grumobilenet and video_control on the official test set** (32 videos, 16 + 16):

| Seed | AUC grumobilenet original | AUC grumobilenet background only | AUC grumobilenet first frame | AUC video_control (row 0) |
|---|---|---|---|---|
| 0 | 0.66 [0.45, 0.85] | 0.65 [0.43, 0.83] | 0.60 [0.39, 0.79] | 0.65 [0.44, 0.84] |
| 1 | 0.71 [0.50, 0.89] | 0.66 [0.46, 0.83] | 0.64 [0.43, 0.83] | 0.64 [0.43, 0.84] |
| 2 | 0.64 [0.43, 0.83] | 0.68 [0.47, 0.86] | 0.61 [0.41, 0.80] | 0.62 [0.41, 0.82] |

- **Validation × test:**
  - grumobilenet original: 0.89–0.96 on validation, 0.64–0.71 on test;
  - video_control: 0.87–0.93 on validation, 0.62–0.65 on test.

  The performance of grumobilenet drops on the test set together with that of video_control, and on the test set it is
  at the same level as the grumobilenet trained on 100% of the training set (official-split experiment: 0.54–0.68).
- **AUC of video_control (row 0 repeated):** validation 0.87, 0.93 and 0.91; test 0.65, 0.64 and 0.62.
- **Composition of the validation set by batch** (the split changes between seeds):

  | Seed | Accidents | Normals 20220518 | Normals 20220620 | Normals 20230818 |
  |---|---|---|---|---|
  | 0 | 51 | 4 | 4 | 14 |
  | 1 | 51 | 5 | 4 | 13 |
  | 2 | 51 | 7 | 3 | 12 |

- **Swap background:** not computed. The pre-registration leaves it optional and not validated on TAD.

## 5. Validation accidents with fewer than 21 rows

**Seed 0: 0 of 51. Seed 1: 0 of 51. Seed 2: 1 of 51.** The one in seed 2 was left out of the control measurements and
of the grumobilenet retentions (n = 50). It remains in the per-video AUC.

No validation video has K < 10. Hence the rule "background = median of [0, K)" was not needed. The caveats of the
official-split experiment apply: without a keyframe, the background of [0, 10) may contain the accident, a bias in
favor of "scene".

## 6. CPU time

| Step | CPU (s) | Wall (s) |
|---|---|---|
| Smoke 1 + smoke 2 (training side only) | 29 | 15 |
| Aborted run (bug in the check; see Deviations), estimate | ~210 | ~45 |
| Diagnosis of the bug (2 short scripts), estimate | ~20 | — |
| Valid full run (3 seeds) | 731 | 139 |
| **Total of the holdout experiment** | **~990 (~17 min, of the 3 h cap)** | |

The cache and features come from the official-split experiment, which took about 5,220 s and is not added here.
Configuration: 6 threads and `tools/run_guarded.sh` with a 2500 MB limit.

## 7. Deviations

**No criterion or threshold was changed.**

1. **Run aborted by a bug in my code, already fixed.**
   - The first full run (20261005-110508) aborted at the split consistency check.
   - The marked-features function also planted the mark on the validation **normals**, which the official-split
     experiment never does. Hence the recomputed C0 was 0.637.
   - Only the check was affected: the split was right (same counts, no missing file), and the control measurements
     used the function only on accidents.
   - I fixed the function (normals and K ≤ 10 get no mark, as in `official_split.marked_train_feats`), ran the smoke
     test again and ran everything again (`tad_diagnosis/results/holdout/20261005-110712/`).
   - The aborted folder was kept on disk (not included). It has only the grumobilenet head of seed 0 and no validation
     or test metric of grumobilenet.
2. **Seed of the grumobilenet head: s + 4,** the same as the grumobilenet of the official-split experiment. The
   pre-registration asks for "the same training" and does not fix the seed.
3. **Marking of the controls on the validation set.** Uses the same cell as the official-split experiment for these
   training videos (`rng([s, k])`) and the same marked features that event_control saw on the C0 validation of the
   official-split experiment, read from `config.TAD_FEATURE_CACHE/marked_s<s>/train/`. Since the score uses only rows
   ≤ 20, the result is equivalent to marking the clip on rows [10, 21).
4. **Extra check, not a criterion.** The C0 of the saved event_control head was recomputed to prove that the split is
   the same as in the official-split experiment. If it did not match, the run aborted.
5. **Swap background omitted.** It is optional in the pre-registration and not validated on TAD.

## 8. Files

- `tad_diagnosis/results/holdout/20261005-110712/summary.json`: verdict, reading and numbers per seed.
- `tad_diagnosis/results/holdout/20261005-110712/seed<s>/`:
  - `val_per_video.csv`: grumobilenet and video_control on the 73 videos;
  - `controls_val_per_video.csv`;
  - `test_per_video.csv`;
  - `seed_summary.json`;
  - `grumobilenet_head.pt`.
