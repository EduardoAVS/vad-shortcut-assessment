# Official-split experiment: result

(Called T3-TAD in the working notes.) Run on 2026-10-05, following `tad_diagnosis/results/official_split/PREREG.md`,
which is binding. No criterion or threshold was changed.

- **Full run:** `tad_diagnosis/results/official_split/20261005-101615/` (`summary.json`, `table.md`, `seed{0,1,2}/`).
- **Scripts:** `tad_diagnosis/build_cache.py` (cache) and `tad_diagnosis/official_split.py` (models, interventions and
  criteria). Both import from the detector and from the SO-TAD diagnosis, without copying code.
- **Smoke test:** TRAINING videos only, smoke run 20261005-101519-smoke (not included). No test metric was computed
  before the full run.

## Summary

- **C0 valid in the 3 seeds:** event_control window AUC of 0.993, 0.999 and 0.999.
- **Intervention module on TAD: GO** (3 of 3 seeds; the rule requires 2):
  - event_control retains 0.00 with the background;
  - video_control retains 0.81, 1.08 and 1.00.
- **Reading of grumobilenet: "no diagnosis".** The per-video test AUC was 0.64, 0.68 and 0.54 (median 0.645), below
  the 0.70 gate. The module works on TAD, but the grumobilenet head trained on TAD does not detect accidents on the
  official test set well enough to say what it relies on.

## 1. C0 per seed (event_control window AUC on 20% of the training set held out; threshold 0.90)

| Seed | C0 | Passes | Control training / validation |
|---|---|---|---|
| 0 | 0.993 | yes | 297 / 73 videos |
| 1 | 0.999 | yes | 297 / 73 videos |
| 2 | 0.999 | yes | 297 / 73 videos |

## 2. Module GO/NO-GO (pre-registered rule)

Rule: GO iff, in at least 2 of the 3 seeds, event_control has background-only retention ≤ 0.3 and video_control has
background-only retention ≥ 0.8. Here, each seed also had to pass C0.

| Seed | C0 ≥ 0.90 | event_control: ret. background only ≤ 0.3 | video_control: ret. background only ≥ 0.8 | Seed passes |
|---|---|---|---|---|
| 0 | 0.993 ✓ | 0.00 [0.00, 0.00] ✓ | 0.81 [0.52, 1.08] ✓ | yes |
| 1 | 0.999 ✓ | 0.00 [0.00, 0.00] ✓ | 1.08 [1.00, 1.33] ✓ | yes |
| 2 | 0.999 ✓ | 0.00 [0.00, 0.00] ✓ | 1.00 [1.00, 1.00] ✓ | yes |

**Verdict: GO (3/3).** The literal rule, without C0, gives the same result (3/3).

Notes:
- In seed 0, video_control passes at the limit (0.81, CI down to 0.52). The GO does not depend on it, because seeds 1
  and 2 pass comfortably.
- video_control saturates: most per-video scores are close to 0 or to 1. This explains the degenerate CIs
  ([1.00, 1.00]).
- event_control gives a score < 0.01 to any background without the mark (the maximum was 0.0063), against a median of
  ≈ 1.0 on the marked clip.

## 3. Reading of grumobilenet

The gate requires GO and an original per-video test AUC ≥ 0.70, by the median of the 3 seeds.

- GO: yes.
- Original AUC: 0.64, 0.68 and 0.54, with median 0.645 < 0.70. The gate fails, and it also fails in each seed alone.

**Reading: "no diagnosis"** (in the 3 seeds and in the median).

The background-only retention of grumobilenet was 0.71, 0.54 and 0.60 (median 0.60). This value is only descriptive
and does not count as a reading. It would fall in the "inconclusive" range (between 0.3 and 0.8) if the gate had
passed.

## 4. Table per seed

Per-video AUC on the 32 test videos (16 accidents and 16 normals). The 95% CI comes from a class-stratified bootstrap,
with B = 1000.

| Seed | C0 (event_control window AUC, val) | AUC grumobilenet original | AUC grumobilenet background only | AUC grumobilenet first frame | AUC video_control (row 0) |
|---|---|---|---|---|---|
| 0 | 0.993 | 0.64 [0.43, 0.86] | 0.66 [0.47, 0.85] | 0.61 [0.39, 0.80] | 0.65 [0.44, 0.84] |
| 1 | 0.999 | 0.68 [0.48, 0.88] | 0.64 [0.43, 0.83] | 0.61 [0.40, 0.80] | 0.64 [0.43, 0.84] |
| 2 | 0.999 | 0.54 [0.32, 0.76] | 0.64 [0.41, 0.84] | 0.58 [0.37, 0.79] | 0.62 [0.41, 0.82] |

Retentions on the 16 test accidents: mean with intervention / original mean, with a 95% CI by bootstrap over videos
(B = 1000).
- Control score: mean of the windows ending at rows 12, 16 and 20 of the marked clip.
- grumobilenet score: maximum over the windows of the clip without the mark.

| Seed | Model | n | Original score | Ret. background only | Ret. first frame | Ret. swap background |
|---|---|---|---|---|---|---|
| 0 | event_control | 16 | 0.89 [0.72, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.95 [0.72, 1.27] |
| 0 | video_control | 16 | 0.62 [0.40, 0.83] | 0.81 [0.52, 1.08] | 1.11 [0.86, 1.45] | 0.36 [0.10, 0.63] |
| 0 | grumobilenet | 16 | 0.88 [0.69, 1.00] | 0.71 [0.47, 0.93] | 0.69 [0.44, 0.91] | 0.93 [0.69, 1.16] |
| 1 | event_control | 16 | 0.93 [0.80, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.94 [0.69, 1.17] |
| 1 | video_control | 16 | 0.52 [0.29, 0.73] | 1.08 [1.00, 1.33] | 1.08 [1.00, 1.33] | 0.90 [0.42, 1.85] |
| 1 | grumobilenet | 16 | 0.99 [0.96, 1.00] | 0.54 [0.31, 0.76] | 0.57 [0.32, 0.79] | 0.82 [0.64, 1.00] |
| 2 | event_control | 16 | 0.88 [0.70, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.99 [0.76, 1.31] |
| 2 | video_control | 16 | 0.56 [0.31, 0.81] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.59 [0.23, 1.27] |
| 2 | grumobilenet | 16 | 0.94 [0.82, 1.00] | 0.60 [0.34, 0.84] | 0.80 [0.57, 1.00] | 0.86 [0.67, 1.00] |

## 5. Test accident videos with fewer than 21 rows

**0 of 16.** The shortest has 42 rows (8.4 s); the median is 100. No test accident was left out of the control
measurements.

In the training set, 8 of the 261 accidents have fewer than 21 rows, and 2 have K ≤ 10 (see Deviations, item 2).

## 6. Descriptive (outside the verdict)

- **grumobilenet also fires on the test normals.** Mean maximum score:
  - normals: 0.61, 0.54 and 0.75;
  - accidents: 0.88, 0.99 and 0.94.

  Since the score of each video is the maximum over its windows, many normals are also close to 1. Hence the low AUC.
- **On the test set, one static image per video predicts as well as the whole model:**
  - background-only AUC: 0.64–0.66;
  - first-frame AUC: 0.58–0.61;
  - video_control AUC: 0.62–0.65;
  - original AUC: 0.54–0.68.

  With n = 32, all CIs include 0.5 or come close to it (lower limits from 0.32 to 0.48). The test set does not separate
  any of these values from chance nor from each other.
- **The scene separates the classes within the training set, but little on the official test set.** video_control has
  a window AUC of 0.86, 0.92 and 0.94 on the 20% validation of the training set, against 0.62–0.65 on the test set.
  - The training normals come from three batches named `*_acci-bg*`.
  - The test videos are named `videox_test_T*` (accidents) and `videox_test_F*` (normals).

  This is the pattern expected if scene and class are linked within the training set, but not between training and
  test.
- **Swap background:**
  - grumobilenet retains 0.82–0.93, unlike GRUMobileNet on SO-TAD (0.09);
  - on this dataset, however, the instrument does not separate the controls clearly: event_control retains 0.94–0.99,
    but video_control retains 0.36, 0.90 and 0.59, against 0.13 on SO-TAD.

  Swap background does not enter the criterion, and grumobilenet should not be read from it.

## 7. CPU time

| Step | CPU (s) | Wall (s) |
|---|---|---|
| Cache smoke (4 training videos) | 21 | 13 |
| Full cache (404 videos, 35,031 rows, 1.7 GB) | 1,904 | 498 |
| Smoke of the official-split experiment (training only) | 34 | 8 |
| Full run (features + 3 seeds) | 3,217 | 550 |
| Exploratory checks (headers, timestamps, identity with the SO-TAD cache), estimate | ~40 | — |
| **Total** | **~5,220 (~1 h 27 min, of the 3 h cap)** | |

Configuration: at most 6 threads and `tools/run_guarded.sh` with a 2500 MB limit. The memory of the cache build was
~1.5 GB, summed over the 3 processes.

## 8. Deviations

**No criterion or threshold was changed, and nothing in the pre-registration was left unimplemented.** The items below
are operationalization decisions where the pre-registration is silent or ambiguous. They were fixed in the code
(docstring of `official_split.py`) before the full run.

1. **Training sets.**
   - event_control and video_control train on the same 80% of the training set, like the SO-TAD controls, because C0
     requires 20% held out.
   - grumobilenet trains on the 372 videos (100%).
   - "All 111 normals" holds literally for grumobilenet. The controls see 89 normals in training and 22 in validation.
2. **Accidents without any row ≥ 10 (K ≤ 10).** There are 2 training videos. They are left out of the training and
   validation of the controls, by the rule of the SO-TAD controls for clips without a positive window. grumobilenet
   uses them.
3. **C0 in the GO.** The GO rule of the pre-registration lists only the two retentions. Here, each seed also needed C0,
   since it is a control-validity criterion. The literal version was computed and gives the same verdict (GO 3/3).
4. **Seeds.** "Fixed seed" was read as a fixed seed per run, derived from seed s as for the SO-TAD controls:
   - 80/20 split: `rng(s+1)`;
   - mark cell: `rng([s, k])` in train and `rng([s, 100000+k])` in test;
   - heads: s+2 (event_control), s+3 (video_control) and s+4 (grumobilenet);
   - swap-background normal: `rng([s, 300000+k])`.

   The 80/20 split and the cells vary between seeds.
5. **video_control on the test set.**
   - For the criterion, it gets the real windows of the marked clip (rows 12, 16 and 20), as in the SO-TAD controls
     and interventions.
   - For the descriptive AUC ("ceiling of predictability from the scene"), it uses the input it was trained on (row 0
     repeated 16 times), on the 32 videos.
6. **CI of the AUC.** Class-stratified bootstrap (B = 1000, seed 0). The pre-registration only says "bootstrap".
7. **AUC gate by the median of the seeds.** Follows the sentence "the reading is taken from the median". The AUC was
   below 0.70 in every seed, so this choice changes nothing.
8. **Swap background for grumobilenet.** Uses the same score as the original (maximum over the windows), so that the
   retention compares like with like.
9. **Cache.**
   - The time of each frame is its own `CAP_PROP_POS_MSEC`.
   - K = number of targets r/5 that do not go past the time of the last frame.
   - The 404 videos are CFR (maximum deviation of 4.7e-5 s from i/fps; 0 fallbacks). Hence the chosen frame is
     round(r·fps/5).
   - On a 25-fps video, pixel identity with `detector/frame_cache.extract_video` was verified.
10. **Factual correction of a declared limitation (descriptive).** The training normals are not all named
    `20220518_acci-bg*`. There are 3 batches, all `*_acci-bg*`: 20220518 (31), 20220620 (21) and 20230818 (59).

Event ablation was not applied, as pre-registered (there is no keyframe nor region).

## 9. Files

- `tad_diagnosis/results/official_split/20261005-101615/summary.json`: verdict, reading and numbers per seed.
- `tad_diagnosis/results/official_split/20261005-101615/seed<s>/`:
  - `grumobilenet_per_video.csv`: scores of grumobilenet and video_control on the 32 videos;
  - `controls_per_video.csv`: controls on the 16 accidents;
  - `seed_summary.json`;
  - heads `*_head.pt` (`event_control_head.pt`, `video_control_head.pt`, `grumobilenet_head.pt`).
- `config.TAD_FRAME_CACHE` (`index.csv` and `cpu_log.csv`) and `config.TAD_FEATURE_CACHE` (deterministic features,
  236 MB); not versioned.
