# Official-split experiment: pre-registration

(Called T3-TAD in the working notes.) Script: `tad_diagnosis/official_split.py`; cache: `tad_diagnosis/build_cache.py`.

Written on 2026-10-05, BEFORE any run. Nothing here changes after seeing results; necessary deviations go to the
"Deviations" section of RESULT.md, with the reason.

**Question:** does the intervention module, validated on SO-TAD (the SO-TAD interventions,
`sotad_diagnosis/interventions.py`), flag a scene shortcut on the TAD benchmark of Xu et al., or not? Both outcomes are
a result.

## Data

- `config.TAD_ROOT` (TAD-benchmark), official split. Ignore the `*Zone.Identifier` files.
  - train: 372 videos (261 accidents in `train/accident_*` and 111 normals in `train/normal`);
  - test: 32 videos (16 accidents and 16 normals).
- No annotation: no keyframe, no box.
- Metadata already measured (data assessment): length +0.00 pp and black bars −2.24 pp against the null. **Do not
  redo.**
- Measured on 2026-10-05:
  - 1.94 h in total; median 13.5 s (max. 89 s);
  - fps from 14 to 54 (279 videos at 25 fps and 77 at 30 fps);
  - 201 videos at 1920×1080;
  - ~35 thousand rows at 5 fps.

## Cache

- 5 rows per second, by time: row r is the frame closest to t = r/5 s.
- Resizing identical to the SO-TAD frame cache of the detector (`detector/frame_cache.py`, 128 px), uint8, one file per
  video.
- No cap on the number of rows (the maximum is ~447).

## Architecture and training

Same as GRUMobileNet and the SO-TAD controls (`sotad_diagnosis/controls.py`):
- MobileNetV3-Small ImageNet, frozen, with the weights of the GRUMobileNet checkpoint;
- GRU(576→128) + Linear(128→2);
- causal window of 16 rows padded with frame 0, decision at the last step;
- 6 epochs, Adam 1e-3, class-weighted cross-entropy;
- all 111 training normals;
- seeds 0, 1 and 2.

## Models

- **grumobilenet:** window label = video label (there is no keyframe). Window sampling as in the training of the SO-TAD
  controls.
- **video_control (negative control):** the same head, with input = row 0 repeated 16 times, and the video label.
- **event_control (positive control):**
  - the SO-TAD mark (8-px checkerboard on one cell of the 4×4 grid, drawn per video) appears on rows r ≥ 10 of the
    accident videos, i.e. an artificial keyframe at row 10 (2 s);
  - window label = 1 iff the video is an accident and the last row of the window is ≥ 10.
- Accident videos with fewer than 21 rows are left out of the control measurements, because these use rows 12, 16
  and 20.

## Scores

- **Controls** (as in the SO-TAD interventions): mean of the windows ending at rows 12, 16 and 20.
- **grumobilenet** (as in the SO-TAD interventions for the real models): maximum over the windows of the video.

## Interventions (those of the SO-TAD interventions, without ablation)

- **Background:** per-pixel median of rows [0, 10). By construction, without the mark.
- **Background only:** the whole window becomes the background.
- **Frozen first frame:** the whole window becomes row 0.
- **Swap background:** on all rows, the pixels with |gray − gray of its own background| > 25 are pasted over the
  background of a normal test video drawn at random (fixed seed).
- **Event ablation:** NOT APPLICABLE, because there is no keyframe nor region.
- **Retention:** mean(score with intervention) / mean(original score) over the accident videos of the test set, with
  a 95% CI by bootstrap over videos (B = 1000).

## Criteria (those of the SO-TAD interventions, without the ablation part)

- **C0, control validity** (as for the SO-TAD controls): event_control window AUC ≥ 0.90 on 20% of the training set
  held out (stratified, fixed seed).
- **GO of the intervention module on TAD** iff, in at least 2 of the 3 seeds:
  - event_control has background-only retention ≤ 0.3;
  - video_control has background-only retention ≥ 0.8.
- **Reading of grumobilenet.** Only valid if the module is GO and the original per-video test AUC is ≥ 0.70; otherwise
  the reading is "no diagnosis". The thresholds are the same that classify the controls in the SO-TAD interventions:
  - background-only retention ≥ 0.8 → "scene" (shortcut);
  - retention ≤ 0.3 → "event";
  - in between → "inconclusive".

  The reading is taken from the median of the 3 seeds, and the values of each seed are reported too.
- **Descriptive, outside the verdict:**
  - per-video test AUC of grumobilenet with original, background-only and frozen-first-frame input, with a 95% CI by
    bootstrap;
  - test AUC of video_control (the ceiling of predictability from the scene);
  - swap-background retention;
  - results per seed.

## Limitations declared in advance

- The test set has only 32 videos, so the CI of the AUC is wide (SE ≈ 0.06 at AUC 0.9).
- Without a keyframe, rows [0, 10) and row 0 may contain the accident in clips that already start at the event. This
  inflates the background-only retention, a bias in favor of "scene". Report how many test accident videos have fewer
  than 21 rows.
- The training normals are all named `20220518_acci-bg*`: the source is probably a single one. This is descriptive and
  does not enter the criterion.
- The TAD paper is inconsistent (344 × 404 videos). We use the 404 files present.
- Attribution (lift) does not enter, because there is no event region.

## Execution

- Smoke test on a few TRAINING videos, without computing any metric on the test set.
- `tools/run_guarded.sh`, at most 6 threads, total cap of 3 h of CPU.
- Outputs in `<out-dir>/<timestamp>/` (recorded run: `tad_diagnosis/results/official_split/<timestamp>/`). Nothing is
  overwritten.
