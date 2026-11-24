# Attribution experiment: Grad-CAM attribution on TAD, without keyframe — pre-registration

(Called T3-TAD-c in the working notes.) Script: `tad_diagnosis/attribution.py`.

Written on 2026-10-05, BEFORE running. Deviations go to the "Deviations" section of RESULT.md, with the reason.

**Motivation (declared):** in the holdout experiment (`../holdout/RESULT.md`), grumobilenet (the GRUMobileNet head
trained on TAD) was read as "scene" on the 80/20 validation split of the training set. On SO-TAD, the Grad-CAM of the
real models concentrated mass on the motion regions (lift ~1.9 against a null of ~1.0) while the decision came from the
scene. This test checks whether that dissociation repeats on TAD. It is a new analysis, requested by the user after the
holdout experiment.

**Question:** does the Grad-CAM of grumobilenet concentrate mass on the motion regions above calibrated nulls?

## Videos and models

- The same validation videos as the holdout experiment: per-seed 80/20 split, `rng(s+1)`, seeds 0, 1 and 2. The
  validation accidents with at least 21 rows are measured; the validation normals are only descriptive.
- **grumobilenet:** the heads saved in `../holdout/20261005-110712/seed<s>/`.
- **Null 1, random_head (randomized head):** same architecture, initialized with seed s, untrained.
- **Null 2, video_control:** the head of the official-split experiment in `../official_split/20261005-101615/seed<s>/`,
  which only learned the scene.
- **Instrument validation:** the event_control of the official-split experiment, on the validation accidents marked as in
  the official-split experiment.

## Measurements

- **Motion region** (keyframe-free version of the motion pseudo-region of the SO-TAD controls,
  `sotad_diagnosis/controls.py`):
  - energy of |gray(row r) − gray(row r−1)| at 128 px, summed over ALL rows of the clip, in each cell of the 4×4 grid;
  - the region is the smallest set of cells that sums to at least 50% of the energy.
- **Attribution:** Grad-CAM, with target logit[1] − logit[0], summed over the 16 steps of the window, as in the SO-TAD
  controls. Grad-CAM++ and LayerCAM are only descriptive.
- **Windows:** up to 8 per video, evenly spaced among the windows ending at rows 15 to K−1.
- **Lift of a region** = attribution mass in the region / area fraction of the region, as in the SO-TAD controls (lift
  is the paper's µw). The mean over the windows of each video is taken.

## Criteria (those of the SO-TAD controls and interventions)

- **Instrument validity:** in at least 2 of the 3 seeds, event_control has mean lift (Grad-CAM) ≥ 2.0 on the mark cell,
  in the windows with the mark. This is the threshold of criterion C1 of the SO-TAD controls. If it fails, the reading
  is "no diagnosis".
- **Attribution reading of grumobilenet**, the same rule as in the SO-TAD interventions. grumobilenet has attribution
  evidence of looking at the motion iff its Grad-CAM lift exceeds EACH of the two nulls with the 95% CI of the paired
  difference > 0 (bootstrap over videos, B = 1000). The overall reading requires this in at least 2 of the 3 seeds.
- **Reading combined with the holdout experiment** (decision = "scene"):
  - if grumobilenet looks at the motion → "dissociation replicated on TAD";
  - if it does not → "no dissociation on TAD: attribution agrees with the cause".

## Descriptive (outside the verdict)

- Lift per seed and per model, with CI.
- Grad-CAM++ and LayerCAM.
- Lift on the validation normals.
- Mean area fraction of the motion region.

## Limitations declared in advance

- Without a keyframe, the motion region includes normal traffic, not only the accident. So this test is looser than the
  SO-TAD one: "looking at the motion" here is not "looking at the accident".
- The CAM has only 4×4 cells.
- The validation set is small (about 51 accidents per seed).

## Execution

- Reuse `official_split.py`, `holdout.py` and the functions of the SO-TAD controls by import, without editing anything.
- `tools/run_guarded.sh`, at most 6 threads, cap of 3 h (expected: less than 30 min).
- Smoke test only on videos of the train side, with no validation metric.
- Outputs in `tad_diagnosis/results/attribution/<timestamp>/`. Nothing is overwritten.
