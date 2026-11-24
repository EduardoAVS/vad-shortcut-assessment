# Assessing Shortcut Learning in Video Anomaly Detectors: An Application to Traffic Accident Detection

Code, annotations and recorded results of the paper ([paper.pdf](paper.pdf)).

The paper proposes a three-module procedure to check whether a video anomaly detector relies on the event or on
shortcuts:

1. **Data assessment**: clip length, active area (black bars), first frame and temporal position of the event.
2. **Attribution**: Grad-CAM mass on the event region, measured by µw (Kohlbrenner et al. 2020; called `lift` in the
   code), compared against two null models.
3. **Interventions**: background sufficiency (does the score survive when everything that moves is removed?) and event
   necessity (does it drop when only the event region is removed?).

Each test is first validated on two control models whose answer is known:

- the **event control** (`event_control`): trained on videos with a 32-px checkerboard mark placed on the frames from
  the keyframe on, so it can only use the mark;
- the **video control** (`video_control`): sees only the first frame, repeated, so it can only use the scene.

The evaluated detector, **GRUMobileNet** (`grumobilenet`: frozen ImageNet MobileNetV3-Small + GRU, 128-px input,
5 fps, windows of 16 frames), is diagnosed on SO-TAD and TAD.

## Repository layout

```
config.py              all paths (datasets, caches, detector checkpoint); each one can be set by an environment variable
tools/run_guarded.sh   runs a job with nice, 2 threads and a memory watchdog (used for every experiment)
detector/              GRUMobileNet
  model.py               MobileNetV3-Small (+GRU) models
  frame_cache.py         128-px frame cache (one row = 5 source frames, 5 fps)
  extract_frames.py      builds the SO-TAD frame cache
  train.py               trains GRUMobileNet on SO-TAD (defaults = the paper run)
  score_test_videos.py   per-window scores on the SO-TAD test videos
  weights/grumobilenet.pt  the trained detector used in the paper
data_assessment/       module 1 (Table 2, Sec. 5.1)
fixed_time_alarm/      fixed-time alarm vs GRUMobileNet, within-video AUC (Sec. 5.1)
sotad_diagnosis/       modules 2 and 3 on SO-TAD (controls, µw, interventions; Table 3, Secs. 5.2-5.4)
  annotations/           the 90 accident bounding boxes drawn for this work (+ NOTES.md)
tad_diagnosis/         modules 2 and 3 on TAD (Table 4)
figures/               scripts that draw the paper figures; output/ holds the figures as used in the paper
```

Each experiment folder has a `results/` folder with the runs cited in the paper (`results/<script>/<timestamp>/` when a
script makes one folder per run). The TAD experiments were pre-registered: `tad_diagnosis/results/<experiment>/PREREG.md`
was written before running, and `RESULT.md` records the outcome and any deviation. `fixed_time_alarm/RESULT.md` records
the fixed-time alarm analysis. For the SO-TAD experiments, the criteria are in each script's docstring.

## Setup

Python 3.12, CPU only (no GPU was used in the paper).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

`requirements.txt` has the exact versions used for the paper. Re-drawing the PNGs of the schematic figures also needs
Node.js and `npm install @resvg/resvg-js` (see `figures/schematic/render_svg.js`).

### Data

No video is included. Both datasets are public; get them from their authors:

- **SO-TAD**: Chen et al., *SO-TAD: A surveillance-oriented benchmark for traffic accident detection*,
  Neurocomputing 618:129061, 2025. Expected layout: `so-tad/{train,test}/<id>.mp4` and `so-tad/Appendix.txt`
  (keyframes, provided with the dataset).
- **TAD**: Xu et al., *TAD: A large-scale benchmark for traffic accidents detection from video surveillance*,
  IEEE Access, 2025. Expected layout: `TAD-benchmark/{train,test}/<class folder>/<video>.mp4`.

Put them in `data/` (`data/so-tad`, `data/TAD-benchmark`) or point to them:

```bash
export SOTAD_ROOT=/path/to/so-tad TAD_ROOT=/path/to/TAD-benchmark
```

Caches and other regenerable files go to `work/` (git-ignored; change it with `WORK_DIR`). See `config.py` for every
variable.

## Reproducing the results

Run everything from the repository root. The times are from the original machine (CPU, 2–6 threads). Every script
accepts `--help`; most accept `--smoke` (a quick run on a few videos) and `--out <dir>` (write somewhere else than
`results/`). New runs go to a new `<timestamp>` folder, so the recorded ones are never overwritten. Seeds are fixed
inside the scripts.

**1. Detector** (`detector/`)

```bash
tools/run_guarded.sh 3000 work/logs/extract_frames.log -- python detector/extract_frames.py        # frame cache, ~9.4 GB
tools/run_guarded.sh 6000 work/logs/train.log -- python detector/train.py                          # optional: retrains GRUMobileNet
tools/run_guarded.sh 2000 work/logs/score_test_videos.log -- python detector/score_test_videos.py  # ~45 min
```

The other steps use the included checkpoint `detector/weights/grumobilenet.pt`, so retraining is optional. `train.py`
writes to `work/runs/grumobilenet/`; to use the new model, set `DETECTOR_CKPT=work/runs/grumobilenet/best.pt`.

**2. Data assessment** (`data_assessment/`)

```bash
python data_assessment/active_area.py                    # active area of every SO-TAD clip, ~26 min
python data_assessment/tad_clips.py                      # length and active area of every TAD clip, ~10 min
python data_assessment/length_and_area_rules.py          # Table 2, length columns
python data_assessment/active_area_null.py               # Table 2, active-area columns
python data_assessment/sotad_clip_lengths.py             # Sec. 5.1 statistics
python data_assessment/sotad_area_and_length_checks.py   # Sec. 5.1 checks
```

The last four read the outputs of the first two; the recorded ones are included, so they can run alone.

**3. Fixed-time alarm** (`fixed_time_alarm/`, needs the frame cache and the test scores of step 1)

```bash
tools/run_guarded.sh 2000 work/logs/train_scores.log -- python fixed_time_alarm/train_scores.py --check   # ~6 min
python fixed_time_alarm/evaluate.py
python fixed_time_alarm/tables.py
```

**4. SO-TAD diagnosis** (`sotad_diagnosis/`, needs the frame cache)

```bash
tools/run_guarded.sh 2000 work/logs/controls.log -- python sotad_diagnosis/controls.py               # ~10 min
tools/run_guarded.sh 2000 work/logs/interventions.log -- python sotad_diagnosis/interventions.py     # ~5 min
python sotad_diagnosis/motion_region.py
python sotad_diagnosis/box_interventions.py
python sotad_diagnosis/box_attribution.py
```

`controls.py` trains the two control models and saves their heads. The later scripts load the heads of the cited run
(`results/controls/20260929-223547`). To use the heads of a new run, pass `--controls-run <run dir>`.

`annotate_keyframes.py` is the interactive tool (needs a display) that was used to draw the boxes in
`annotations/keyframe_boxes.csv`; you do not need to run it.

**5. TAD diagnosis** (`tad_diagnosis/`)

```bash
tools/run_guarded.sh 2500 work/logs/tad_build_cache.log -- python tad_diagnosis/build_cache.py --workers 6
tools/run_guarded.sh 2500 work/logs/tad_official_split.log -- python tad_diagnosis/official_split.py
tools/run_guarded.sh 2500 work/logs/tad_holdout.log -- python tad_diagnosis/holdout.py
tools/run_guarded.sh 2500 work/logs/tad_attribution.log -- python tad_diagnosis/attribution.py
tools/run_guarded.sh 2500 work/logs/tad_cross_validation.log -- python tad_diagnosis/cross_validation.py
```

New TAD runs go to `work/tad_diagnosis/<experiment>/<timestamp>/` (option `--out-dir`). The later experiments read the
heads and splits of the recorded runs in `tad_diagnosis/results/`; to use new runs, pass `--official-run` /
`--holdout-run` (see each script's docstring).

**6. Figures** (`figures/`)

```bash
python figures/dataset/dataset_figures.py        # Figs. 5, 6, 8 (real frames; need the datasets and the frame cache)
python figures/schematic/schematic_figures.py    # SVG of Figs. 1, 2, 3, 4, 7, 9
NODE_PATH=<dir with node_modules> node figures/schematic/render_svg.js   # SVG -> PNG
```

The numbers drawn in Fig. 7 are typed in `schematic_figures.py`.

## Where each paper result comes from

Paths below are relative to the experiment folder; `a.b` is a key inside a JSON file.

| Paper result | Script | Recorded output |
|---|---|---|
| **Table 2**, length AUC (SO-TAD test / SO-TAD all / TAD all: 1.0000 / 0.9996 / 0.5605); length rule vs majority (100.00 vs 81.21; 99.63 vs 87.10; 68.56 vs 68.56) | `data_assessment/length_and_area_rules.py` | `results/length_and_area_rules.json` -> `auc`, `separability.best_single_threshold_accuracy` vs `separability.majority_class` |
| **Table 2**, active-area rule vs mean of 2,000 label shuffles (99.58 vs 82.07; 98.95 vs 87.48; 74.50 vs 76.75); 88/90 test and 259/282 accidents in an active area that no regular clip has | `data_assessment/active_area_null.py` | `results/active_area_null.json` -> `table_rule_observed` vs `table_rule_null_mean`, `exclusive_observed` |
| Sec. 5.1: 98.6% of regular clips last exactly 20 s; 29.4% of accident clips < 6 s; shortest 2.72 s; test length ranges 79–375 vs 388–514 frames; median test keyframe 2.10 s | `data_assessment/sotad_clip_lengths.py` | `results/sotad_clip_lengths.json` |
| Fixed-time alarm at **3.18 s**: within-video AUC **0.822** vs **0.711** for GRUMobileNet | `fixed_time_alarm/evaluate.py` | `results/reference_predictors.csv` (row "(ii) median train keyframe": `T_s`, `auc`); `results/variants.csv` (`grumobilenet`, `warmup5`); `RESULT.md` |
| Video control, window-level AUC **0.945** | `sotad_diagnosis/controls.py` | `results/controls/20260929-223547/summary.json` -> `controls_validation.window_auc_video_control` |
| **Table 3**, event control µw **5.9** on the mark; Sec. 5.4: video control **1.31**, randomized network **1.92** on the mark | `sotad_diagnosis/controls.py` | same file -> `controls.event_control/gradcam.lift_post`, `controls.video_control/gradcam.lift_post`, `controls.random_head/gradcam.lift_post` |
| **Table 3**, event control retention **11%**, necessity **0.73**; video control retention **99%**, necessity **0.02** | `sotad_diagnosis/interventions.py` | `results/interventions/20260929-233839/summary.json` -> `causal.event_control.*`, `causal.video_control.*` |
| Background sufficiency of GRUMobileNet: retention **94%** [90, 98], AUC **0.999**; first frame repeated **93%**, AUC **0.997**; detection AUC 0.999 | `sotad_diagnosis/interventions.py` | same file -> `causal.grumobilenet.retention_background_only`, `retention_first_frame`, `per_video_auc.grumobilenet.*` |
| Motion region contains the box center in **81%** [73, 89] of the videos vs **22%** by chance | `sotad_diagnosis/motion_region.py` | `results/motion_region/20261006-092505/summary.json` -> `primary.cache.pointing`, `pointing_ci95`, `chance` |
| Event necessity of GRUMobileNet **0.015** [−0.01, 0.04] (annotated box vs random region) | `sotad_diagnosis/box_interventions.py` | `results/box_interventions/20261006-093346/summary.json` -> `all.grumobilenet.box_cells.event_minus_random` |
| µw on the annotated box (Fig. 7): GRUMobileNet **1.64** [1.46, 1.82]; null 1 (randomized head) **0.67**; null 2 (video control) **0.85**; paired differences **0.98** [0.77, 1.16] and **0.80** [0.58, 1.02] | `sotad_diagnosis/box_attribution.py` | `results/box_attribution/20261006-100109/summary.json` -> `primary.box_cells.lift.*`, `primary.box_cells.evidence.grumobilenet.paired_difference.*` |
| **Table 4**, official test: GRUMobileNet AUC **0.54–0.68**, video control **0.62–0.65** (3 seeds) | `tad_diagnosis/official_split.py` | `tad_diagnosis/results/official_split/20261005-101615/table.md` (+ `summary.json`; `../RESULT.md`) |
| **Table 4**, cross-validation: AUC **0.92** [0.89, 0.95]; video control **0.88**; retention GRUMobileNet **90%** [87, 94], first frame **93%** [90, 96]; event control **11%** [8, 15]; video control **95%** [91, 98] | `tad_diagnosis/cross_validation.py` | `tad_diagnosis/results/cross_validation/20261005-120959/table.md` (+ `summary.json`; `../RESULT.md`) |
| **Table 4**, µw (20% of the training set held out): event control **9.7–10.6**, video control 1.1–1.2, GRUMobileNet 1.0–1.2, randomized network 0.9–1.2 | `tad_diagnosis/attribution.py` (uses the split of `holdout.py`) | `tad_diagnosis/results/attribution/20261005-120205/table.md` (+ `summary.json`; `../RESULT.md`) |
| (supporting) TAD 80/20 holdout diagnosis that motivated the attribution and cross-validation runs | `tad_diagnosis/holdout.py` | `tad_diagnosis/results/holdout/20261005-110712/table.md` (+ `summary.json`; `../RESULT.md`) |
| GRUMobileNet | `detector/train.py` | `detector/weights/grumobilenet.pt` (its `config` key holds the training configuration) |
| The 90 SO-TAD accident boxes | drawn with `sotad_diagnosis/annotate_keyframes.py` | `sotad_diagnosis/annotations/keyframe_boxes.csv` (+ `NOTES.md`) |
| Figs. 1, 2, 3, 4, 7, 9 | `figures/schematic/schematic_figures.py` -> `render_svg.js` | `figures/output/fig{1,2,3,4,7,9}_*.{svg,png}` |
| Figs. 5, 6, 8 | `figures/dataset/dataset_figures.py` | `figures/output/fig5_first_frames.png`, `fig6_motion_region.png`, `fig8_gradcam.png` |

## Notes on reproducibility

- Every script above was re-run from this layout, and the outputs were compared with the recorded ones. Most came out
  identical. The rest differ only by float round-off (≤ 1e-6), which does not change any number at the precision of
  the paper. Feature caches are recomputed from the frame cache, which explains the round-off.
- The full 6-epoch training of GRUMobileNet was not re-run. A short run gives the same output as the original training
  code. Bit-exact retraining is not guaranteed: data loading uses 4 worker processes.
- The TAD experiments use 6 threads (`--threads`, as in the recorded runs). A different thread count can change the
  trained heads by float round-off.
