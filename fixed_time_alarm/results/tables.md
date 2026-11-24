### Fixed-time alarm vs GRUMobileNet (PRIMARY: threshold at 99.9% of the TRAINING regular videos)

| row | S [95% CI] | hit@1s | hit@2s | hit@4s | % alarm < tau-0.5s | AUC macro [95% CI] | AUC placebo (n=9) | excess S [paired 95% CI] | test frame FPR |
|---|---|---|---|---|---|---|---|---|---|
| (i) first frame (T = 0.00 s) | 0.022 [0.000; 0.056] | 0.022 | 0.022 | 0.022 | 97.8 | 0.500 [0.500; 0.500] | 0.500 | -0.536 [-0.601; -0.467] | - |
| (ii) median train keyframe **(strongest null)** (T = 3.18 s) | 0.558 [0.498; 0.617] | 0.289 | 0.678 | 0.844 | 14.4 | 0.822 [0.801; 0.844] | 0.803 | - | - |
| (iii) clock T* (train) (T = 3.60 s) | 0.508 [0.452; 0.566] | 0.178 | 0.544 | 0.856 | 10.0 | 0.780 [0.754; 0.806] | 0.762 | -0.050 [-0.098; 0.004] | - |
| grumobilenet/warmup5 | 0.234 [0.159; 0.317] | 0.200 | 0.244 | 0.289 | 57.8 | 0.711 [0.647; 0.774] | 0.724 [0.532; 0.889] | -0.324 [-0.421; -0.230] | 2.05% |
| grumobilenet/raw | 0.020 [0.000; 0.051] | 0.022 | 0.022 | 0.022 | 97.8 | 0.711 [0.647; 0.774] | 0.724 [0.532; 0.889] | -0.538 [-0.602; -0.470] | 3.83% |

### Fixed-time alarm vs GRUMobileNet (SENSITIVITY: threshold at 99.9% of the TEST regular videos)

| row | S [95% CI] | hit@1s | hit@2s | hit@4s | % alarm < tau-0.5s | AUC macro [95% CI] | AUC placebo (n=9) | excess S [paired 95% CI] | test frame FPR |
|---|---|---|---|---|---|---|---|---|---|
| (i) first frame (T = 0.00 s) | 0.022 [0.000; 0.056] | 0.022 | 0.022 | 0.022 | 97.8 | 0.500 [0.500; 0.500] | 0.500 | -0.536 [-0.601; -0.467] | - |
| (ii) median train keyframe **(strongest null)** (T = 3.18 s) | 0.558 [0.498; 0.617] | 0.289 | 0.678 | 0.844 | 14.4 | 0.822 [0.801; 0.844] | 0.803 | - | - |
| (iii) clock T* (train) (T = 3.60 s) | 0.508 [0.452; 0.566] | 0.178 | 0.544 | 0.856 | 10.0 | 0.780 [0.754; 0.806] | 0.762 | -0.050 [-0.098; 0.004] | - |
| grumobilenet/warmup5 (ineligible) | 0.201 [0.130; 0.273] | 0.167 | 0.233 | 0.267 | 12.2 | 0.711 [0.647; 0.774] | 0.724 [0.532; 0.889] | -0.357 [-0.444; -0.265] | 0.10% |
| grumobilenet/raw (ineligible) | 0.018 [0.000; 0.047] | 0.011 | 0.022 | 0.022 | 96.7 | 0.711 [0.647; 0.774] | 0.724 [0.532; 0.889] | -0.540 [-0.603; -0.473] | 0.10% |

### Fixed-time alarm vs GRUMobileNet (pre-registered SENSITIVITY: threshold at 99% of the training regular videos)

| row | S [95% CI] | hit@1s | hit@2s | hit@4s | % alarm < tau-0.5s | AUC macro [95% CI] | AUC placebo (n=9) | excess S [paired 95% CI] | test frame FPR |
|---|---|---|---|---|---|---|---|---|---|
| (i) first frame (T = 0.00 s) | 0.022 [0.000; 0.056] | 0.022 | 0.022 | 0.022 | 97.8 | 0.500 [0.500; 0.500] | 0.500 | -0.536 [-0.601; -0.467] | - |
| (ii) median train keyframe **(strongest null)** (T = 3.18 s) | 0.558 [0.498; 0.617] | 0.289 | 0.678 | 0.844 | 14.4 | 0.822 [0.801; 0.844] | 0.803 | - | - |
| (iii) clock T* (train) (T = 3.60 s) | 0.508 [0.452; 0.566] | 0.178 | 0.544 | 0.856 | 10.0 | 0.780 [0.754; 0.806] | 0.762 | -0.050 [-0.098; 0.004] | - |
| grumobilenet/warmup5 | 0.133 [0.066; 0.210] | 0.133 | 0.133 | 0.133 | 86.7 | 0.711 [0.647; 0.774] | 0.724 [0.532; 0.889] | -0.425 [-0.526; -0.321] | 5.10% |
| grumobilenet/raw | 0.022 [0.000; 0.056] | 0.022 | 0.022 | 0.022 | 97.8 | 0.711 [0.647; 0.774] | 0.724 [0.532; 0.889] | -0.536 [-0.601; -0.467] | 8.32% |

### Realized false alarms on the test regular videos with the primary threshold (nominal 0.1% per frame)

| model | variant | thr | test_frame_fpr | test_video_fa | calib |
|---|---|---|---|---|---|
| grumobilenet | smooth5 | 0.0066226597105506 | 3.85 | 8.2 | train_normals(200) |
| grumobilenet | raw | 0.0064358912073079 | 3.83 | 8.5 | train_normals(200) |
| grumobilenet | runmin | 0.0061375119651602 | 2.69 | 8.0 | train_normals(200) |
| grumobilenet | warmup5 | 0.0060113577703014 | 2.05 | 7.2 | train_normals(200) |
| grumobilenet | cusum | 0.4218537147880326 | 1.38 | 2.8 | train_normals(200) |
