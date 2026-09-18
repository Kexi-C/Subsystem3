# Rail Corrugation Detection (Subsystem 3)

Classifies each 1-second axle-box recording as **Normal**, **Side I** or **Side II** rail corrugation.
The competition metric is **macro F1** over the three classes.

## Results

All figures are cross-validated on the 272 training files. Low-speed files are included and scored under the rule below.

| Evaluation | Macro F1 | Side I F1 | Side II F1 | Normal F1 |
|---|---|---|---|---|
| Nested CV, 3 x 5-fold, seed 42 | 0.861 (95% bootstrap CI 0.783-0.935) | 0.741 | 0.863 | 0.979 |
| Multi-seed check, 10 seeds x 5-fold, single model per file | 0.803 +/- 0.040 (range 0.736-0.850) | 0.600 | - | - |

The multi-seed figure is the more realistic estimate for unseen data. The seed-42 figure benefits from averaging out-of-fold scores over three repeats, which the single deployed model does not have.

## Quick start

Requirements: Python 3.13 (Anaconda), numpy 2.3, pandas 2.3, scipy 1.16, scikit-learn 1.7, joblib. matplotlib is needed only for `eda.py` and `evaluate.py`.
The model file was saved with scikit-learn 1.7.2; use the same version to load it.

### Predict

```bash
python code/predict.py --input <csv file or directory> --output <output.csv or directory> [--details details.csv]
```

Input files must have the same format as the provided `Train/` and `Test/` files:

- a header row;
- 129 columns: tooth-wheel speed pulse, then vibration and shock for car 1-8, position 1-8;
- sampled at 10 kHz, normally 10,000 rows (1 s).

The output has one row per input file, with columns `file_id` and `prediction`. This matches `04_Example_Submission/rail_predictions.csv`.
The optional `--details` file adds speed, the low-speed flag and both side scores.

Other programs, such as an app, can call `predict_files()` in `code/predict.py` directly.

### Retrain

```bash
python code/build_cache.py   # raw signal cache + per-axle-box features (about 1 min)
python code/train.py         # cross-validation, model selection, final model (about 6 min)
python code/evaluate.py      # metrics, confusion matrix, error list for the selected scheme
```

By default the dataset is expected in the sibling repository `../NebulaX-Hackathon-ProblemStatement/PS3/02_Datasets/Rail_Corrugation`, and the cache (about 1.7 GB) is written to `../Subsystem3_cache`.
Both can be overridden with the environment variables `RAIL_DATA_DIR` and `RAIL_CACHE_DIR`.

## Repository structure

| Path | Purpose |
|---|---|
| `code/config.py` | Paths, physical constants, data layout, feature and model settings |
| `code/data_io.py` | CSV reading and reshaping into a (car, position, channel, time) tensor |
| `code/features.py` | Per-axle-box features, aggregation, sensor baseline |
| `code/build_cache.py` | Builds the raw `.npy` cache and feature tensors |
| `code/eda.py` | Exploratory analysis (speed, NaN, single-feature AUC, side PSD difference) |
| `code/train.py` | Cross-validation, scheme selection, final model |
| `code/evaluate.py` | Evaluation of out-of-fold predictions |
| `code/predict.py` | Inference |
| `model/rail_model.joblib` | Trained model: both component models, sensor baseline, threshold, settings |

## Method

1. **Speed.** Speed is computed from the 0/1 transitions of the 90-tooth wheel pulse: v = N / 180 x pi x 0.85 m/s, where N is the number of transitions in one second.

2. **Low-speed rule.** Files below 1 m/s are predicted Normal and are not passed to the model. All 44 such training files are Normal, and a stationary wheel cannot excite corrugation.

3. **Per-axle-box features.** 43 features are computed for each of the 128 signals, after removing the DC offset:
   - time domain: log RMS, log peak-to-peak, crest factor, kurtosis, skewness;
   - absolute and relative Welch band energy in 8 frequency bands (10-5000 Hz);
   - absolute and relative energy in 10 half-octave wavelength bands (2-64 cm), mapped with lambda = v / f so that they do not depend on speed;
   - dominant frequency and dominant wavelength.

4. **Sensor baseline.** The per-sensor median over the training files is subtracted. This removes systematic differences between sensors.

5. **Aggregation.**
   - Per side: mean, median, max and std over the 32 axle boxes.
   - Side contrast: Side I minus Side II for mean, median, max and top-3 mean.
   - Left-right differences of the 32 wheelsets: max, min, top-3 mean, bottom-3 mean, 90th and 10th percentiles.
   - Car-level differences (mean left-right difference of each car's 4 wheelsets) over the 8 cars: max, top-2 mean, min, bottom-2 mean, and max/min over adjacent car pairs.
   - The file-level model also uses the 8 per-car values.

6. **Models.** Two HistGradientBoosting classifiers are used, both with class-balanced weights, learning rate 0.05, 200 iterations, 15 leaves, at least 5 samples per leaf and L2 regularization 1.0.
   - **Side-view model (1,721 features).** Each file becomes two samples: a Side I view and a mirrored Side II view, with all differences expressed as "this side minus the other side". One binary model learns "this side is corrugated", so Side I and Side II faults are pooled for training (38 positive samples).
   - **File-level model (2,752 features).** A direct three-class model that also uses the per-car features, so it can learn class-specific car patterns.

7. **Decision.** The two models' Side I and Side II fault scores are averaged. A side is predicted when its score is at least **0.22**; if both sides reach the threshold, the higher score wins; otherwise the prediction is Normal.

## Validation protocol

- Repeated stratified 5-fold cross-validation (3 repeats, fixed seed 42), so that every code version is compared on identical splits.
- The sensor baseline and all fitted steps are estimated inside each training fold.
- Thresholds are evaluated with nested tuning: each fold's threshold is tuned only on the other folds. The shared threshold of 0.22 was selected in every fold.
- Robustness was checked with 10 additional seeds. The ensemble beat the side-view model in 7 of 10 seeds and the file-level model in 8 of 10 seeds.

## Tried and not adopted

| Change | Outcome |
|---|---|
| Random forest | Scores depended almost entirely on a tuned threshold (default-threshold macro F1 0.48) |
| Logistic regression | Consistently weaker; degraded further as car-level features were added |
| Speed residualization (feature = a + b ln v) | No gain (0.809 vs 0.807) |
| Separate thresholds for Side I and Side II | Higher tuned score but lower nested score (0.842 vs 0.861) |
| Up-weighting Side I fault samples | Lower nested score (0.814 vs 0.826) |
| Univariate feature selection, with or without bagging | About 0.04 lower in every seed tested |

## Known limitations

- **Side I is the weakest class.** It has only 14 training files, and its F1 varies between about 0.46 and 0.71 across data splits.
- **High-speed Side I files are hard to detect.** Two of the three Side I files at about 18.5 m/s are still missed. Their only distinctive signal found so far is in the shock channel, in the 4-8 cm wavelength band.
- **Recurring false alarms.** A few Normal files show single-car left-right patterns similar to the fault classes and are repeatedly predicted as faulty.
- **Wheelset pairing assumption.** Positions (1,2), (3,4), (5,6) and (7,8) are assumed to be the left and right ends of the same wheelset. This is set in `AXLE_PAIRS` in `config.py`.
- **Command-line interface.** The `--input`/`--output` interface follows the Info Kit description; no more detailed specification was found in the provided documents.
