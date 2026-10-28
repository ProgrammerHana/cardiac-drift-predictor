# Cardiac Drift Prediction from Early-Run Telemetry

**An end-to-end Machine Learning project for predicting aerobic decoupling using the first 15 minutes of running telemetry.**

## Project Overview

Cardiovascular drift describes changes in cardiovascular response during prolonged exercise, often including an increase in heart rate despite relatively stable external workload. These changes may be influenced by factors such as exercise intensity, fatigue, hydration, and environmental conditions.

This project investigates a central question:

**Can we predict a runner's full-run cardiac drift using only telemetry collected during the first 15 minutes, even for runners the model has never encountered?**

Using running data from the Endomondo/FitRec dataset, the project implements a complete machine learning workflow: raw telemetry processing, feature engineering, model benchmarking, group-aware validation, hyperparameter tuning, and model explainability.

The long-term concept is an early-warning tool for runners. The current implementation is an experimental regression model, not a validated system for detecting physiological fatigue or preventing performance deterioration.

## Interactive Dashboard — Streamlit

A Streamlit application provides a local interface for exploring running data and demonstrating the prediction workflow.

To run the application locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard serves as a demonstration of the ML pipeline rather than a live, watch-connected monitoring service.

## Data Processing & Feature Engineering

The raw dataset contains time-series measurements such as heart rate, speed, GPS coordinates, timestamps, and elevation.

The preprocessing pipeline performs data cleaning, speed-unit consistency checks, validation of running intervals, and extraction of numerical features.

Each processed training example represents one running session.

The model uses nine input features:

| Feature                | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| `gender`               | Runner metadata                                            |
| `hr_mean_5_15`         | Average heart rate during minutes 5–15                     |
| `speed_mean_5_15`      | Average running speed during minutes 5–15                  |
| `hr_slope_5_15`        | Heart-rate trend during the early running window           |
| `speed_slope_5_15`     | Speed trend during the same window                         |
| `ratio_slope_5_15`     | Trend in the heart-rate-to-speed ratio                     |
| `early_decoupling_pct` | Change in HR-to-speed ratio between minutes 5–10 and 10–15 |
| `elev_gain_15m`        | Cumulative elevation gain during the first 15 minutes      |
| `speed_std_15m`        | Variability in running speed during the early window       |

### Prediction Target

The target, `target_drift_pct`, measures the percentage change in the heart-rate-to-speed ratio between the two halves of a completed run.

For each half:

`R = mean(HR) / mean(Speed)`

The target is calculated as:

`Drift(%) = 100 * ((R_second / R_first) - 1)`

The first five minutes are excluded from the first-half calculation to reduce the influence of the initial heart-rate adjustment period.

This is an operational measure of HR-to-speed decoupling. It is not a direct measurement of fatigue and may also be affected by pacing changes, elevation, and measurement noise.

## Key Engineering Decisions & Methodology

### 1. Data Quality and Unit Consistency

A central challenge was transforming heterogeneous running telemetry into consistent numerical features.

The dataset includes recorded speed measurements and GPS-derived speed estimates. Their units were investigated by comparing recorded values against speeds independently calculated from GPS distances and timestamps.

Additional preprocessing reduces the impact of invalid measurements, unusually low speeds, and implausible running data.

### 2. Preventing Feature Leakage

The prediction task is defined at the 15-minute mark.

Features that require knowledge of the completed run, such as `total_duration_min` and duration-derived interaction terms, were excluded from the model inputs.

The completed run is used to calculate the training target, but not to provide future-derived prediction features.

Some dataset eligibility filters still depend on completed-run information. Consequently, reported performance applies to the selected population of runs rather than all possible live running sessions.

### 3. Runner-Aware Validation

Running sessions from the same person are not independent observations. Randomly splitting individual workouts could place sessions from the same runner in both training and validation sets.

To evaluate generalization to unseen runners, the project uses group-aware splitting based on `user_id`.

The evaluated dataset contains **1,846 runs from 420 runners**, divided into:

| Dataset                              |  Runs | Unique runners |
| ------------------------------------ | ----: | -------------: |
| Development (80% of runners)         | 1,478 |            336 |
| Held-out evaluation (20% of runners) |   368 |             84 |

There is no runner overlap between the two partitions.

Within the development set, five-fold `GroupKFold` cross-validation is used for model comparison and hyperparameter tuning.

### 4. Model Benchmarking and Hyperparameter Tuning

Four regression approaches were compared:

- Dummy Regressor — constant mean prediction
- Ridge Regression — regularized linear baseline
- Huber Regression — robust linear baseline
- XGBoost — gradient-boosted decision trees

Model selection was based primarily on Mean Absolute Error (MAE).

Hyperparameter tuning for XGBoost was performed using `GridSearchCV` within the development dataset.

The selected XGBoost configuration was:

```python
{
    "learning_rate": 0.01,
    "max_depth": 3,
    "n_estimators": 300
}
```

### 5. Feature Ablation Study

An ablation experiment investigated whether the engineered `hr_slope_per_elev` feature improved predictive performance.

Both configurations were evaluated on identical development folds using the same XGBoost hyperparameters.

| Configuration               | Development CV MAE |
| --------------------------- | -----------------: |
| With `hr_slope_per_elev`    |             4.4291 |
| Without `hr_slope_per_elev` |         **4.4197** |

Removing the feature produced a small average improvement of approximately 0.21%, although the results were not consistent across folds.

The simpler nine-feature configuration was therefore selected. This experiment does not establish that elevation-related interactions are inherently uninformative.

### 6. Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to investigate how individual features influence XGBoost predictions.

The explainability workflow includes:

- **Summary / Beeswarm Plot:** Ranks features by the magnitude of their contributions across predictions.
- **Dependence Plots:** Show how feature values relate to their SHAP contributions, including heart-rate slope and early decoupling.
- **Feature-level interpretation:** Examines patterns learned by the model and identifies relationships worth investigating further.

The analysis highlighted relationships involving elevation gain, running speed, and early heart-rate trends. Some learned relationships were not straightforward to interpret physiologically.

SHAP explains model behavior, not physiological causation. These visualizations are therefore treated as diagnostic tools rather than proof of underlying biological mechanisms.

## Model Performance

The final selected model is an XGBoost Regressor.

### Development Cross-Validation

| Model       |        MAE |       RMSE |         R² |
| ----------- | ---------: | ---------: | ---------: |
| Dummy       |     4.5184 |     6.2013 |    -0.0032 |
| Ridge       |     4.4634 |     6.1690 |     0.0068 |
| Huber       |     4.4353 |     6.1550 |     0.0117 |
| **XGBoost** | **4.4197** | **6.0815** | **0.0348** |

### Held-out Runner Evaluation

| Metric |   Dummy |    XGBoost |
| ------ | ------: | ---------: |
| MAE    |  4.3519 | **4.2758** |
| RMSE   |  5.8749 | **5.6664** |
| R²     | -0.0002 | **0.0695** |

XGBoost reduced MAE by approximately **1.75% relative to the mean baseline** on the held-out runner dataset.

The improvement is modest and does not establish strong predictive accuracy. However, it provides preliminary evidence that early-run telemetry contains some information useful for predicting the defined decoupling target.

The held-out runner partition was also evaluated for an earlier feature configuration during development; these results should therefore not be interpreted as a completely untouched, single-use confirmatory test.

## Limitations

The project has several important limitations:

- Heart-rate response varies substantially between individuals, and the available data does not include reliable individualized physiological thresholds.
- Running conditions, route changes, and measurement noise may influence the calculated decoupling target.
- The dataset is restricted to selected running sessions, including a completed-run duration window of approximately 30–120 minutes.
- The current predictive improvement over a simple baseline is small.
- The system has not been prospectively validated as a real-time warning or coaching tool.

These limitations motivate further work on target quality, feature engineering, and validation across runners and running conditions.

## Project Structure

```text
cardiac-drift/
├── data/
│   └── processed/
│       └── dataset_sample.csv
├── models/
│   └── xgb_cardiac_drift.joblib
├── scripts/
│   ├── build_dataset.py
│   ├── train_baseline.py
│   ├── explain_model.py
│   └── export_model.py
├── visualizations/
├── app.py
├── requirements.txt
└── README.md
```

## Tech Stack

| Area             | Technologies                                |
| ---------------- | ------------------------------------------- |
| Data processing  | Python, Pandas, NumPy                       |
| Machine learning | Scikit-learn, XGBoost                       |
| Validation       | GroupKFold, GroupShuffleSplit, GridSearchCV |
| Explainability   | SHAP                                        |
| Visualization    | Matplotlib, Plotly                          |
| Application      | Streamlit                                   |

## Key Takeaways

The project demonstrates a practical approach to developing and evaluating machine learning models from noisy real-world telemetry.

The most important engineering challenges involved ensuring consistent data preprocessing, defining a meaningful prediction target, preventing future-derived features from entering the model, and evaluating generalization across unseen runners.

Rather than relying on model complexity alone, the workflow uses baseline comparisons, controlled feature experiments, and explainability analysis to understand both the capabilities and limitations of the resulting system.
