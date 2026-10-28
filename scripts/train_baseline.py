import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def load_data(file_path: Path):
    df = pd.read_csv(file_path)
    df_clean = df.dropna(subset=["target_drift_pct", "user_id"]).copy()

    valid_mask = (
        (df_clean["total_duration_min"] <= 120.0)
        & (df_clean["elev_gain_15m"] <= 300.0)
        & (df_clean["speed_std_15m"] <= 5.0)
    )
    df_filtered = df_clean[valid_mask].copy()

    df_filtered["hr_slope_per_elev"] = df_filtered["hr_slope_5_15"] / (
        df_filtered["elev_gain_15m"] + 10.0
    )

    return df_filtered


def regression_metrics(y_true, predictions):
    return {
        "MAE": mean_absolute_error(y_true, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_true, predictions)),
        "R2": r2_score(y_true, predictions),
    }


def run_hr_slope_per_elev_ablation(X_dev, y_dev, cv_splits, features):
    feature_without_ratio = [
        feature for feature in features if feature != "hr_slope_per_elev"
    ]
    if len(feature_without_ratio) != len(features) - 1:
        raise ValueError("Expected exactly one hr_slope_per_elev feature.")

    forbidden_features = {"total_duration_min", "pacing_drift_risk"}
    if forbidden_features.intersection(features):
        raise ValueError("Full-run duration features must not be model inputs.")

    params = {
        "learning_rate": 0.01,
        "max_depth": 3,
        "n_estimators": 300,
    }
    fold_results = []
    for fold_number, (train_idx, validation_idx) in enumerate(cv_splits, start=1):
        model_a = XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
            **params,
        )
        model_b = XGBRegressor(
            random_state=42,
            objective="reg:squarederror",
            n_jobs=-1,
            **params,
        )
        model_a.fit(X_dev.iloc[train_idx][features], y_dev.iloc[train_idx])
        model_b.fit(
            X_dev.iloc[train_idx][feature_without_ratio],
            y_dev.iloc[train_idx],
        )
        mae_a = mean_absolute_error(
            y_dev.iloc[validation_idx],
            model_a.predict(X_dev.iloc[validation_idx][features]),
        )
        mae_b = mean_absolute_error(
            y_dev.iloc[validation_idx],
            model_b.predict(
                X_dev.iloc[validation_idx][feature_without_ratio]
            ),
        )
        fold_results.append(
            {
                "Fold": fold_number,
                "MAE A (with feature)": mae_a,
                "MAE B (without feature)": mae_b,
                "B - A (positive favors A)": mae_b - mae_a,
            }
        )

    fold_table = pd.DataFrame(fold_results)
    mean_mae_a = fold_table["MAE A (with feature)"].mean()
    mean_mae_b = fold_table["MAE B (without feature)"].mean()
    mae_improvement = mean_mae_b - mean_mae_a
    relative_improvement = mae_improvement / mean_mae_b * 100.0

    print("\n=== XGBoost hr_slope_per_elev ablation: development CV only ===")
    print(f"Model A features: {', '.join(features)}")
    print(f"Model B features: {', '.join(feature_without_ratio)}")
    print(f"Fixed XGBoost parameters: {params}")
    print(
        "Future-derived input features in either model: "
        f"{bool(forbidden_features.intersection(features))}"
    )
    print(
        "All telemetry-derived input features are available by minute 15: yes "
        "(gender is static metadata)."
    )
    print(
        fold_table.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    print("\n=== Fold summary ===")
    print(
        f"Model A mean MAE: {mean_mae_a:.4f}; "
        f"fold SD: {fold_table['MAE A (with feature)'].std(ddof=1):.4f}"
    )
    print(
        f"Model B mean MAE: {mean_mae_b:.4f}; "
        f"fold SD: {fold_table['MAE B (without feature)'].std(ddof=1):.4f}"
    )
    print(
        f"MAE improvement from including hr_slope_per_elev: "
        f"{mae_improvement:.4f} points "
        f"({relative_improvement:.2f}% relative to Model B)."
    )
    print(
        "Positive B - A favors including the feature. This comparison uses "
        "fixed XGBoost hyperparameters and is not proof that the feature helps "
        "every model."
    )
    print(
        "The final test set was not used for predictions or metrics. "
        "The existing total_duration_min <= 120 filter still conditions the "
        "evaluated population on full-run information."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation-hr-slope-per-elev",
        action="store_true",
        help="Compare XGBoost with and without hr_slope_per_elev on development CV only.",
    )
    args = parser.parse_args()

    csv_path = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "dataset_sample.csv"
    )
    df = load_data(csv_path)

    features = [
        "gender",
        "hr_mean_5_15",
        "speed_mean_5_15",
        "early_decoupling_pct",
        "ratio_slope_5_15",
        "speed_slope_5_15",
        "hr_slope_5_15",
        "elev_gain_15m",
        
        "speed_std_15m",
    ]
    target = "target_drift_pct"
    forbidden_features = {"total_duration_min", "pacing_drift_risk"}
    if forbidden_features.intersection(features):
        raise ValueError("Full-run duration features must not be model inputs.")

    X = df[features]
    groups = df["user_id"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    development_idx, test_idx = next(splitter.split(X, groups=groups))
    X_dev = X.iloc[development_idx]
    y_dev = df[target].iloc[development_idx]
    groups_dev = groups.iloc[development_idx]
    groups_test = groups.iloc[test_idx]

    overlap = set(groups_dev).intersection(groups_test)
    if overlap:
        raise RuntimeError(f"Runner overlap between development and test: {len(overlap)}")

    gkf = GroupKFold(n_splits=5)
    cv_splits = list(gkf.split(X_dev, y_dev, groups=groups_dev))

    if args.ablation_hr_slope_per_elev:
        print("=== Reused runner split ===")
        print(
            f"Development: {len(development_idx)} runs, "
            f"{groups_dev.nunique()} runners; final test: {len(test_idx)} runs, "
            f"{groups_test.nunique()} runners."
        )
        print(f"Runner overlap: {len(overlap)}")
        run_hr_slope_per_elev_ablation(X_dev, y_dev, cv_splits, features)
        return

    X_test = X.iloc[test_idx]
    y_test = df[target].iloc[test_idx]

    print("=== Data split ===")
    print(
        f"Development: {len(X_dev)} runs, {groups_dev.nunique()} runners; "
        f"final test: {len(X_test)} runs, {groups_test.nunique()} runners."
    )
    print(f"Runner overlap: {len(overlap)}")
    print(f"Model features: {', '.join(features)}")
    print(
        "Full-run duration and duration-derived features in X: "
        f"{bool(forbidden_features.intersection(features))}"
    )
    print(
        "Note: gender is static record metadata; all telemetry-derived inputs "
        "are computed from the first 15 minutes."
    )

    models = {
        "Dummy (Mean Baseline)": DummyRegressor(strategy="mean"),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Huber": make_pipeline(
            StandardScaler(), HuberRegressor(epsilon=1.35)
        ),
    }

    cv_results = []
    for name, model in models.items():
        fold_metrics = []
        for train_idx, validation_idx in cv_splits:
            fold_model = clone(model)
            fold_model.fit(X_dev.iloc[train_idx], y_dev.iloc[train_idx])
            predictions = fold_model.predict(X_dev.iloc[validation_idx])
            fold_metrics.append(
                regression_metrics(y_dev.iloc[validation_idx], predictions)
            )
        cv_results.append(
            {
                "Model": name,
                **{
                    metric: np.mean([fold[metric] for fold in fold_metrics])
                    for metric in ("MAE", "RMSE", "R2")
                },
            }
        )

    xgb = XGBRegressor(
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300],
    }
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=cv_splits,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
        refit="mae",
        n_jobs=1,
        return_train_score=False,
    )
    grid.fit(X_dev, y_dev)

    best_index = grid.best_index_
    xgb_fold_metrics = [
        {
            "MAE": -grid.cv_results_[f"split{fold}_test_mae"][best_index],
            "RMSE": -grid.cv_results_[f"split{fold}_test_rmse"][best_index],
            "R2": grid.cv_results_[f"split{fold}_test_r2"][best_index],
        }
        for fold in range(len(cv_splits))
    ]
    cv_results.append(
        {
            "Model": "XGBoost (GridSearchCV)",
            **{
                metric: np.mean([fold[metric] for fold in xgb_fold_metrics])
                for metric in ("MAE", "RMSE", "R2")
            },
        }
    )

    cv_table = pd.DataFrame(cv_results).sort_values("MAE").reset_index(drop=True)
    selected_name = cv_table.loc[0, "Model"]
    selected_cv_model = (
        clone(grid.best_estimator_)
        if selected_name == "XGBoost (GridSearchCV)"
        else clone(models[selected_name])
    )

    print("\n=== Development 5-fold GroupKFold CV ===")
    print(cv_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSelected by development CV mean MAE: {selected_name}")
    if selected_name == "XGBoost (GridSearchCV)":
        print(f"Selected XGBoost parameters: {grid.best_params_}")

    selected_model = clone(selected_cv_model)
    selected_model.fit(X_dev, y_dev)
    dummy = clone(models["Dummy (Mean Baseline)"])
    dummy.fit(X_dev, y_dev)

    test_results = []
    for name, model in (
        ("Dummy (Mean Baseline)", dummy),
        (selected_name, selected_model),
    ):
        predictions = model.predict(X_test)
        test_results.append(
            {"Model": name, **regression_metrics(y_test, predictions)}
        )

    test_table = pd.DataFrame(test_results)
    dummy_mae = test_table.loc[
        test_table["Model"] == "Dummy (Mean Baseline)", "MAE"
    ].iloc[0]
    selected_mae = test_table.loc[test_table["Model"] == selected_name, "MAE"].iloc[0]
    mae_improvement = dummy_mae - selected_mae
    mae_improvement_pct = mae_improvement / dummy_mae * 100.0

    print("\n=== Final held-out runner test ===")
    print(test_table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(
        f"MAE improvement over Dummy: {mae_improvement:.4f} percentage points "
        f"({mae_improvement_pct:.2f}%)."
    )
    print(
        "\nPopulation limitation: the existing total_duration_min <= 120 filter "
        "uses full-run information, so these results do not estimate performance "
        "for all possible runs."
    )


if __name__ == "__main__":
    main()
