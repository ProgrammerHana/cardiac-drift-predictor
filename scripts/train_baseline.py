import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

def load_data(file_path: Path):
    df = pd.read_csv(file_path)
    df_clean = df.dropna(subset=["target_drift_pct", "user_id"]).copy()

    valid_mask = (
        (df_clean["total_duration_min"] <= 120.0) &
        (df_clean["elev_gain_15m"] <= 300.0) &
        (df_clean["speed_std_15m"] <= 5.0) 
    )
    df_filtered = df_clean[valid_mask].copy()
    
    df_filtered["hr_slope_per_elev"] = df_filtered["hr_slope_5_15"] / (df_filtered["elev_gain_15m"] + 10.0)
    df_filtered["pacing_drift_risk"] = df_filtered["hr_slope_5_15"] * df_filtered["total_duration_min"]

    return df_filtered

def main():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "dataset_sample.csv"
    df = load_data(csv_path)

    features = [
        "gender", "total_duration_min", "hr_mean_5_15", "speed_mean_5_15",
        "early_decoupling_pct", "ratio_slope_5_15", "speed_slope_5_15",
        "hr_slope_5_15", "elev_gain_15m", "hr_slope_per_elev",
        "pacing_drift_risk", "speed_std_15m"
    ]
    target = "target_drift_pct"

    X = df[features]
    y = df[target]
    groups = df["user_id"]

    gkf = GroupKFold(n_splits=5)
    
    print(f"Working dataset: {len(df)} runs across {groups.nunique()} runners.")
    print("Evaluating Baselines with 5-Fold GroupKFold...\n")

    # 1. אימון קווי הבסיס (Baselines)
    baselines = {
        "Dummy (Mean Baseline)": DummyRegressor(strategy="mean"),
        "Ridge (Linear Baseline)": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "Huber (Robust Baseline)": make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35))
    }

    baseline_results = []
    for name, model in baselines.items():
        maes = []
        for train_idx, test_idx in gkf.split(X, y, groups=groups):
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = model.predict(X.iloc[test_idx])
            maes.append(mean_absolute_error(y.iloc[test_idx], preds))
        
        baseline_results.append({"Model": name, "MAE (% Drift)": np.mean(maes)})
        print(f"{name} MAE: {np.mean(maes):.3f}")

    # 2. אימון מתקדם - XGBoost עם Hyperparameter Tuning
    print("\nStarting Hyperparameter Tuning for XGBoost...")
    xgb = XGBRegressor(random_state=42, objective="reg:squarederror", n_jobs=-1)
    
    param_grid = {
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300]
    }

    # GridSearchCV משתמש ב-GroupKFold כדי למנוע זליגת נתונים בבחירת הפרמטרים
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=gkf,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    
    grid.fit(X, y, groups=groups)
    
    best_mae = -grid.best_score_
    best_params = grid.best_params_
    
    print("\n=== Final Benchmark Results ===")
    results_df = pd.DataFrame(baseline_results)
    results_df = pd.concat([results_df, pd.DataFrame([{"Model": "Tuned XGBoost", "MAE (% Drift)": best_mae}])], ignore_index=True)
    
    print(results_df.to_string(index=False))
    print(f"\nBest XGBoost Parameters: {best_params}")

    print("\n=== XGBoost Feature Importances ===")
    best_xgb = grid.best_estimator_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": np.round(best_xgb.feature_importances_, 4)
    }).sort_values(by="Importance", ascending=False)
    
    print(importance_df.to_string(index=False))

if __name__ == "__main__":
    main()