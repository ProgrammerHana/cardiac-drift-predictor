import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBRegressor


def load_data(file_path: Path):
    df = pd.read_csv(file_path)
    df_clean = df.dropna(subset=["target_drift_pct", "user_id"]).copy()

    valid_mask = (
        (df_clean["total_duration_min"] <= 120.0)
        & (df_clean["elev_gain_15m"] <= 300.0)
        & (df_clean["speed_std_15m"] <= 5.0)
    )
    return df_clean[valid_mask].copy()


def main():
    project_root = Path(__file__).resolve().parent.parent
    data_path = project_root / "data" / "processed" / "dataset_sample.csv"
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)

    df = load_data(data_path)

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

    X = df[features]
    y = df["target_drift_pct"]

    print(f"Training final production XGBoost model on {len(X)} runs...")
    model = XGBRegressor(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=300,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model.fit(X, y)

    model_path = models_dir / "xgb_cardiac_drift.joblib"
    joblib.dump(model, model_path)
    print(f"Success! Model exported to: {model_path}")


if __name__ == "__main__":
    main()
