import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import shap
from sklearn.model_selection import GroupShuffleSplit
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
    csv_path = project_root / "data" / "processed" / "dataset_sample.csv"
    output_dir = project_root / "visualizations"
    output_dir.mkdir(exist_ok=True)

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
    groups = df["user_id"]

    # שימוש באותו פיצול דטרמיניסטי כמו בשלב האימון
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    dev_idx, _ = next(splitter.split(df[features], groups=groups))

    X_dev = df[features].iloc[dev_idx]
    y_dev = df[target].iloc[dev_idx]

    print(f"Training tuned XGBoost on {len(X_dev)} development runs...")
    model = XGBRegressor(
        learning_rate=0.01,
        max_depth=3,
        n_estimators=300,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=-1,
    )
    model.fit(X_dev, y_dev)

    print("Computing SHAP values via TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_dev)

    # 1. Summary Plot (Beeswarm)
    print("Generating Summary Plot...")
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.title("SHAP Summary Plot - Feature Impact on Aerobic Drift", fontsize=12)
    plt.tight_layout()
    summary_path = output_dir / "shap_summary.png"
    plt.savefig(summary_path, dpi=300)
    plt.close()
    print(f"Saved: {summary_path}")

    # 2. Dependence Plot: hr_slope_5_15
    print("Generating Dependence Plot for hr_slope_5_15...")
    plt.figure(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, "hr_slope_5_15"], color=shap_values[:, "elev_gain_15m"], show=False)
    plt.title("SHAP Dependence: HR Slope (5-15 min) colored by Elevation Gain", fontsize=11)
    plt.tight_layout()
    dep_hr_path = output_dir / "shap_dependence_hr_slope.png"
    plt.savefig(dep_hr_path, dpi=300)
    plt.close()
    print(f"Saved: {dep_hr_path}")

    # 3. Dependence Plot: early_decoupling_pct
    print("Generating Dependence Plot for early_decoupling_pct...")
    plt.figure(figsize=(8, 5))
    shap.plots.scatter(shap_values[:, "early_decoupling_pct"], show=False)
    plt.title("SHAP Dependence: Early Decoupling %", fontsize=11)
    plt.tight_layout()
    dep_decoupling_path = output_dir / "shap_dependence_early_decoupling.png"
    plt.savefig(dep_decoupling_path, dpi=300)
    plt.close()
    print(f"Saved: {dep_decoupling_path}")

    print("\nAll SHAP visualizations generated successfully.")


if __name__ == "__main__":
    main()