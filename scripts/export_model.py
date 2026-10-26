import pandas as pd
from pathlib import Path
import joblib
from xgboost import XGBRegressor

def export_final_model():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_path = PROJECT_ROOT / "data" / "processed" / "dataset_sample.csv"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # סינון וניקוי
    valid_mask = (
        (df["total_duration_min"] <= 120.0) &
        (df["elev_gain_15m"] <= 300.0) &
        (df["speed_std_15m"] <= 5.0) 
    )
    df_filtered = df[valid_mask].copy()
    
    # מאפיינים מהונדסים
    df_filtered["hr_slope_per_elev"] = df_filtered["hr_slope_5_15"] / (df_filtered["elev_gain_15m"] + 10.0)
    df_filtered["pacing_drift_risk"] = df_filtered["hr_slope_5_15"] * df_filtered["total_duration_min"]

    features = [
        "gender", "total_duration_min", "hr_mean_5_15", "speed_mean_5_15",
        "early_decoupling_pct", "ratio_slope_5_15", "speed_slope_5_15",
        "hr_slope_5_15", "elev_gain_15m", "hr_slope_per_elev",
        "pacing_drift_risk", "speed_std_15m"
    ]
    
    X = df_filtered[features]
    y = df_filtered["target_drift_pct"]
    
    print("Training final XGBoost model with optimized parameters...")
    # שימוש בפרמטרים המנצחים שמצאנו
    model = XGBRegressor(
        max_depth=3, 
        learning_rate=0.01, 
        n_estimators=300, 
        random_state=42, 
        objective="reg:squarederror"
    )
    model.fit(X, y)
    
    export_path = models_dir / "xgb_cardiac_drift.joblib"
    joblib.dump(model, export_path)
    print(f"Success! Model successfully exported to: {export_path}")

if __name__ == "__main__":
    export_final_model()