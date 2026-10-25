import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_data(file_path: Path):
    df = pd.read_csv(file_path)
    df_clean = df.dropna(subset=["target_drift_pct", "user_id"]).copy()

    valid_mask = (
        (df_clean["total_duration_min"] <= 120.0) &
        (df_clean["elev_gain_15m"] <= 300.0) &
        (df_clean["speed_std_15m"] <= 3.0)
    )
    df_filtered = df_clean[valid_mask].copy()

    # --- הוספת מאפייני אינטראקציה (Feature Engineering) ---
    # 1. שיפוע דופק מנוקה מעליות שטח
    df_filtered["hr_slope_per_elev"] = df_filtered["hr_slope_5_15"] / (df_filtered["elev_gain_15m"] + 10.0)
    
    # 2. סיכון מצטבר: שילוב בין עצימות ההתחלה למשך הריצה
    df_filtered["pacing_drift_risk"] = df_filtered["hr_slope_5_15"] * df_filtered["total_duration_min"]
    
    # 3. יציבות הקצב ביחס ליחס הדופק/מהירות הבסיסי
    df_filtered["speed_stability_index"] = df_filtered["speed_std_15m"] / df_filtered["initial_ratio"]

    return df_filtered

def train_and_evaluate(csv_path: Path):
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path.resolve()}")

    df = load_data(csv_path)

    # רשימת המאפיינים המורחבת הכוללת את האינטראקציות
    # רשימת המאפיינים המנופה (Feature Selection)
    features = [
        "gender",
        "total_duration_min",
        "initial_ratio",
        "hr_slope_per_elev",      # מחליף את השיפוע והגובה הגולמיים
        "pacing_drift_risk",      # משלב שיפוע וזמן
        "speed_stability_index"   # מחליף את סטיית התקן הגולמית של המהירות
    ]
    target = "target_drift_pct"

    X = df[features]
    y = df[target]
    groups = df["user_id"]

    n_unique_users = groups.nunique()
    n_splits = min(5, n_unique_users)
    print(f"Working dataset: {len(df)} runs across {n_unique_users} unique runners.")
    print(f"Evaluating with {n_splits}-Fold GroupKFold...\n")

    gkf = GroupKFold(n_splits=n_splits)

    models = {
        "Dummy (Mean Baseline)": DummyRegressor(strategy="mean"),
        "Ridge (Standardized)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest (Depth=5)": RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    }

    cv_metrics = {name: {"mae": [], "rmse": [], "r2": []} for name in models}

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            cv_metrics[name]["mae"].append(mean_absolute_error(y_test, preds))
            cv_metrics[name]["rmse"].append(np.sqrt(mean_squared_error(y_test, preds)))
            cv_metrics[name]["r2"].append(r2_score(y_test, preds))

    summary = []
    for name in models:
        summary.append({
            "Model": name,
            "MAE (% Drift)": round(np.mean(cv_metrics[name]["mae"]), 3),
            "RMSE": round(np.mean(cv_metrics[name]["rmse"]), 3),
            "R2 Score": round(np.mean(cv_metrics[name]["r2"]), 4)
        })

    print("=== Baseline Benchmark Results (With Engineered Features) ===")
    print(pd.DataFrame(summary).to_string(index=False))

    # בדיקת משקולות וחשיבות עדכנית
    print("\n=== Feature Importances (Trained on full dataset) ===")
    ridge_pipe = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge_pipe.fit(X, y)
    ridge_weights = ridge_pipe.named_steps["ridge"].coef_
    
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X, y)
    rf_importances = rf_model.feature_importances_
    
    importance_df = pd.DataFrame({
        "Feature": features,
        "Ridge Weight": np.round(ridge_weights, 4),
        "RF Importance": np.round(rf_importances, 4)
    }).sort_values(by="RF Importance", ascending=False)
    
    print(importance_df.to_string(index=False))

if __name__ == "__main__":
    DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "dataset_sample.csv"
    train_and_evaluate(DATA_PATH)