import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import joblib

# הגדרת תצורת העמוד
st.set_page_config(page_title="Cardiac Drift Predictor", layout="wide", page_icon="🏃‍♂️")

# 1. טעינת נתונים (עבור הגרפים והסטטיסטיקות)
@st.cache_data
def load_data():
    data_path = Path("data/processed/dataset_sample.csv")
    df = pd.read_csv(data_path)
    
    valid_mask = (
        (df["total_duration_min"] <= 120.0) &
        (df["elev_gain_15m"] <= 300.0) &
        (df["speed_std_15m"] <= 5.0) 
    )
    df_filtered = df[valid_mask].copy()
    
    # אין צורך יותר לחשב כאן את hr_slope_per_elev או pacing_drift_risk!
    return df_filtered

df = load_data()

# 2. טעינת המודל המוכן (Inference Only)
@st.cache_resource
def load_model():
    model_path = Path("models/xgb_cardiac_drift.joblib")
    return joblib.load(model_path)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found! Please run 'python scripts/export_model.py' first.")
    st.stop()

# --- ממשק המשתמש הראשי ---
st.title("🏃‍♂️ Cardiac Drift Early Warning System")
st.markdown("Predicting late-stage aerobic decoupling based **strictly on the first 15 minutes** of running data.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Dataset Overview")
    st.metric(label="Valid Runs", value=f"{len(df):,}")
    st.metric(label="Unique Runners", value=f"{df['user_id'].nunique():,}")
    st.metric(label="Avg Drift", value=f"{df['target_drift_pct'].mean():.1f}%")
    st.dataframe(df[["total_duration_min", "elev_gain_15m", "hr_slope_5_15", "target_drift_pct"]].head(10))

with col2:
    st.header("Physiological Signal")
    fig = px.scatter(
        df, x="hr_slope_5_15", y="target_drift_pct", color="elev_gain_15m",
        color_continuous_scale="Viridis", title="Cardiac Drift vs. Initial HR Slope",
        labels={"hr_slope_5_15": "HR Slope (bpm/min)", "target_drift_pct": "Final Aerobic Drift (%)"},
        trendline="ols"
    )
    fig.update_yaxes(range=[-10, 30])
    st.plotly_chart(fig, use_container_width=True)

# --- סימולטור חיזוי (Sidebar) ---
st.sidebar.header("⚙️ Live ML Simulator")
st.sidebar.markdown("Adjust warmup metrics (first 15 min) to predict final drift.")

sim_elev = st.sidebar.slider("Elevation Gain in 15m (m)", 0, 150, 10)
sim_hr_mean = st.sidebar.slider("Avg HR (bpm)", 110, 180, 145)
sim_speed = st.sidebar.slider("Avg Speed (km/h)", 8.0, 16.0, 11.0)
sim_hr_slope = st.sidebar.slider("HR Slope (bpm/min)", -1.0, 3.0, 1.5)

# הכנת הנתונים למודל (9 מאפיינים בלבד - תואם ב-100% לאימון)
input_features = [
    "gender",
    "hr_mean_5_15",
    "speed_mean_5_15",
    "early_decoupling_pct",
    "ratio_slope_5_15",
    "speed_slope_5_15",
    "hr_slope_5_15",
    "elev_gain_15m",
    "speed_std_15m"
]

input_df = pd.DataFrame([[
    1,                                      # gender (male default)
    sim_hr_mean,                            # hr_mean_5_15
    sim_speed,                              # speed_mean_5_15
    df["early_decoupling_pct"].median(),    # early_decoupling_pct
    df["ratio_slope_5_15"].median(),        # ratio_slope_5_15
    df["speed_slope_5_15"].median(),        # speed_slope_5_15
    sim_hr_slope,                           # hr_slope_5_15
    sim_elev,                               # elev_gain_15m
    df["speed_std_15m"].median()            # speed_std_15m
]], columns=input_features)

# הרצת התחזית בעזרת XGBoost
predicted_drift = model.predict(input_df)[0]

st.sidebar.divider()
st.sidebar.subheader("🤖 Model Prediction")

# עיצוב התוצאה לפי רמת סיכון
if predicted_drift < 3.0:
    st.sidebar.success(f"Low Risk\n\nPredicted Drift: {predicted_drift:.1f}%")
elif 3.0 <= predicted_drift < 6.0:
    st.sidebar.warning(f"Moderate Risk\n\nPredicted Drift: {predicted_drift:.1f}%")
else:
    st.sidebar.error(f"High Risk (Crash Imminent)\n\nPredicted Drift: {predicted_drift:.1f}%")

st.sidebar.markdown("*Powered by Tuned XGBoost*")