from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Cardiac Drift Explorer",
    page_icon="🏃",
    layout="wide",
)

st.title("🏃 Cardiac Drift — Dataset Explorer")
st.caption("Exploratory analysis of running telemetry")

@st.cache_data
def load_data():
    DATA_PATH = Path(__file__).parent / "data" / "processed" / "dataset_sample.csv"
    return pd.read_csv(DATA_PATH)

df = load_data()

target = "target_drift_pct"

# Basic metrics
c1, c2, c3 = st.columns(3)

c1.metric("Valid runs", f"{len(df):,}")
c2.metric(
    "Unique runners",
    f"{df['user_id'].nunique():,}"
    if "user_id" in df.columns else "N/A"
)
c3.metric(
    "Median cardiac drift",
    f"{df[target].median():.2f}%"
)

st.divider()

# Target distribution
st.subheader("1. Target distribution")

fig = px.histogram(
    df,
    x=target,
    nbins=25,
    marginal="box",
    labels={target: "Cardiac Drift (%)"},
    title="Distribution of Cardiac Drift",
)

st.plotly_chart(fig, use_container_width=True)

# Feature vs target
st.subheader("2. Feature relationships")

excluded = {"user_id", "gender", target}
features = [
    col for col in df.select_dtypes(include="number").columns
    if col not in excluded
]

if features:
    default_feature = (
        "hr_slope_5_15"
        if "hr_slope_5_15" in features
        else features[0]
    )

    selected_feature = st.selectbox(
        "Select a feature",
        features,
        index=features.index(default_feature),
    )

    fig = px.scatter(
        df,
        x=selected_feature,
        y=target,
        hover_data=["user_id"] if "user_id" in df.columns else None,
        labels={target: "Cardiac Drift (%)"},
        title=f"{selected_feature} vs Cardiac Drift",
    )

    st.plotly_chart(fig, use_container_width=True)

# Correlation matrix
st.subheader("3. Feature correlations")

corr_data = df.select_dtypes(include="number").drop(
    columns=["user_id", "gender"],
    errors="ignore",
)

corr = corr_data.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    zmin=-1,
    zmax=1,
    color_continuous_scale="RdBu_r",
    aspect="auto",
    title="Correlation Matrix",
)

st.plotly_chart(fig, use_container_width=True)

# Data table
with st.expander("View dataset and summary statistics"):
    st.dataframe(df, use_container_width=True, hide_index=True)

    numeric_data = df.select_dtypes(include="number").drop(
        columns=["user_id", "gender"],
        errors="ignore",
    )

    st.subheader("Summary statistics")
    st.dataframe(
        numeric_data.describe().T.round(2),
        use_container_width=True,
    )

    st.subheader("Missing values")
    missing = df.isna().sum().rename("Missing values")
    st.dataframe(missing.to_frame(), use_container_width=True)
