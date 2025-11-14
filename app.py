import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
import os

import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

st.set_page_config(page_title="Nepal Agri Price Analyzer", layout="wide")


# ============================
# Load Dataset
# ============================
@st.cache_data
def load_data(source):
    try:
        if hasattr(source, "read"):
            return pd.read_csv(source, parse_dates=['date'])
        else:
            return pd.read_csv(source, parse_dates=['date'])
    except Exception as e:
        st.error(f"Dataset load error: {e}")
        return pd.DataFrame()


# ============================
# Load Nepal Province GeoJSON (Stable Version)
# ============================
@st.cache_data
def load_nepal_province_geojson():

    url = "https://raw.githubusercontent.com/mesaugat/geoJSON-Nepal/master/nepal-province.geojson"

    try:
        r = requests.get(url)
        return r.json()
    except:
        st.error("Failed to load Nepal GeoJSON.")
        return None


# ============================
# Fix Province Name Mapping
# ============================
def normalize_province_name(x):
    """Match dataset province names to GeoJSON names."""
    if pd.isna(x):
        return x

    x = x.strip()

    mapping = {
        "Sudurpashchim Province": "Sudurpashchim",
        "Karnali Province": "Karnali",
        "Lumbini Province": "Lumbini",
        "Gandaki Province": "Gandaki",
        "Bagmati Province": "Bagmati",
        "Madhesh Province": "Madhesh",
        "Province 1": "Koshi",
        "Koshi Province": "Koshi"
    }

    return mapping.get(x, x)


# ============================
# Train Models (All Pipelines)
# ============================
def train_all_models(df):

    df = df.copy()
    df["month"] = df["date"].dt.month

    cat_cols = ["state", "city", "crop_type", "season", "month"]
    num_cols = ["rainfall_mm", "temperature_c"]

    X = df[cat_cols + num_cols]
    y = df["price_₹/ton"]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
    ])

    models = {
        "XGBoost": xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42
        ),
        "Random Forest": RandomForestRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }

    trained = {}
    performance = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        pipe = Pipeline([("pre", pre), ("reg", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        trained[name] = pipe
        performance[name] = {"MAE": mae, "R2": r2}

    # save
    with open("models.pkl", "wb") as f:
        pickle.dump({"models": trained}, f)

    return performance


# ============================
# Streamlit UI
# ============================
def main():

    st.title("🌾 Nepal Agricultural Market Price Analyzer")

    st.sidebar.header("Upload Dataset")
    file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = load_data(file)
    else:
        df = load_data("cleaned_dataset.csv")

    if df.empty:
        st.warning("Please upload a dataset.")
        return

    st.sidebar.success(f"Loaded {len(df):,} rows")

    df["province_norm"] = df["state"].apply(normalize_province_name)

    tabs = st.tabs(["Dashboard", "Training", "Prediction", "Regional Analysis"])

    # =======================
    # Dashboard
    # =======================
    with tabs[0]:
        st.subheader("📈 Latest Entries")
        st.dataframe(df.tail(20))

    # =======================
    # Training
    # =======================
    with tabs[1]:
        st.subheader("🔧 Train All Models")

        if st.button("Train Models"):
            with st.spinner("Training..."):
                result = train_all_models(df)
            st.success("Models Trained Successfully!")
            st.json(result)

    # =======================
    # Prediction
    # =======================
    with tabs[2]:
        st.subheader("🎯 Predict Price")

        if not os.path.exists("models.pkl"):
            st.info("Train models first.")
        else:
            with open("models.pkl", "rb") as f:
                saved = pickle.load(f)
                models = saved["models"]

            model_name = st.selectbox("Select Model", list(models.keys()))
            model = models[model_name]

            state = st.selectbox("State", sorted(df["state"].unique()))
            city = st.selectbox("City", sorted(df["city"].unique()))
            crop = st.selectbox("Crop", sorted(df["crop_type"].unique()))
            season = st.selectbox("Season", sorted(df["season"].unique()))
            rainfall = st.number_input("Rainfall (mm)", value=float(df["rainfall_mm"].mean()))
            temp = st.number_input("Temperature (°C)", value=float(df["temperature_c"].mean()))
            month = st.number_input("Month", 1, 12, 6)

            X_in = pd.DataFrame([[
                state, city, crop, season, month, rainfall, temp
            ]], columns=["state", "city", "crop_type", "season", "month", "rainfall_mm", "temperature_c"])

            if st.button("Predict"):
                pred = model.predict(X_in)[0]
                st.success(f"Predicted Price → ₹{pred:.2f} / ton")

    # =======================
    # Regional Analysis (Map)
    # =======================
    with tabs[3]:
        st.subheader("🗺️ Regional Price Distribution (Nepal Provinces)")

        geojson = load_nepal_province_geojson()

        if geojson is None:
            st.error("GeoJSON could not be loaded.")
        else:
            # compute avg price by province
            avgp = df.groupby("province_norm")["price_₹/ton"].mean().reset_index()

            fig = px.choropleth(
                avgp,
                geojson=geojson,
                featureidkey="properties.PROVINCE",
                locations="province_norm",
                color="price_₹/ton",
                color_continuous_scale="YlOrBr",
                title="Average Price by Province (Nepal)",
                hover_name="province_norm"
            )

            fig.update_geos(fitbounds="locations", visible=False)

            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
