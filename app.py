# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import requests
import os
from io import BytesIO, StringIO

import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

st.set_page_config(page_title="AgriPrice Analyzer (Nepal)", layout="wide")

# -------------------------
# Utility: load CSV (file path or UploadedFile)
# -------------------------
@st.cache_data
def load_data(file_input='cleaned_dataset.csv'):
    try:
        # uploaded_file has .read
        if hasattr(file_input, "read"):
            # Streamlit's UploadedFile -> bytes. Use pandas read_csv directly
            return pd.read_csv(file_input, parse_dates=['date'])
        # string path
        if isinstance(file_input, str) and os.path.exists(file_input):
            return pd.read_csv(file_input, parse_dates=['date'])
        # not found -> empty DF
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# -------------------------
# Robust GeoJSON loader for Nepal provinces
# returns (geojson_obj, property_key) where property_key is like 'name' or 'PROVINCE'
# -------------------------
def load_nepal_geojson():
    # Try local file first
    local_path = os.path.join("assets", "nepal_provinces.geojson")
    candidates = [
        local_path,
        "https://raw.githubusercontent.com/sandeshchapagain/Nepal-GeoJSON/master/Provinces/nepal-provinces.geojson",
        "https://raw.githubusercontent.com/sandeshchapagain/nepal-geojson/main/nepal-provinces.geojson",
        # older/alternate raw links (kept for robustness)
        "https://raw.githubusercontent.com/sandeshchapagain/Nepal-GeoJSON/main/Provinces/nepal-provinces.geojson",
    ]

    for src in candidates:
        try:
            if src == local_path:
                if os.path.exists(src):
                    with open(src, "r", encoding="utf-8") as f:
                        geo = json.load(f)
                        pk = detect_feature_property_key(geo)
                        return geo, pk
                else:
                    continue
            else:
                resp = requests.get(src, timeout=10)
                if resp.status_code != 200:
                    continue
                # Try JSON parse
                try:
                    geo = resp.json()
                    pk = detect_feature_property_key(geo)
                    return geo, pk
                except Exception:
                    # maybe content is HTML or not JSON; skip
                    continue
        except Exception:
            continue

    # Fallback: embedded simplified polygons (very coarse; just to show map)
    embedded = {
        "type": "FeatureCollection",
        "features": [
            {"type":"Feature","properties":{"PROVINCE":"Province 1"},"geometry":{"type":"Polygon","coordinates":[[[86.5,26.8],[87.8,26.8],[87.8,27.9],[86.5,27.9],[86.5,26.8]]]}},
            {"type":"Feature","properties":{"PROVINCE":"Madhesh Province"},"geometry":{"type":"Polygon","coordinates":[[[85.0,26.4],[86.5,26.4],[86.5,27.2],[85.0,27.2],[85.0,26.4]]]}},
            {"type":"Feature","properties":{"PROVINCE":"Bagmati Province"},"geometry":{"type":"Polygon","coordinates":[[[85.2,27.3],[86.4,27.3],[86.4,28.2],[85.2,28.2],[85.2,27.3]]]}},
            {"type":"Feature","properties":{"PROVINCE":"Gandaki Province"},"geometry":{"type":"Polygon","coordinates":[[[84.0,27.5],[85.2,27.5],[85.2,28.4],[84.0,28.4],[84.0,27.5]]]}},
            {"type":"Feature","properties":{"PROVINCE":"Lumbini Province"},"geometry":{"type":"Polygon","coordinates":[[[82.8,26.5],[84.2,26.5],[84.2,27.6],[82.8,27.6],[82.8,26.5]]]}},
            {"type":"Feature","properties":{"PROVINCE":"Karnali Province"},"geometry":{"type":"Polygon","coordinates":[[[81.8,28.0],[83.3,28.0],[83.3,29.1],[81.8,29.1],[81.8,28.0]]]}},
            {"type":"Feature","properties":{"PROVINCE":"Sudurpashchim Province"},"geometry":{"type":"Polygon","coordinates":[[[80.0,28.0],[81.6,28.0],[81.6,29.2],[80.0,29.2],[80.0,28.0]]]}}
        ]
    }
    return embedded, "PROVINCE"

def detect_feature_property_key(geojson_obj):
    # Inspect first non-empty feature properties to find likely property key storing province name
    for feat in geojson_obj.get("features", []):
        props = feat.get("properties", {})
        # common keys: 'name', 'NAME_1', 'province', 'PROVINCE'
        for k in ["PROVINCE", "NAME_1", "name", "province", "Name", "NAME"]:
            if k in props:
                return k
    # fallback
    return "name"

# -------------------------
# Build preprocessing + models (pipelines)
# -------------------------
def build_and_train_models(df):
    """
    Train three models (XGB, RF, Linear) using a ColumnTransformer + Pipeline.
    Save pipelines to 'models.pkl' as {'models': {...}, 'feature_cols': feature_cols}
    """
    df = df.copy()
    df['month'] = df['date'].dt.month

    # define feature columns
    cat_cols = ['state', 'city', 'crop_type', 'season', 'month']
    num_cols = ['rainfall_mm', 'temperature_c']

    # ensure columns exist
    for c in cat_cols + num_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in dataset: {c}")

    X = df[cat_cols + num_cols]
    y = df['price_₹/ton']

    # Column transformer: scale numeric, one-hot categorical
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
        ],
        remainder='drop'
    )

    # Define pipelines
    pipelines = {
        "XGBoost": Pipeline([("preproc", preprocessor), ("reg", xgb.XGBRegressor(objective='reg:squarederror', random_state=42))]),
        "Random Forest": Pipeline([("preproc", preprocessor), ("reg", RandomForestRegressor(random_state=42))]),
        "Linear Regression": Pipeline([("preproc", preprocessor), ("reg", LinearRegression())])
    }

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained = {}
    perf = {}
    for name, pipe in pipelines.items():
        # fit
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        trained[name] = pipe
        perf[name] = {"MAE": float(mae), "R2": float(r2)}

    # save pipelines and metadata
    with open("models.pkl", "wb") as f:
        pickle.dump({"models": trained, "feature_cols": X.columns.tolist()}, f)

    return perf

# -------------------------
# Helpers: normalize province strings to expected forms
# -------------------------
def normalize_province_name(s):
    if pd.isna(s):
        return s
    s2 = str(s).strip()
    # many dataset values already like 'Madhesh Province' or 'Sudurpashchim Province'
    # just ensure consistent capitalization
    return s2

# -------------------------
# App UI
# -------------------------
def main():
    st.title("AgriPrice Analyzer — Nepal (XGB / RF / Linear)")

    st.sidebar.header("Data")
    uploaded = st.sidebar.file_uploader("Upload cleaned CSV", type=["csv"])

    if uploaded:
        df = load_data(uploaded)
    else:
        df = load_data("cleaned_dataset.csv")

    if df.empty:
        st.warning("No dataset loaded — upload a CSV in the sidebar or place 'cleaned_dataset.csv' next to the app.")
        return

    st.sidebar.success(f"Loaded dataset with {len(df):,} rows")

    # quick preview
    if st.sidebar.checkbox("Show data sample"):
        st.dataframe(df.head(20))

    # Train models button
    if st.sidebar.button("Train 3 Models (XGB, RF, Linear)"):
        with st.spinner("Training models — this may take a while..."):
            try:
                perf = build_and_train_models(df)
                st.success("Models trained and saved to models.pkl")
                st.json(perf)
            except Exception as e:
                st.error(f"Training failed: {e}")

    # If models exist, load
    models_exist = os.path.exists("models.pkl")
    models = {}
    feature_cols = None
    if models_exist:
        with open("models.pkl", "rb") as f:
            data = pickle.load(f)
            models = data.get("models", {})
            feature_cols = data.get("feature_cols", None)

    # Tabs
    tabs = st.tabs(["Dashboard", "Historical", "Weather", "Predict", "Regional Map"])

    # Dashboard
    with tabs[0]:
        st.header("Market Overview")
        st.metric("Avg price (₹/ton)", f"{df['price_₹/ton'].mean():.2f}")
        st.write("Top 10 latest rows")
        st.dataframe(df.sort_values("date", ascending=False).head(10))

    # Historical trends
    with tabs[1]:
        st.header("Historical Trends")
        crop = st.selectbox("Choose crop", df['crop_type'].unique())
        dr = st.date_input("Date range", value=(df['date'].min().date(), df['date'].max().date()))
        mask = (df['crop_type']==crop) & (df['date'].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1])))
        sub = df[mask]
        if sub.empty:
            st.warning("No data for selection — showing full crop history")
            sub = df[df['crop_type']==crop]
        fig = px.line(sub, x='date', y='price_₹/ton', title=f"{crop} price trend")
        st.plotly_chart(fig, use_container_width=True)

    # Weather
    with tabs[2]:
        st.header("Weather vs Price")
        factor = st.selectbox("Weather factor", ['rainfall_mm','temperature_c'])
        fig = px.scatter(df, x=factor, y='price_₹/ton', color='crop_type', trendline='ols')
        st.plotly_chart(fig, use_container_width=True)

    # Predict
    with tabs[3]:
        st.header("Predict price (choose model)")
        if not models:
            st.info("No trained models found — train models from the sidebar.")
        else:
            model_name = st.selectbox("Model", list(models.keys()))
            pipe = models[model_name]

            # Build input fields from feature_cols if available else derive from df
            if feature_cols:
                # feature_cols include categorical state,city,crop_type,season,month and numeric rainfall_mm,temperature_c
                # We'll ask user for state, city, crop_type, season and optional rainfall,temp; month can be auto
                st.subheader("Input region & conditions")
                state = st.selectbox("State/Province", sorted(df['state'].dropna().unique()))
                city = st.selectbox("City", sorted(df['city'].dropna().unique()))
                ctype = st.selectbox("Crop Type", sorted(df['crop_type'].dropna().unique()))
                season = st.selectbox("Season", sorted(df['season'].dropna().unique()))
                # get averages for month, rainfall, temp from dataset for that region if available
                region = df[(df['state']==state) & (df['city']==city)]
                if not region.empty:
                    default_month = int(region['date'].dt.month.mode()[0])
                    default_rain = float(region['rainfall_mm'].mean())
                    default_temp = float(region['temperature_c'].mean())
                else:
                    default_month = 6
                    default_rain = float(df['rainfall_mm'].mean())
                    default_temp = float(df['temperature_c'].mean())

                month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=int(default_month))
                rainfall = st.number_input("Rainfall (mm)", value=float(default_rain))
                temp = st.number_input("Temperature (°C)", value=float(default_temp))

                # prepare input
                X_in = pd.DataFrame([[
                    state, city, ctype, season, month, rainfall, temp
                ]], columns=['state','city','crop_type','season','month','rainfall_mm','temperature_c'])

                if st.button("Predict"):
                    try:
                        pred = pipe.predict(X_in)[0]
                        st.success(f"[{model_name}] Predicted price: ₹{pred:.2f}/ton")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    # Regional Map
    with tabs[4]:
        st.header("Nepal Province Price Map")

        # load geojson robustly
        geojson_obj, prop_key = load_nepal_geojson()

        # normalize province column to match geojson property values
        df_map = df.copy()
        df_map['province'] = df_map['state'].astype(str).str.strip()

        # compute avg price per province
        avgp = df_map.groupby('province', as_index=False)['price_₹/ton'].mean().rename(columns={'price_₹/ton':'avg_price'})

        # detect which column in geojson to use
        # property key returned by loader is likely 'PROVINCE' or 'name' etc.
        # Build set of geojson names to compare and report match rate
        geo_names = set()
        for feat in geojson_obj.get('features', []):
            props = feat.get('properties', {})
            val = props.get(prop_key)
            if val:
                geo_names.add(str(val).strip())

        # match
        avgp['matched'] = avgp['province'].apply(lambda x: x in geo_names)
        match_rate = avgp['matched'].mean() * 100.0 if len(avgp)>0 else 0.0

        st.write(f"Province name match rate with GeoJSON: {match_rate:.1f}%")
        if match_rate < 50:
            st.warning("Low match rate — ensure your 'state' values match GeoJSON province names (e.g. 'Madhesh Province').")
            st.dataframe(avgp.sort_values('matched', ascending=False).head(20))
        # filter only matched
        mapped = avgp[avgp['matched']].copy()
        if mapped.empty:
            # fallback: show bar chart
            st.info("No matched province names found; showing bar chart of top provinces by price.")
            st.bar_chart(avgp.sort_values('avg_price', ascending=False).set_index('province')['avg_price'])
        else:
            # Use featureidkey based on property key
            featureid = f"properties.{prop_key}"
            try:
                fig = px.choropleth(
                    mapped,
                    geojson=geojson_obj,
                    locations='province',
                    featureidkey=featureid,
                    color='avg_price',
                    color_continuous_scale='YlOrBr',
                    hover_name='province',
                    title='Average Price by Province (Nepal)'
                )
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Choropleth render error: {e}")
                st.info("Showing fallback bar chart:")
                st.bar_chart(avgp.sort_values('avg_price', ascending=False).set_index('province')['avg_price'])


if __name__ == "__main__":
    main()
