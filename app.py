# app.py
import os
import json
import pickle
from io import BytesIO

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import requests

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# -------------------------
# Config
# -------------------------
st.set_page_config(page_title="AgriPrice Analyzer (Nepal)", layout="wide")
st.title("🌾 Agricultural Market Price Analyzer (Nepal)")

# Base directories (absolute path so Streamlit finds assets)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEOJSON_PATH = os.path.join(BASE_DIR, "assets", "nepal_provinces.geojson")
MODELS_PATH = os.path.join(BASE_DIR, "models.pkl")
DEFAULT_DATA_PATH = os.path.join(BASE_DIR, "cleaned_dataset.csv")

# -------------------------
# Utility functions
# -------------------------
@st.cache_data
def load_data(file_input=None):
    """
    Loads CSV either from uploaded file-like object or local file path.
    Returns DataFrame (date parsed) or empty DF on error.
    """
    try:
        if file_input is None:
            path = DEFAULT_DATA_PATH
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=["date"])
            else:
                return pd.DataFrame()
        elif hasattr(file_input, "read"):  # uploaded file
            df = pd.read_csv(file_input, parse_dates=["date"])
        elif isinstance(file_input, str) and os.path.exists(file_input):
            df = pd.read_csv(file_input, parse_dates=["date"])
        else:
            return pd.DataFrame()

        # Ensure date column exists & parsed
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def detect_geojson_name_key(geojson_obj):
    """Return the likely property key that holds province name in geojson features."""
    for feat in geojson_obj.get("features", []):
        props = feat.get("properties", {})
        for candidate in ("PROVINCE", "NAME_1", "name", "province", "Name", "NAME"):
            if candidate in props:
                return candidate
    return None

def load_local_geojson(path=GEOJSON_PATH):
    """Load a local geojson file (returns object or None)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            geo = json.load(f)
        key = detect_geojson_name_key(geo)
        return geo, key
    except Exception as e:
        return None, None

def normalize_state_for_matching(s: str) -> str:
    """Simple normalization helpers: lower, remove punctuation, remove 'province' suffix."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    # remove trailing 'Province' or 'province' (common in your data)
    lower = s.lower()
    # common patterns: 'lumbini province', 'province 1', 'province no. 1'
    lower = lower.replace("province ", "province ").replace("province", "")
    # remove extra dots/commas
    for ch in [".", ",", "-"]:
        lower = lower.replace(ch, " ")
    # collapse spaces
    lower = " ".join(lower.split())
    return lower.title()  # Title case

def find_best_geo_match(state_value, geo_names_set):
    """
    Try matching strategies, return the matched geo name or None:
      1) exact match
      2) strip suffix 'Province' variants
      3) match using heuristic mapping
    """
    if pd.isna(state_value):
        return None

    s_raw = str(state_value).strip()
    # 1) exact
    if s_raw in geo_names_set:
        return s_raw

    # 2) common normalization attempts
    s1 = s_raw.replace(" Province", "").replace(" province", "").strip()
    if s1 in geo_names_set:
        return s1

    s2 = s_raw.replace("Province", "").replace("province", "").replace("Province ", "").strip()
    if s2 in geo_names_set:
        return s2

    # 3) normalized title without 'Province'
    s3 = normalize_state_for_matching(s_raw)
    # try variants in geo names
    for name in geo_names_set:
        if s3.lower() == str(name).lower() or s3.lower() in str(name).lower() or str(name).lower() in s3.lower():
            return name

    # 4) small explicit mapping for known cases
    explicit = {
        "Province 1": "Province 1",
        "Province No.1": "Province 1",
        "Province No. 1": "Province 1",
        "Province One": "Province 1",
        "Sudurpaschim": "Sudurpashchim",
        "Sudurpaschim Province": "Sudurpashchim",
        "Madesh Province": "Madhesh",
        "Madesh": "Madhesh",
        "Madhes Province": "Madhesh",
        "Koshi Province": "Province 1",
        "Far West": "Sudurpashchim",
        "Far-West": "Sudurpashchim"
    }
    if s_raw in explicit and explicit[s_raw] in geo_names_set:
        return explicit[s_raw]

    # no match
    return None

# -------------------------
# Model training & persistence
# -------------------------
def build_and_train_models(df):
    """
    Train three pipelines and save to MODELS_PATH:
      - XGBoost
      - RandomForest
      - LinearRegression
    Returns performance dict.
    """
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("Dataset must contain 'date' column")

    df["month"] = df["date"].dt.month

    cat_cols = ["state", "city", "crop_type", "season", "month"]
    num_cols = ["rainfall_mm", "temperature_c"]

    # Check required columns
    for c in cat_cols + num_cols + ["price_₹/ton"]:
        if c not in df.columns:
            raise ValueError(f"Missing column in dataset: {c}")

    X = df[cat_cols + num_cols]
    y = df["price_₹/ton"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols)
        ],
        remainder="drop"
    )

    pipelines = {
        "XGBoost": Pipeline([("pre", preprocessor), ("reg", xgb.XGBRegressor(objective="reg:squarederror", random_state=42))]),
        "Random Forest": Pipeline([("pre", preprocessor), ("reg", RandomForestRegressor(random_state=42, n_estimators=200))]),
        "Linear Regression": Pipeline([("pre", preprocessor), ("reg", LinearRegression())])
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained = {}
    perf = {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        trained[name] = pipe
        perf[name] = {"MAE": float(mae), "R2": float(r2)}

    # Save pipelines and column order
    with open(MODELS_PATH, "wb") as f:
        pickle.dump({"models": trained, "feature_cols": X.columns.tolist()}, f)

    return perf

# -------------------------
# App UI
# -------------------------
def main():
    # Sidebar data upload
    st.sidebar.header("Upload Dataset")
    uploaded = st.sidebar.file_uploader("Upload cleaned CSV (Nepal dataset)", type=["csv"])

    if uploaded:
        df = load_data(uploaded)
    else:
        df = load_data(DEFAULT_DATA_PATH)

    if df.empty:
        st.warning("No dataset loaded. Upload a CSV or place 'cleaned_dataset.csv' in the app folder.")
        return

    st.sidebar.success(f"Loaded {len(df):,} rows")

    # Quick preview
    if st.sidebar.checkbox("Show data sample"):
        st.dataframe(df.head(20))

    # Train button
    if st.sidebar.button("Train 3 Models (XGBoost, RF, Linear)"):
        with st.spinner("Training models..."):
            try:
                perf = build_and_train_models(df)
                st.success("Models trained and saved to models.pkl")
                st.json(perf)
            except Exception as e:
                st.error(f"Training failed: {e}")

    # Load models if exist
    models = {}
    feature_cols = None
    if os.path.exists(MODELS_PATH):
        with open(MODELS_PATH, "rb") as f:
            models_data = pickle.load(f)
            models = models_data.get("models", {})
            feature_cols = models_data.get("feature_cols", None)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "📈 Historical Trends",
        "🌦 Weather Impact",
        "🔮 Prediction",
        "🗺️ Regional Analysis"
    ])

    # Tab 1: Dashboard
    with tab1:
        st.header("Market Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Price (₹/ton)", f"{df['price_₹/ton'].mean():.2f}")
        with col2:
            # compute safe supply-demand ratio
            ratio = np.nan
            if "supply_volume_tons" in df.columns and "demand_volume_tons" in df.columns:
                denom = (df['demand_volume_tons'] == 0)
                if denom.all():
                    ratio = np.nan
                else:
                    ratio = (df['supply_volume_tons'] / df['demand_volume_tons']).replace([np.inf, -np.inf], np.nan).mean()
            st.metric("Supply-Demand Ratio", f"{ratio:.2f}" if pd.notna(ratio) else "N/A")
        with col3:
            st.metric("Active Provinces", df['state'].nunique())

        st.subheader("Latest Entries")
        st.dataframe(df.sort_values("date", ascending=False).head(10), use_container_width=True)

    # Tab 2: Historical Trends
    with tab2:
        st.header("Historical Price Analysis")
        crop_filter = st.selectbox("Select Crop", df['crop_type'].unique())
        default_start = df['date'].min().date()
        default_end = df['date'].max().date()
        date_range = st.date_input("Select Date Range", value=(default_start, default_end),
                                   min_value=default_start, max_value=default_end)
        filtered_df = df[(df['crop_type']==crop_filter) & (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))]
        if filtered_df.empty:
            st.warning("No data in selected range — showing full crop history.")
            filtered_df = df[df['crop_type']==crop_filter]
        fig = px.line(filtered_df, x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    # Tab 3: Weather Impact
    with tab3:
        st.header("Climate Correlation")
        factor = st.selectbox("Weather Factor", ['rainfall_mm', 'temperature_c'])
        try:
            fig = px.scatter(df, x=factor, y='price_₹/ton', color='crop_type', trendline="ols", title=f"Price vs {factor}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Plot error: {e}")

    # Tab 4: Prediction
    with tab4:
        st.header("Price Prediction")
        if not models:
            st.info("No trained models found. Train from the sidebar first.")
        else:
            model_choice = st.selectbox("Select Model", list(models.keys()))
            pipe = models[model_choice]

            st.subheader("Input for prediction")
            state = st.selectbox("State/Province", sorted(df['state'].dropna().unique()))
            city = st.selectbox("City", sorted(df['city'].dropna().unique()))
            crop_type = st.selectbox("Crop Type", sorted(df['crop_type'].dropna().unique()))
            season = st.selectbox("Season", sorted(df['season'].dropna().unique()))

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
            rainfall = st.number_input("Rainfall (mm)", value=float(round(default_rain,1)))
            temperature = st.number_input("Temperature (°C)", value=float(round(default_temp,1)))

            if st.button("Predict"):
                X_in = pd.DataFrame([[state, city, crop_type, season, month, rainfall, temperature]],
                                     columns=['state','city','crop_type','season','month','rainfall_mm','temperature_c'])
                try:
                    pred = pipe.predict(X_in)[0]
                    st.success(f"[{model_choice}] Predicted Price: ₹{pred:.2f}/ton")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Tab 5: Regional Analysis (Nepal)
    with tab5:
        st.header("Nepal Province Price Map")

        geojson_obj, prop_key = load_local_geojson(GEOJSON_PATH)
        if geojson_obj is None or prop_key is None:
            st.error(f"Could not load local GeoJSON at {GEOJSON_PATH}. Ensure file exists and is valid.")
            st.stop()

        # Build set of geojson province names
        geo_names = set()
        for feat in geojson_obj.get('features', []):
            v = feat.get('properties', {}).get(prop_key)
            if v:
                geo_names.add(str(v).strip())

        # Create normalized province column and attempt to map to geojson names
        df_map = df.copy()
        df_map['province_raw'] = df_map['state'].astype(str).str.strip()

        # Try to find the best match for each province in dataset
        mapped_names = []
        for val in df_map['province_raw'].unique():
            matched = find_best_geo_match(val, geo_names)
            mapped_names.append((val, matched))
        # mapping dict
        mapping = {orig: matched if matched is not None else orig for orig, matched in mapped_names}

        # Apply mapping for display and grouping
        df_map['province_mapped'] = df_map['province_raw'].map(mapping)

        avg_prices = df_map.groupby('province_mapped', as_index=False)['price_₹/ton'].mean().rename(columns={'price_₹/ton':'avg_price'})

        # Show diagnostics
        matched_count = avg_prices['province_mapped'].isin(geo_names).sum()
        total_count = len(avg_prices)
        match_rate = 100.0 * matched_count / total_count if total_count>0 else 0.0

        with st.expander("Map diagnostics"):
            st.write(f"GeoJSON property key used: `{prop_key}`")
            st.write(f"Unique data provinces (raw): {len(df_map['province_raw'].unique())}")
            st.write(f"Unique mapped provinces: {len(avg_prices)}")
            st.write(f"Match rate vs GeoJSON: {match_rate:.1f}%")
            st.write("Sample mapping (data -> mapped):")
            sample_map = pd.DataFrame(mapped_names, columns=['data_value','mapped_to']).head(50)
            st.dataframe(sample_map)

        # If few matches, show bar chart fallback
        if match_rate < 30:
            st.warning("Low match rate between dataset province names and GeoJSON names. Showing bar chart fallback.")
            st.bar_chart(avg_prices.sort_values('avg_price', ascending=False).set_index('province_mapped')['avg_price'])
        else:
            # Filter only those that matched to avoid blank areas
            avg_matched = avg_prices[avg_prices['province_mapped'].isin(geo_names)].copy()
            if avg_matched.empty:
                st.warning("No matched provinces — showing bar chart.")
                st.bar_chart(avg_prices.sort_values('avg_price', ascending=False).set_index('province_mapped')['avg_price'])
            else:
                # Draw choropleth
                featureid = f"properties.{prop_key}"
                try:
                    fig = px.choropleth(
                        avg_matched,
                        geojson=geojson_obj,
                        locations='province_mapped',
                        featureidkey=featureid,
                        color='avg_price',
                        color_continuous_scale='YlOrBr',
                        hover_name='province_mapped',
                        title='Average Price by Province (Nepal)'
                    )
                    fig.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Choropleth rendering error: {e}")
                    st.bar_chart(avg_prices.sort_values('avg_price', ascending=False).set_index('province_mapped')['avg_price'])

    # Report download
    st.sidebar.header("Report")
    if st.sidebar.button("Download Summary Report"):
        out = df.describe().T
        csv = out.to_csv().encode('utf-8')
        st.sidebar.download_button("Download CSV", data=csv, file_name="market_summary.csv", mime="text/csv")


if __name__ == "__main__":
    main()
