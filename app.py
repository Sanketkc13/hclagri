# app.py
import os
import io
import pickle
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# Configuration
# -----------------------
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")
st.title("Agricultural Market Price Analyzer 🌾")

DEFAULT_LOCAL_PATH = "/mnt/data/cleaned_dataset.csv"
NEPAL_GEOJSON_LOCAL = "assets/nepal_provinces.geojson"  # local preferred
NEPAL_GEOJSON_REMOTE = "https://raw.githubusercontent.com/sandeshchapagain/nepal-geojson/main/nepal-provinces.geojson"
INDIA_GEOJSON_REMOTE = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"

# -----------------------
# Utility: column detection & normalization
# -----------------------
def detect_and_standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Detect price/supply/demand and normalize column names to internal schema.
       Ensures required columns exist (fill/make reasonable defaults).
    """
    df = df.copy()
    cols = list(df.columns)

    # mapping heuristics
    mapping = {}
    for c in cols:
        cl = c.lower()
        if "price" in cl:
            mapping[c] = "price"
        if cl in ("state", "province", "region"):
            mapping[c] = "state"
        if cl in ("city", "district", "municipality"):
            mapping[c] = "city"
        if "crop" in cl:
            mapping[c] = "crop_type"
        if "season" in cl:
            mapping[c] = "season"
        if "rain" in cl:
            mapping[c] = "rainfall_mm"
        if "temp" in cl or "temperature" in cl:
            mapping[c] = "temperature_c"
        if cl.startswith("supply"):
            mapping[c] = "supply_volume_tons"
        if cl.startswith("demand"):
            mapping[c] = "demand_volume_tons"
        if cl in ("date",):
            mapping[c] = "date"

    df = df.rename(columns=mapping)

    # ensure date exists
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        # create dummy date if missing
        df["date"] = pd.to_datetime("2020-01-01")

    # required categorical columns - create if missing
    for c in ["state", "city", "crop_type", "season"]:
        if c not in df.columns:
            df[c] = "Unknown"
        df[c] = df[c].astype(str).fillna("Unknown")

    # ensure numeric columns exist and are numeric
    for c in ["rainfall_mm", "temperature_c", "supply_volume_tons", "demand_volume_tons"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")
        # fill missing with mean if available else zero
        if df[c].isna().all():
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(df[c].mean())

    # price column: if not found, create and fill 0 (but training requires price)
    if "price" not in df.columns:
        # try to detect alternative price-like column name (rare)
        found = None
        for c in cols:
            if "price" in c.lower():
                found = c
                break
        if found:
            df = df.rename(columns={found: "price"})
        else:
            df["price"] = np.nan

    # add month column for convenience
    if "month" not in df.columns:
        df["month"] = df["date"].dt.month.fillna(6).astype(int)

    return df

# -----------------------
# Robust loader (uploaded file or default local)
# -----------------------
@st.cache_data
def load_data(file_obj=None):
    try:
        if file_obj is None:
            if os.path.exists(DEFAULT_LOCAL_PATH):
                df = pd.read_csv(DEFAULT_LOCAL_PATH)
            elif os.path.exists("cleaned_dataset.csv"):
                df = pd.read_csv("cleaned_dataset.csv")
            else:
                return pd.DataFrame()
        else:
            # file_obj may be UploadedFile or path string
            if hasattr(file_obj, "read"):
                file_obj.seek(0)
                df = pd.read_csv(file_obj)
            elif isinstance(file_obj, str) and os.path.exists(file_obj):
                df = pd.read_csv(file_obj)
            else:
                # attempt to read from file-like string
                try:
                    df = pd.read_csv(io.StringIO(file_obj.decode("utf-8")))
                except Exception:
                    return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

# -----------------------
# Build preprocess + model pipelines
# -----------------------
def build_pipelines(cat_cols, num_cols):
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
            ("num", StandardScaler(), num_cols)
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    pipelines = {
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", XGBRegressor(objective="reg:squarederror",
                                       n_estimators=300, learning_rate=0.05, max_depth=6,
                                       subsample=0.85, colsample_bytree=0.85,
                                       reg_alpha=0.5, reg_lambda=1.0,
                                       random_state=42, n_jobs=-1))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(n_estimators=400, max_depth=18,
                                               min_samples_split=4, min_samples_leaf=2,
                                               n_jobs=-1, random_state=42))
        ]),
        "Linear Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ])
    }
    return pipelines

# -----------------------
# Train (with optional XGB randomized search + early stopping)
# -----------------------
def train_and_save(df, model_choice="XGBoost", do_xgb_search=False, xgb_iter=12):
    # require a price column to train
    if "price" not in df.columns or df["price"].isna().all():
        raise ValueError("No valid 'price' column present in dataset for training.")

    # define columns
    cat_cols = ["state", "city", "crop_type", "season", "month"]
    num_cols = ["rainfall_mm", "temperature_c", "supply_volume_tons", "demand_volume_tons"]
    feature_cols = cat_cols + num_cols

    # prepare X,y
    df2 = df.copy()
    df2["month"] = df2["date"].dt.month.fillna(6).astype(int)
    X = df2[feature_cols]
    y = df2["price"].astype(float)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # build pipelines
    pipelines = build_pipelines(cat_cols, num_cols)
    pipeline = pipelines[model_choice]

    if model_choice == "XGBoost" and do_xgb_search:
        st.info("Running RandomizedSearchCV for XGBoost (this might take a while)...")
        param_dist = {
            "regressor__n_estimators": [100, 200, 300, 400],
            "regressor__learning_rate": [0.01, 0.03, 0.05, 0.08],
            "regressor__max_depth": [3, 5, 6, 8],
            "regressor__subsample": [0.6, 0.75, 0.85, 1.0],
            "regressor__colsample_bytree": [0.6, 0.75, 0.85, 1.0],
            "regressor__reg_alpha": [0, 0.1, 0.5],
            "regressor__reg_lambda": [0.5, 1.0, 2.0]
        }
        search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                    n_iter=max(6, min(xgb_iter, 40)), scoring="neg_mean_absolute_error",
                                    cv=3, random_state=42, n_jobs=-1, verbose=0)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        st.success("RandomizedSearchCV complete.")
        st.write("Best params (XGBoost pipeline):", search.best_params_)

    # For XGBoost, fit with early stopping using pipeline.fit with fit params
    if model_choice == "XGBoost":
        # provide eval_set via pipeline fit params (prefixed by regressor__)
        pipeline.fit(X_train, y_train,
                     regressor__eval_set=[(X_test, y_test)],
                     regressor__early_stopping_rounds=30,
                     regressor__verbose=False)
    else:
        pipeline.fit(X_train, y_train)

    # evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # feature importance if available (for tree models)
    feat_importance = None
    if model_choice in ("XGBoost", "Random Forest"):
        try:
            # build feature names after preprocessing
            # use pipeline.named_steps['preprocessor'] to get output feature names (sklearn >=1.0)
            pre = pipeline.named_steps['preprocessor']
            # After OneHotEncoder+sparse=False, feature names are available via get_feature_names_out
            try:
                cat_names = pre.named_transformers_['cat'].get_feature_names_out(cat_cols)
            except Exception:
                # fallback
                cat_names = [f"{c}__{i}" for c in cat_cols]
            num_names = num_cols
            feature_names = list(cat_names) + list(num_names)
            # get regressor feature importance
            reg = pipeline.named_steps['regressor']
            if hasattr(reg, "feature_importances_"):
                imp = reg.feature_importances_
                feat_importance = pd.DataFrame({"feature": feature_names[:len(imp)], "importance": imp}).sort_values("importance", ascending=False)
        except Exception:
            feat_importance = None

    # Save pipeline payload
    payload = {
        "pipeline": pipeline,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "model_choice": model_choice
    }
    os.makedirs("models", exist_ok=True)
    with open("model.pkl", "wb") as f:
        pickle.dump(payload, f)
    with open(os.path.join("models", f"{model_choice.lower().replace(' ','_')}_pipeline.pkl"), "wb") as f:
        pickle.dump(payload, f)

    # RF node stats if applicable
    rf_stats = None
    if model_choice == "Random Forest":
        try:
            rf = pipeline.named_steps['regressor']
            total_nodes = sum([est.tree_.node_count for est in rf.estimators_])
            n_trees = len(rf.estimators_)
            avg_nodes = int(total_nodes / n_trees) if n_trees else 0
            rf_stats = {"n_trees": n_trees, "total_nodes": int(total_nodes), "avg_nodes": avg_nodes}
        except Exception:
            rf_stats = None

    return {"MAE": float(mae), "R2": float(r2), "feat_importance": feat_importance, "rf_stats": rf_stats}

# -----------------------
# Prepare single-row input for prediction (auto-fill)
# -----------------------
def prepare_input_row_for_pipeline(df_original, state, city, crop, season, month=None, rainfall=None, temp=None, supply=None, demand=None):
    df2 = df_original.copy()
    if month is None:
        if "date" in df2.columns:
            month = int(df2['date'].dt.month.mode()[0])
        else:
            month = 6
    # select subset for that location to compute averages
    subset = df2[
        (df2['state'].astype(str) == str(state)) &
        (df2['city'].astype(str) == str(city)) &
        (df2['crop_type'].astype(str) == str(crop)) &
        (df2['season'].astype(str) == str(season))
    ]
    if subset.shape[0] > 0:
        rainfall = float(subset['rainfall_mm'].mean()) if rainfall is None else rainfall
        temp = float(subset['temperature_c'].mean()) if temp is None else temp
        supply = float(subset['supply_volume_tons'].mean()) if supply is None else supply
        demand = float(subset['demand_volume_tons'].mean()) if demand is None else demand
    else:
        # fallback to dataset means
        rainfall = float(df2['rainfall_mm'].mean()) if rainfall is None else rainfall
        temp = float(df2['temperature_c'].mean()) if temp is None else temp
        supply = float(df2['supply_volume_tons'].mean()) if supply is None else supply
        demand = float(df2['demand_volume_tons'].mean()) if demand is None else demand

    row = {
        "state": str(state),
        "city": str(city),
        "crop_type": str(crop),
        "season": str(season),
        "month": int(month),
        "rainfall_mm": float(rainfall),
        "temperature_c": float(temp),
        "supply_volume_tons": float(supply),
        "demand_volume_tons": float(demand)
    }
    X_row = pd.DataFrame([row])
    return X_row, row

# -----------------------
# Justifications
# -----------------------
JUST = {
    "Linear Regression": "Baseline, interpretable — good for quick checks and simple linear relationships.",
    "Random Forest": "Robust ensemble capturing non-linear interactions; less sensitive to outliers.",
    "XGBoost": "High-performance gradient boosting with regularization and early stopping — strong for tabular data."
}

# -----------------------
# MAIN app UI
# -----------------------
def main():
    st.sidebar.header("Data & Training Controls")

    uploaded = st.sidebar.file_uploader("Upload CSV (optional). If uploaded it will be saved to the default path.", type=["csv"])
    # if upload provided, save to DEFAULT_LOCAL_PATH so subsequent runs use it automatically
    if uploaded is not None:
        try:
            # save uploaded file to default path for persistence
            bytes_data = uploaded.getvalue()
            os.makedirs(os.path.dirname(DEFAULT_LOCAL_PATH), exist_ok=True)
            with open(DEFAULT_LOCAL_PATH, "wb") as f:
                f.write(bytes_data)
            st.sidebar.success(f"Uploaded and saved to {DEFAULT_LOCAL_PATH}")
            df_raw = load_data(DEFAULT_LOCAL_PATH)
        except Exception as e:
            st.sidebar.error(f"Failed to save uploaded file: {e}")
            df_raw = load_data(uploaded)
    else:
        df_raw = load_data(None)

    if df_raw.empty:
        st.warning("No data found — upload a CSV or place 'cleaned_dataset.csv' at the default path.")
        st.stop()

    # standardize columns & fill
    df = detect_and_standardize(df_raw)

    st.sidebar.success(f"Loaded dataset ({len(df):,} rows)")

    # check that price exists for training
    if "price" not in df.columns or df["price"].isna().all():
        st.sidebar.error("No valid 'price' column detected. Training and prediction require a numeric price column.")
    else:
        st.sidebar.info("Price column detected and ready.")

    # Model controls
    st.sidebar.markdown("### Choose model to train")
    model_choice = st.sidebar.selectbox("Model", ["XGBoost", "Random Forest", "Linear Regression"])
    st.sidebar.info(JUST[model_choice])
    do_xgb_search = False
    xgb_iter = 12
    if model_choice == "XGBoost":
        do_xgb_search = st.sidebar.checkbox("Enable XGBoost RandomizedSearchCV (optional, slower)", value=False)
        if do_xgb_search:
            xgb_iter = st.sidebar.number_input("Search iterations", min_value=4, max_value=40, value=12, step=2)

    if st.sidebar.button("Train & Save Model"):
        try:
            with st.spinner("Training — this may take some time"):
                perf = train_and_save(df, model_choice, do_xgb_search, xgb_iter)
            st.sidebar.success(f"Trained {model_choice} — MAE: {perf['MAE']:.2f}, R²: {perf['R2']:.3f}")
            if perf.get("rf_stats"):
                st.sidebar.write("RF stats:", perf["rf_stats"])
            if perf.get("feat_importance") is not None:
                st.sidebar.write("Top features:")
                st.sidebar.dataframe(perf["feat_importance"].head(10).reset_index(drop=True))
        except Exception as e:
            st.sidebar.error(f"Training failed: {e}")

    # App tabs
    tabs = st.tabs(["📊 Dashboard", "📈 Historical Trends", "🌦 Weather Impact", "🔮 Prediction", "🗺 Regional Analysis", "⚙️ Model"])

    # TAB Dashboard
    with tabs[0]:
        st.header("Market Overview")
        c1, c2, c3 = st.columns(3)
        if "price" in df.columns:
            c1.metric("Average Price", f"₹{df['price'].mean():.2f}/ton")
        else:
            c1.metric("Average Price", "N/A")
        if {'supply_volume_tons', 'demand_volume_tons'}.issubset(df.columns):
            ratio = (df['supply_volume_tons'] / df['demand_volume_tons']).replace([np.inf, -np.inf], np.nan)
            c2.metric("Supply/Demand Ratio", f"{ratio.mean():.2f}")
        else:
            c2.metric("Supply/Demand Ratio", "N/A")
        c3.metric("Active Regions", int(df['state'].nunique()) if 'state' in df.columns else "N/A")
        st.subheader("Latest entries")
        if 'date' in df.columns:
            st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)
        else:
            st.dataframe(df.head(10), use_container_width=True)

    # TAB Historical Trends
    with tabs[1]:
        st.header("Historical Price Analysis")
        if 'crop_type' in df.columns:
            crop = st.selectbox("Select Crop", df['crop_type'].unique())
            if 'date' in df.columns:
                if 'date_range' not in st.session_state:
                    st.session_state['date_range'] = (df['date'].min().date(), df['date'].max().date())
                dr = st.date_input("Date Range", value=st.session_state['date_range'],
                                   min_value=df['date'].min().date(), max_value=df['date'].max().date())
                st.session_state['date_range'] = dr
                filtered = df[(df['crop_type'] == crop) & (df['date'].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1])))]
                if filtered.empty:
                    st.error("No data for that range — showing full crop history")
                    filtered = df[df['crop_type'] == crop]
                fig = px.line(filtered, x='date', y='price', title=f"{crop} Price Trend")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No date column to plot time series.")
        else:
            st.info("No crop_type column present.")

    # TAB Weather Impact
    with tabs[2]:
        st.header("Climate / Weather Impact")
        if set(['rainfall_mm', 'price']).issubset(df.columns):
            try:
                fig = px.scatter(df, x='rainfall_mm', y='price', color='crop_type', trendline="ols", title="Price vs Rainfall")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Trendline requires statsmodels. Showing scatter only.")
                fig = px.scatter(df, x='rainfall_mm', y='price', color='crop_type', title="Price vs Rainfall")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("rainfall_mm or price column missing.")

        if set(['temperature_c', 'price']).issubset(df.columns):
            try:
                fig = px.scatter(df, x='temperature_c', y='price', color='crop_type', trendline="ols", title="Price vs Temperature")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Trendline requires statsmodels. Showing scatter only.")
                fig = px.scatter(df, x='temperature_c', y='price', color='crop_type', title="Price vs Temperature")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("temperature_c or price column missing.")

        st.subheader("Heatmap: Avg Price by Rainfall × Temperature")
        try:
            if set(['rainfall_mm', 'temperature_c', 'price']).issubset(df.columns):
                df_heat = df[['rainfall_mm', 'temperature_c', 'price']].dropna().copy()
                if df_heat.shape[0] < 20:
                    st.info("Not enough data for a robust heatmap.")
                else:
                    df_heat['rain_bin'] = pd.cut(df_heat['rainfall_mm'], bins=10)
                    df_heat['temp_bin'] = pd.cut(df_heat['temperature_c'], bins=10)
                    heat = df_heat.groupby(['temp_bin', 'rain_bin'])['price'].mean().reset_index()
                    pivot = heat.pivot(index='temp_bin', columns='rain_bin', values='price')
                    pivot.index = pivot.index.astype(str)
                    pivot.columns = pivot.columns.astype(str)
                    fig_heat = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                                         labels={'x': 'Rainfall bin', 'y': 'Temperature bin', 'color': 'Avg Price'},
                                         title="Average Price by Rainfall & Temperature bins")
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Heatmap requires rainfall_mm, temperature_c and price columns.")
        except Exception as e:
            st.error(f"Heatmap generation error: {e}")

    # TAB Prediction
    with tabs[3]:
        st.header("Price Prediction")
        if not os.path.exists("model.pkl"):
            st.warning("No trained model available. Train a model from the sidebar first.")
        else:
            try:
                with open("model.pkl", "rb") as f:
                    payload = pickle.load(f)
                pipeline = payload["pipeline"]
                feature_cols = payload["feature_cols"]
            except Exception as e:
                st.error(f"Failed to load saved model: {e}")
                pipeline = None

            if pipeline is not None:
                # dependent city dropdown
                state_choice = st.selectbox("State", sorted(df['state'].astype(str).unique()))
                cities_in_state = sorted(df[df['state'].astype(str) == str(state_choice)]['city'].astype(str).unique())
                if len(cities_in_state) == 0:
                    st.warning("No cities found for this state; type a city")
                    city_choice = st.text_input("City", value="")
                else:
                    city_choice = st.selectbox("City", cities_in_state)

                crop_choice = st.selectbox("Crop Type", sorted(df['crop_type'].astype(str).unique()))
                season_choice = st.selectbox("Season", sorted(df['season'].astype(str).unique()))

                with st.expander("Advanced inputs (auto-filled but editable)"):
                    # auto-filled
                    X_row_auto, raw_auto = prepare_input_row_for_pipeline(df, state_choice, city_choice, crop_choice, season_choice)
                    month_val = st.number_input("Month (1-12)", min_value=1, max_value=12, value=int(X_row_auto.loc[0, "month"]))
                    rainfall_val = st.number_input("Rainfall (mm)", value=float(X_row_auto.loc[0, "rainfall_mm"]))
                    temp_val = st.number_input("Temperature (°C)", value=float(X_row_auto.loc[0, "temperature_c"]))
                    supply_val = st.number_input("Supply Volume (tons)", value=float(X_row_auto.loc[0, "supply_volume_tons"]))
                    demand_val = st.number_input("Demand Volume (tons)", value=float(X_row_auto.loc[0, "demand_volume_tons"]))

                # build input df
                input_df = pd.DataFrame([{
                    "state": state_choice,
                    "city": city_choice,
                    "crop_type": crop_choice,
                    "season": season_choice,
                    "month": int(month_val),
                    "rainfall_mm": float(rainfall_val),
                    "temperature_c": float(temp_val),
                    "supply_volume_tons": float(supply_val),
                    "demand_volume_tons": float(demand_val)
                }])

                if st.button("Predict Price"):
                    try:
                        pred = pipeline.predict(input_df)[0]
                        st.success(f"Predicted Price: ₹{pred:.2f}/ton")
                        # Optional: show local statistics used
                        st.caption(f"Using averages for {city_choice}, {state_choice} where available.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

    # TAB Regional Analysis (Nepal provinces if available)
    with tabs[4]:
        st.header("Regional Analysis (Province map fallback)")
        # compute province-level average price
        if set(["state", "price"]).issubset(df.columns):
            avg_prices = df.groupby(["state", "crop_type"])["price"].mean().reset_index()
            # try local Nepal geojson first
            geojson = None
            if os.path.exists(NEPAL_GEOJSON_LOCAL):
                try:
                    with open(NEPAL_GEOJSON_LOCAL, "r", encoding="utf-8") as f:
                        geojson = json.load(f)
                    featureid = "properties.name"
                except Exception:
                    geojson = None
            if geojson is None:
                # remote Nepal attempt
                try:
                    import requests
                    resp = requests.get(NEPAL_GEOJSON_REMOTE, timeout=6)
                    geojson = resp.json()
                    featureid = "properties.name"
                except Exception:
                    geojson = None

            if geojson is None:
                # fallback to India map remote (original)
                try:
                    import requests
                    resp = requests.get(INDIA_GEOJSON_REMOTE, timeout=6)
                    geojson = resp.json()
                    featureid = "properties.NAME_1"
                except Exception:
                    geojson = None

            if geojson is None:
                st.info("GeoJSON not available. Showing bar chart instead.")
                fig = px.bar(avg_prices, x="state", y="price", color="crop_type", title="Avg Price by State (fallback)")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.choropleth(avg_prices,
                                    geojson=geojson,
                                    locations="state",
                                    featureidkey=featureid,
                                    color="price",
                                    hover_name="state",
                                    animation_frame="crop_type",
                                    color_continuous_scale="YlOrBr",
                                    title="Province-wise Average Price")
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("state and price columns required for map; showing aggregated table.")
            st.dataframe(df.groupby("state")["price"].mean().reset_index().sort_values("price", ascending=False))

    # TAB Model details
    with tabs[5]:
        st.header("Model & Artifacts")
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                payload = pickle.load(f)
            st.write("Model choice:", payload.get("model_choice"))
            st.write("Feature columns:", payload.get("feature_cols"))
            if payload.get("pipeline") is not None:
                pipe = payload["pipeline"]
                try:
                    reg = pipe.named_steps["regressor"]
                    if hasattr(reg, "feature_importances_"):
                        st.subheader("Feature importances (top 15)")
                        # attempt to reconstruct feature names:
                        pre = pipe.named_steps["preprocessor"]
                        try:
                            cat_cols = payload["cat_cols"]
                            cfn = pre.named_transformers_['cat'].get_feature_names_out(payload["cat_cols"])
                            feat_names = list(cfn) + payload["num_cols"]
                        except Exception:
                            feat_names = payload.get("feature_cols", [])
                        try:
                            importances = reg.feature_importances_
                            imp_df = pd.DataFrame({"feature": feat_names[:len(importances)], "importance": importances})
                            st.dataframe(imp_df.sort_values("importance", ascending=False).head(15).reset_index(drop=True))
                        except Exception:
                            st.info("Could not extract feature importances.")
                except Exception:
                    st.info("No model internals available for inspection.")
            # allow download
            with open("model.pkl", "rb") as f:
                st.download_button("Download model.pkl", f, file_name="model.pkl", mime="application/octet-stream")
        else:
            st.info("No trained model saved yet. Train one from the sidebar.")

    # Sidebar: quick report export
    st.sidebar.markdown("---")
    if st.sidebar.button("Download Data Summary CSV"):
        try:
            tmp = df.describe().T
            st.sidebar.download_button("Download summary", data=tmp.to_csv().encode("utf-8"), file_name="market_summary.csv", mime="text/csv")
        except Exception as e:
            st.sidebar.error(f"Failed to prepare report: {e}")

if __name__ == "__main__":
    main()
