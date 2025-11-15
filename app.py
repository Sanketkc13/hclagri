# app.py
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")

# -------------------------
# Helper: load data
# -------------------------
@st.cache_data
def load_data(path="cleaned_dataset.csv", uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif os.path.exists(path):
            df = pd.read_csv(path)
        else:
            return pd.DataFrame()
        # parse date
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

# -------------------------
# Helper: encode categorical features, return df and label encoders
# -------------------------
def fit_label_encoders(df, cat_cols):
    le_dict = {}
    df_enc = df.copy()
    for c in cat_cols:
        le = LabelEncoder()
        # fill NA with string '___NA___' to ensure no error
        df_enc[c] = df_enc[c].fillna("___NA___").astype(str)
        df_enc[c] = le.fit_transform(df_enc[c])
        le_dict[c] = le
    return df_enc, le_dict

# -------------------------
# Train & save single model
# -------------------------
def train_and_save_model(df, model_name):
    """
    Train selected model on the dataset and save to model.pkl
    model_name in ("XGBoost","Random Forest","Linear Regression")
    """
    # check required columns
    req = {"state","city","crop_type","season","rainfall_mm","temperature_c","price_₹/ton","date","supply_volume_tons","demand_volume_tons"}
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df2 = df.copy()
    df2['month'] = df2['date'].dt.month

    # features
    cat_cols = ['state','city','crop_type','season','month']
    num_cols = ['rainfall_mm','temperature_c','supply_volume_tons','demand_volume_tons']
    feature_cols = cat_cols + num_cols

    # label encode categorical features
    df_enc, le_dict = fit_label_encoders(df2, cat_cols)

    X = df_enc[feature_cols]
    y = df_enc['price_₹/ton']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_name == "XGBoost":
        model = XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=200)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, random_state=42)
    else:
        model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # save model and encoders and feature metadata
    payload = {
        "model": model,
        "le_dict": le_dict,
        "feature_columns": feature_cols,
        "model_name": model_name
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(payload, f)

    return {"MAE": float(mae), "R2": float(r2)}

# -------------------------
# Helper: prepare input for prediction (auto-fill weather/supply/demand/month)
# -------------------------
def prepare_prediction_input(df, le_dict, feature_columns, state, city, crop_type, season):
    """
    For the selected state/city/crop/season derive month, rainfall, temperature, supply, demand
    by taking averages from dataset for the matching rows. If none found, use dataset global averages as fallback.
    Returns a DataFrame ready for model.predict()
    """
    # ensure date month exists
    df2 = df.copy()
    if 'month' not in df2.columns:
        if 'date' in df2.columns:
            df2['month'] = df2['date'].dt.month
        else:
            df2['month'] = 6

    subset = df2[
        (df2['state'] == state) &
        (df2['city'] == city) &
        (df2['crop_type'] == crop_type) &
        (df2['season'] == season)
    ]

    if not subset.empty:
        month = int(subset['date'].dt.month.mode()[0]) if 'date' in subset.columns else int(subset['month'].mode()[0])
        rainfall = float(subset['rainfall_mm'].mean())
        temp = float(subset['temperature_c'].mean())
        supply = float(subset['supply_volume_tons'].mean()) if 'supply_volume_tons' in subset.columns else float(df2['supply_volume_tons'].mean())
        demand = float(subset['demand_volume_tons'].mean()) if 'demand_volume_tons' in subset.columns else float(df2['demand_volume_tons'].mean())
    else:
        # global fallback
        month = int(df2['date'].dt.month.mode()[0]) if 'date' in df2.columns else 6
        rainfall = float(df2['rainfall_mm'].mean())
        temp = float(df2['temperature_c'].mean())
        supply = float(df2['supply_volume_tons'].mean()) if 'supply_volume_tons' in df2.columns else 0.0
        demand = float(df2['demand_volume_tons'].mean()) if 'demand_volume_tons' in df2.columns else 0.0

    # build raw row in original string categories (to encode)
    raw_row = {
        'state': state,
        'city': city,
        'crop_type': crop_type,
        'season': season,
        'month': month,
        'rainfall_mm': rainfall,
        'temperature_c': temp,
        'supply_volume_tons': supply,
        'demand_volume_tons': demand
    }

    # encode categorical values using le_dict
    encoded = []
    for col in feature_columns:
        if col in ['rainfall_mm','temperature_c','supply_volume_tons','demand_volume_tons']:
            encoded.append(raw_row[col])
        else:
            # column is categorical: transform using fitted LabelEncoder
            le = le_dict[col]
            # labelencoder expects string input; handle unseen values by mapping to nearest or raising
            val = raw_row[col]
            # if unseen label, add fallback '___NA___' if encoder was fit with that
            try:
                transformed = le.transform([str(val)])[0]
            except Exception:
                # fallback: if the encoder has seen '___NA___', use that index; else use 0
                classes = list(le.classes_)
                if "___NA___" in classes:
                    transformed = int(np.where(np.array(classes) == "___NA___")[0][0])
                else:
                    # best effort: use 0
                    transformed = 0
            encoded.append(transformed)

    X_in = pd.DataFrame([encoded], columns=feature_columns)
    return X_in, raw_row

# -------------------------
# Weather analytics functions
# -------------------------
def weather_summary(df, factor):
    """Return mean, median, std, correlation with price for the chosen factor."""
    s = df[factor].dropna()
    mean = float(s.mean())
    median = float(s.median())
    std = float(s.std())
    # correlation with price
    if 'price_₹/ton' in df.columns:
        corr = float(df[[factor, 'price_₹/ton']].dropna().corr().iloc[0,1])
    else:
        corr = float('nan')
    return {"mean": mean, "median": median, "std": std, "corr_with_price": corr}

# -------------------------
# App UI
# -------------------------
def main():
    st.title("Agricultural Market Price Analyzer 🌾 (Original Look)")

    # Sidebar: upload or use local csv
    st.sidebar.header("Data Management & Model")
    uploaded = st.sidebar.file_uploader("Upload cleaned CSV", type=['csv'])
    if uploaded:
        df = load_data(uploaded_file=uploaded)
    else:
        df = load_data()

    if df.empty:
        st.warning("No dataset loaded. Upload a CSV or place 'cleaned_dataset.csv' next to this app.")
        return

    st.sidebar.success(f"Loaded dataset with {len(df):,} rows")

    # Sidebar: choose algorithm to train (single saved model)
    algo = st.sidebar.selectbox("Choose algorithm to train & save (single model used for prediction)", ["XGBoost","Random Forest","Linear Regression"])
    if st.sidebar.button("Train & Save Model"):
        with st.spinner("Training..."):
            try:
                perf = train_and_save_model(df, algo)
                st.sidebar.success(f"Trained {algo} — MAE: {perf['MAE']:.2f}, R²: {perf['R2']:.3f}")
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")

    # Main tabs (original style)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "📈 Historical Trends", "🌦 Weather Impact", "🔮 Prediction"])

    # TAB 1 - Dashboard
    with tab1:
        st.header("Market Overview")
        cols = st.columns(3)
        with cols[0]:
            if 'price_₹/ton' in df.columns:
                st.metric("Average Price (₹/ton)", f"{df['price_₹/ton'].mean():.2f}")
            else:
                st.metric("Average Price (₹/ton)", "N/A")
        with cols[1]:
            st.metric("Active Regions", int(df['state'].nunique()) if 'state' in df.columns else "N/A")
        with cols[2]:
            st.metric("Crops Tracked", int(df['crop_type'].nunique()) if 'crop_type' in df.columns else "N/A")

        st.subheader("Latest market entries")
        st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)

    # TAB 2 - Historical Trends
    with tab2:
        st.header("Historical Price Analysis")
        crop = st.selectbox("Select Crop", df['crop_type'].unique())
        # date range
        if 'date' in df.columns:
            min_date = df['date'].min().date()
            max_date = df['date'].max().date()
            dr = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
            filtered = df[(df['crop_type']==crop) & (df['date'].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1])))]
        else:
            filtered = df[df['crop_type']==crop]

        if filtered.empty:
            st.warning("No data for selected range/crop — showing full crop history.")
            filtered = df[df['crop_type']==crop]

        fig = px.line(filtered, x='date', y='price_₹/ton', title=f"{crop} Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3 - Weather Impact (FULL analytics)
    with tab3:
        st.header("Weather Impact — Full Climate Analytics (Original Look)")

        # factor 1: rainfall
        st.subheader("Price vs Rainfall")
        try:
            fig1 = px.scatter(df, x='rainfall_mm', y='price_₹/ton', color='crop_type', trendline="ols",
                              title="Price vs Rainfall (scatter + OLS)")
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.warning("Could not add trendline (statsmodels may be missing). Showing scatter only.")
            fig1 = px.scatter(df, x='rainfall_mm', y='price_₹/ton', color='crop_type', title="Price vs Rainfall (scatter)")
            st.plotly_chart(fig1, use_container_width=True)

        # factor 2: temperature
        st.subheader("Price vs Temperature")
        try:
            fig2 = px.scatter(df, x='temperature_c', y='price_₹/ton', color='crop_type', trendline="ols",
                              title="Price vs Temperature (scatter + OLS)")
            st.plotly_chart(fig2, use_container_width=True)
        except Exception:
            st.warning("Could not add trendline (statsmodels may be missing). Showing scatter only.")
            fig2 = px.scatter(df, x='temperature_c', y='price_₹/ton', color='crop_type', title="Price vs Temperature (scatter)")
            st.plotly_chart(fig2, use_container_width=True)

        # heatmap (binned rainfall x temperature average price)
        st.subheader("Price Heatmap (binned Rainfall × Temperature)")
        try:
            # create bins
            df_heat = df[['rainfall_mm','temperature_c','price_₹/ton']].dropna().copy()
            # create bins - 10 bins each
            df_heat['rain_bin'] = pd.cut(df_heat['rainfall_mm'], bins=10)
            df_heat['temp_bin'] = pd.cut(df_heat['temperature_c'], bins=10)
            heat = df_heat.groupby(['rain_bin','temp_bin'])['price_₹/ton'].mean().reset_index()
            # pivot to matrix
            heat_pivot = heat.pivot(index='temp_bin', columns='rain_bin', values='price_₹/ton')
            # convert index/columns to string for plotting
            heat_pivot.index = heat_pivot.index.astype(str)
            heat_pivot.columns = heat_pivot.columns.astype(str)
            fig_heat = px.imshow(heat_pivot.values,
                                 x=heat_pivot.columns,
                                 y=heat_pivot.index,
                                 labels={'x':'Rainfall bin','y':'Temperature bin','color':'Avg Price'},
                                 title="Average Price by Rainfall & Temperature bins")
            st.plotly_chart(fig_heat, use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap generation failed: {e}")

        # Weather summary stats
        st.subheader("Weather Summary Statistics")
        rain_stats = weather_summary(df, 'rainfall_mm')
        temp_stats = weather_summary(df, 'temperature_c')

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Rainfall (mm)**")
            st.write(f"Mean: {rain_stats['mean']:.2f}")
            st.write(f"Median: {rain_stats['median']:.2f}")
            st.write(f"Std: {rain_stats['std']:.2f}")
            st.write(f"Correlation with price: {rain_stats['corr_with_price']:.3f}")
        with col2:
            st.markdown("**Temperature (°C)**")
            st.write(f"Mean: {temp_stats['mean']:.2f}")
            st.write(f"Median: {temp_stats['median']:.2f}")
            st.write(f"Std: {temp_stats['std']:.2f}")
            st.write(f"Correlation with price: {temp_stats['corr_with_price']:.3f}")

    # TAB 4 - Prediction (auto month/weather)
    with tab4:
        st.header("Price Prediction")
        # check model exists
        if not os.path.exists("model.pkl"):
            st.warning("No trained model found. Train a model from the sidebar to enable prediction.")
        else:
            # load model
            with open("model.pkl","rb") as f:
                payload = pickle.load(f)
            model = payload["model"]
            le_dict = payload["le_dict"]
            feature_cols = payload["feature_columns"]

            # user picks region/crop/season only
            col1, col2 = st.columns(2)
            with col1:
                state = st.selectbox("State", sorted(df['state'].dropna().unique()))
                city = st.selectbox("City", sorted(df['city'].dropna().unique()))
            with col2:
                crop_type = st.selectbox("Crop Type", sorted(df['crop_type'].dropna().unique()))
                season = st.selectbox("Season", sorted(df['season'].dropna().unique()))

            # show the auto-extracted values for transparency (but greyed)
            X_in, raw_row = prepare_prediction_input(df, le_dict, feature_cols, state, city, crop_type, season)
            st.markdown("**Auto-filled inputs used for prediction (from dataset averages)**")
            st.write({
                "month": raw_row['month'],
                "rainfall_mm": f"{raw_row['rainfall_mm']:.1f}",
                "temperature_c": f"{raw_row['temperature_c']:.1f}",
                "supply_volume_tons": f"{raw_row['supply_volume_tons']:.1f}",
                "demand_volume_tons": f"{raw_row['demand_volume_tons']:.1f}"
            })

            if st.button("Predict Price"):
                try:
                    pred = model.predict(X_in)[0]
                    st.success(f"Predicted Price: ₹{pred:.2f} / ton")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

    # Sidebar: download model option
    st.sidebar.markdown("---")
    if os.path.exists("model.pkl"):
        if st.sidebar.button("Download trained model (model.pkl)"):
            with open("model.pkl","rb") as f:
                bin_data = f.read()
            st.sidebar.download_button("Download model.pkl", data=bin_data, file_name="model.pkl", mime="application/octet-stream")

if __name__ == "__main__":
    main()
