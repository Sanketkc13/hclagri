# app.py
import os
import json
import pickle
import requests
from io import BytesIO

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="AgriPrice Analyzer (Nepal)", layout="wide")

# -------------------------
# Utils: load data safely
# -------------------------
@st.cache_data
def load_data(file_input='cleaned_dataset.csv'):
    try:
        if hasattr(file_input, "read"):  # uploaded file-like
            df = pd.read_csv(file_input, parse_dates=['date'])
        elif isinstance(file_input, str) and os.path.exists(file_input):
            df = pd.read_csv(file_input, parse_dates=['date'])
        else:
            return pd.DataFrame()
        # ensure date column parsed
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# -------------------------
# Utils: robust Nepal geojson loader
# returns (geojson_obj, property_key)
# -------------------------
@st.cache_data
def load_nepal_geojson():
    # prefer a stable raw URL known to work
    candidates = [
        "https://raw.githubusercontent.com/mesaugat/geoJSON-Nepal/master/nepal-province.geojson",
        "https://raw.githubusercontent.com/sandeshchapagain/Nepal-GeoJSON/master/Provinces/nepal-provinces.geojson",
        "https://raw.githubusercontent.com/sandeshchapagain/nepal-geojson/main/nepal-provinces.geojson",
    ]
    for url in candidates:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                geo = r.json()
                key = detect_geojson_name_key(geo)
                return geo, key
        except Exception:
            continue
    # fallback: None
    return None, None

def detect_geojson_name_key(geo):
    # inspect properties of first feature
    for feat in geo.get('features', []):
        props = feat.get('properties', {})
        for k in ['PROVINCE', 'NAME_1', 'name', 'province', 'Name']:
            if k in props:
                return k
    # fallback
    return 'name'

# -------------------------
# Build & train models (pipelines) and save to models.pkl
# -------------------------
def build_and_train_models(df):
    df = df.copy()
    if 'date' not in df.columns:
        raise ValueError("Dataset must have 'date' column.")
    # create month column
    df['month'] = df['date'].dt.month

    cat_cols = ['state', 'city', 'crop_type', 'season', 'month']
    num_cols = ['rainfall_mm', 'temperature_c']

    # sanity check columns
    for c in cat_cols + num_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in dataset: {c}")

    X = df[cat_cols + num_cols]
    y = df['price_₹/ton']

    # Preprocessor: scale numeric, one-hot categorical
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols)
    ])

    pipelines = {
        'XGBoost': Pipeline([('pre', preprocessor), ('reg', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))]),
        'Random Forest': Pipeline([('pre', preprocessor), ('reg', RandomForestRegressor(random_state=42, n_estimators=200))]),
        'Linear Regression': Pipeline([('pre', preprocessor), ('reg', LinearRegression())])
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
        perf[name] = {'MAE': float(mae), 'R2': float(r2)}

    # save trained
    with open('models.pkl', 'wb') as f:
        pickle.dump({'models': trained, 'feature_cols': X.columns.tolist()}, f)

    return perf

# -------------------------
# Helper: draw learning curve (R²)
# -------------------------
def plot_learning_curve_for_pipeline(pipe, X, y, cv=3):
    train_sizes, train_scores, val_scores = learning_curve(pipe, X, y, cv=cv, scoring='r2', n_jobs=-1,
                                                          train_sizes=np.linspace(0.1,1.0,5), shuffle=True, random_state=42)
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, marker='o', label='Train R²')
    ax.plot(train_sizes, val_mean, marker='o', label='Validation R²')
    ax.set_xlabel('Training samples')
    ax.set_ylabel('R²')
    ax.legend()
    st.pyplot(fig)

# -------------------------
# Main app
# -------------------------
def main():
    st.title("Agricultural Market Price Analyzer 🌾 (Nepal)")

    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])
    if uploaded_file:
        df = load_data(uploaded_file)
        st.sidebar.success(f"Loaded {len(df):,} rows from uploaded file")
    else:
        df = load_data('cleaned_dataset.csv')
        if not df.empty:
            st.sidebar.success(f"Loaded {len(df):,} rows from cleaned_dataset.csv")
        else:
            st.sidebar.info("No dataset found. Please upload a CSV.")

    if df.empty:
        st.warning("No data available — upload a CSV to proceed.")
        return

    # quick sample view
    if st.sidebar.checkbox("Show data sample"):
        st.dataframe(df.head(20))

    # Train models
    if st.sidebar.button("Train 3 Models (XGB, RF, Linear)"):
        with st.spinner("Training models — this may take a minute..."):
            try:
                perf = build_and_train_models(df)
                st.success("Models trained and saved to models.pkl")
                st.json(perf)
            except Exception as e:
                st.error(f"Training failed: {e}")

    # If models exist, load
    models = {}
    feature_cols = None
    if os.path.exists('models.pkl'):
        with open('models.pkl','rb') as f:
            data = pickle.load(f)
            models = data.get('models', {})
            feature_cols = data.get('feature_cols', None)

    # tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Dashboard",
        "📈 Historical Trends",
        "🌦️ Weather Impact",
        "🔮 Price Prediction",
        "🗺️ Regional Analysis"
    ])

    # TAB 1: Dashboard
    with tab1:
        st.header("Market Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Average Price", f"₹{df['price_₹/ton'].mean():.2f}/ton")
        with col2:
            # avoid division by zero
            if (df['demand_volume_tons'] == 0).all():
                ratio = np.nan
            else:
                ratio = (df['supply_volume_tons']/df['demand_volume_tons']).replace([np.inf, -np.inf], np.nan).mean()
            st.metric("Supply-Demand Ratio", f"{ratio:.2f}" if pd.notna(ratio) else "N/A")
        with col3:
            st.metric("Active Regions", df['state'].nunique())

        st.subheader("Latest Market Entries")
        st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)

    # TAB 2: Historical Trends
    with tab2:
        st.header("Historical Price Analysis")
        crop_filter = st.selectbox("Select Crop", df['crop_type'].unique())
        if 'date_range' not in st.session_state:
            st.session_state.date_range = (df['date'].min().date(), df['date'].max().date())
        date_range = st.date_input("Select Date Range", value=st.session_state.date_range,
                                   min_value=df['date'].min().date(), max_value=df['date'].max().date())
        st.session_state.date_range = date_range

        filtered_df = df[(df['crop_type'] == crop_filter) & (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))]
        if filtered_df.empty:
            st.warning("No data available for that range — showing full trend for the selected crop.")
            filtered_df = df[df['crop_type']==crop_filter]

        fig = px.line(filtered_df, x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    # TAB 3: Weather Impact
    with tab3:
        st.header("Climate Correlation Analysis")
        weather_factor = st.selectbox("Select Weather Factor", ['rainfall_mm', 'temperature_c'])
        try:
            fig = px.scatter(df, x=weather_factor, y='price_₹/ton', color='crop_type', trendline="ols",
                             title=f"Price vs {weather_factor.replace('_',' ').title()}")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Error generating plot: {e}")

    # TAB 4: Prediction
    with tab4:
        st.header("Price Prediction Model")
        if not models:
            st.info("No trained models found. Use sidebar to train models.")
        else:
            model_choice = st.selectbox("Select Model", list(models.keys()))
            pipe = models[model_choice]

            # inputs
            col1, col2 = st.columns(2)
            with col1:
                state = st.selectbox("State/Province", sorted(df['state'].dropna().unique()))
                city = st.selectbox("City", sorted(df['city'].dropna().unique()))
                crop_type = st.selectbox("Crop Type", sorted(df['crop_type'].dropna().unique()))
            with col2:
                season = st.selectbox("Season", sorted(df['season'].dropna().unique()))
                # auto fill averages
                sub = df[(df['state']==state)&(df['city']==city)]
                if not sub.empty:
                    avg_month = int(sub['date'].dt.month.mode()[0])
                    avg_rain = float(sub['rainfall_mm'].mean())
                    avg_temp = float(sub['temperature_c'].mean())
                else:
                    avg_month = 6
                    avg_rain = float(df['rainfall_mm'].mean())
                    avg_temp = float(df['temperature_c'].mean())

                month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=int(avg_month))
                rainfall = st.number_input("Rainfall (mm)", value=float(round(avg_rain,1)))
                temperature = st.number_input("Temperature (°C)", value=float(round(avg_temp,1)))

            if st.button("Predict Price"):
                X_in = pd.DataFrame([[state, city, crop_type, season, month, rainfall, temperature]],
                                    columns=['state','city','crop_type','season','month','rainfall_mm','temperature_c'])
                try:
                    pred = pipe.predict(X_in)[0]
                    st.success(f"[{model_choice}] Predicted Price: ₹{pred:.2f}/ton")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

            # show basic learning curve for selected pipeline
            if st.checkbox("Show learning curve for selected model"):
                # build X,y consistent with pipeline training
                df_local = df.copy()
                df_local['month'] = df_local['date'].dt.month
                X_full = df_local[['state','city','crop_type','season','month','rainfall_mm','temperature_c']]
                y_full = df_local['price_₹/ton']
                with st.spinner("Computing learning curve..."):
                    plot_learning_curve_for_pipeline(pipe, X_full, y_full, cv=3)

    # TAB 5: Regional Analysis (Nepal)
    with tab5:
        st.header("Geographical Price Distribution (Nepal Provinces)")

        geojson_obj, prop_key = load_nepal_geojson()
        if geojson_obj is None:
            st.error("Could not load Nepal GeoJSON. Check internet or add a local file 'assets/nepal_provinces.geojson'.")
        else:
            # normalize province names - dataset has 'Madhesh Province' etc.
            df_map = df.copy()
            # remove trailing/leading whitespace
            df_map['province'] = df_map['state'].astype(str).str.strip()

            avg_prices = df_map.groupby('province', as_index=False)['price_₹/ton'].mean().rename(columns={'price_₹/ton':'avg_price'})
            # compute match diagnostics
            geo_names = set()
            for feat in geojson_obj.get('features', []):
                val = feat.get('properties', {}).get(prop_key)
                if val:
                    geo_names.add(str(val).strip())
            avg_prices['matched'] = avg_prices['province'].apply(lambda x: str(x).strip() in geo_names)
            match_rate = avg_prices['matched'].mean() * 100 if len(avg_prices)>0 else 0.0

            with st.expander("Map diagnostics"):
                st.write(f"Detected GeoJSON property key: `{prop_key}`")
                st.write(f"Province match rate: {match_rate:.1f}%")
                st.write("GeoJSON province names sample:", sorted(list(geo_names))[:10])
                st.dataframe(avg_prices.sort_values('matched', ascending=False).head(20))

            # if no matches, fallback to bar chart
            if avg_prices['matched'].sum() == 0:
                st.warning("No matching province names found between data and GeoJSON — showing bar chart.")
                st.bar_chart(avg_prices.sort_values('avg_price', ascending=False).set_index('province')['avg_price'])
            else:
                mapped = avg_prices[avg_prices['matched']].copy()
                featureid = f"properties.{prop_key}"
                try:
                    fig = px.choropleth(
                        mapped,
                        geojson=geojson_obj,
                        locations='province',
                        featureidkey=featureid,
                        color='avg_price',
                        color_continuous_scale=px.colors.sequential.YlOrBr,
                        hover_name='province',
                        title='Average Price by Province (Nepal)'
                    )
                    fig.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Choropleth render error: {e}")
                    st.bar_chart(avg_prices.sort_values('avg_price', ascending=False).set_index('province')['avg_price'])

    # Report generation
    st.sidebar.header("Report")
    if st.sidebar.button("📥 Generate Summary Report"):
        report = df.describe().T
        csv = report.to_csv().encode('utf-8')
        st.sidebar.download_button(label="Download Summary CSV", data=csv, file_name="market_summary.csv", mime="text/csv")

if __name__ == "__main__":
    main()
