# app.py
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# -----------------------
# App config
# -----------------------
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")
st.title("Agricultural Market Price Analyzer 🌾")

# -----------------------
# Utility: load CSV (uploaded or default)
# -----------------------
@st.cache_data
def load_data(uploaded_file=None, path="cleaned_dataset.csv"):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif os.path.exists(path):
            df = pd.read_csv(path)
        else:
            return pd.DataFrame()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

# -----------------------
# Encode categorical columns (fit LabelEncoders)
# -----------------------
def fit_label_encoders(df, cat_cols):
    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder()
        # convert to str and fill na with placeholder
        arr = df[c].astype(str).fillna("___NA___")
        le.fit(arr)
        le_dict[c] = le
    return le_dict

# -----------------------
# Train model & save
# -----------------------
def train_and_save(df, model_choice, do_search=False):
    """
    Trains chosen model on df and saves payload to model.pkl
    Payload contains: model, le_dict, feature_cols, model_choice
    Returns performance dict.
    """
    # required columns
    required = {'state','city','crop_type','season','rainfall_mm','temperature_c',
                'supply_volume_tons','demand_volume_tons','price_₹/ton','date'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")

    df2 = df.copy()
    df2['month'] = df2['date'].dt.month.fillna(6).astype(int)

    cat_cols = ['state','city','crop_type','season','month']
    num_cols = ['rainfall_mm','temperature_c','supply_volume_tons','demand_volume_tons']
    feature_cols = cat_cols + num_cols

    # Fit label encoders on string representation (preserve order)
    df_cat = df2[cat_cols].astype(str).fillna("___NA___")
    le_dict = fit_label_encoders(df_cat, cat_cols)

    # Build encoded X
    X_enc = pd.DataFrame()
    for c in cat_cols:
        X_enc[c] = le_dict[c].transform(df_cat[c])
    for c in num_cols:
        X_enc[c] = pd.to_numeric(df2[c], errors='coerce').fillna(df2[c].mean())

    y = pd.to_numeric(df2['price_₹/ton'], errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

    # Model specialization / defaults
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model_path = "models/lr_model.pkl"
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
        model_path = "models/rf_model.pkl"
    else:  # XGBoost defaults
        model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.08,
                             max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        model_path = "models/xgb_model.pkl"

    # Optional small randomized search for XGBoost
    if model_choice == "XGBoost" and do_search:
        param_dist = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.08, 0.1],
            'max_depth': [3, 5, 6],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=6,
            scoring='neg_mean_absolute_error',
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_

    # fit final
    model.fit(X_train, y_train)

    # evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # save payload
    payload = {
        'model': model,
        'le_dict': le_dict,
        'feature_cols': feature_cols,
        'model_choice': model_choice
    }
    os.makedirs("models", exist_ok=True)
    with open("model.pkl","wb") as f:
        pickle.dump(payload, f)
    # also save specialized model file for convenience
    with open(model_path,"wb") as f:
        pickle.dump(payload, f)

    return {'MAE': float(mae), 'R2': float(r2)}

# -----------------------
# Prepare single input from selection (auto-fill)
# -----------------------
def prepare_input(df, le_dict, feature_cols, state, city, crop_type, season):
    df2 = df.copy()
    if 'date' in df2.columns and 'month' not in df2.columns:
        df2['month'] = df2['date'].dt.month.fillna(6).astype(int)

    subset = df2[
        (df2['state'].astype(str) == str(state)) &
        (df2['city'].astype(str) == str(city)) &
        (df2['crop_type'].astype(str) == str(crop_type)) &
        (df2['season'].astype(str) == str(season))
    ]

    if not subset.empty:
        month = int(subset['date'].dt.month.mode()[0]) if 'date' in subset.columns else int(subset['month'].mode()[0])
        rainfall = float(subset['rainfall_mm'].mean())
        temp = float(subset['temperature_c'].mean())
        supply = float(subset['supply_volume_tons'].mean()) if 'supply_volume_tons' in subset.columns else float(df2['supply_volume_tons'].mean())
        demand = float(subset['demand_volume_tons'].mean()) if 'demand_volume_tons' in subset.columns else float(df2['demand_volume_tons'].mean())
    else:
        month = int(df2['date'].dt.month.mode()[0]) if 'date' in df2.columns else 6
        rainfall = float(df2['rainfall_mm'].mean())
        temp = float(df2['temperature_c'].mean())
        supply = float(df2['supply_volume_tons'].mean()) if 'supply_volume_tons' in df2.columns else 0.0
        demand = float(df2['demand_volume_tons'].mean()) if 'demand_volume_tons' in df2.columns else 0.0

    raw = {
        'state': str(state),
        'city': str(city),
        'crop_type': str(crop_type),
        'season': str(season),
        'month': int(month),
        'rainfall_mm': float(rainfall),
        'temperature_c': float(temp),
        'supply_volume_tons': float(supply),
        'demand_volume_tons': float(demand)
    }

    encoded = []
    for col in feature_cols:
        if col in ['rainfall_mm','temperature_c','supply_volume_tons','demand_volume_tons']:
            encoded.append(raw[col])
        else:
            le = le_dict.get(col)
            if le is None:
                # fallback 0
                encoded.append(0)
            else:
                try:
                    transformed = le.transform([raw[col]])[0]
                except Exception:
                    # fallback if unseen label
                    if "___NA___" in list(le.classes_):
                        transformed = int(np.where(np.array(le.classes_)=="___NA___")[0][0])
                    else:
                        transformed = 0
                encoded.append(transformed)
    X_row = pd.DataFrame([encoded], columns=feature_cols)
    return X_row, raw

# -----------------------
# Model justification strings (for report/slides)
# -----------------------
JUSTIFICATIONS = {
    "Linear Regression": "Baseline, very fast, interpretable — good reference model.",
    "Random Forest": "Captures non-linearities, robust, low tuning required — good default for tabular data.",
    "XGBoost": "State-of-the-art gradient boosting, highly accurate with tuning; good for top performance."
}

# -----------------------
# MAIN UI
# -----------------------
def main():
    # Sidebar: Load / train controls
    st.sidebar.header("Data & Model Controls")
    uploaded = st.sidebar.file_uploader("Upload CSV (cleaned_dataset.csv format)", type=['csv'])
    df = load_data(uploaded_file=uploaded) if uploaded else load_data()

    if df.empty:
        st.warning("No data found — upload a CSV or place 'cleaned_dataset.csv' beside this app.")
        return
    else:
        st.sidebar.success(f"Loaded dataset ({len(df):,} rows)")

    st.sidebar.markdown("### Model selection & justifications")
    model_choice = st.sidebar.selectbox("Choose model to train", ["XGBoost","Random Forest","Linear Regression"])
    st.sidebar.info(JUSTIFICATIONS[model_choice])

    do_search = False
    if model_choice == "XGBoost":
        do_search = st.sidebar.checkbox("Run small RandomizedSearchCV for XGBoost (longer)", value=False)

    if st.sidebar.button("Train & Save Model"):
        with st.spinner("Training..."):
            try:
                perf = train_and_save(df, model_choice, do_search)
                st.sidebar.success(f"Trained {model_choice} — MAE: {perf['MAE']:.2f}, R²: {perf['R2']:.3f}")
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")

    # tabs (original layout)
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Dashboard",
        "📈 Historical Trends",
        "🌦 Weather Impact",
        "🔮 Price Prediction",
        "🗺 Regional Analysis"
    ])

    # Tab1: Dashboard
    with tab1:
        st.header("Market Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            if 'price_₹/ton' in df.columns:
                st.metric("Current Average Price", f"₹{df['price_₹/ton'].mean():.2f}/ton")
            else:
                st.metric("Current Average Price", "N/A")
        with c2:
            if 'supply_volume_tons' in df.columns and 'demand_volume_tons' in df.columns:
                ratio = (df['supply_volume_tons']/df['demand_volume_tons']).replace([np.inf,-np.inf], np.nan)
                st.metric("Supply-Demand Ratio", f"{ratio.mean():.2f}")
            else:
                st.metric("Supply-Demand Ratio", "N/A")
        with c3:
            st.metric("Active Regions", int(df['state'].nunique()) if 'state' in df.columns else "N/A")
        st.subheader("Latest Market Entries")
        if 'date' in df.columns:
            st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)
        else:
            st.dataframe(df.head(10), use_container_width=True)

    # Tab2: Historical Trends
    with tab2:
        st.header("Historical Price Analysis")
        if 'crop_type' in df.columns:
            crop_filter = st.selectbox("Select Crop", df['crop_type'].unique())
            if 'date' in df.columns:
                if 'date_range' not in st.session_state:
                    st.session_state['date_range'] = (df['date'].min().date(), df['date'].max().date())
                date_range = st.date_input("Select Date Range", value=st.session_state['date_range'],
                                           min_value=df['date'].min().date(), max_value=df['date'].max().date(), key='date_range_selector')
                st.session_state['date_range'] = date_range
                filtered = df[(df['crop_type']==crop_filter) & (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))]
                if filtered.empty:
                    st.error("No data available for selected date range. Showing full historical trend.")
                    filtered = df[df['crop_type']==crop_filter]
                fig = px.line(filtered, x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.line(df[df['crop_type']==crop_filter], x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No crop_type column in dataset.")

    # Tab3: Weather Impact (full analytics)
    with tab3:
        st.header("Climate Correlation Analysis (Full)")
        # Price vs Rainfall
        if 'rainfall_mm' in df.columns and 'price_₹/ton' in df.columns:
            st.subheader("Price vs Rainfall (scatter + OLS)")
            try:
                fig = px.scatter(df, x='rainfall_mm', y='price_₹/ton', color='crop_type', trendline="ols", title="Price vs Rainfall")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Trendline requires statsmodels; showing scatter only.")
                fig = px.scatter(df, x='rainfall_mm', y='price_₹/ton', color='crop_type', title="Price vs Rainfall")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("rainfall_mm or price_₹/ton column missing.")

        # Price vs Temperature
        if 'temperature_c' in df.columns and 'price_₹/ton' in df.columns:
            st.subheader("Price vs Temperature (scatter + OLS)")
            try:
                fig2 = px.scatter(df, x='temperature_c', y='price_₹/ton', color='crop_type', trendline="ols", title="Price vs Temperature")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                st.warning("Trendline requires statsmodels; showing scatter only.")
                fig2 = px.scatter(df, x='temperature_c', y='price_₹/ton', color='crop_type', title="Price vs Temperature")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("temperature_c or price_₹/ton column missing.")

        # Heatmap: binned rainfall x temperature -> avg price
        st.subheader("Heatmap: Average Price by Rainfall × Temperature bins")
        try:
            if set(['rainfall_mm','temperature_c','price_₹/ton']).issubset(df.columns):
                df_heat = df[['rainfall_mm','temperature_c','price_₹/ton']].dropna().copy()
                if df_heat.empty or df_heat.shape[0] < 20:
                    st.info("Not enough data for heatmap.")
                else:
                    df_heat['rain_bin'] = pd.cut(df_heat['rainfall_mm'], bins=10)
                    df_heat['temp_bin'] = pd.cut(df_heat['temperature_c'], bins=10)
                    heat = df_heat.groupby(['temp_bin','rain_bin'])['price_₹/ton'].mean().reset_index()
                    pivot = heat.pivot(index='temp_bin', columns='rain_bin', values='price_₹/ton')
                    pivot.index = pivot.index.astype(str)
                    pivot.columns = pivot.columns.astype(str)
                    fig_heat = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                                         labels={'x':'Rainfall bin','y':'Temperature bin','color':'Avg Price'},
                                         title="Average Price by Rainfall & Temperature bins")
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Heatmap requires rainfall_mm, temperature_c and price_₹/ton columns.")
        except Exception as e:
            st.error(f"Heatmap generation error: {e}")

        # Weather summary stats
        st.subheader("Weather Summary Statistics")
        try:
            if 'rainfall_mm' in df.columns:
                rain_mean, rain_med, rain_std = df['rainfall_mm'].mean(), df['rainfall_mm'].median(), df['rainfall_mm'].std()
                rain_corr = df[['rainfall_mm','price_₹/ton']].dropna().corr().iloc[0,1] if 'price_₹/ton' in df.columns else None
            else:
                rain_mean = rain_med = rain_std = rain_corr = None

            if 'temperature_c' in df.columns:
                temp_mean, temp_med, temp_std = df['temperature_c'].mean(), df['temperature_c'].median(), df['temperature_c'].std()
                temp_corr = df[['temperature_c','price_₹/ton']].dropna().corr().iloc[0,1] if 'price_₹/ton' in df.columns else None
            else:
                temp_mean = temp_med = temp_std = temp_corr = None

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rainfall (mm)**")
                st.write(f"Mean: {rain_mean:.2f}" if rain_mean is not None else "N/A")
                st.write(f"Median: {rain_med:.2f}" if rain_med is not None else "N/A")
                st.write(f"Std: {rain_std:.2f}" if rain_std is not None else "N/A")
                st.write(f"Correlation with price: {rain_corr:.3f}" if rain_corr is not None else "N/A")
            with c2:
                st.markdown("**Temperature (°C)**")
                st.write(f"Mean: {temp_mean:.2f}" if temp_mean is not None else "N/A")
                st.write(f"Median: {temp_med:.2f}" if temp_med is not None else "N/A")
                st.write(f"Std: {temp_std:.2f}" if temp_std is not None else "N/A")
                st.write(f"Correlation with price: {temp_corr:.3f}" if temp_corr is not None else "N/A")
        except Exception as e:
            st.error(f"Weather stats error: {e}")

    # Tab4: Prediction (Option A: auto-fill)
    with tab4:
        st.header("Price Prediction (Auto-filled month/weather/supply/demand)")

        if not os.path.exists("model.pkl"):
            st.warning("No trained model found. Train a model (sidebar) to enable predictions.")
        else:
            try:
                with open("model.pkl","rb") as f:
                    payload = pickle.load(f)
                model = payload['model']
                le_dict = payload['le_dict']
                feature_cols = payload['feature_cols']
            except Exception as e:
                st.error(f"Failed to load saved model: {e}")
                model = le_dict = feature_cols = None

            if model is not None:
                # populate selection boxes using dataset unique values (string)
                state_choices = sorted(df['state'].astype(str).unique()) if 'state' in df.columns else []
                city_choices = sorted(df['city'].astype(str).unique()) if 'city' in df.columns else []
                crop_choices = sorted(df['crop_type'].astype(str).unique()) if 'crop_type' in df.columns else []
                season_choices = sorted(df['season'].astype(str).unique()) if 'season' in df.columns else []

                col1, col2 = st.columns(2)
                with col1:
                    state = st.selectbox("State", state_choices)
                    city = st.selectbox("City", city_choices)
                with col2:
                    crop_type = st.selectbox("Crop Type", crop_choices)
                    season = st.selectbox("Season", season_choices)

                X_row, raw = prepare_input(df, le_dict, feature_cols, state, city, crop_type, season)
                st.markdown("**Auto-filled inputs used for prediction**")
                st.write({
                    "month": raw['month'],
                    "rainfall_mm": round(raw['rainfall_mm'],2),
                    "temperature_c": round(raw['temperature_c'],2),
                    "supply_volume_tons": round(raw['supply_volume_tons'],2),
                    "demand_volume_tons": round(raw['demand_volume_tons'],2)
                })

                if st.button("Predict Price"):
                    try:
                        pred = model.predict(X_row)[0]
                        st.success(f"Predicted Price: ₹{pred:.2f} / ton")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    # Tab5: Regional Analysis (unchanged default behavior)
    with tab5:
        st.header("Geographical Price Distribution (unchanged)")
        try:
            # attempt to display India choropleth as before
            india_geojson = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
            avg_prices = df.groupby(['state','crop_type'])['price_₹/ton'].mean().reset_index()
            if avg_prices.empty:
                st.info("No aggregated data for map.")
            else:
                fig = px.choropleth(
                    avg_prices,
                    geojson=india_geojson,
                    locations="state",
                    featureidkey="properties.NAME_1",
                    color="price_₹/ton",
                    color_continuous_scale=px.colors.sequential.Plasma,
                    hover_name="state",
                    animation_frame="crop_type",
                    scope="asia",
                    title="State-wise Price Variations"
                )
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Map rendering error: {e}")

    # Sidebar: download model if exists, and report generation
    st.sidebar.markdown("---")
    if os.path.exists("model.pkl"):
        with open("model.pkl","rb") as f:
            bin_model = f.read()
        st.sidebar.download_button("Download trained model (model.pkl)", data=bin_model, file_name="model.pkl", mime="application/octet-stream")

    st.sidebar.header("Reports")
    if st.sidebar.button("📥 Generate Summary CSV"):
        try:
            report = df.describe().T
            csv = report.to_csv().encode('utf-8')
            st.sidebar.download_button("Download Summary", data=csv, file_name="market_summary.csv", mime="text/csv")
        except Exception as e:
            st.sidebar.error(f"Report generation failed: {e}")

if __name__ == "__main__":
    main()
