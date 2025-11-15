# app.py
import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")
st.title("Agricultural Market Price Analyzer 🌾")

# -------------------------
# Data loader (accepts uploaded files or local path)
# -------------------------
@st.cache_data
def load_data(file_input=None):
    """
    Accept either a file-like (uploaded) object or a local filepath string.
    Returns a DataFrame with parsed dates where present.
    """
    try:
        if file_input is None:
            path = 'cleaned_dataset.csv'
            if os.path.exists(path):
                df = pd.read_csv(path)
            else:
                return pd.DataFrame()
        elif hasattr(file_input, "read"):
            # uploaded file-like object
            df = pd.read_csv(file_input)
        elif isinstance(file_input, str) and os.path.exists(file_input):
            df = pd.read_csv(file_input)
        else:
            return pd.DataFrame()

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# -------------------------
# Train function (specialized per model)
# -------------------------
def train_model_and_save(df, model_choice, do_random_search=False):
    """
    Train selected model (one of 'XGBoost','Random Forest','Linear Regression').
    Returns training metrics dict and saves model + encoders to model.pkl
    """
    # required columns
    required_cols = {'state','city','crop_type','season','rainfall_mm','temperature_c',
                     'supply_volume_tons','demand_volume_tons','price_₹/ton','date'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df_local = df.copy()
    # create month
    df_local['month'] = df_local['date'].dt.month.fillna(6).astype(int)

    # feature lists
    cat_cols = ['state','city','crop_type','season','month']  # month included as category to encode consistently
    num_cols = ['rainfall_mm','temperature_c','supply_volume_tons','demand_volume_tons']
    feature_cols = cat_cols + num_cols

    # Fit LabelEncoders on raw string columns WITHOUT modifying the original df used elsewhere
    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder()
        # convert to string to avoid issues
        df_local[c] = df_local[c].astype(str).fillna('___NA___')
        le.fit(df_local[c])
        le_dict[c] = le

    # Build numeric X, y for training (encode categorical columns)
    X_enc = pd.DataFrame()
    for c in cat_cols:
        X_enc[c] = le_dict[c].transform(df_local[c].astype(str))

    for c in num_cols:
        X_enc[c] = pd.to_numeric(df_local[c], errors='coerce').fillna(df_local[c].mean())

    y = pd.to_numeric(df_local['price_₹/ton'], errors='coerce').fillna(0)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

    # Select base model with specialized defaults
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model_path = "models/lr_model.pkl"
    elif model_choice == "Random Forest":
        # tuned-ish defaults for accuracy while remaining reasonably fast
        model = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
        model_path = "models/rf_model.pkl"
    else:
        # XGBoost: use performant defaults; optionally run a lightweight randomized search if requested
        model = XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.08,
                             max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs= -1)
        model_path = "models/xgb_model.pkl"

    # Optional quick hyperparameter tuning for XGBoost (kept small to avoid long runs)
    if model_choice == "XGBoost" and do_random_search:
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.08, 0.1],
            'max_depth': [3, 5, 6, 8],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=8,
            scoring='neg_mean_absolute_error',
            cv=3,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_

    # Fit final model
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Save model + encoders + metadata as single model.pkl (overwrite)
    os.makedirs("models", exist_ok=True)
    payload = {
        'model': model,
        'le_dict': le_dict,
        'feature_cols': feature_cols,
        'model_choice': model_choice
    }
    with open("model.pkl", "wb") as f:
        pickle.dump(payload, f)

    # Also save separate specialized file (optional)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    return {'MAE': float(mae), 'R2': float(r2)}

# -------------------------
# Prepare single-row input for prediction (auto-fetch month/weather/supply/demand)
# -------------------------
def prepare_input_row(df, le_dict, feature_cols, state, city, crop_type, season):
    # Use dataset averages for the selection
    df_local = df.copy()
    if 'date' in df_local.columns and 'month' not in df_local.columns:
        df_local['month'] = df_local['date'].dt.month.fillna(6).astype(int)

    subset = df_local[
        (df_local['state'].astype(str) == str(state)) &
        (df_local['city'].astype(str) == str(city)) &
        (df_local['crop_type'].astype(str) == str(crop_type)) &
        (df_local['season'].astype(str) == str(season))
    ]

    if not subset.empty:
        month_val = int(subset['date'].dt.month.mode()[0]) if 'date' in subset.columns else int(subset['month'].mode()[0])
        rainfall_val = float(subset['rainfall_mm'].mean())
        temp_val = float(subset['temperature_c'].mean())
        supply_val = float(subset['supply_volume_tons'].mean()) if 'supply_volume_tons' in subset.columns else float(df_local['supply_volume_tons'].mean())
        demand_val = float(subset['demand_volume_tons'].mean()) if 'demand_volume_tons' in subset.columns else float(df_local['demand_volume_tons'].mean())
    else:
        # fallback to global averages
        month_val = int(df_local['date'].dt.month.mode()[0]) if 'date' in df_local.columns else 6
        rainfall_val = float(df_local['rainfall_mm'].mean())
        temp_val = float(df_local['temperature_c'].mean())
        supply_val = float(df_local['supply_volume_tons'].mean()) if 'supply_volume_tons' in df_local.columns else 0.0
        demand_val = float(df_local['demand_volume_tons'].mean()) if 'demand_volume_tons' in df_local.columns else 0.0

    # Build raw row dictionary
    raw = {
        'state': str(state),
        'city': str(city),
        'crop_type': str(crop_type),
        'season': str(season),
        'month': int(month_val),
        'rainfall_mm': float(rainfall_val),
        'temperature_c': float(temp_val),
        'supply_volume_tons': float(supply_val),
        'demand_volume_tons': float(demand_val)
    }

    # Encode categorical values using le_dict
    encoded = []
    for col in feature_cols:
        if col in ['rainfall_mm','temperature_c','supply_volume_tons','demand_volume_tons']:
            encoded.append(raw[col])
        else:
            le = le_dict[col]
            # handle unseen labels: if unseen, map to nearest or to index 0
            try:
                val_enc = le.transform([raw[col]])[0]
            except Exception:
                classes = list(le.classes_)
                if '___NA___' in classes:
                    val_enc = int(np.where(np.array(classes) == '___NA___')[0][0])
                else:
                    val_enc = 0
            encoded.append(val_enc)

    X_row = pd.DataFrame([encoded], columns=feature_cols)
    return X_row, raw

# -------------------------
# Short justification strings for report/slides
# -------------------------
MODEL_JUSTIFICATIONS = {
    "Linear Regression": (
        "Linear Regression — baseline, extremely fast and interpretable. "
        "Useful as a reference model and for understanding linear relationships between features and price."
    ),
    "Random Forest": (
        "Random Forest — ensemble of decision trees that captures non-linearities and interactions without much tuning. "
        "Robust to outliers and works well on tabular data; good balance between accuracy and training time."
    ),
    "XGBoost": (
        "XGBoost — gradient boosting implementation optimized for speed and accuracy. "
        "Performs well on structured data; supports many hyperparameters to improve accuracy. "
        "We use tuned defaults and optional small randomized search for further improvement."
    )
}

# -------------------------
# UI & Main
# -------------------------
def main():
    st.sidebar.header("Data Management & Model Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])
    df = load_data(uploaded_file if uploaded_file else None)

    if df.empty:
        st.warning("No data loaded. Upload CSV or place 'cleaned_dataset.csv' next to this app.")
        return

    st.sidebar.success(f"Loaded dataset: {len(df):,} rows")

    # Show justification and model choices in sidebar
    st.sidebar.markdown("### Choose model to train (single saved model used for predictions)")
    model_choice = st.sidebar.selectbox("Algorithm", ["XGBoost","Random Forest","Linear Regression"])
    st.sidebar.markdown("**Why this model?**")
    st.sidebar.info(MODEL_JUSTIFICATIONS[model_choice])

    # Toggle to run a short randomized search for XGBoost (kept small)
    do_random_search = False
    if model_choice == "XGBoost":
        do_random_search = st.sidebar.checkbox("Run small RandomizedSearchCV for XGBoost (longer)", value=False)

    if st.sidebar.button("Train & Save Selected Model"):
        with st.spinner("Training model — this may take a while..."):
            try:
                perf = train_model_and_save(df, model_choice, do_random_search=do_random_search)
                st.sidebar.success(f"Trained {model_choice} — MAE: {perf['MAE']:.2f}, R²: {perf['R2']:.3f}")
            except Exception as e:
                st.sidebar.error(f"Training failed: {e}")

    # Main tabs identical to your original layout
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Dashboard",
        "📈 Historical Trends",
        "🌦️ Weather Impact",
        "🔮 Price Prediction",
        "🗺️ Regional Analysis"
    ])

    # Tab1: Dashboard
    with tab1:
        st.header("Market Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'price_₹/ton' in df.columns:
                st.metric("Current Average Price", f"₹{df['price_₹/ton'].mean():.2f}/ton", "")
            else:
                st.metric("Current Average Price", "N/A", "")
        with col2:
            if 'supply_volume_tons' in df.columns and 'demand_volume_tons' in df.columns:
                ratio_series = (df['supply_volume_tons'] / df['demand_volume_tons']).replace([np.inf, -np.inf], np.nan)
                st.metric("Supply-Demand Ratio", f"{ratio_series.mean():.2f}", "")
            else:
                st.metric("Supply-Demand Ratio", "N/A", "")
        with col3:
            st.metric("Active Regions", int(df['state'].nunique()) if 'state' in df.columns else "N/A", "")

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
        else:
            st.warning("No crop_type column in dataset.")
            crop_filter = None

        if 'date' in df.columns and crop_filter is not None:
            if 'date_range' not in st.session_state:
                st.session_state['date_range'] = (df['date'].min().date(), df['date'].max().date())
            date_range = st.date_input("Select Date Range", value=st.session_state['date_range'],
                                       min_value=df['date'].min().date(), max_value=df['date'].max().date(), key="date_range_selector")
            st.session_state['date_range'] = date_range

            filtered_df = df[(df['crop_type'] == crop_filter) &
                             (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))]
            if filtered_df.empty:
                st.error("No data available for selected date range. Showing full historical trend.")
                filtered_df = df[df['crop_type'] == crop_filter]
            fig = px.line(filtered_df, x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
            st.plotly_chart(fig, use_container_width=True)
        elif crop_filter is not None:
            fig = px.line(df[df['crop_type'] == crop_filter], x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No crop data to show.")

    # Tab3: Weather Impact
    with tab3:
        st.header("Climate Correlation Analysis (Full)")
        # Price vs rainfall with trendline
        if 'rainfall_mm' in df.columns and 'price_₹/ton' in df.columns:
            st.subheader("Price vs Rainfall (scatter + OLS)")
            try:
                fig = px.scatter(df, x='rainfall_mm', y='price_₹/ton', color='crop_type', trendline="ols",
                                 title="Price vs Rainfall")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Statsmodels not available — showing scatter without OLS.")
                fig = px.scatter(df, x='rainfall_mm', y='price_₹/ton', color='crop_type', title="Price vs Rainfall")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("rainfall_mm or price_₹/ton column missing.")

        # Price vs temperature with trendline
        if 'temperature_c' in df.columns and 'price_₹/ton' in df.columns:
            st.subheader("Price vs Temperature (scatter + OLS)")
            try:
                fig = px.scatter(df, x='temperature_c', y='price_₹/ton', color='crop_type', trendline="ols",
                                 title="Price vs Temperature")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Statsmodels not available — showing scatter without OLS.")
                fig = px.scatter(df, x='temperature_c', y='price_₹/ton', color='crop_type', title="Price vs Temperature")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("temperature_c or price_₹/ton column missing.")

        # Heatmap: binned rainfall x temperature -> average price
        if all(c in df.columns for c in ['rainfall_mm','temperature_c','price_₹/ton']):
            st.subheader("Heatmap: Average Price (Rainfall × Temperature bins)")
            df_heat = df[['rainfall_mm','temperature_c','price_₹/ton']].dropna().copy()
            if not df_heat.empty:
                df_heat['rain_bin'] = pd.cut(df_heat['rainfall_mm'], bins=10)
                df_heat['temp_bin'] = pd.cut(df_heat['temperature_c'], bins=10)
                heat = df_heat.groupby(['temp_bin','rain_bin'])['price_₹/ton'].mean().reset_index()
                pivot = heat.pivot(index='temp_bin', columns='rain_bin', values='price_₹/ton')
                # create readable labels
                pivot.index = pivot.index.astype(str)
                pivot.columns = pivot.columns.astype(str)
                fig_heat = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                                     labels={'x':'Rainfall bin','y':'Temperature bin','color':'Avg Price'},
                                     title="Average Price by Rainfall & Temperature bins")
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Not enough data for heatmap.")
        else:
            st.info("Heatmap columns missing.")

        # Weather summary stats
        st.subheader("Weather Summary Statistics")
        if 'rainfall_mm' in df.columns:
            rain_mean = df['rainfall_mm'].mean()
            rain_median = df['rainfall_mm'].median()
            rain_std = df['rainfall_mm'].std()
            rain_corr = df[['rainfall_mm','price_₹/ton']].dropna().corr().iloc[0,1] if 'price_₹/ton' in df.columns else None
        else:
            rain_mean = rain_median = rain_std = rain_corr = None

        if 'temperature_c' in df.columns:
            temp_mean = df['temperature_c'].mean()
            temp_median = df['temperature_c'].median()
            temp_std = df['temperature_c'].std()
            temp_corr = df[['temperature_c','price_₹/ton']].dropna().corr().iloc[0,1] if 'price_₹/ton' in df.columns else None
        else:
            temp_mean = temp_median = temp_std = temp_corr = None

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Rainfall (mm)**")
            st.write(f"Mean: {rain_mean:.2f}" if rain_mean is not None else "N/A")
            st.write(f"Median: {rain_median:.2f}" if rain_median is not None else "N/A")
            st.write(f"Std: {rain_std:.2f}" if rain_std is not None else "N/A")
            st.write(f"Corr with price: {rain_corr:.3f}" if rain_corr is not None else "N/A")
        with col2:
            st.markdown("**Temperature (°C)**")
            st.write(f"Mean: {temp_mean:.2f}" if temp_mean is not None else "N/A")
            st.write(f"Median: {temp_median:.2f}" if temp_median is not None else "N/A")
            st.write(f"Std: {temp_std:.2f}" if temp_std is not None else "N/A")
            st.write(f"Corr with price: {temp_corr:.3f}" if temp_corr is not None else "N/A")

    # Tab4: Prediction (Option A behavior: auto month/weather)
    with tab4:
        st.header("Price Prediction")
        if not os.path.exists("model.pkl"):
            st.warning("No trained model found. Train a model from the sidebar.")
        else:
            with open("model.pkl","rb") as f:
                payload = pickle.load(f)
            model = payload['model']
            le_dict = payload['le_dict']
            feature_cols = payload['feature_cols']

            # use encoder classes to populate select boxes (keeps UI tied to trained encoders)
            # If user trained with different encoders, these reflect the training set values
            try:
                state_choices = list(le_dict['state'].classes_)
                city_choices = list(le_dict['city'].classes_)
                crop_choices = list(le_dict['crop_type'].classes_)
                season_choices = list(le_dict['season'].classes_)
            except Exception:
                # fallback to dataset values
                state_choices = sorted(df['state'].astype(str).unique())
                city_choices = sorted(df['city'].astype(str).unique())
                crop_choices = sorted(df['crop_type'].astype(str).unique())
                season_choices = sorted(df['season'].astype(str).unique())

            col1, col2 = st.columns(2)
            with col1:
                state = st.selectbox("State", state_choices)
                city = st.selectbox("City", city_choices)
            with col2:
                crop_type = st.selectbox("Crop Type", crop_choices)
                season = st.selectbox("Season", season_choices)

            # prepare auto-filled input (for transparency, show values)
            X_in, raw_row = prepare_input_row(df, le_dict, feature_cols, state, city, crop_type, season)
            st.markdown("**Auto-filled features (from historical averages for selected region/crop/season)**")
            st.write({
                "month": raw_row['month'],
                "rainfall_mm": round(raw_row['rainfall_mm'], 2),
                "temperature_c": round(raw_row['temperature_c'], 2),
                "supply_volume_tons": round(raw_row['supply_volume_tons'], 2),
                "demand_volume_tons": round(raw_row['demand_volume_tons'], 2)
            })

            if st.button("Predict Price"):
                try:
                    pred = model.predict(X_in)[0]
                    st.success(f"Predicted Price: ₹{pred:.2f} / ton")
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    # Tab5: Regional Analysis (unchanged map behavior)
    with tab5:
        st.header("Geographical Price Distribution")
        try:
            india_geojson = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
            avg_prices = df.groupby(['state','crop_type'])['price_₹/ton'].mean().reset_index()
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
                title="India State-wise Price Variations"
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Map rendering error: {str(e)}")

    # Report generation (download)
    st.sidebar.header("Report Generation")
    if st.sidebar.button("📥 Generate Full Report"):
        report = df.describe().T
        csv = report.to_csv().encode('utf-8')
        st.sidebar.download_button(label="Download Summary Report", data=csv, file_name="market_summary.csv", mime="text/csv")

if __name__ == "__main__":
    main()
