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

st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")
st.title("Agricultural Market Price Analyzer 🌾")

# -------------------
# Helpers to detect & normalize columns
# -------------------
def detect_price_column(cols):
    # find column name containing 'price' (case-insensitive)
    for c in cols:
        if 'price' in c.lower():
            return c
    return None

def detect_supply_column(cols):
    for c in cols:
        if c.lower().startswith('supply'):
            return c
    return None

def detect_demand_column(cols):
    for c in cols:
        if c.lower().startswith('demand'):
            return c
    return None

def standardize_dataframe(df):
    # copy to avoid mutating original
    df = df.copy()
    cols = df.columns.tolist()

    # detect important columns
    price_col = detect_price_column(cols)
    supply_col = detect_supply_column(cols)
    demand_col = detect_demand_column(cols)

    # check existence
    # Required: date, state, city, crop_type, season, rainfall_mm, temperature_c, price, supply, demand
    # We'll rename detected columns to standard names used by app
    mapping = {}
    if 'date' in cols:
        mapping['date'] = 'date'
    if 'state' in cols:
        mapping['state'] = 'state'
    if 'city' in cols:
        mapping['city'] = 'city'
    if 'crop_type' in cols:
        mapping['crop_type'] = 'crop_type'
    if 'season' in cols:
        mapping['season'] = 'season'
    if 'rainfall_mm' in cols:
        mapping['rainfall_mm'] = 'rainfall_mm'
    if 'temperature_c' in cols:
        mapping['temperature_c'] = 'temperature_c'

    if price_col:
        mapping[price_col] = 'price'
    if supply_col:
        mapping[supply_col] = 'supply_volume_tons'
    if demand_col:
        mapping[demand_col] = 'demand_volume_tons'

    df = df.rename(columns=mapping)

    return df, {'price_col': price_col, 'supply_col': supply_col, 'demand_col': demand_col, 'mapping': mapping}

# -------------------
# Data loader
# -------------------
@st.cache_data
def load_data(uploaded_file=None, default_path="cleaned_dataset.csv"):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        elif os.path.exists(default_path):
            df = pd.read_csv(default_path)
        else:
            return pd.DataFrame()
        # make sure date parse
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()

# -------------------
# Label encoders
# -------------------
def fit_label_encoders(df, cat_cols):
    le_dict = {}
    for c in cat_cols:
        le = LabelEncoder()
        arr = df[c].astype(str).fillna("___NA___")
        le.fit(arr)
        le_dict[c] = le
    return le_dict

# -------------------
# Train & save model
# -------------------
def train_and_save(df, model_choice, do_search=False):
    required = {'state','city','crop_type','season','rainfall_mm','temperature_c',
                'supply_volume_tons','demand_volume_tons','price','date'}
    if not required.issubset(set(df.columns)):
        missing = required - set(df.columns)
        raise ValueError(f"Dataset missing required columns (after normalization): {missing}")

    df2 = df.copy()
    df2['month'] = df2['date'].dt.month.fillna(6).astype(int)

    cat_cols = ['state','city','crop_type','season','month']
    num_cols = ['rainfall_mm','temperature_c','supply_volume_tons','demand_volume_tons']
    feature_cols = cat_cols + num_cols

    # fit encoders
    df_cat = df2[cat_cols].astype(str).fillna("___NA___")
    le_dict = fit_label_encoders(df_cat, cat_cols)

    # encode X
    X_enc = pd.DataFrame()
    for c in cat_cols:
        X_enc[c] = le_dict[c].transform(df_cat[c])
    for c in num_cols:
        X_enc[c] = pd.to_numeric(df2[c], errors='coerce').fillna(df2[c].mean())

    y = pd.to_numeric(df2['price'], errors='coerce').fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

    # choose model
    if model_choice == "Linear Regression":
        model = LinearRegression()
        model_path = "models/lr_model.pkl"
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=None, n_jobs=-1, random_state=42)
        model_path = "models/rf_model.pkl"
    else:
        model = XGBRegressor(objective='reg:squarederror',
                             n_estimators=200, learning_rate=0.08, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
        model_path = "models/xgb_model.pkl"

    if model_choice == "XGBoost" and do_search:
        param_dist = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.08],
            'max_depth': [3, 5, 6],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
        }
        search = RandomizedSearchCV(model, param_dist, n_iter=6, scoring='neg_mean_absolute_error', cv=3, random_state=42, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    payload = {
        'model': model,
        'le_dict': le_dict,
        'feature_cols': feature_cols,
        'model_choice': model_choice
    }
    os.makedirs("models", exist_ok=True)
    with open("model.pkl", "wb") as f:
        pickle.dump(payload, f)
    with open(model_path, "wb") as f:
        pickle.dump(payload, f)

    return {'MAE': float(mae), 'R2': float(r2)}

# -------------------
# Prepare prediction row (auto-fill)
# -------------------
def prepare_input_row(df, le_dict, feature_cols, state, city, crop_type, season):
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
        month_val = int(subset['date'].dt.month.mode()[0]) if 'date' in subset.columns else int(subset['month'].mode()[0])
        rainfall_val = float(subset['rainfall_mm'].mean())
        temp_val = float(subset['temperature_c'].mean())
        supply_val = float(subset['supply_volume_tons'].mean()) if 'supply_volume_tons' in subset.columns else float(df2['supply_volume_tons'].mean())
        demand_val = float(subset['demand_volume_tons'].mean()) if 'demand_volume_tons' in subset.columns else float(df2['demand_volume_tons'].mean())
    else:
        month_val = int(df2['date'].dt.month.mode()[0]) if 'date' in df2.columns else 6
        rainfall_val = float(df2['rainfall_mm'].mean())
        temp_val = float(df2['temperature_c'].mean())
        supply_val = float(df2['supply_volume_tons'].mean()) if 'supply_volume_tons' in df2.columns else 0.0
        demand_val = float(df2['demand_volume_tons'].mean()) if 'demand_volume_tons' in df2.columns else 0.0

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

    encoded = []
    for col in feature_cols:
        if col in ['rainfall_mm','temperature_c','supply_volume_tons','demand_volume_tons']:
            encoded.append(raw[col])
        else:
            le = le_dict.get(col)
            if le is None:
                encoded.append(0)
            else:
                try:
                    transformed = le.transform([raw[col]])[0]
                except Exception:
                    # fallback for unseen labels
                    classes = list(le.classes_)
                    if "___NA___" in classes:
                        transformed = int(np.where(np.array(classes) == "___NA___")[0][0])
                    else:
                        transformed = 0
                encoded.append(transformed)

    X_row = pd.DataFrame([encoded], columns=feature_cols)
    return X_row, raw

# -------------------
# Model justifications
# -------------------
JUST = {
    "Linear Regression": "Fast baseline, easy to interpret. Use for quick checks and comparisons.",
    "Random Forest": "Robust tree ensemble that captures non-linear relationships with little tuning.",
    "XGBoost": "High-performance gradient boosting; tuned defaults provide strong accuracy for tabular data."
}

# -------------------
# Main app UI
# -------------------
def main():
    st.sidebar.header("Data & Model Controls")
    uploaded = st.sidebar.file_uploader("Upload CSV (or leave to use cleaned_dataset.csv)", type=['csv'])
    df_raw = load_data(uploaded if uploaded else None)

    if df_raw.empty:
        st.warning("No data loaded. Upload a CSV or place 'cleaned_dataset.csv' in the same folder.")
        return

    # Normalize column names to standard internal names
    df, detect_info = standardize_dataframe(df_raw)

    st.sidebar.success(f"Loaded dataset ({len(df):,} rows)")

    # Check required fields existence after normalization
    # We'll allow training only if core fields exist; otherwise warn
    core_fields = {'date','state','city','crop_type','season','rainfall_mm','temperature_c','price'}
    missing_core = core_fields - set(df.columns)
    if missing_core:
        st.sidebar.error(f"Missing required columns (after auto-detection): {missing_core}")
        st.sidebar.info("Make sure your CSV has columns for date,state,city,crop_type,season,rainfall_mm,temperature_c and a price column (name containing 'price').")
        # still allow viewing but block training/prediction
    else:
        pass

    # MODEL SIDE BAR
    st.sidebar.markdown("### Choose algorithm to train")
    model_choice = st.sidebar.selectbox("Model", ["XGBoost","Random Forest","Linear Regression"])
    st.sidebar.info(JUST[model_choice])
    do_search = False
    if model_choice == "XGBoost":
        do_search = st.sidebar.checkbox("Run small hyperparameter search (XGBoost)", value=False)

    if st.sidebar.button("Train & Save Model"):
        if missing_core:
            st.sidebar.error("Cannot train: dataset missing required columns.")
        else:
            with st.spinner("Training model..."):
                try:
                    perf = train_and_save(df, model_choice, do_search)
                    st.sidebar.success(f"Trained {model_choice} — MAE: {perf['MAE']:.2f}, R²: {perf['R2']:.3f}")
                except Exception as e:
                    st.sidebar.error(f"Training failed: {e}")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "📈 Historical Trends", "🌦 Weather Impact", "🔮 Prediction", "🗺 Regional Analysis"
    ])

    # TAB 1 - Dashboard
    with tab1:
        st.header("Market Overview")
        c1, c2, c3 = st.columns(3)
        with c1:
            if 'price' in df.columns:
                st.metric("Average Price", f"₹{df['price'].mean():.2f}/ton")
            else:
                st.metric("Average Price", "N/A")
        with c2:
            if 'supply_volume_tons' in df.columns and 'demand_volume_tons' in df.columns:
                ratio = (df['supply_volume_tons'] / df['demand_volume_tons']).replace([np.inf,-np.inf], np.nan)
                st.metric("Supply/Demand Ratio", f"{ratio.mean():.2f}")
            else:
                st.metric("Supply/Demand Ratio", "N/A")
        with c3:
            st.metric("Active Regions", int(df['state'].nunique()) if 'state' in df.columns else "N/A")
        st.subheader("Latest Entries")
        if 'date' in df.columns:
            st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)
        else:
            st.dataframe(df.head(10), use_container_width=True)

    # TAB 2 - Historical Trends
    with tab2:
        st.header("Historical Price Analysis")
        if 'crop_type' in df.columns:
            crop = st.selectbox("Select Crop", df['crop_type'].unique())
            if 'date' in df.columns:
                if 'date_range' not in st.session_state:
                    st.session_state['date_range'] = (df['date'].min().date(), df['date'].max().date())
                dr = st.date_input("Date Range", value=st.session_state['date_range'],
                                   min_value=df['date'].min().date(), max_value=df['date'].max().date())
                st.session_state['date_range'] = dr
                filtered = df[(df['crop_type']==crop) & (df['date'].between(pd.to_datetime(dr[0]), pd.to_datetime(dr[1])))]
                if filtered.empty:
                    st.error("No data for that range — showing full crop history")
                    filtered = df[df['crop_type']==crop]
                fig = px.line(filtered, x='date', y='price', title=f"{crop} Price Trend")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No date column to plot time series.")
        else:
            st.info("No crop_type column present.")

    # TAB 3 - Weather Impact
    with tab3:
        st.header("Weather Impact (Full analytics)")
        if set(['rainfall_mm','price']).issubset(df.columns):
            st.subheader("Price vs Rainfall (scatter + OLS)")
            try:
                fig = px.scatter(df, x='rainfall_mm', y='price', color='crop_type', trendline="ols", title="Price vs Rainfall")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Trendline requires statsmodels. Showing scatter only.")
                fig = px.scatter(df, x='rainfall_mm', y='price', color='crop_type', title="Price vs Rainfall")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("rainfall_mm or price column missing.")

        if set(['temperature_c','price']).issubset(df.columns):
            st.subheader("Price vs Temperature (scatter + OLS)")
            try:
                fig = px.scatter(df, x='temperature_c', y='price', color='crop_type', trendline="ols", title="Price vs Temperature")
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.warning("Trendline requires statsmodels. Showing scatter only.")
                fig = px.scatter(df, x='temperature_c', y='price', color='crop_type', title="Price vs Temperature")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("temperature_c or price column missing.")

        st.subheader("Heatmap: Average Price over Rainfall × Temperature bins")
        try:
            if set(['rainfall_mm','temperature_c','price']).issubset(df.columns):
                df_heat = df[['rainfall_mm','temperature_c','price']].dropna().copy()
                if df_heat.shape[0] < 20:
                    st.info("Not enough data for a robust heatmap.")
                else:
                    df_heat['rain_bin'] = pd.cut(df_heat['rainfall_mm'], bins=10)
                    df_heat['temp_bin'] = pd.cut(df_heat['temperature_c'], bins=10)
                    heat = df_heat.groupby(['temp_bin','rain_bin'])['price'].mean().reset_index()
                    pivot = heat.pivot(index='temp_bin', columns='rain_bin', values='price')
                    pivot.index = pivot.index.astype(str)
                    pivot.columns = pivot.columns.astype(str)
                    fig_heat = px.imshow(pivot.values, x=pivot.columns, y=pivot.index,
                                         labels={'x':'Rainfall bin','y':'Temperature bin','color':'Avg Price'},
                                         title="Average Price by Rainfall & Temperature bins")
                    st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Heatmap requires rainfall_mm, temperature_c and price columns.")
        except Exception as e:
            st.error(f"Heatmap generation error: {e}")

        # summary stats
        st.subheader("Weather summary stats")
        try:
            if 'rainfall_mm' in df.columns:
                rmean, rmed, rstd = df['rainfall_mm'].mean(), df['rainfall_mm'].median(), df['rainfall_mm'].std()
                rcorr = df[['rainfall_mm','price']].dropna().corr().iloc[0,1] if 'price' in df.columns else None
            else:
                rmean = rmed = rstd = rcorr = None
            if 'temperature_c' in df.columns:
                tmean, tmed, tstd = df['temperature_c'].mean(), df['temperature_c'].median(), df['temperature_c'].std()
                tcorr = df[['temperature_c','price']].dropna().corr().iloc[0,1] if 'price' in df.columns else None
            else:
                tmean = tmed = tstd = tcorr = None
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Rainfall (mm)**")
                st.write(f"Mean: {rmean:.2f}" if rmean is not None else "N/A")
                st.write(f"Median: {rmed:.2f}" if rmed is not None else "N/A")
                st.write(f"Std: {rstd:.2f}" if rstd is not None else "N/A")
                st.write(f"Corr with price: {rcorr:.3f}" if rcorr is not None else "N/A")
            with c2:
                st.markdown("**Temperature (°C)**")
                st.write(f"Mean: {tmean:.2f}" if tmean is not None else "N/A")
                st.write(f"Median: {tmed:.2f}" if tmed is not None else "N/A")
                st.write(f"Std: {tstd:.2f}" if tstd is not None else "N/A")
                st.write(f"Corr with price: {tcorr:.3f}" if tcorr is not None else "N/A")
        except Exception as e:
            st.error(f"Weather summary error: {e}")

    # TAB 4: Prediction
    with tab4:
        st.header("Price Prediction (auto-filled features)")

        if not os.path.exists("model.pkl"):
            st.warning("No trained model found. Train & save a model in the sidebar.")
        else:
            try:
                with open("model.pkl","rb") as f:
                    payload = pickle.load(f)
                model = payload['model']
                le_dict = payload['le_dict']
                feature_cols = payload['feature_cols']
            except Exception as e:
                st.error(f"Failed to load model: {e}")
                model = le_dict = feature_cols = None

            if model is not None:
                # dependent city dropdown: filter cities by selected state (use actual df values)
                state_choice = st.selectbox("State", sorted(df['state'].astype(str).unique()))
                cities_in_state = sorted(df[df['state'].astype(str) == str(state_choice)]['city'].astype(str).unique())
                if len(cities_in_state) == 0:
                    st.warning("No cities found for selected state. Choose a different state.")
                    city_choice = st.text_input("City (type)", "")
                else:
                    city_choice = st.selectbox("City", cities_in_state)

                crop_choice = st.selectbox("Crop Type", sorted(df['crop_type'].astype(str).unique()))
                season_choice = st.selectbox("Season", sorted(df['season'].astype(str).unique()))

                # prepare encoded row (no debug auto-fill shown)
                X_row, raw = prepare_input_row(df, le_dict, feature_cols, state_choice, city_choice, crop_choice, season_choice)

                if st.button("Predict Price"):
                    try:
                        pred = model.predict(X_row)[0]
                        st.success(f"Predicted Price: ₹{pred:.2f} / ton")
                        st.caption(f"Prediction uses historical averages for {city_choice}, {state_choice}.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")

    # TAB 5: Regional Analysis (map)
    with tab5:
        st.header("Regional Analysis (map fallback)")
        try:
            # attempt to draw India map as before — safe fallback to message if issues
            india_geojson = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
            if set(['state','price']).issubset(df.columns):
                avg_prices = df.groupby(['state','crop_type'])['price'].mean().reset_index()
                if avg_prices.empty:
                    st.info("Not enough aggregated data for map.")
                else:
                    fig = px.choropleth(
                        avg_prices,
                        geojson=india_geojson,
                        locations="state",
                        featureidkey="properties.NAME_1",
                        color="price",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        hover_name="state",
                        animation_frame="crop_type",
                        scope="asia",
                        title="State-wise Price Variations"
                    )
                    fig.update_geos(fitbounds="locations", visible=False)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Map requires 'state' and a price column in the dataset.")
        except Exception as e:
            st.error(f"Map rendering error: {e}")

    # Sidebar: download model/report
    st.sidebar.markdown("---")
    if os.path.exists("model.pkl"):
        with open("model.pkl","rb") as f:
            bin_m = f.read()
        st.sidebar.download_button("Download trained model (model.pkl)", data=bin_m, file_name="model.pkl", mime="application/octet-stream")

    st.sidebar.header("Reports")
    if st.sidebar.button("Download summary CSV"):
        try:
            report = df.describe().T
            st.sidebar.download_button("Download summary", data=report.to_csv().encode('utf-8'),
                                       file_name="market_summary.csv", mime="text/csv")
        except Exception as e:
            st.sidebar.error(f"Failed to prepare report: {e}")

if __name__ == "__main__":
    main()
