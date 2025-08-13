import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import json
import os

# =============================
# App Configuration
# =============================
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")

# =============================
# Province helpers (names + centroids + optional embedded GeoJSON)
# =============================
NEPAL_PROVINCE_ALIASES = {
    'Koshi': 'Province 1', 'Province 1': 'Province 1', 'Province No. 1': 'Province 1',
    'Madhesh': 'Province 2', 'Province 2': 'Province 2', 'Province No. 2': 'Province 2',
    'Bagmati': 'Bagmati', 'Province 3': 'Bagmati', 'Bagmati Province': 'Bagmati',
    'Gandaki': 'Gandaki', 'Province 4': 'Gandaki', 'Gandaki Province': 'Gandaki',
    'Lumbini': 'Lumbini', 'Province 5': 'Lumbini', 'Lumbini Province': 'Lumbini',
    'Karnali': 'Karnali', 'Province 6': 'Karnali', 'Karnali Province': 'Karnali',
    'Sudurpashchim': 'Sudurpashchim', 'Province 7': 'Sudurpashchim', 'Sudurpashchim Province': 'Sudurpashchim'
}

NEPAL_PROVINCE_CENTROIDS = {
    'Province 1': (27.2, 87.3),
    'Province 2': (26.8, 85.2),
    'Bagmati': (27.6, 85.4),
    'Gandaki': (28.2, 84.2),
    'Lumbini': (27.7, 83.3),
    'Karnali': (29.1, 82.6),
    'Sudurpashchim': (29.2, 80.9)
}

EMBEDDED_NEPAL_GEOJSON = {
    "type": "FeatureCollection",
    "features": [
        {"type":"Feature","properties":{"name":"Province 1"},"geometry":{"type":"Polygon","coordinates":[[[87.9,26.4],[87.9,27.8],[86.5,27.8],[86.5,26.4],[87.9,26.4]]]}},
        {"type":"Feature","properties":{"name":"Province 2"},"geometry":{"type":"Polygon","coordinates":[[[86.5,26.3],[86.5,27.1],[84.9,27.1],[84.9,26.3],[86.5,26.3]]]}},
        {"type":"Feature","properties":{"name":"Bagmati"},"geometry":{"type":"Polygon","coordinates":[[[86.6,27.1],[86.6,28.1],[85.2,28.1],[85.2,27.1],[86.6,27.1]]]}},
        {"type":"Feature","properties":{"name":"Gandaki"},"geometry":{"type":"Polygon","coordinates":[[[85.2,27.3],[85.2,28.6],[83.9,28.6],[83.9,27.3],[85.2,27.3]]]}},
        {"type":"Feature","properties":{"name":"Lumbini"},"geometry":{"type":"Polygon","coordinates":[[[84.0,26.8],[84.0,27.8],[82.9,27.8],[82.9,26.8],[84.0,26.8]]]}},
        {"type":"Feature","properties":{"name":"Karnali"},"geometry":{"type":"Polygon","coordinates":[[[83.2,28.2],[83.2,29.6],[81.9,29.6],[81.9,28.2],[83.2,28.2]]]}},
        {"type":"Feature","properties":{"name":"Sudurpashchim"},"geometry":{"type":"Polygon","coordinates":[[[81.9,28.0],[81.9,29.6],[80.0,29.6],[80.0,28.0],[81.9,28.0]]]}},
    ]
}

def normalize_state_name(x: str) -> str:
    if pd.isna(x):
        return x
    x = str(x).strip()
    return NEPAL_PROVINCE_ALIASES.get(x, x)

# =============================
# Data Loading
# =============================
@st.cache_data
def load_data(file_input='cleaned_dataset.csv'):
    try:
        if isinstance(file_input, str) and os.path.exists(file_input):
            df = pd.read_csv(file_input)
        elif hasattr(file_input, 'read'):
            df = pd.read_csv(file_input)
        else:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        if 'state' in df.columns:
            df['state'] = df['state'].apply(normalize_state_name)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# =============================
# Model Training Functions
# =============================
def build_preprocessor():
    categorical_cols = ['state', 'city', 'crop_type', 'season', 'month']
    numeric_cols = ['rainfall_mm', 'temperature_c']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
            ('num', StandardScaler(), numeric_cols)
        ]
    )
    return preprocessor, categorical_cols + numeric_cols

def train_model(df: pd.DataFrame):
    try:
        df = df.copy()
        df['month'] = df['date'].dt.month

        preprocessor, feature_cols = build_preprocessor()
        X = df[feature_cols]
        y = df['price_₹/ton']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'XGBoost': XGBRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }

        trained_models = {}
        model_performance = {}

        for name, model in models.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

            if isinstance(model, XGBRegressor):
                # Special handling for XGBoost
                X_train_trans = pipe.named_steps['preprocessor'].fit_transform(X_train)
                X_test_trans = pipe.named_steps['preprocessor'].transform(X_test)
                
                if hasattr(X_train_trans, 'toarray'):
                    X_train_trans = X_train_trans.toarray()
                    X_test_trans = X_test_trans.toarray()
                
                model.fit(
                    X_train_trans, y_train,
                    eval_set=[(X_test_trans, y_test)],
                    early_stopping_rounds=30,
                    verbose=False,
                    eval_metric='rmse'
                )
                
                pipe.named_steps['regressor'] = model
                y_pred = model.predict(X_test_trans)
            else:
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            trained_models[name] = pipe
            model_performance[name] = {'MAE': mae, 'R2': r2}

        with open('model.pkl', 'wb') as f:
            pickle.dump({'models': trained_models, 'columns': X.columns.tolist()}, f)

        for name, scores in model_performance.items():
            st.sidebar.write(f"🔹 {name} → MAE: ₹{scores['MAE']:.2f}, R²: {scores['R2']:.2f}")

        return True
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False

# =============================
# Main App
# =============================
def main():
    st.title("Agricultural Market Price Analyzer 🌾")

    # Data Loading
    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])
    df = load_data(uploaded_file if uploaded_file else "cleaned_dataset.csv")

    if not df.empty:
        st.sidebar.success("Data loaded successfully!")
        with st.expander("🔍 View Uploaded Data Sample"):
            st.dataframe(df.head(20), use_container_width=True)

        if st.sidebar.button("Retrain Prediction Model"):
            with st.spinner("Training new model..."):
                if train_model(df):
                    st.sidebar.success("Model updated successfully!")

    tabs = st.tabs([
        "📊 Real-Time Dashboard",
        "📈 Historical Trends",
        "🌦️ Weather Impact",
        "🔮 Price Prediction",
        "🗺️ Regional Analysis",
        "🧪 Model Diagnostics"
    ])

    # Dashboard Tab
    with tabs[0]:
        st.header("Market Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Average Price", f"₹{df['price_₹/ton'].mean():.2f}/ton", "5.2% vs last month")
        with col2:
            st.metric("Supply-Demand Ratio", f"{(df['supply_volume_tons']/df['demand_volume_tons']).mean():.2f}", "Market Balance")
        with col3:
            st.metric("Active Regions", df['state'].nunique(), "Provinces tracking prices")
        st.subheader("Latest Market Entries")
        st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)

    # Historical Trends Tab
    with tabs[1]:
        st.header("Historical Price Analysis")
        if 'date_range' not in st.session_state:
            st.session_state.date_range = [df['date'].min().date(), df['date'].max().date()]
        
        col1, col2 = st.columns(2)
        with col1:
            crop_filter = st.selectbox("Select Crop", df['crop_type'].unique())
        with col2:
            date_range = st.date_input(
                "Select Date Range",
                value=st.session_state.date_range,
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date()
            )
        
        filtered_df = df[(df['crop_type'] == crop_filter) & 
                        (df['date'].between(pd.to_datetime(date_range[0]), 
                         pd.to_datetime(date_range[1])))]
        
        if filtered_df.empty:
            st.error("No data available for selected date range. Showing full historical trend.")
            filtered_df = df[df['crop_type'] == crop_filter]
        
        fig = px.line(filtered_df, x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    # Price Prediction Tab
    with tabs[3]:
        st.header("Price Prediction Model")
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            models = model_data['models']
            columns = model_data['columns']

            col1, col2 = st.columns(2)
            with col1:
                model_choice = st.selectbox("Select Model", list(models.keys()))
                state = st.selectbox("State", sorted(df['state'].dropna().unique()))
                city = st.selectbox("City", sorted(df['city'].dropna().unique()))
                crop_type = st.selectbox("Crop Type", sorted(df['crop_type'].dropna().unique()))
            with col2:
                season = st.selectbox("Season", sorted(df['season'].dropna().unique()))
                filtered_data = df[(df['state'] == state) & (df['city'] == city) & (df['season'] == season)]
                
                if not filtered_data.empty:
                    avg_month = int(filtered_data['date'].dt.month.mode()[0])
                    avg_rainfall = float(filtered_data['rainfall_mm'].mean())
                    avg_temp = float(filtered_data['temperature_c'].mean())
                else:
                    avg_month = 6
                    avg_rainfall = 100.0
                    avg_temp = 25.0

            if st.button("Predict Price"):
                input_df = pd.DataFrame([[state, city, crop_type, season, avg_month, avg_rainfall, avg_temp]], 
                                      columns=columns)
                selected_model = models[model_choice]
                
                if isinstance(selected_model.named_steps['regressor'], XGBRegressor):
                    input_trans = selected_model.named_steps['preprocessor'].transform(input_df)
                    if hasattr(input_trans, 'toarray'):
                        input_trans = input_trans.toarray()
                    prediction = selected_model.named_steps['regressor'].predict(input_trans)
                else:
                    prediction = selected_model.predict(input_df)
                
                st.success(f"[{model_choice}] Predicted Price: ₹{prediction[0]:.2f}/ton")
                st.caption(f"Based on {state}'s {season} season averages: {avg_rainfall:.1f}mm rainfall, {avg_temp:.1f}°C")
        else:
            st.warning("No trained model found. Upload data and train model first.")

    # Regional Analysis Tab
    with tabs[4]:
        st.header("Geographical Price Distribution (Nepal Provinces)")
        try:
            local_path = os.path.join('assets', 'nepal_provinces.geojson')
            geojson_obj = EMBEDDED_NEPAL_GEOJSON
            if os.path.exists(local_path):
                with open(local_path, 'r', encoding='utf-8') as f:
                    geojson_obj = json.load(f)

            avg_prices = df.groupby(['state', 'crop_type'])['price_₹/ton'].mean().reset_index()
            avg_prices['state'] = avg_prices['state'].apply(normalize_state_name)

            try:
                fig = px.choropleth(
                    avg_prices,
                    geojson=geojson_obj,
                    locations="state",
                    featureidkey="properties.name",
                    color="price_₹/ton",
                    color_continuous_scale=px.colors.sequential.YlOrBr,
                    hover_name="state",
                    animation_frame="crop_type"
                )
                fig.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                scatter_df = avg_prices.copy()
                scatter_df['lat'] = scatter_df['state'].map(lambda s: NEPAL_PROVINCE_CENTROIDS.get(s, (27.7, 85.3))[0])
                scatter_df['lon'] = scatter_df['state'].map(lambda s: NEPAL_PROVINCE_CENTROIDS.get(s, (27.7, 85.3))[1])
                fig2 = px.scatter_geo(scatter_df, lat='lat', lon='lon', color='price_₹/ton', 
                                     hover_name='state', animation_frame='crop_type')
                fig2.update_geos(fitbounds="locations", visible=False)
                st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Map rendering error: {str(e)}")

    # Model Diagnostics Tab
    with tabs[5]:
        st.header("Model Diagnostics")
        if not os.path.exists('model.pkl'):
            st.info("Train the models first from the sidebar, then return here.")
        else:
            model_data = pickle.load(open('model.pkl', 'rb'))
            models = model_data['models']
            columns = model_data['columns']
            X = df[columns]
            y = df['price_₹/ton']

            st.subheader("PCA Analysis")
            color_by = st.selectbox("Color PCA by", ['crop_type', 'state', 'season', 'city'])
            n_comp = st.slider("Number of components", 2, 20, 10)
            if st.button("Run PCA"):
                preproc = list(models.values())[0].named_steps['preprocessor']
                feature_df = df[columns]
                pca_out = run_pca_analysis(preproc, feature_df, color_by=color_by, n_components=n_comp)
                st.plotly_chart(pca_out['scree'], use_container_width=True)
                st.plotly_chart(pca_out['cumulative'], use_container_width=True)
                if pca_out['scatter']:
                    st.plotly_chart(pca_out['scatter'], use_container_width=True)

    # Sidebar Notes
    with st.sidebar.expander("🗺️ Map Data Source"):
        st.markdown("""- Using **local** `./assets/nepal_provinces.geojson` if present.
- Otherwise falling back to **embedded simplified** shapes.
- For production, drop a full-precision GeoJSON at that path for accurate borders.""")

if __name__ == "__main__":
    main()