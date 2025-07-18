import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime
import os

# Configuration
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")

# Load initial or uploaded data
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
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Model training function
def train_model(df):
    try:
        df = df.copy()
        df['month'] = df['date'].dt.month

        categorical_cols = ['state', 'city', 'crop_type', 'season', 'month']
        numeric_cols = ['rainfall_mm', 'temperature_c']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ])

        X = df[categorical_cols + numeric_cols]
        y = df['price_₹/ton']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'XGBoost': XGBRegressor(),
            'Random Forest': RandomForestRegressor(),
            'Linear Regression': LinearRegression()
        }

        trained_models = {}
        model_performance = {}

        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            trained_models[name] = pipeline
            model_performance[name] = {'MAE': mae, 'R2': r2}

        with open('model.pkl', 'wb') as f:
            pickle.dump({'models': trained_models, 'columns': X.columns}, f)

        for name, scores in model_performance.items():
            st.sidebar.write(f"🔹 {name} → MAE: ₹{scores['MAE']:.2f}, R²: {scores['R2']:.2f}")

        return True
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False

# Main app
def main():
    st.title("Agricultural Market Price Analyzer 🌾")

    # Sidebar controls
    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])

    if uploaded_file:
        df = load_data(uploaded_file)
        df.to_csv("uploaded_dataset.csv", index=False)
        active_file = "uploaded_dataset.csv"
    else:
        df = load_data("cleaned_dataset.csv")
        active_file = "cleaned_dataset.csv"

    if not df.empty:
        st.sidebar.success("Data loaded successfully!")
        with st.expander("🔍 View Uploaded Data Sample"):
            st.dataframe(df.head(20), use_container_width=True)

        if st.sidebar.button("Retrain Prediction Model"):
            with st.spinner("Training new model..."):
                if train_model(df):
                    st.sidebar.success("Model updated successfully!")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Dashboard", "📈 Historical Trends", "🌦️ Weather Impact", "🔮 Price Prediction", "🗺️ Regional Analysis"])

    with tab1:
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

    with tab2:
        st.header("Historical Price Analysis")
        if 'date_range' not in st.session_state:
            st.session_state.date_range = [df['date'].min().date(), df['date'].max().date()]
        col1, col2 = st.columns(2)
        with col1:
            crop_filter = st.selectbox("Select Crop", df['crop_type'].unique())
        with col2:
            date_range = st.date_input("Select Date Range", value=st.session_state.date_range,
                                       min_value=df['date'].min().date(), max_value=df['date'].max().date(),
                                       key="date_range_selector")
        st.session_state.date_range = date_range
        filtered_df = df[(df['crop_type'] == crop_filter) & (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))]
        if filtered_df.empty:
            st.error("No data available for selected date range. Showing full historical trend.")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                if st.button("Reset to Default Date Range"):
                    st.session_state.date_range = [df['date'].min().date(), df['date'].max().date()]
                    st.experimental_rerun()
            filtered_df = df[df['crop_type'] == crop_filter]
        fig = px.line(filtered_df, x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Climate Correlation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            weather_factor = st.selectbox("Select Weather Factor", ['rainfall_mm', 'temperature_c'])
        fig = px.scatter(df, x=weather_factor, y='price_₹/ton', color='crop_type', trendline="ols",
                         title=f"Price vs {weather_factor.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Price Prediction Model")
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            models = model_data['models']
            columns = model_data['columns']

            col1, col2 = st.columns(2)
            with col1:
                model_choice = st.selectbox("Select Model", list(models.keys()))
                state = st.selectbox("State", df['state'].unique())
                city = st.selectbox("City", df['city'].unique())
                crop_type = st.selectbox("Crop Type", df['crop_type'].unique())
            with col2:
                season = st.selectbox("Season", df['season'].unique())
                filtered_data = df[(df['state'] == state) & (df['city'] == city) & (df['season'] == season)]
                if not filtered_data.empty:
                    avg_month = int(filtered_data['date'].dt.month.mode()[0])
                    avg_rainfall = filtered_data['rainfall_mm'].mean()
                    avg_temp = filtered_data['temperature_c'].mean()
                else:
                    avg_month = 6
                    avg_rainfall = 100.0
                    avg_temp = 25.0

            if st.button("Predict Price"):
                input_df = pd.DataFrame([[state, city, crop_type, season, avg_month, avg_rainfall, avg_temp]],
                                        columns=columns)
                selected_model = models[model_choice]
                prediction = selected_model.predict(input_df)
                st.success(f"[{model_choice}] Predicted Price: ₹{prediction[0]:.2f}/ton")
                st.caption(f"Based on {state}'s {season} season averages: {avg_rainfall:.1f}mm rainfall, {avg_temp:.1f}°C")
        else:
            st.warning("No trained model found. Upload data and train model first.")

    with tab5:
        st.header("Geographical Price Distribution (Nepal Provinces)")
        try:
            nepal_geojson = "https://raw.githubusercontent.com/sandeshchapagain/nepal-geojson/main/nepal-provinces.geojson"
            avg_prices = df.groupby(['state', 'crop_type'])['price_₹/ton'].mean().reset_index()
            fig = px.choropleth(avg_prices, geojson=nepal_geojson, locations="state", featureidkey="properties.name",
                                color="price_₹/ton", color_continuous_scale=px.colors.sequential.YlOrBr,
                                hover_name="state", animation_frame="crop_type",
                                title="Nepal Province-wise Price Variations")
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Map rendering error: {str(e)}")

    st.sidebar.header("Report Generation")
    if st.sidebar.button("📥 Generate Full Report"):
        report = df.describe().T
        st.sidebar.download_button(label="Download Summary Report", data=report.to_csv(),
                                   file_name="market_summary.csv", mime="text/csv")

if __name__ == "__main__":
    main()
