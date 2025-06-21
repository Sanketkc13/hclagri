import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import datetime
import os

# Configuration
st.set_page_config(page_title="Nepal AgriPrice Analyzer", layout="wide")

# Load initial data
@st.cache_data
def load_data(file_path='cleaned_dataset.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

# Model training
def train_model(df):
    try:
        df = df.copy()
        df['month'] = df['date'].dt.month
        
        le_dict = {}
        categorical_cols = ['state', 'city', 'crop_type', 'season', 'month']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        
        X = df[categorical_cols + ['rainfall_mm', 'temperature_c']]
        y = df['price_₹/ton']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'le_dict': le_dict}, f)
        
        return True
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False

# Main app
def main():
    st.title("🌾 Nepal Agricultural Market Price Analyzer")

    # Sidebar
    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])
    df = load_data(uploaded_file if uploaded_file else 'cleaned_dataset.csv')

    if not df.empty:
        st.sidebar.success("✅ Data loaded successfully!")
        if st.sidebar.button("🔁 Retrain Prediction Model"):
            with st.spinner("Training model..."):
                if train_model(df):
                    st.sidebar.success("✅ Model retrained!")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Dashboard", 
        "📈 Historical Trends", 
        "🌦️ Weather Impact", 
        "🔮 Price Prediction", 
        "🗺️ Regional Analysis"
    ])

    with tab1:
        st.header("Market Overview (Nepal)")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Average Price", f"₹{df['price_₹/ton'].mean():.2f}/ton")
        with col2:
            ratio = (df['supply_volume_tons'] / df['demand_volume_tons']).mean()
            st.metric("Supply-Demand Ratio", f"{ratio:.2f}")
        with col3:
            st.metric("Active Provinces", df['state'].nunique())

        st.subheader("Latest Market Records")
        st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)

    with tab2:
        st.header("Historical Price Analysis")

        if 'date_range' not in st.session_state:
            st.session_state.date_range = [
                df['date'].min().date(),
                df['date'].max().date()
            ]

        col1, col2 = st.columns(2)
        with col1:
            crop_filter = st.selectbox("Select Crop", df['crop_type'].unique())
        with col2:
            date_range = st.date_input(
                "Select Date Range",
                value=st.session_state.date_range,
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date(),
                key="date_range_selector"
            )

        st.session_state.date_range = date_range

        filtered_df = df[
            (df['crop_type'] == crop_filter) &
            (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
        ]

        if filtered_df.empty:
            st.error("No data in range. Resetting filter.")
            if st.button("Reset Date Filter"):
                st.session_state.date_range = [
                    df['date'].min().date(),
                    df['date'].max().date()
                ]
                st.experimental_rerun()
            filtered_df = df[df['crop_type'] == crop_filter]

        fig = px.line(filtered_df, x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.header("Weather Impact on Prices")
        col1, col2 = st.columns(2)
        with col1:
            weather_factor = st.selectbox("Select Weather Factor", ['rainfall_mm', 'temperature_c'])

        fig = px.scatter(df, x=weather_factor, y='price_₹/ton', color='crop_type',
                         trendline="ols", title=f"Price vs {weather_factor.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.header("Price Prediction")

        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            model, le_dict = model_data['model'], model_data['le_dict']

            col1, col2 = st.columns(2)
            with col1:
                state = st.selectbox("Province", le_dict['state'].classes_)
                city = st.selectbox("City", le_dict['city'].classes_)
                crop_type = st.selectbox("Crop Type", le_dict['crop_type'].classes_)
            with col2:
                season = st.selectbox("Season", le_dict['season'].classes_)

                filtered_data = df[
                    (df['state'] == state) & 
                    (df['city'] == city) & 
                    (df['season'] == season)
                ]

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
                                        columns=['state', 'city', 'crop_type', 'season', 'month', 'rainfall_mm', 'temperature_c'])
                for col in ['state', 'city', 'crop_type', 'season', 'month']:
                    input_df[col] = le_dict[col].transform(input_df[col])

                prediction = model.predict(input_df)
                st.success(f"Estimated Price: ₹{prediction[0]:.2f}/ton")
        else:
            st.warning("Train the model first by uploading data.")

    with tab5:
        st.header("Regional Price Map (Coming Soon)")
        st.info("Nepali province-level mapping feature is under development.")

    # Report generation
    st.sidebar.header("📥 Generate Summary Report")
    if st.sidebar.button("Download Summary CSV"):
        report = df.describe().T
        st.sidebar.download_button(
            label="📄 Download Report",
            data=report.to_csv(),
            file_name="nepal_agri_summary.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
