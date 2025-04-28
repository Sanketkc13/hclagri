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
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")

# Load initial data
@st.cache_data
def load_data(file_path='cleaned_dataset.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    return pd.DataFrame()

# Model training function
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
    st.title("Agricultural Market Price Analyzer 🌾")
    
    # Sidebar controls
    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])
    df = load_data(uploaded_file if uploaded_file else 'cleaned_dataset.csv')
    
    if not df.empty:
        st.sidebar.success("Data loaded successfully!")
        if st.sidebar.button("Retrain Prediction Model"):
            with st.spinner("Training new model..."):
                if train_model(df):
                    st.sidebar.success("Model updated successfully!")
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Dashboard", 
        "📈 Historical Trends", 
        "🌦️ Weather Impact", 
        "🔮 Price Prediction", 
        "🗺️ Regional Analysis"
    ])

    with tab1:  # Real-Time Dashboard
        st.header("Market Overview")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Average Price", 
                     f"₹{df['price_₹/ton'].mean():.2f}/ton",
                     "5.2% vs last month")
        
        with col2:
            st.metric("Supply-Demand Ratio", 
                     f"{(df['supply_volume_tons']/df['demand_volume_tons']).mean():.2f}",
                     "Market Balance")
        
        with col3:
            st.metric("Active Regions", 
                     df['state'].nunique(),
                     "States tracking prices")
        
        st.subheader("Latest Market Entries")
        st.dataframe(df.sort_values('date', ascending=False).head(10), 
                    use_container_width=True)

    with tab2:  # Historical Trends
        st.header("Historical Price Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            crop_filter = st.selectbox("Select Crop", df['crop_type'].unique())
        
        with col2:
            date_range = st.date_input("Select Date Range", 
                                      [df['date'].min(), df['date'].max()])
        
        filtered_df = df[(df['crop_type'] == crop_filter) & 
                        (df['date'].between(pd.to_datetime(date_range[0]), 
                                         pd.to_datetime(date_range[1])))]
        
        fig = px.line(filtered_df, x='date', y='price_₹/ton', 
                     title=f"{crop_filter} Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:  # Weather Impact
        st.header("Climate Correlation Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            weather_factor = st.selectbox("Select Weather Factor", 
                                        ['rainfall_mm', 'temperature_c'])
        
        try:
            fig = px.scatter(df, x=weather_factor, y='price_₹/ton', 
                            color='crop_type', trendline="ols",
                            title=f"Price vs {weather_factor.replace('_', ' ').title()}")
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.error("Statsmodels required for trendlines. Install with: pip install statsmodels")
        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")

    with tab4:  # Price Prediction
        st.header("Price Prediction Model")
        
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            model, le_dict = model_data['model'], model_data['le_dict']
            
            col1, col2 = st.columns(2)
            with col1:
                state = st.selectbox("State", le_dict['state'].classes_)
                city = st.selectbox("City", le_dict['city'].classes_)
                crop_type = st.selectbox("Crop Type", le_dict['crop_type'].classes_)
            
            with col2:
                season = st.selectbox("Season", le_dict['season'].classes_)
                month = st.selectbox("Month", range(1, 13))
                rainfall = st.number_input("Rainfall (mm)", value=100.0)
                temp = st.number_input("Temperature (°C)", value=25.0)
            
            if st.button("Predict Price"):
                input_data = pd.DataFrame([[
                    state, city, crop_type, season, month, rainfall, temp
                ]], columns=['state', 'city', 'crop_type', 'season', 
                           'month', 'rainfall_mm', 'temperature_c'])
                
                for col in ['state', 'city', 'crop_type', 'season', 'month']:
                    input_data[col] = le_dict[col].transform(input_data[col])
                
                prediction = model.predict(input_data)
                st.success(f"Predicted Price: ₹{prediction[0]:.2f}/ton")
        else:
            st.warning("No trained model found. Upload data and train model first.")

   with tab5:  # Regional Analysis
    st.header("Geographical Price Distribution")
    
    try:
        # Load India states GeoJSON
        india_geojson = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
        
        avg_prices = df.groupby(['state', 'crop_type'])['price_₹/ton'].mean().reset_index()
        
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

    # Report generation
    st.sidebar.header("Report Generation")
    if st.sidebar.button("📥 Generate Full Report"):
        report = df.describe().T
        st.sidebar.download_button(
            label="Download Summary Report",
            data=report.to_csv(),
            file_name="market_summary.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()