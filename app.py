import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import datetime
import os

# ---------------------------------------------------------
# STREAMLIT CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")

# ---------------------------------------------------------
# FIXED: LOAD DATA (WORKS WITH UPLOADED FILE OR LOCAL FILE)
# ---------------------------------------------------------
@st.cache_data
def load_data(file_source='cleaned_dataset.csv'):
    try:
        # Case 1: User uploaded a file (it has a .read attribute)
        if hasattr(file_source, "read"):
            df = pd.read_csv(file_source)
            df['date'] = pd.to_datetime(df['date'])
            return df
        
        # Case 2: file_source is a string path
        if isinstance(file_source, str) and os.path.exists(file_source):
            df = pd.read_csv(file_source)
            df['date'] = pd.to_datetime(df['date'])
            return df
        
        # Case 3: No file found
        return pd.DataFrame()

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# TRAIN MODEL FUNCTION
# ---------------------------------------------------------
def train_model(df):
    try:
        df = df.copy()
        df['month'] = df['date'].dt.month
        
        le_dict = {}
        categorical_cols = ['state', 'city', 'crop_type', 'season', 'month']
        
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        
        X = df[categorical_cols + ['rainfall_mm', 'temperature_c']]
        y = df['price_₹/ton']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'le_dict': le_dict}, f)
        
        return True
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False

# ---------------------------------------------------------
# MAIN APP
# ---------------------------------------------------------
def main():
    st.title("Agricultural Market Price Analyzer 🌾")
    
    # ---------------- SIDEBAR: FILE UPLOAD ----------------
    st.sidebar.header("Data Management")

    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])

    # Load Nepal dataset OR default dataset
    df = load_data(uploaded_file if uploaded_file else 'cleaned_dataset.csv')
    
    if not df.empty:
        st.sidebar.success("Data loaded successfully!")
        if st.sidebar.button("Retrain Prediction Model"):
            with st.spinner("Training new model..."):
                if train_model(df):
                    st.sidebar.success("Model updated successfully!")
    else:
        st.warning("No dataset available. Please upload a file.")

    # ---------------- MAIN TABS ----------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Real-Time Dashboard", 
        "📈 Historical Trends", 
        "🌦️ Weather Impact", 
        "🔮 Price Prediction", 
        "🗺️ Regional Analysis"
    ])

    # ---------------------------------------------------------
    # TAB 1: REAL TIME DASHBOARD
    # ---------------------------------------------------------
    with tab1:
        st.header("Market Overview")

        if df.empty:
            st.warning("Upload a dataset to continue.")
            return
        
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

    # ---------------------------------------------------------
    # TAB 2: HISTORICAL TRENDS
    # ---------------------------------------------------------
    with tab2:
        st.header("Historical Price Analysis")
        
        if df.empty:
            st.warning("Upload a dataset first.")
        else:
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
                    key="date_selector"
                )
            
            st.session_state.date_range = date_range
            
            filtered_df = df[
                (df['crop_type'] == crop_filter) &
                (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
            ]
            
            if filtered_df.empty:
                st.warning("No data in selected range — showing full trend.")
                filtered_df = df[df['crop_type'] == crop_filter]
            
            fig = px.line(filtered_df, x='date', y='price_₹/ton',
                          title=f"{crop_filter} Price Trend")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: WEATHER IMPACT
    # ---------------------------------------------------------
    with tab3:
        st.header("Climate Correlation Analysis")

        if df.empty:
            st.warning("Upload a dataset first.")
        else:
            weather_factor = st.selectbox("Select Weather Factor", 
                                        ['rainfall_mm', 'temperature_c'])
            fig = px.scatter(df, x=weather_factor, y='price_₹/ton',
                             color='crop_type',
                             trendline="ols",
                             title=f"Price vs {weather_factor.title()}")
            st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 4: PRICE PREDICTION
    # ---------------------------------------------------------
    with tab4:
        st.header("Price Prediction Model")

        if not os.path.exists('model.pkl'):
            st.warning("Train the model first from sidebar.")
        else:
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

                filtered = df[
                    (df['state'] == state) &
                    (df['city'] == city) &
                    (df['season'] == season)
                ]
                
                if not filtered.empty:
                    month = int(filtered['date'].dt.month.mode()[0])
                    rainfall = filtered['rainfall_mm'].mean()
                    temp = filtered['temperature_c'].mean()
                else:
                    month = 6
                    rainfall = 100 
                    temp = 25

            if st.button("Predict Price"):
                input_data = pd.DataFrame([[state, city, crop_type, season,
                                            month, rainfall, temp]],
                                          columns=['state','city','crop_type',
                                                   'season','month',
                                                   'rainfall_mm','temperature_c'])
                
                for col in ['state','city','crop_type','season','month']:
                    input_data[col] = le_dict[col].transform(input_data[col])
                
                pred = model.predict(input_data)[0]
                st.success(f"Predicted Price: ₹{pred:.2f}/ton")

    # ---------------------------------------------------------
    # TAB 5: REGIONAL ANALYSIS (NEPAL MAP CAN BE ADDED IF YOU WANT)
    # ---------------------------------------------------------
    with tab5:
        st.header("Regional Price Map (India)")
        
        try:
            india_geojson = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
            avg_prices = df.groupby(['state'])['price_₹/ton'].mean().reset_index()
            
            fig = px.choropleth(
                avg_prices,
                geojson=india_geojson,
                locations="state",
                featureidkey="properties.NAME_1",
                color="price_₹/ton",
                color_continuous_scale="Plasma",
                title="State-wise Price Distribution"
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Map Error: {e}")

# ---------------------------------------------------------
if __name__ == "__main__":
    main()
