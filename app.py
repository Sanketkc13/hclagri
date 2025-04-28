import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

# 1. Preprocessing function
def preprocess_data(df, le_dict=None, scaler=None, training=True):
    df = df.copy()
    categorical_cols = ['state', 'city', 'crop_type', 'season']

    if training:
        le_dict = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        scaler = StandardScaler()
        X = df.drop(['price_₹/ton'], axis=1)
        X_scaled = scaler.fit_transform(X)
        df[X.columns] = X_scaled
        return df, le_dict, scaler
    else:
        for col in categorical_cols:
            le = le_dict[col]
            df[col] = le.transform(df[col])
        df[df.columns] = scaler.transform(df)
        return df

# 2. Save model
def save_model(model, le_dict, scaler):
    data = {
        'model': model,
        'le_dict': le_dict,
        'scaler': scaler
    }
    joblib.dump(data, 'model.pkl')

# 3. Load model
def load_model():
    data = joblib.load('model.pkl')
    return data['model'], data['le_dict'], data['scaler']

# 4. Train or Load
def train_model():
    if not os.path.exists('model.pkl'):
        st.write("Training model...")

        # Load dataset
        df = pd.read_csv('your_dataset.csv')

        # Preprocess
        df_processed, le_dict, scaler = preprocess_data(df, training=True)

        # Train model
        X = df_processed.drop(['price_₹/ton'], axis=1)
        y = df_processed['price_₹/ton']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(n_estimators=200, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        # Save model
        save_model(model, le_dict, scaler)

        # Evaluate
        y_pred = model.predict(X_test)
        st.write("Model trained. Test R2 Score:", r2_score(y_test, y_pred))
    else:
        st.write("Model found. Loading...")
    return load_model()

# 5. Main Streamlit App
st.title('Agriculture Price Prediction')

# Load model
model, le_dict, scaler = train_model()

# Load dataset for dropdowns
df = pd.read_csv('your_dataset.csv')

# UI for user input
state = st.selectbox('Select State', df['state'].unique())
city = st.selectbox('Select City', df[df['state'] == state]['city'].unique())
crop_type = st.selectbox('Select Crop Type', df['crop_type'].unique())
season = st.selectbox('Select Season', df['season'].unique())

temperature_c = st.number_input('Temperature (°C)', value=30.0)
rainfall_mm = st.number_input('Rainfall (mm)', value=100.0)
supply_volume_tons = st.number_input('Supply Volume (tons)', value=2000.0)
demand_volume_tons = st.number_input('Demand Volume (tons)', value=1500.0)
transportation_cost = st.number_input('Transportation Cost (₹/ton)', value=300.0)
fertilizer_usage = st.number_input('Fertilizer Usage (kg/hectare)', value=150.0)
pest_infestation = st.slider('Pest Infestation (0-1)', 0.0, 1.0, 0.2)
market_competition = st.slider('Market Competition (0-1)', 0.0, 1.0, 0.5)

# Prepare input
input_dict = {
    'state': [state],
    'city': [city],
    'crop_type': [crop_type],
    'season': [season],
    'temperature_c': [temperature_c],
    'rainfall_mm': [rainfall_mm],
    'supply_volume_tons': [supply_volume_tons],
    'demand_volume_tons': [demand_volume_tons],
    'transportation_cost_₹/ton': [transportation_cost],
    'fertilizer_usage_kg/hectare': [fertilizer_usage],
    'pest_infestation_0-1': [pest_infestation],
    'market_competition_0-1': [market_competition]
}

input_df = pd.DataFrame(input_dict)

# Preprocess input
input_df_processed = preprocess_data(input_df, le_dict, scaler, training=False)

# Predict
if st.button('Predict Price'):
    predicted_price = model.predict(input_df_processed)
    st.success(f"Predicted Price: ₹{predicted_price[0]:.2f} per ton")
