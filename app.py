import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Title
st.title("Agriculture Price Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_dataset.csv')
    return df

# Save model, encoders, scaler
def save_model(model, le_dict, scaler):
    joblib.dump({'model': model, 'le_dict': le_dict, 'scaler': scaler}, 'saved_model.pkl')

# Load model, encoders, scaler
def load_saved_model():
    data = joblib.load('saved_model.pkl')
    return data['model'], data['le_dict'], data['scaler']

# Preprocessing during training
def preprocess_data(df):
    df = df.dropna()
    
    le_dict = {}
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le  # store each column's LabelEncoder separately

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df, le_dict, scaler

# Preprocessing during prediction
def preprocess_input(input_df, le_dict, scaler):
    input_df_encoded = input_df.copy()
    for col, le in le_dict.items():
        if col in input_df_encoded.columns:
            input_df_encoded[col] = le.transform(input_df_encoded[col])

    input_df_scaled = scaler.transform(input_df_encoded)
    return input_df_scaled

# Train model
def train_model(df):
    X = df.drop(['price_₹/ton'], axis=1)
    y = df['price_₹/ton']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective='reg:squarederror')

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.6, 0.8, 1.0]
    }

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='r2', verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return best_model, mae, mse, r2

# Main app
df = load_data()
st.write("### Loaded Cleaned Dataset")
st.dataframe(df.head())

# Check if model is already saved
model_exists = os.path.exists('saved_model.pkl')

if not model_exists:
    st.warning("Model not found! Please train the model first.")

# Train model button
if st.button('Train Model'):
    with st.spinner('Training in progress...'):
        df_processed, le_dict, scaler = preprocess_data(df)
        model, mae, mse, r2 = train_model(df_processed)
        save_model(model, le_dict, scaler)

        st.success("Model training completed and saved!")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

    st.write("### Correlation Matrix Heatmap")
    plt.figure(figsize=(10,8))
    sns.heatmap(df_processed.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)

# Prediction section
st.write("---")
st.header("📈 Predict Crop Price")

if model_exists:
    model, le_dict, scaler = load_saved_model()

    # Now ask for inputs
    st.subheader("Enter details to predict price:")
    
    # Important: Manually enter the columns you expect based on your dataset
    state = st.text_input('State', 'Maharashtra')
    city = st.text_input('City', 'Mumbai')
    crop_type = st.text_input('Crop Type', 'Wheat')
    season = st.text_input('Season', 'Kharif')
    temperature_c = st.number_input('Temperature (°C)', 0.0, 50.0, 25.0)
    rainfall_mm = st.number_input('Rainfall (mm)', 0.0, 500.0, 100.0)
    supply_volume_tons = st.number_input('Supply Volume (tons)', 0.0, 10000.0, 500.0)
    demand_volume_tons = st.number_input('Demand Volume (tons)', 0.0, 10000.0, 550.0)
    transportation_cost = st.number_input('Transportation Cost (₹/ton)', 0.0, 2000.0, 400.0)
    fertilizer_usage = st.number_input('Fertilizer Usage (kg/hectare)', 0.0, 500.0, 120.0)
    pest_infestation = st.slider('Pest Infestation (0-1)', 0.0, 1.0, 0.2)
    market_competition = st.slider('Market Competition (0-1)', 0.0, 1.0, 0.5)

    if st.button('Predict Price'):
        input_data = {
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
        input_df = pd.DataFrame(input_data)

        input_df_scaled = preprocess_input(input_df, le_dict, scaler)
        predicted_price = model.predict(input_df_scaled)

        st.success(f"Predicted Price: ₹{predicted_price[0]:.2f} per ton")
