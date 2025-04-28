import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Functions
def save_model(model, le_dict, scaler):
    with open('saved_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'le_dict': le_dict, 'scaler': scaler}, f)

def load_model():
    with open('saved_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_dict'], data['scaler']

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
        X_scaled = scaler.transform(df)
        df[df.columns] = X_scaled
        return df

def train_model(df):
    X = df.drop(['price_₹/ton'], axis=1)
    y = df['price_₹/ton']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(objective='reg:squarederror', n_jobs=-1)

    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 5]
    }

    random_search = RandomizedSearchCV(
        model, 
        param_distributions=param_grid,
        n_iter=5,  # Try only 5 random combinations
        cv=3,
        scoring='r2',
        verbose=1,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return best_model, mae, mse, r2

# Load dataset
df = pd.read_csv('your_dataset.csv')  # <- Update your file path here

# Streamlit App
st.title('Agriculture Price Prediction')

# Training
if st.button('Train Model'):
    st.write('Training started...')
    df_processed, le_dict, scaler = preprocess_data(df, training=True)
    model, mae, mse, r2 = train_model(df_processed)
    save_model(model, le_dict, scaler)
    st.success('Model trained and saved successfully!')
    st.metric('MAE', f'{mae:.2f}')
    st.metric('MSE', f'{mse:.2f}')
    st.metric('R² Score', f'{r2:.2f}')

# Load model if exists
if os.path.exists('saved_model.pkl'):
    model, le_dict, scaler = load_model()
else:
    model = None

# Prediction
st.header('Predict Price')
if model:
    states = df['state'].unique()
    cities = df['city'].unique()
    crops = df['crop_type'].unique()
    seasons = df['season'].unique()

    state = st.selectbox('State', states)
    city = st.selectbox('City', cities)
    crop_type = st.selectbox('Crop Type', crops)
    season = st.selectbox('Season', seasons)
    temperature_c = st.number_input('Temperature (°C)')
    rainfall_mm = st.number_input('Rainfall (mm)')
    supply_volume_tons = st.number_input('Supply Volume (tons)')
    demand_volume_tons = st.number_input('Demand Volume (tons)')
    transportation_cost = st.number_input('Transportation Cost (₹/ton)')
    fertilizer_usage = st.number_input('Fertilizer Usage (kg/hectare)')
    pest_infestation = st.slider('Pest Infestation (0-1)', 0.0, 1.0)
    market_competition = st.slider('Market Competition (0-1)', 0.0, 1.0)

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
        'market_competition_0-1': [market_competition],
    }

    input_df = pd.DataFrame(input_dict)

    if st.button('Predict'):
        input_df_processed = preprocess_data(input_df, le_dict=le_dict, scaler=scaler, training=False)
        predicted_price = model.predict(input_df_processed)
        st.success(f'Predicted Price (₹/ton): {predicted_price[0]:.2f}')
else:
    st.warning('Train the model first!')

