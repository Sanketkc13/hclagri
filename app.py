import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import pickle

# Title
st.title("Agriculture Price Prediction App")

# Load your cleaned dataset directly
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_dataset.csv')
    return df

# Smart fill values 
def smart_fill_values(crop, month, state, city):
    return {
        'temperature_c': 18 if month in [12,1,2] else 32 if month in [3,4,5] else 27 if month in [6,7,8,9] else 24,
        'rainfall_mm': 300 if month in [6,7,8,9] else 50,
        'supply_volume_tons': 500,
        'demand_volume_tons': 520,
        'transportation_cost_₹/ton': 400,
        'fertilizer_usage_kg/hectare': 110,
        'pest_infestation_0-1': 0.3,
        'market_competition_0-1': 0.5,
        'crop_type': crop,
        'month': month,
        'state': state,
        'city': city
    }

# Modified preprocessing
@st.cache_data
def preprocess_data(df, le_dict=None, scaler=None, fit=True):
    df = df.copy()
    
    if fit:
        le_dict = {}
        cat_cols = df.select_dtypes(include='object').columns
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le
        
        scaler = StandardScaler()
        df[df.columns] = scaler.fit_transform(df[df.columns])
        return df, le_dict, scaler
    else:
        cat_cols = le_dict.keys()
        for col in cat_cols:
            df[col] = le_dict[col].transform(df[col])
        df[df.columns] = scaler.transform(df[df.columns])
        return df

# Save model with preprocessing artifacts
def save_model(model, le_dict, scaler):
    with open('xgboost_agricultural_price_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'le_dict': le_dict, 'scaler': scaler}, f)

# Load model with preprocessing artifacts
def load_model():
    with open('xgboost_agricultural_price_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_dict'], data['scaler']

# Model training
def train_model(df, le_dict, scaler):
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
    save_model(best_model, le_dict, scaler)

    y_pred = best_model.predict(X_test)
    return best_model, (
        mean_absolute_error(y_test, y_pred),
        mean_squared_error(y_test, y_pred),
        r2_score(y_test, y_pred)
    )

# Main app
df = load_data()
st.write("### Loaded Cleaned Dataset")
st.dataframe(df.head())

# Preprocess data
df_processed, le_dict, scaler = preprocess_data(df, fit=True)

if st.button('Train Model'):
    with st.spinner('Training...'):
        model, (mae, mse, r2) = train_model(df_processed, le_dict, scaler)
        
        st.success("Model trained!")
        st.metric("MAE", f"₹{mae:.2f}")
        st.metric("MSE", f"₹{mse:.2f}")
        st.metric("R²", f"{r2:.2f}")

    # Visualization
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df_processed.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.write("### Correlation Matrix")
    st.pyplot(fig)

# Prediction section
if st.button('Predict Price'):
    model, le_dict, scaler = load_model()
    
    # Get user input
    crop = st.selectbox('Crop', df['crop_type'].unique())
    month = st.selectbox('Month', range(1,13))
    state = st.selectbox('State', df['state'].unique())
    city = st.selectbox('City', df['city'].unique())

    # Create input DataFrame
    input_df = pd.DataFrame([smart_fill_values(crop, month, state, city)])
    
    # Preprocess input
    input_processed = preprocess_data(input_df, le_dict, scaler, fit=False)
    input_processed = input_processed[df_processed.drop('price_₹/ton', axis=1).columns]

    # Predict
    price = model.predict(input_processed)[0]
    st.success(f"Predicted price for {crop}: ₹{price:.2f}/ton")
