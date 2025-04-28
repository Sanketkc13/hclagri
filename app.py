import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def preprocess_data(df, training=True):
    le_dict = {}
    categorical_columns = ['state', 'city', 'crop_type', 'season']
    
    # Encode categorical features
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Scale numerical features
    numerical_columns = ['temperature_c', 'rainfall_mm']
    scaler = StandardScaler()
    
    X = df[categorical_columns + numerical_columns]
    y = df['price_₹/ton']  # Target variable
    
    if training:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
        
    return X_scaled, le_dict, scaler, y

def train_model():
    # Load and prepare data
    df = pd.read_csv('cleaned_dataset.csv')
    
    # Select relevant columns
    selected_cols = [
        'state', 'city', 'crop_type', 'season',
        'temperature_c', 'rainfall_mm', 'price_₹/ton'
    ]
    
    # Validation check
    missing = [col for col in selected_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    
    df = df[selected_cols]
    
    # Preprocess data
    X_processed, le_dict, scaler, y = preprocess_data(df, training=True)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    # Save artifacts
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'le_dict': le_dict, 'scaler': scaler}, f)
        
    return model, le_dict, scaler

def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_dict'], data['scaler']

def app():
    model, le_dict, scaler = load_model()
    
    st.header("Agricultural Price Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        state = st.selectbox("State", le_dict['state'].classes_)
        city = st.selectbox("City", le_dict['city'].classes_)
        crop_type = st.selectbox("Crop Type", le_dict['crop_type'].classes_)
    
    with col2:
        season = st.selectbox("Season", le_dict['season'].classes_)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)

    # Create input dataframe
    input_data = {
        'state': [state],
        'city': [city],
        'crop_type': [crop_type],
        'season': [season],
        'rainfall_mm': [rainfall],
        'temperature_c': [temperature]
    }
    
    input_df = pd.DataFrame(input_data)
    
    # Transform categorical features
    for col in ['state', 'city', 'crop_type', 'season']:
        input_df[col] = le_dict[col].transform(input_df[col])
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    # Prediction
    if st.button("Predict Price"):
        price = model.predict(input_scaled)[0]
        st.success(f"Predicted Price: ₹{price:.2f}/ton")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        st.error(f"Initialization failed: {str(e)}")
    app()