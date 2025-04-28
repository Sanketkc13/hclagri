import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def preprocess_data(df):
    # Create time features from date
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Encode categorical features
    le_dict = {}
    categorical_cols = ['state', 'city', 'crop_type', 'season', 'month']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    return df[['state', 'city', 'crop_type', 'season', 'month']], df['price_₹/ton'], le_dict

def train_model():
    # Load and prepare data
    df = pd.read_csv('cleaned_dataset.csv')
    
    # Process data
    X, y, le_dict = preprocess_data(df)
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = XGBRegressor()
    model.fit(X_train, y_train)
    
    # Save artifacts
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'le_dict': le_dict}, f)
    
    return model, le_dict

def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_dict']

def app():
    model, le_dict = load_model()
    
    st.header("Crop Price Predictor")
    col1, col2 = st.columns(2)
    
    with col1:
        state = st.selectbox("State", le_dict['state'].classes_)
        city = st.selectbox("City", le_dict['city'].classes_)
        crop_type = st.selectbox("Crop Type", le_dict['crop_type'].classes_)
    
    with col2:
        season = st.selectbox("Season", le_dict['season'].classes_)
        month = st.selectbox("Month", options=range(1, 13))
    
    # Create input array
    input_data = pd.DataFrame({
        'state': [state],
        'city': [city],
        'crop_type': [crop_type],
        'season': [season],
        'month': [month]
    })
    
    # Transform categorical features
    for col in ['state', 'city', 'crop_type', 'season', 'month']:
        input_data[col] = le_dict[col].transform(input_data[col])
    
    # Make prediction
    if st.button("Predict Price"):
        price = model.predict(input_data)
        st.success(f"Predicted Price: ₹{price[0]:.2f}/ton")

if __name__ == "__main__":
    try:
        train_model()
    except Exception as e:
        st.error(f"Initialization error: {str(e)}")
    app()