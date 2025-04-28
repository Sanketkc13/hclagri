import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import os

def preprocess_data(df):
    # Extract time features from date
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    # Encode categorical features
    le_dict = {}
    categorical_cols = ['state', 'city', 'crop_type', 'season', 'month']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    return df[categorical_cols], df['price_₹/ton'], le_dict

def train_model():
    try:
        if not os.path.exists('cleaned_dataset.csv'):
            st.error("Dataset file not found!")
            return None, None
        
        df = pd.read_csv('cleaned_dataset.csv')
        X, y, le_dict = preprocess_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = XGBRegressor()
        model.fit(X_train, y_train)
        
        with open('model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'le_dict': le_dict}, f)
        
        st.success("Model trained successfully!")
        return model, le_dict
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return None, None

def load_model():
    try:
        if not os.path.exists('model.pkl'):
            st.error("No trained model found!")
            return None, None
            
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['le_dict']
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return None, None

def main():
    st.title("Agricultural Price Predictor")
    
    # Model management
    st.sidebar.header("Model Operations")
    if st.sidebar.button("Train New Model"):
        model, le_dict = train_model()
    else:
        model, le_dict = load_model()
    
    # Prediction interface
    if model and le_dict:
        st.header("Make Prediction")
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                state = st.selectbox("State", le_dict['state'].classes_)
                city = st.selectbox("City", le_dict['city'].classes_)
                crop_type = st.selectbox("Crop Type", le_dict['crop_type'].classes_)
            
            with col2:
                season = st.selectbox("Season", le_dict['season'].classes_)
                month = st.selectbox("Month", range(1, 13))
            
            if st.form_submit_button("Predict Price"):
                try:
                    # Create input data with correct feature order
                    input_data = pd.DataFrame({
                        'state': [state],
                        'city': [city],
                        'crop_type': [crop_type],
                        'season': [season],
                        'month': [month]
                    }, columns=['state', 'city', 'crop_type', 'season', 'month'])
                    
                    # Transform categorical features
                    for col in ['state', 'city', 'crop_type', 'season', 'month']:
                        input_data[col] = le_dict[col].transform(input_data[col])
                    
                    # Make prediction
                    price = model.predict(input_data)
                    st.success(f"Predicted Price: ₹{price[0]:.2f}/ton")
                except Exception as e:
                    st.error(f"Prediction failed: {str(e)}")
    else:
        st.warning("Please train or load a model first")

if __name__ == "__main__":
    main()