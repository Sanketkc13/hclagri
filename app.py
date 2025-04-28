import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor

# Preprocess function
def preprocess_data(df, training=True, le_dict=None, scaler=None):
    df = df.copy()

    # Separate features and target
    X = df.drop(columns=['price_₹/ton'])
    y = df['price_₹/ton']

    # Identify categorical and numerical features
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(exclude=['object']).columns.tolist()

    # Initialize label encoders if training
    if training:
        le_dict = {col: LabelEncoder() for col in cat_features}
        for col in cat_features:
            X[col] = le_dict[col].fit_transform(X[col])
    else:
        for col in cat_features:
            X[col] = le_dict[col].transform(X[col])

    # Initialize scaler if training
    if training:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    if training:
        return X_scaled_df, le_dict, scaler, y
    else:
        return X_scaled_df

# Train model
def train_model():
    df = pd.read_csv('cleaned_dataset.csv')
    df_processed, le_dict, scaler, y = preprocess_data(df, training=True)

    X_train, X_test, y_train, y_test = train_test_split(df_processed, y, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Save model
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'le_dict': le_dict, 'scaler': scaler}, f)

    return model, le_dict, scaler

# Load model
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_dict'], data['scaler']

# Streamlit App
st.title('Agricultural Price Prediction 🚜🌾')

# Train or Load
if st.button('Train Model'):
    model, le_dict, scaler = train_model()
    st.success('Model trained successfully!')
else:
    try:
        model, le_dict, scaler = load_model()
        st.success('Model loaded successfully!')
    except:
        st.warning('Model not found. Please train the model first.')
        st.stop()

# User input
st.header('Enter crop details for prediction:')

state = st.text_input('State')
district = st.text_input('District')
market = st.text_input('Market')
commodity = st.text_input('Commodity')
variety = st.text_input('Variety')
grade = st.text_input('Grade')
min_price = st.number_input('Minimum Price (₹/quintal)', min_value=0)
max_price = st.number_input('Maximum Price (₹/quintal)', min_value=0)
modal_price = st.number_input('Modal Price (₹/quintal)', min_value=0)

if st.button('Predict Price'):
    # Create input DataFrame
    input_data = {
        'state': [state],
        'district': [district],
        'market': [market],
        'commodity': [commodity],
        'variety': [variety],
        'grade': [grade],
        'min_price': [min_price],
        'max_price': [max_price],
        'modal_price': [modal_price]
    }
    input_df = pd.DataFrame(input_data)

    try:
        input_df_processed = preprocess_data(input_df, training=False, le_dict=le_dict, scaler=scaler)
        predicted_price = model.predict(input_df_processed)
        st.success(f'Predicted Price: ₹{predicted_price[0]:,.2f} per ton')
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
