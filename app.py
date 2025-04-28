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

    if training:
        X = df.drop(columns=['price_₹/ton'])
        y = df['price_₹/ton']
    else:
        X = df

    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    num_features = X.select_dtypes(exclude=['object']).columns.tolist()

    if training:
        le_dict = {col: LabelEncoder() for col in cat_features}
        for col in cat_features:
            X[col] = le_dict[col].fit_transform(X[col])
    else:
        for col in cat_features:
            X[col] = le_dict[col].transform(X[col])

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

    selected_cols = ['state', 'district', 'crop_type', 'time', 'rainfall', 'temperature', 'price_₹/ton']
    df = df[selected_cols]

    df_processed, le_dict, scaler, y = preprocess_data(df, training=True)

    X_train, X_test, y_train, y_test = train_test_split(df_processed, y, test_size=0.2, random_state=42)

    model = XGBRegressor()
    model.fit(X_train, y_train)

    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'le_dict': le_dict, 'scaler': scaler}, f)

    return model, le_dict, scaler

# Load model
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_dict'], data['scaler']

# Streamlit App
st.title('🌾 Agri Price Predictor 🌾')

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

# Load dataset to get dropdown options and rainfall/temp values
df = pd.read_csv('cleaned_dataset.csv')
selected_cols = ['state', 'district', 'crop_type', 'time', 'rainfall', 'temperature', 'price_₹/ton']
df = df[selected_cols]

st.header('Enter details for prediction:')

# Dropdown menus
state = st.selectbox('Select State', sorted(df['state'].unique()))
district = st.selectbox('Select District', sorted(df[df['state'] == state]['district'].unique()))
crop_type = st.selectbox('Select Crop Type', sorted(df['crop_type'].unique()))
time = st.selectbox('Select Time', sorted(df['time'].unique()))

# Prediction
if st.button('Predict Price'):
    # Find matching row in dataset
    match = df[
        (df['state'] == state) &
        (df['district'] == district) &
        (df['crop_type'] == crop_type) &
        (df['time'] == time)
    ]

    if match.empty:
        st.error('No matching data found. Please try different inputs.')
    else:
        # Use rainfall and temperature from the dataset
        rainfall = match.iloc[0]['rainfall']
        temperature = match.iloc[0]['temperature']

        input_data = {
            'state': [state],
            'district': [district],
            'crop_type': [crop_type],
            'time': [time],
            'rainfall': [rainfall],
            'temperature': [temperature]
        }
        input_df = pd.DataFrame(input_data)

        try:
            input_df_processed = preprocess_data(input_df, training=False, le_dict=le_dict, scaler=scaler)
            predicted_price = model.predict(input_df_processed)
            st.success(f'Predicted Price: ₹{predicted_price[0]:,.2f} per ton')
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
