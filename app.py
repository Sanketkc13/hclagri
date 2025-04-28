import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

# Function to preprocess the data
def preprocess_data(df, training=True):
    # Encoding categorical columns
    le_dict = {}
    categorical_columns = ['State', 'District', 'Crop Type', 'Time']

    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Scaling the numerical columns (if any)
    scaler = StandardScaler()
    numerical_columns = ['Rainfall', 'Temperature']

    X = df[categorical_columns + numerical_columns]
    y = df['Price (₹/ton)']

    if training:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, le_dict, scaler, y

# Function to train the model
def train_model():
    # Load dataset
    df = pd.read_csv('cleaned_dataset.csv')

    # Define the columns to be selected (adjust according to actual dataset)
    selected_cols = ['State', 'District', 'Crop Type', 'Time', 'Rainfall', 'Temperature', 'Price (₹/ton)']

    # Selecting the relevant columns from the dataset
    df = df[selected_cols]

    # Preprocess the data
    df_processed, le_dict, scaler, y = preprocess_data(df, training=True)

    # Split the data into training and testing
    X_train, X_test, y_train, y_test = train_test_split(df_processed, y, test_size=0.2, random_state=42)

    # Initialize and train the XGBoost model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Save the model, le_dict, and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'le_dict': le_dict, 'scaler': scaler}, f)

    return model, le_dict, scaler

# Function to load the model
def load_model():
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['le_dict'], data['scaler']

# Streamlit application logic
def app():
    # Load the trained model and preprocessing components
    model, le_dict, scaler = load_model()

    # Input fields for state, district, crop type, and time
    state = st.selectbox("Select State", le_dict['State'].classes_)
    district = st.selectbox("Select District", le_dict['District'].classes_)
    crop_type = st.selectbox("Select Crop Type", le_dict['Crop Type'].classes_)
    time = st.selectbox("Select Time", le_dict['Time'].classes_)

    # Create input data based on selections
    input_data = {
        'State': [state],
        'District': [district],
        'Crop Type': [crop_type],
        'Time': [time],
        'Rainfall': [30],  # Default value for rainfall
        'Temperature': [25]  # Default value for temperature
    }

    input_df = pd.DataFrame(input_data)

    # Apply the same preprocessing as during training
    input_df_encoded = input_df.copy()
    for col in ['State', 'District', 'Crop Type', 'Time']:
        input_df_encoded[col] = le_dict[col].transform(input_df_encoded[col])

    # Scaling the input data (excluding price)
    input_df_processed = scaler.transform(input_df_encoded[['State', 'District', 'Crop Type', 'Time', 'Rainfall', 'Temperature']])

    # Predict the price using the trained model
    predicted_price = model.predict(input_df_processed)

    # Display the predicted price
    st.write(f"Predicted Price (₹/ton): {predicted_price[0]:.2f}")

if __name__ == "__main__":
    app()
