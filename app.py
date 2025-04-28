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
    categorical_columns = ['State', 'District', 'Crop_Type', 'Season']  # Updated column names
    
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le

    # Scaling the numerical columns
    scaler = StandardScaler()
    numerical_columns = ['Rainfall', 'Temperature']

    X = df[categorical_columns + numerical_columns]
    y = df['Price']  # Updated target column name

    if training:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, le_dict, scaler, y

# Function to train the model
def train_model():
    try:
        # Load dataset
        df = pd.read_csv('cleaned_dataset.csv')

        # Debugging: Show actual columns
        st.write('Actual columns in dataset:', df.columns.tolist())

        # Updated column names based on typical CSV conventions
        selected_cols = ['State', 'District', 'Crop_Type', 'Season', 'Rainfall', 'Temperature', 'Price']
        
        # Verify columns exist in dataframe
        missing_cols = [col for col in selected_cols if col not in df.columns]
        if missing_cols:
            raise KeyError(f"Columns missing from dataset: {missing_cols}")

        df = df[selected_cols]

        # Preprocess the data
        df_processed, le_dict, scaler, y = preprocess_data(df, training=True)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(df_processed, y, test_size=0.2, random_state=42)

        # Initialize and train model
        model = XGBRegressor()
        model.fit(X_train, y_train)

        # Save model components
        with open('model.pkl', 'wb') as f:
            pickle.dump({'model': model, 'le_dict': le_dict, 'scaler': scaler}, f)

        return model, le_dict, scaler

    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        raise

# Function to load the model
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            data = pickle.load(f)
        return data['model'], data['le_dict'], data['scaler']
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        raise
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        raise

# Streamlit application
def app():
    try:
        model, le_dict, scaler = load_model()

        # Input fields with proper labels
        st.header("Crop Price Prediction")
        
        # Create two columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            state = st.selectbox("State", le_dict['State'].classes_)
            district = st.selectbox("District", le_dict['District'].classes_)
            crop_type = st.selectbox("Crop Type", le_dict['Crop_Type'].classes_)
            
        with col2:
            season = st.selectbox("Season", le_dict['Season'].classes_)
            rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=100.0)
            temperature = st.number_input("Temperature (°C)", min_value=0.0, value=25.0)

        # Create input dataframe
        input_data = {
            'State': [state],
            'District': [district],
            'Crop_Type': [crop_type],
            'Season': [season],
            'Rainfall': [rainfall],
            'Temperature': [temperature]
        }

        input_df = pd.DataFrame(input_data)

        # Transform categorical features
        for col in ['State', 'District', 'Crop_Type', 'Season']:
            input_df[col] = le_dict[col].transform(input_df[col])

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Make prediction
        if st.button("Predict Price"):
            predicted_price = model.predict(input_scaled)
            st.success(f"Predicted Price: ₹{predicted_price[0]:.2f}/ton")

    except Exception as e:
        st.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    # First train the model
    try:
        train_model()
    except:
        st.error("Could not train model. Check if cleaned_dataset.csv exists with correct columns.")
    
    # Then run the app
    app()