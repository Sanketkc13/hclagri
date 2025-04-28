import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Title
st.title("Agriculture Price Prediction App")

# Load your cleaned dataset directly
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_dataset.csv')  # <-- Here we load directly
    return df

# Smart fill values (optional for later)
def smart_fill_values(crop, month, state, city):
    smart_data = {}
    if month in [12, 1, 2]:
        smart_data['temperature_c'] = 18
    elif month in [3, 4, 5]:
        smart_data['temperature_c'] = 32
    elif month in [6, 7, 8, 9]:
        smart_data['temperature_c'] = 27
    else:
        smart_data['temperature_c'] = 24

    if month in [6, 7, 8, 9]: 
        smart_data['rainfall_mm'] = 300
    else:
        smart_data['rainfall_mm'] = 50

    smart_data['supply_volume_tons'] = 500
    smart_data['demand_volume_tons'] = 520
    smart_data['transportation_cost_₹/ton'] = 400
    smart_data['fertilizer_usage_kg/hectare'] = 110
    smart_data['pest_infestation_0-1'] = 0.3
    smart_data['market_competition_0-1'] = 0.5

    return smart_data

# Preprocessing
@st.cache_data
def preprocess_data(df):
    df = df.dropna()

    le = LabelEncoder()
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    return df

# Model training
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

df_processed = preprocess_data(df)

if st.button('Train Model'):
    with st.spinner('Training in progress...'):
        model, mae, mse, r2 = train_model(df_processed)

        st.success("Model training completed!")
        st.write(f"**Mean Absolute Error (MAE):** {mae:.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

    st.write("### Correlation Matrix Heatmap")
    plt.figure(figsize=(10,8))
    sns.heatmap(df_processed.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    st.pyplot(plt)
