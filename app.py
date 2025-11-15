import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os

st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")

# Load dataset
@st.cache_data
def load_data(file_path='cleaned_dataset.csv'):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df.dropna(subset=['date'])
    return pd.DataFrame()

# Train selected model
def train_model(df, model_choice):
    try:
        df = df.copy()
        df['month'] = df['date'].dt.month

        features = ['state', 'city', 'crop_type', 'season', 'month',
                    'rainfall_mm', 'temperature_c',
                    'supply_volume_tons', 'demand_volume_tons']

        le_dict = {}
        cat_cols = ['state', 'city', 'crop_type', 'season']

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le

        X = df[features]
        y = df['price_₹/ton']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if model_choice == "XGBoost":
            model = XGBRegressor()
        elif model_choice == "Random Forest":
            model = RandomForestRegressor()
        else:
            model = LinearRegression()

        model.fit(X_train, y_train)

        with open("model.pkl", "wb") as f:
            pickle.dump({"model": model, "le_dict": le_dict, "features": features}, f)

        return True

    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False

def main():
    st.title("🌾 Agricultural Market Price Analyzer")

    # Sidebar Upload & Train
    st.sidebar.header("Data & Model Controls")
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    df = load_data(uploaded_file if uploaded_file else "cleaned_dataset.csv")

    if not df.empty:
        st.sidebar.success("Data Loaded Successfully!")

        model_choice = st.sidebar.selectbox(
            "Choose Prediction Model",
            ["XGBoost", "Random Forest", "Linear Regression"]
        )

        if st.sidebar.button("Train Model"):
            with st.spinner("Training Model..."):
                if train_model(df, model_choice):
                    st.sidebar.success(f"{model_choice} Model Trained Successfully")

    # Main Tabs
    tabs = st.tabs(["📊 Dashboard", "📈 Trends", "🌦 Weather Impact", "🔮 Prediction"])

    with tabs[0]:  
        st.header("Market Overview")
        st.dataframe(df.head(), use_container_width=True)

        if 'price_₹/ton' in df.columns:
            fig = px.histogram(df, x="price_₹/ton", title="Price Distribution")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.header("Historical Trends")
        crop = st.selectbox("Select Crop", df["crop_type"].unique())
        fig = px.line(df[df["crop_type"] == crop], x="date", y="price_₹/ton",
                      title=f"Historical Price Trend - {crop}")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.header("Weather vs Price Correlation")
        fig = px.scatter(df, x="rainfall_mm", y="price_₹/ton", color="crop_type",
                         title="Price vs Rainfall Impact")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.header("Predict Commodity Price")

        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                model_data = pickle.load(f)

            model, le_dict, features = model_data["model"], model_data["le_dict"], model_data["features"]

            state = st.selectbox("Select State", le_dict["state"].classes_)
            city = st.selectbox("Select City", le_dict["city"].classes_)
            crop = st.selectbox("Select Crop Type", le_dict["crop_type"].classes_)
            season = st.selectbox("Select Season", le_dict["season"].classes_)

            filtered = df[(df["state"] == state) &
                          (df["city"] == city) &
                          (df["crop_type"] == crop) &
                          (df["season"] == season)]

            if not filtered.empty:
                avg = filtered.mean(numeric_only=True)
                month = int(filtered["date"].dt.month.mode()[0])
            else:
                avg = {"rainfall_mm": 100, "temperature_c": 25,
                       "supply_volume_tons": 500, "demand_volume_tons": 450}
                month = 6

            if st.button("Predict Price"):
                input_data = pd.DataFrame([[
                    le_dict["state"].transform([state])[0],
                    le_dict["city"].transform([city])[0],
                    le_dict["crop_type"].transform([crop])[0],
                    le_dict["season"].transform([season])[0],
                    month,
                    avg.get("rainfall_mm", 100),
                    avg.get("temperature_c", 25),
                    avg.get("supply_volume_tons", 500),
                    avg.get("demand_volume_tons", 450)
                ]], columns=features)

                price = model.predict(input_data)[0]
                st.success(f"Predicted Price: ₹{price:.2f} / ton")
        else:
            st.warning("Please train a model first.")

if __name__ == "__main__":
    main()
