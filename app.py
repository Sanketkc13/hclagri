import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np

st.set_page_config(page_title="Agricultural Price Analyzer", layout="wide")

# --------------------------------------------------------------------
# 📌 LOAD DATA
# --------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data.csv")

data = load_data()

# Ensure required columns exist
required_cols = {"date","state","city","crop_type","season","temperature","rainfall","price"}
if not required_cols.issubset(set(data.columns)):
    st.error(f"❌ Dataset missing required columns.\n\nRequired: {required_cols}")
    st.stop()

# Convert date
data["date"] = pd.to_datetime(data["date"])

# --------------------------------------------------------------------
# 📌 FIX PROVINCE NAME MAPPING (standardize)
# --------------------------------------------------------------------
province_mapping = {
    "Koshi Province": "Koshi",
    "Madhesh Province": "Madhesh",
    "Bagmati Province": "Bagmati",
    "Gandaki Province": "Gandaki",
    "Lumbini Province": "Lumbini",
    "Karnali Province": "Karnali",
    "Sudurpashchim Province": "Sudurpashchim"
}

data["province_clean"] = data["state"].map(province_mapping)

# --------------------------------------------------------------------
# 📌 LOAD GEOJSON
# --------------------------------------------------------------------
geojson_path = "assets/nepal_provinces.geojson"
geojson_data = None
geo_loaded = False

if os.path.exists(geojson_path):
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        geo_loaded = True
    except Exception as e:
        st.warning(f"⚠ Could not load GeoJSON: {e}")

# --------------------------------------------------------------------
# 📌 MAP — fallback to bar chart when low match rate
# --------------------------------------------------------------------
st.subheader("Average Price by Province (Nepal)")

avg_df = data.groupby("province_clean")["price"].mean().reset_index()

# Count matches vs GeoJSON names
if geo_loaded:
    geo_provinces = [f["properties"]["name"] for f in geojson_data["features"]]
    matches = avg_df["province_clean"].isin(geo_provinces).sum()
    match_rate = matches / len(avg_df)

    if match_rate < 0.70:
        st.warning("Low match rate between dataset province names and GeoJSON. Showing bar chart instead.")

        fig = px.bar(
            avg_df,
            x="province_clean",
            y="price",
            title="Average Price by Province"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.choropleth(
            avg_df,
            geojson=geojson_data,
            featureidkey="properties.name",
            locations="province_clean",
            color="price",
            color_continuous_scale="OrRd",
            title="Average Price by Province",
        )
        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)
else:
    st.error("❌ GeoJSON file not found. Showing bar chart instead.")
    fig = px.bar(avg_df, x="province_clean", y="price")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------
# 📌 MODEL TRAINING SECTION
# --------------------------------------------------------------------
st.header("Train Prediction Models")

model_choice = st.selectbox(
    "Select model to train:",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

# Encode all categorical columns
df = data.copy()
label_cols = ["state", "city", "crop_type", "season"]

le_dict = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Features and target
X = df[["state", "city", "crop_type", "season", "temperature", "rainfall"]]
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model selection
def train_model(model_choice):
    if model_choice == "Linear Regression":
        return LinearRegression()
    if model_choice == "Random Forest":
        return RandomForestRegressor(n_estimators=200)
    if model_choice == "XGBoost":
        return XGBRegressor(n_estimators=300, learning_rate=0.1)

if st.button("Train Selected Model"):
    model = train_model(model_choice)
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    st.success(f"🎉 {model_choice} Trained Successfully!")
    st.write(f"📌 R² Score: **{score:.3f}**")

    st.session_state["trained_model"] = model
    st.session_state["label_encoders"] = le_dict

# --------------------------------------------------------------------
# 📌 PREDICTION
# --------------------------------------------------------------------
st.header("Make a Prediction")

if "trained_model" not in st.session_state:
    st.info("Train a model first.")
else:
    with st.form("predict_form"):
        p_state = st.selectbox("Province", sorted(data["state"].unique()))
        p_city = st.selectbox("City", sorted(data["city"].unique()))
        p_crop = st.selectbox("Crop Type", sorted(data["crop_type"].unique()))
        p_season = st.selectbox("Season", sorted(data["season"].unique()))
        p_temp = st.number_input("Temperature", value=25.0)
        p_rain = st.number_input("Rainfall", value=10.0)
        submit = st.form_submit_button("Predict Price")

    if submit:
        enc = st.session_state["label_encoders"]

        inp = np.array([[
            enc["state"].transform([p_state])[0],
            enc["city"].transform([p_city])[0],
            enc["crop_type"].transform([p_crop])[0],
            enc["season"].transform([p_season])[0],
            p_temp,
            p_rain
        ]])

        model = st.session_state["trained_model"]
        pred = model.predict(inp)[0]

        st.success(f"🌾 Predicted Price: **{pred:.2f}**")

