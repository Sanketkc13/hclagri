import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import json
import os

# -------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
@st.cache_data
def load_data(file_path="cleaned_dataset.csv"):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return pd.DataFrame()

df = load_data()

# -------------------------------------------------------
# TRAINING FUNCTION
# -------------------------------------------------------
def train_model(df, model_type):
    try:
        df = df.copy()
        df["month"] = df["date"].dt.month

        # Label Encoding
        le_dict = {}
        cat_cols = ["state", "city", "crop_type", "season", "month"]

        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            le_dict[col] = le

        X = df[cat_cols + ["rainfall_mm", "temperature_c"]]
        y = df["price_₹/ton"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Select Model
        if model_type == "XGBoost":
            model = XGBRegressor()
            model_path = "models/xgb_model.pkl"

        elif model_type == "Random Forest":
            model = RandomForestRegressor()
            model_path = "models/rf_model.pkl"

        elif model_type == "Linear Regression":
            model = LinearRegression()
            model_path = "models/lr_model.pkl"

        # Train
        model.fit(X_train, y_train)

        # Save
        os.makedirs("models", exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "le_dict": le_dict}, f)

        return True

    except Exception as e:
        st.error(f"Training Error: {str(e)}")
        return False


# -------------------------------------------------------
# MAIN UI
# -------------------------------------------------------
st.title("🌾 Agricultural Market Price Analyzer")


# -------------------------------------------------------
# SIDEBAR – Select & Train Model
# -------------------------------------------------------
st.sidebar.header("Model Training")

model_choice = st.sidebar.selectbox(
    "Choose Model to Train",
    ["XGBoost", "Random Forest", "Linear Regression"]
)

if st.sidebar.button("Train Selected Model"):
    with st.spinner(f"Training {model_choice}..."):
        if train_model(df, model_choice):
            st.sidebar.success(f"{model_choice} trained successfully!")


# -------------------------------------------------------
# TABS
# -------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "📈 Trends",
    "🌦️ Weather Impact",
    "🔮 Prediction",
    "🗺️ Nepal Price Map"
])

# -------------------------------------------------------
# TAB 1 – DASHBOARD
# -------------------------------------------------------
with tab1:
    st.header("Real-Time Dashboard")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Avg Price", f"₹{df['price_₹/ton'].mean():.2f}")

    with col2:
        st.metric("Active Provinces", df["state"].nunique())

    with col3:
        st.metric("Total Records", len(df))

    st.subheader("Recent Data")
    st.dataframe(df.tail(10), use_container_width=True)


# -------------------------------------------------------
# TAB 2 – PRICE TRENDS
# -------------------------------------------------------
with tab2:
    st.header("Historical Price Trends")

    crop = st.selectbox("Select Crop", df["crop_type"].unique())

    filtered = df[df["crop_type"] == crop]

    fig = px.line(filtered, x="date", y="price_₹/ton",
                  title=f"{crop} Price Trend")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------
# TAB 3 – WEATHER IMPACT
# -------------------------------------------------------
with tab3:
    st.header("Weather vs Price")

    factor = st.selectbox("Select Weather Factor",
                          ["rainfall_mm", "temperature_c"])

    fig = px.scatter(df, x=factor, y="price_₹/ton",
                     color="crop_type",
                     title=f"Price vs {factor}")
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------
# TAB 4 – PRICE PREDICTION
# -------------------------------------------------------
with tab4:
    st.header("Predict Future Price")

    pred_model = st.selectbox(
        "Choose Model",
        ["XGBoost", "Random Forest", "Linear Regression"]
    )

    model_path = f"models/{pred_model.lower().replace(' ', '_')}_model.pkl"

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        model = model_data["model"]
        le_dict = model_data["le_dict"]

        col1, col2 = st.columns(2)

        with col1:
            state = st.selectbox("State", le_dict["state"].classes_)
            city = st.selectbox("City", le_dict["city"].classes_)
            crop = st.selectbox("Crop", le_dict["crop_type"].classes_)

        with col2:
            season = st.selectbox("Season", le_dict["season"].classes_)

            subset = df[
                (df["state"] == state) &
                (df["city"] == city) &
                (df["season"] == season)
            ]

            if len(subset) > 0:
                month = subset["date"].dt.month.mode()[0]
                rain = subset["rainfall_mm"].mean()
                temp = subset["temperature_c"].mean()
            else:
                month = 6
                rain = 100
                temp = 25

        if st.button("Predict Price"):
            input_df = pd.DataFrame([[
                state, city, crop, season, month, rain, temp
            ]], columns=[
                "state", "city", "crop_type", "season",
                "month", "rainfall_mm", "temperature_c"
            ])

            for col in ["state", "city", "crop_type", "season", "month"]:
                input_df[col] = le_dict[col].transform(input_df[col])

            pred = model.predict(input_df)[0]

            st.success(f"Predicted Price: ₹{pred:.2f} / ton")

    else:
        st.warning("Train this model first.")


# -------------------------------------------------------
# TAB 5 – NEPAL MAP
# -------------------------------------------------------
with tab5:
    st.header("Average Price by Province (Nepal)")

    try:
        geo_path = "assets/nepal_provinces.geojson"

        with open(geo_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)

        avg_df = df.groupby("state")["price_₹/ton"].mean().reset_index()
        avg_df = avg_df.rename(columns={"price_₹/ton": "avg_price"})

        fig = px.choropleth(
            avg_df,
            geojson=geojson,
            locations="state",
            featureidkey="properties.PROVINCE",
            color="avg_price",
            color_continuous_scale="YlOrBr",
            title="Nepal Province Price Map"
        )

        fig.update_geos(fitbounds="locations", visible=False)
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Map Load Error: {str(e)}")
