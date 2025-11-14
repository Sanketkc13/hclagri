import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import datetime
import requests
import os

# ------------------------------------
# STREAMLIT CONFIG
# ------------------------------------
st.set_page_config(page_title="Agricultural Market Price Analyzer", layout="wide")


# ------------------------------------
# LOAD DATA
# ------------------------------------
@st.cache_data
def load_data(file_path='cleaned_dataset.csv'):
    try:
        if isinstance(file_path, str):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path)

        df['date'] = pd.to_datetime(df['date'])
        return df

    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


# ------------------------------------
# TRAIN MODEL
# ------------------------------------
def train_model(df):
    try:
        df = df.copy()
        df["month"] = df["date"].dt.month

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

        model = XGBRegressor()
        model.fit(X_train, y_train)

        with open("model.pkl", "wb") as f:
            pickle.dump({"model": model, "le_dict": le_dict}, f)

        return True

    except Exception as e:
        st.error(f"Training Error: {e}")
        return False


# ------------------------------------
# MAIN APP
# ------------------------------------
def main():
    st.title("🌾 Agricultural Market Price Analyzer")

    # Sidebar Upload
    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=["csv"])

    df = load_data(uploaded_file if uploaded_file else "cleaned_dataset.csv")

    if df.empty:
        st.error("No dataset available.")
        return

    st.sidebar.success("Dataset Loaded Successfully!")

    # Train model button
    if st.sidebar.button("Retrain Prediction Model"):
        with st.spinner("Training new model..."):
            if train_model(df):
                st.sidebar.success("Model Retrained Successfully!")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📊 Real-Time Dashboard",
            "📈 Historical Trends",
            "🌦️ Weather Impact",
            "🔮 Price Prediction",
            "🗺️ Regional Analysis",
        ]
    )

    # ------------------------------------
    # TAB 1 - DASHBOARD
    # ------------------------------------
    with tab1:
        st.header("Market Overview")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Current Avg Price",
                f"₹{df['price_₹/ton'].mean():.2f}/ton",
                "+5.2% vs last month",
            )

        with col2:
            st.metric(
                "Supply-Demand Ratio",
                f"{(df['supply_volume_tons'] / df['demand_volume_tons']).mean():.2f}",
                "Market Balance",
            )

        with col3:
            st.metric("Active Provinces", df["state"].nunique(), "Regions Being Tracked")

        st.subheader("Latest Entries")
        st.dataframe(df.sort_values("date", ascending=False).head(10), use_container_width=True)

    # ------------------------------------
    # TAB 2 - HISTORICAL TRENDS
    # ------------------------------------
    with tab2:
        st.header("Historical Price Trend")

        crop = st.selectbox("Select Crop", df["crop_type"].unique())

        date_range = st.date_input(
            "Select Date Range",
            (df["date"].min().date(), df["date"].max().date()),
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
        )

        filtered = df[
            (df["crop_type"] == crop)
            & (df["date"].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))
        ]

        if filtered.empty:
            st.warning("No data for this date range. Showing full crop history.")
            filtered = df[df["crop_type"] == crop]

        fig = px.line(
            filtered,
            x="date",
            y="price_₹/ton",
            title=f"{crop} Price Trend",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------
    # TAB 3 - WEATHER IMPACT
    # ------------------------------------
    with tab3:
        st.header("Climate Impact on Prices")

        factor = st.selectbox("Weather Factor", ["rainfall_mm", "temperature_c"])

        fig = px.scatter(
            df,
            x=factor,
            y="price_₹/ton",
            color="crop_type",
            trendline="ols",
            title=f"Price vs {factor.replace('_', ' ').title()}",
        )
        st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------
    # TAB 4 - PREDICTION
    # ------------------------------------
    with tab4:
        st.header("Predict Market Price")

        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                model_data = pickle.load(f)

            model = model_data["model"]
            le_dict = model_data["le_dict"]

            col1, col2 = st.columns(2)
            with col1:
                state = st.selectbox("State", le_dict["state"].classes_)
                city = st.selectbox("City", le_dict["city"].classes_)
                crop = st.selectbox("Crop Type", le_dict["crop_type"].classes_)

            with col2:
                season = st.selectbox("Season", le_dict["season"].classes_)

                # Auto-fill regional averages
                region = df[(df["state"] == state) & (df["city"] == city)]

                if not region.empty:
                    month = int(region["date"].dt.month.mode()[0])
                    rain = region["rainfall_mm"].mean()
                    temp = region["temperature_c"].mean()
                else:
                    month = 6
                    rain = 100.0
                    temp = 25.0

            if st.button("Predict Price"):
                input_df = pd.DataFrame(
                    [[state, city, crop, season, month, rain, temp]],
                    columns=[
                        "state",
                        "city",
                        "crop_type",
                        "season",
                        "month",
                        "rainfall_mm",
                        "temperature_c",
                    ],
                )

                # Apply label encoders
                for col in ["state", "city", "crop_type", "season", "month"]:
                    input_df[col] = le_dict[col].transform(input_df[col])

                prediction = model.predict(input_df)[0]

                st.success(f"Predicted Price: ₹{prediction:.2f}/ton")

        else:
            st.warning("No trained model found. Train the model first.")

    # ------------------------------------
    # TAB 5 - NEPAL PROVINCE MAP
    # ------------------------------------
    with tab5:
        st.header("Nepal Province Price Distribution")

        try:
            # Correct Nepal GeoJSON
            URL = "https://raw.githubusercontent.com/sandeshchapagain/nepal-geojson/main/nepal-provinces.geojson"
            geojson_data = requests.get(URL).json()

            df["province"] = df["state"].str.strip()

            avg_prices = df.groupby("province")["price_₹/ton"].mean().reset_index()

            fig = px.choropleth(
                avg_prices,
                geojson=geojson_data,
                locations="province",
                featureidkey="properties.name",
                color="price_₹/ton",
                color_continuous_scale="YlOrBr",
                title="Average Price by Province (Nepal)",
            )

            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Map Error: {e}")


# ------------------------------------
if __name__ == "__main__":
    main()
