import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# =========================
# App Title
# =========================
st.set_page_config(page_title="Agricultural Price Prediction", layout="wide")
st.title("🌾 Agricultural Market Price Analyzer")

# =========================
# Load Data
# =========================
uploaded_file = st.file_uploader("Upload Cleaned Dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Identify categorical and numeric columns
    cat_cols = ['state', 'city', 'crop_type', 'season']
    num_cols = [col for col in df.columns if col not in cat_cols + ['price']]

    # =========================
    # Shared Preprocessor
    # =========================
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ]
    )

    # =========================
    # Models as Pipelines
    # =========================
    models = {
        "XGBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ]),
        "Linear Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
    }

    # =========================
    # Model Selection
    # =========================
    selected_model_name = st.selectbox("Select Prediction Algorithm", list(models.keys()))
    model = models[selected_model_name]

    # =========================
    # Train/Test Split
    # =========================
    X = df[cat_cols + num_cols]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.write(f"### {selected_model_name} Test Predictions")
    results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    st.dataframe(results_df.head())

    # =========================
    # Learning Curve
    # =========================
    st.write("### Learning Curve")
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, 'o-', label="Training score")
    ax.plot(train_sizes, test_mean, 'o-', label="Validation score")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Score (R²)")
    ax.legend()
    st.pyplot(fig)

    # =========================
    # PCA Visualization
    # =========================
    st.write("### PCA Analysis")
    X_processed = preprocessor.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_processed)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['price'] = y.values
    fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color='price', title="PCA Price Distribution")
    st.plotly_chart(fig_pca, use_container_width=True)

    # =========================
    # Geographical Price Distribution (Offline Nepal Map)
    # =========================
    st.write("### Geographical Price Distribution (Nepal Provinces)")

    # Embedded simplified GeoJSON
    nepal_geojson = {
        "type": "FeatureCollection",
        "features": [
            # Minimal placeholder — replace with full geojson if you have it
            {
                "type": "Feature",
                "properties": {"Province": "Bagmati"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[85.0, 27.5], [85.5, 27.5], [85.5, 28.0], [85.0, 28.0], [85.0, 27.5]]]
                }
            }
        ]
    }

    province_avg_price = df.groupby('state')['price'].mean().reset_index()
    fig_map = px.choropleth(
        province_avg_price,
        geojson=nepal_geojson,
        featureidkey="properties.Province",
        locations='state',
        color='price',
        color_continuous_scale="Viridis",
        title="Average Price by Province"
    )
    fig_map.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_map, use_container_width=True)

else:
    st.info("Please upload your cleaned dataset to begin.")
