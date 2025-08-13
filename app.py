import pandas as pd
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os

# =============================
# App Configuration
# =============================
st.set_page_config(page_title="AgriPrice Analyzer", layout="wide")

# =============================
# Data Loading
# =============================
@st.cache_data
def load_data(file_input='cleaned_dataset.csv'):
    try:
        if isinstance(file_input, str) and os.path.exists(file_input):
            df = pd.read_csv(file_input)
        elif hasattr(file_input, 'read'):
            df = pd.read_csv(file_input)
        else:
            return pd.DataFrame()
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# =============================
# Helpers: Diagnostics & Tuning
# =============================
def plot_learning_curve_pipeline(pipeline, X, y, scoring='neg_mean_absolute_error', 
                                 train_sizes=np.linspace(0.1, 1.0, 5), cv=3, model_name='Model'):
    sizes, train_scores, val_scores = learning_curve(
        estimator=pipeline,
        X=X,
        y=y,
        train_sizes=train_sizes,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        shuffle=True,
        random_state=42
    )
    train_mae = -train_scores
    val_mae = -val_scores
    train_mean, train_std = train_mae.mean(axis=1), train_mae.std(axis=1)
    val_mean, val_std = val_mae.mean(axis=1), val_mae.std(axis=1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sizes, y=train_mean, mode='lines+markers', name='Train MAE'))
    fig.add_trace(go.Scatter(x=sizes, y=val_mean, mode='lines+markers', name='Validation MAE'))
    fig.add_trace(go.Scatter(
        x=np.concatenate([sizes, sizes[::-1]]),
        y=np.concatenate([train_mean - train_std, (train_mean + train_std)[::-1]]),
        fill='toself', fillcolor='rgba(0,150,136,0.08)', line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
    ))
    fig.update_layout(title=f'Learning Curve ({model_name})', xaxis_title='Training examples', yaxis_title='MAE (lower is better)')
    return fig


def plot_validation_curve_fig(pipeline, X, y, param_name, param_range, scoring='neg_mean_absolute_error', cv=3):
    train_scores, val_scores = validation_curve(
        estimator=pipeline,
        X=X,
        y=y,
        param_name=param_name,
        param_range=param_range,
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(param_range), y=train_mae, name='Train MAE', mode='lines+markers'))
    fig.add_trace(go.Scatter(x=list(param_range), y=val_mae, name='Validation MAE', mode='lines+markers'))
    fig.update_layout(title=f'Validation Curve ({param_name})', xaxis_title=param_name, yaxis_title='MAE')
    return fig


def tune_model_randomized(pipeline, X, y, param_distributions, n_iter=15, cv=3, scoring='neg_mean_absolute_error'):
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
    search.fit(X, y)
    return search


def run_pca_analysis(preprocessor, X_df, color_by=None, n_components=10):
    X_trans = preprocessor.fit_transform(X_df)
    # to dense if sparse
    try:
        X_arr = X_trans.toarray()
    except Exception:
        X_arr = X_trans
    pca = PCA(n_components=min(n_components, X_arr.shape[1]))
    pcs = pca.fit_transform(X_arr)
    explained = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    # Scree plot
    scree_fig = go.Figure()
    scree_fig.add_trace(go.Bar(x=[f"PC{i+1}" for i in range(len(explained))], y=explained))
    scree_fig.update_layout(title="Scree Plot (Explained Variance Ratio)", xaxis_title='Principal Component', yaxis_title='Variance Ratio')

    # Cumulative variance
    cum_fig = go.Figure()
    cum_fig.add_trace(go.Scatter(x=list(range(1, len(cum_explained)+1)), y=cum_explained, mode='lines+markers', name='Cumulative'))
    cum_fig.update_layout(title='Cumulative Explained Variance', xaxis_title='Components', yaxis_title='Cumulative Variance')

    # 2D scatter
    scatter_fig = None
    if pcs.shape[1] >= 2:
        df_pca = pd.DataFrame({'PC1': pcs[:, 0], 'PC2': pcs[:, 1]})
        if color_by is not None and color_by in X_df.columns:
            df_pca[color_by] = X_df[color_by].values
            scatter_fig = px.scatter(df_pca, x='PC1', y='PC2', color=color_by, title='PCA Projection (PC1 vs PC2)')
        else:
            scatter_fig = px.scatter(df_pca, x='PC1', y='PC2', title='PCA Projection (PC1 vs PC2)')

    return {'pca': pca, 'scree': scree_fig, 'cumulative': cum_fig, 'scatter': scatter_fig, 'explained': explained}

# =============================
# Training
# =============================
def train_model(df):
    try:
        df = df.copy()
        df['month'] = df['date'].dt.month

        categorical_cols = ['state', 'city', 'crop_type', 'season', 'month']
        numeric_cols = ['rainfall_mm', 'temperature_c']

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
                ('num', StandardScaler(), numeric_cols)
            ]
        )

        X = df[categorical_cols + numeric_cols]
        y = df['price_₹/ton']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            'XGBoost': XGBRegressor(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Linear Regression': LinearRegression()
        }

        trained_models = {}
        model_performance = {}

        for name, model in models.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

            # If XGBoost, add early stopping for better generalization and to track eval history
            fit_kwargs = {}
            if isinstance(model, XGBRegressor):
                fit_kwargs = {
                    'regressor__eval_set': [(X_test, y_test)],
                    'regressor__early_stopping_rounds': 20,
                    'regressor__verbose': False,
                    'regressor__eval_metric': 'rmse'
                }

            pipe.fit(X_train, y_train, **fit_kwargs)
            y_pred = pipe.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            trained_models[name] = pipe
            model_performance[name] = {'MAE': mae, 'R2': r2}

        with open('model.pkl', 'wb') as f:
            pickle.dump({'models': trained_models, 'columns': X.columns.tolist()}, f)

        for name, scores in model_performance.items():
            st.sidebar.write(f"🔹 {name} → MAE: ₹{scores['MAE']:.2f}, R²: {scores['R2']:.2f}")

        return True
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False

# =============================
# Main App
# =============================
def main():
    st.title("Agricultural Market Price Analyzer 🌾")

    # Sidebar controls
    st.sidebar.header("Data Management")
    uploaded_file = st.sidebar.file_uploader("Upload CSV Data", type=['csv'])

    if uploaded_file:
        df = load_data(uploaded_file)
        df.to_csv("uploaded_dataset.csv", index=False)
        active_file = "uploaded_dataset.csv"
    else:
        df = load_data("cleaned_dataset.csv")
        active_file = "cleaned_dataset.csv"

    if not df.empty:
        st.sidebar.success("Data loaded successfully!")
        with st.expander("🔍 View Uploaded Data Sample"):
            st.dataframe(df.head(20), use_container_width=True)

        if st.sidebar.button("Retrain Prediction Model"):
            with st.spinner("Training new model..."):
                if train_model(df):
                    st.sidebar.success("Model updated successfully!")

    tabs = st.tabs([
        "📊 Real-Time Dashboard",
        "📈 Historical Trends",
        "🌦️ Weather Impact",
        "🔮 Price Prediction",
        "🗺️ Regional Analysis",
        "🧪 Model Diagnostics"
    ])

    # ================= Dashboard =================
    with tabs[0]:
        st.header("Market Overview")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Average Price", f"₹{df['price_₹/ton'].mean():.2f}/ton", "5.2% vs last month")
        with col2:
            st.metric("Supply-Demand Ratio", f"{(df['supply_volume_tons']/df['demand_volume_tons']).mean():.2f}", "Market Balance")
        with col3:
            st.metric("Active Regions", df['state'].nunique(), "Provinces tracking prices")

        st.subheader("Latest Market Entries")
        st.dataframe(df.sort_values('date', ascending=False).head(10), use_container_width=True)

    # ================= Historical Trends =================
    with tabs[1]:
        st.header("Historical Price Analysis")
        if 'date_range' not in st.session_state:
            st.session_state.date_range = [df['date'].min().date(), df['date'].max().date()]
        col1, col2 = st.columns(2)
        with col1:
            crop_filter = st.selectbox("Select Crop", df['crop_type'].unique())
        with col2:
            date_range = st.date_input(
                "Select Date Range",
                value=st.session_state.date_range,
                min_value=df['date'].min().date(),
                max_value=df['date'].max().date(),
                key="date_range_selector"
            )
        st.session_state.date_range = date_range
        filtered_df = df[(df['crop_type'] == crop_filter) & (df['date'].between(pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])))]
        if filtered_df.empty:
            st.error("No data available for selected date range. Showing full historical trend.")
            c1, c2, c3 = st.columns([1,2,1])
            with c2:
                if st.button("Reset to Default Date Range"):
                    st.session_state.date_range = [df['date'].min().date(), df['date'].max().date()]
                    st.experimental_rerun()
            filtered_df = df[df['crop_type'] == crop_filter]
        fig = px.line(filtered_df, x='date', y='price_₹/ton', title=f"{crop_filter} Price Trend")
        st.plotly_chart(fig, use_container_width=True)

    # ================= Weather Impact =================
    with tabs[2]:
        st.header("Climate Correlation Analysis")
        col1, col2 = st.columns(2)
        with col1:
            weather_factor = st.selectbox("Select Weather Factor", ['rainfall_mm', 'temperature_c'])
        fig = px.scatter(df, x=weather_factor, y='price_₹/ton', color='crop_type', trendline="ols",
                         title=f"Price vs {weather_factor.replace('_', ' ').title()}")
        st.plotly_chart(fig, use_container_width=True)

    # ================= Price Prediction =================
    with tabs[3]:
        st.header("Price Prediction Model")
        if os.path.exists('model.pkl'):
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            models = model_data['models']
            columns = model_data['columns']

            col1, col2 = st.columns(2)
            with col1:
                model_choice = st.selectbox("Select Model", list(models.keys()))
                state = st.selectbox("State", df['state'].unique())
                city = st.selectbox("City", df['city'].unique())
                crop_type = st.selectbox("Crop Type", df['crop_type'].unique())
            with col2:
                season = st.selectbox("Season", df['season'].unique())
                filtered_data = df[(df['state'] == state) & (df['city'] == city) & (df['season'] == season)]
                if not filtered_data.empty:
                    avg_month = int(filtered_data['date'].dt.month.mode()[0])
                    avg_rainfall = filtered_data['rainfall_mm'].mean()
                    avg_temp = filtered_data['temperature_c'].mean()
                else:
                    avg_month = 6
                    avg_rainfall = 100.0
                    avg_temp = 25.0

            if st.button("Predict Price"):
                input_df = pd.DataFrame([[state, city, crop_type, season, avg_month, avg_rainfall, avg_temp]], columns=columns)
                selected_model = models[model_choice]
                prediction = selected_model.predict(input_df)
                st.success(f"[{model_choice}] Predicted Price: ₹{prediction[0]:.2f}/ton")
                st.caption(f"Based on {state}'s {season} season averages: {avg_rainfall:.1f}mm rainfall, {avg_temp:.1f}°C")
        else:
            st.warning("No trained model found. Upload data and train model first.")

    # ================= Regional Analysis (Nepal) =================
    with tabs[4]:
        st.header("Geographical Price Distribution (Nepal Provinces)")
        try:
            nepal_geojson = "https://raw.githubusercontent.com/sandeshchapagain/nepal-geojson/main/nepal-provinces.geojson"
            avg_prices = df.groupby(['state', 'crop_type'])['price_₹/ton'].mean().reset_index()
            fig = px.choropleth(
                avg_prices,
                geojson=nepal_geojson,
                locations="state",
                featureidkey="properties.name",
                color="price_₹/ton",
                color_continuous_scale=px.colors.sequential.YlOrBr,
                hover_name="state",
                animation_frame="crop_type",
                title="Nepal Province-wise Price Variations"
            )
            fig.update_geos(fitbounds="locations", visible=False)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Map rendering error: {str(e)}")

    # ================= Model Diagnostics =================
    with tabs[5]:
        st.header("Model Diagnostics: Learning Curves, Tuning, PCA")
        if not os.path.exists('model.pkl'):
            st.info("Train the models first from the sidebar, then return here.")
        else:
            model_data = pickle.load(open('model.pkl', 'rb'))
            models = model_data['models']
            columns = model_data['columns']

            X = df[columns]
            y = df['price_₹/ton']

            st.subheader("➊ Learning Curve")
            mcol1, mcol2 = st.columns([2,1])
            with mcol1:
                lc_model = st.selectbox("Model for learning curve", list(models.keys()), key='lc_model')
                cv_folds = st.slider("CV folds", 2, 5, 3, key='lc_cv')
            with mcol2:
                if st.button("Generate Learning Curve"):
                    with st.spinner("Computing learning curve..."):
                        fig = plot_learning_curve_pipeline(models[lc_model], X, y, cv=cv_folds, model_name=lc_model)
                        st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("➋ Validation Curve (single hyperparameter)")
            vc_model = st.selectbox("Model for validation curve", list(models.keys()), key='vc_model')
            param_choices = []
            if vc_model.startswith('Random Forest'):
                param_choices = ['regressor__n_estimators', 'regressor__max_depth', 'regressor__min_samples_split']
            elif vc_model.startswith('XGBoost'):
                param_choices = ['regressor__n_estimators', 'regressor__max_depth', 'regressor__learning_rate', 'regressor__subsample']
            else:
                st.info("Validation curves are not very informative for Linear Regression.")
            if param_choices:
                p = st.selectbox("Hyperparameter", param_choices)
                rng = st.text_input("Parameter range (comma-separated)", value="50,100,200")
                if st.button("Plot Validation Curve"):
                    try:
                        vals = [float(v.strip()) for v in rng.split(',')] if 'learning_rate' in p else [int(v.strip()) if v.strip().lower() != 'none' else None for v in rng.split(',')]
                        fig = plot_validation_curve_fig(models[vc_model], X, y, p, vals)
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Validation curve error: {e}")

            st.divider()
            st.subheader("➌ Hyperparameter Tuning (RandomizedSearchCV)")
            tune_model = st.selectbox("Model to tune", ['XGBoost', 'Random Forest'])
            n_iter = st.slider("Search iterations", 5, 40, 15)
            cv_tune = st.slider("CV folds (tuning)", 2, 5, 3)
            if st.button("Run Tuning"):
                with st.spinner("Running randomized search..."):
                    base_pipe = models[tune_model]
                    if tune_model == 'XGBoost':
                        param_dist = {
                            'regressor__n_estimators': [50, 100, 200, 400],
                            'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
                            'regressor__max_depth': [3, 5, 7, 9],
                            'regressor__subsample': [0.6, 0.8, 1.0]
                        }
                    else:
                        param_dist = {
                            'regressor__n_estimators': [50, 100, 200, 400],
                            'regressor__max_depth': [None, 10, 20, 30],
                            'regressor__min_samples_split': [2, 5, 10]
                        }
                    search = tune_model_randomized(base_pipe, X, y, param_dist, n_iter=n_iter, cv=cv_tune)
                    st.success(f"Best CV MAE: {-search.best_score_:.3f}")
                    st.write("Best Params:", search.best_params_)

                    # Allow saving tuned model
                    if st.button("Save tuned model"):
                        models[tune_model + ' (tuned)'] = search.best_estimator_
                        pickle.dump({'models': models, 'columns': columns}, open('model.pkl', 'wb'))
                        st.success("Saved! Reload the page to use it from the prediction tab.")

            st.divider()
            st.subheader("➍ PCA Analysis")
            color_by = st.selectbox("Color PCA by", ['crop_type', 'state', 'season', 'city'])
            n_comp = st.slider("Number of components", 2, 20, 10)
            if st.button("Run PCA"):
                with st.spinner("Running PCA on preprocessed features..."):
                    preproc = list(models.values())[0].named_steps['preprocessor']  # any model's preprocessor is fine
                    feature_df = df[columns]
                    pca_out = run_pca_analysis(preproc, feature_df, color_by=color_by, n_components=n_comp)
                    st.plotly_chart(pca_out['scree'], use_container_width=True)
                    st.plotly_chart(pca_out['cumulative'], use_container_width=True)
                    if pca_out['scatter'] is not None:
                        st.plotly_chart(pca_out['scatter'], use_container_width=True)

    # ================= Report =================
    st.sidebar.header("Report Generation")
    if st.sidebar.button("📥 Generate Full Report"):
        report = df.describe().T
        st.sidebar.download_button(label="Download Summary Report", data=report.to_csv(), file_name="market_summary.csv", mime="text/csv")

if __name__ == "__main__":
    main()
