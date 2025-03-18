
#===================================================#
# Eric Daudrix - Lycée Monnerville Cahors - CMQE IF #
#===================================================#

import streamlit as st
import pandas as pd
import numpy as np

#from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.forecasting.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import plotly.graph_objects as go

# Streamlit App Configuration
st.set_page_config(page_title="Forecasting App", layout="wide")
st.title("📈 Forecasting Application")
st.sidebar.header("⚙️ Configuration du modèle prédictif")

# Sidebar inputs for parameters
data_freq = st.sidebar.number_input("📊 Data Frequency (seconds)", min_value=1, value=60, step=1)
steps = st.sidebar.number_input("📉 Training Steps", min_value=1, value=120, step=1)
lags = st.sidebar.number_input("⏳ Lags", min_value=1, value=15, step=1)
pred_steps = st.sidebar.number_input("🔮 Prediction Steps", min_value=1, value=120, step=1)

# File upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    try:
        # Load data
        data = pd.read_csv(uploaded_file, sep=None, engine='python')
        st.write("📊 **Data Preview:**", data.head())

        # Data preparation
        data.rename(columns={data.columns[0]: 'date', data.columns[1]: 'y'}, inplace=True)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        
        # Resampling (to avoid missing timestamps)
        data = data.resample(f'{data_freq}s').mean().interpolate()

        # Display data summary
        st.write("📈 **Data Summary:**", data.describe())

        # Split data
        train = data[:-steps]
        test = data[-steps:]

        if st.button("🚀 Train Model"):
            if train.empty:
                st.error("⚠️ Not enough training data. Try reducing `Training Steps`.")
            else:
                # Train model
                forecaster = ForecasterAutoreg(
                    regressor=RandomForestRegressor(random_state=123),
                    lags=lags
                )
                forecaster.fit(y=train['y'])

                st.success("✅ Model trained successfully!")

                # Test model
                predictions = forecaster.predict(steps=pred_steps)

                # Plot predictions vs actual with Plotly
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train.index, y=train['y'], mode='lines', name='Train data', line=dict(color='green')))
                fig.add_trace(go.Scatter(x=test.index, y=test['y'], mode='lines', name='Test data', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=predictions.index, y=predictions.values, mode='lines', name='Predictions', line=dict(color='red')))
                fig.update_layout(title="📊 Predictions vs Actual", xaxis_title="Date", yaxis_title="Value")
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error loading file: {e}")

else:
    st.info("📌 Please upload a CSV file to proceed.")
