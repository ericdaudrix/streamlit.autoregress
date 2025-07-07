import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go

# Modèles SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Modèle Prophet
from prophet import Prophet

# --- Application Streamlit ---
st.title("Prévision du temps de fonctionnement")
st.sidebar.header("Paramètres globaux")

# Upload du fichier CSV
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV (Date;Value)", type=["csv"])

if uploaded_file:
    # Chargement et préparation des données
    df = pd.read_csv(uploaded_file, sep=';')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    st.subheader("Aperçu des données")
    st.write(df)

    # Sélection du modèle
    type_modele = st.sidebar.selectbox(
        "Choisir le modèle",
        ["Régression + Fourier", "SARIMA", "Prophet"]
    )

    # Régression + Fourier
    if type_modele == "Régression + Fourier":
        df['Days'] = (df['Date'] - df['Date'].min()).dt.days
        df['DoY'] = df['Date'].dt.dayofyear
        df['sin_year'] = np.sin(2 * np.pi * df['DoY'] / 365)
        df['cos_year'] = np.cos(2 * np.pi * df['DoY'] / 365)
        features = ['Days', 'sin_year', 'cos_year']
        X = df[features]
        y = df['Value']

        sub = st.sidebar.selectbox("Type de régression", ["LinearRegression", "Ridge", "Lasso"])
        alpha = st.sidebar.slider("Alpha pour Ridge/Lasso", 0.0, 10.0, 1.0) if sub in ["Ridge", "Lasso"] else None
        if sub == "LinearRegression":
            model = LinearRegression()
        elif sub == "Ridge":
            model = Ridge(alpha=alpha)
        else:
            model = Lasso(alpha=alpha)

        n_splits = st.sidebar.slider("Folds CV", 2, 10, 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {"MAE": [], "RMSE": [], "R2": []}
        for tr, te in tscv.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            y_pred = model.predict(X.iloc[te])
            metrics['MAE'].append(mean_absolute_error(y.iloc[te], y_pred))
            metrics['RMSE'].append(np.sqrt(mean_squared_error(y.iloc[te], y_pred)))
            metrics['R2'].append(r2_score(y.iloc[te], y_pred))
        st.subheader("Métriques de validation croisée")
        st.write(pd.DataFrame(metrics))

        model.fit(X, y)
        df['Fit'] = model.predict(X)
        h = st.sidebar.number_input("Horizon (jours)", 30, 365, 183)
        future = pd.DataFrame({'Date': df['Date'].max() + pd.to_timedelta(np.arange(1, h+1), 'D')})
        future['Days'] = (future['Date'] - df['Date'].min()).dt.days
        future['DoY'] = future['Date'].dt.dayofyear
        future['sin_year'] = np.sin(2*np.pi*future['DoY']/365)
        future['cos_year'] = np.cos(2*np.pi*future['DoY']/365)
        future['Prediction'] = model.predict(future[features])

    # SARIMA
    elif type_modele == "SARIMA":
        df.set_index('Date', inplace=True)
        p = st.sidebar.number_input("AR order (p)", 0, 5, 1)
        d = st.sidebar.number_input("Diff order (d)", 0, 2, 1)
        q = st.sidebar.number_input("MA order (q)", 0, 5, 1)
        P = st.sidebar.number_input("Seasonal AR (P)", 0, 2, 1)
        D = st.sidebar.number_input("Seasonal diff (D)", 0, 1, 1)
        Q = st.sidebar.number_input("Seasonal MA (Q)", 0, 2, 1)
        s = st.sidebar.number_input("Season length (s)", 2, 365, 365)
        sarima = SARIMAX(df['Value'], order=(p, d, q), seasonal_order=(P, D, Q, s), enforce_stationarity=False, enforce_invertibility=False)
        res = sarima.fit(disp=False)
        df['Fit'] = res.fittedvalues
        h = st.sidebar.number_input("Horizon (jours)", 30, 365, 183)
        pred = res.get_forecast(steps=h)
        future = pred.predicted_mean.reset_index().rename(columns={'index': 'Date', 0: 'Prediction'})
        df.reset_index(inplace=True)

    # Prophet
    else:
        prophet_df = df.rename(columns={'Date': 'ds', 'Value': 'y'})[['ds', 'y']]
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m.fit(prophet_df)
        df['Fit'] = m.predict(prophet_df)['yhat']
        h = st.sidebar.number_input("Horizon (jours)", 30, 365, 183)
        future_pd = m.make_future_dataframe(periods=h)
        forecast = m.predict(future_pd)
        future = forecast[['ds', 'yhat']].tail(h).rename(columns={'ds': 'Date', 'yhat': 'Prediction'})

    # Tracé final
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], mode='lines+markers', name='Historique'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Fit'], mode='lines', name='Fit'))
    fig.add_trace(go.Scatter(x=future['Date'], y=future['Prediction'], mode='lines', name='Prévision future'))
    fig.update_layout(
        title="Prévision du temps de fonctionnement", xaxis_title="Date", yaxis_title="Value",
        hovermode='x unified', template='plotly_white'
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Uploader un CSV avec colonnes Date;Value pour commencer.")
