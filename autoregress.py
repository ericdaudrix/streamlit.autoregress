import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from prophet import Prophet

# --- Application Streamlit ---
st.title("Prévision du temps de fonctionnement")
st.sidebar.header("Configuration du modèle")

# Upload du fichier CSV
uploaded_file = st.sidebar.file_uploader(
    "Uploader un fichier CSV (colonnes: Date;Value)", type=["csv"]
)
if not uploaded_file:
    st.info(
        "Merci d'uploader un fichier CSV avec les colonnes `Date` et `Value` séparées par `;`."
    )
    st.stop()

# Chargement des données
df = pd.read_csv(uploaded_file, sep=';')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
st.subheader("Aperçu des données")
st.write(df)

# Création de feature pour régression
df['Days'] = (df['Date'] - df['Date'].min()).dt.days
X = df[['Days']]
y = df['Value']

# Sélection du modèle
type_model = st.sidebar.selectbox(
    "Sélectionnez un modèle",
    ["LinearRegression", "Prophet"]
)
use_prophet = (type_model == 'Prophet')

# Si Prophet, toggles de saisonnalités
if use_prophet:
    yearly = st.sidebar.checkbox("Saisonnalité annuelle", value=False)
    weekly = st.sidebar.checkbox("Saisonnalité hebdomadaire", value=False)
    daily = st.sidebar.checkbox("Saisonnalité journalière", value=False)
    hourly = st.sidebar.checkbox("Saisonnalité horaire", value=False)

# Horizon de prévision
horizon = st.sidebar.number_input(
    "Horizon de prévision (jours)", 30, 365, 183
)

# Entraînement complet et prévisions
if use_prophet:
    # Préparer DataFrame pour Prophet
    prop_full = df.rename(columns={'Date':'ds','Value':'y'})[['ds','y']]
    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily
    )
    if hourly:
        m.add_seasonality(name='hourly', period=24, fourier_order=5)
    m.fit(prop_full)

    # Fit et forecast avec intervalles
    forecast = m.predict(m.make_future_dataframe(periods=horizon, freq='D'))
    df['Fit'] = forecast.set_index('ds')['yhat'][:len(df)].values
    future = forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(horizon)
    future = future.rename(columns={'ds':'Date','yhat':'Prediction','yhat_lower':'Lower','yhat_upper':'Upper'})
else:
    # Régression linéaire
    model = LinearRegression()
    model.fit(X, y)
    df['Fit'] = model.predict(X)

    # Calculer écart-type des résidus pour intervalle
    resid_std = (y - df['Fit']).std()
    # Prévision future
    last_date = df['Date'].max()
    future_dates = last_date + pd.to_timedelta(np.arange(1, horizon+1), unit='D')
    future_days = ((future_dates - df['Date'].min()) / np.timedelta64(1, 'D')).values.reshape(-1, 1)
    preds = model.predict(future_days)
    # bornes à 95%
    lower = preds - 1.96 * resid_std
    upper = preds + 1.96 * resid_std
    future = pd.DataFrame({
        'Date': future_dates,
        'Prediction': preds,
        'Lower': lower,
        'Upper': upper
    })

# Visualisation interactive avec intervalle de confiance
fig = go.Figure()
# Historique
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Value'], mode='markers', name='Historique'
))
# Fit
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Fit'], mode='lines', name='Fit'
))
# Intervalle historique / pas nécessaire ici
# Prévisions futures
fig.add_trace(go.Scatter(
    x=future['Date'], y=future['Upper'],
    mode='lines', line=dict(width=0), showlegend=False
))
fig.add_trace(go.Scatter(
    x=future['Date'], y=future['Lower'],
    mode='lines', fill='tonexty', fillcolor='rgba(0,100,80,0.2)',
    line=dict(width=0), name='Intervalle 95%'
))
# Courbe de prévision
fig.add_trace(go.Scatter(
    x=future['Date'], y=future['Prediction'], mode='lines', name='Prévisions'
))

fig.update_layout(
    title="Prévision du temps de fonctionnement avec intervalle de confiance",
    xaxis_title="Date",
    yaxis_title="Valeur",
    hovermode='x unified',
    template='plotly_white'
)

st.subheader("Visualisation des résultats")
st.plotly_chart(fig, use_container_width=True)
