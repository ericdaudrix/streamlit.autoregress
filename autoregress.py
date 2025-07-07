import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
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

# Feature numérique pour régression
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
    yearly = st.sidebar.checkbox("Saisonnalité annuelle", value=True)
    weekly = st.sidebar.checkbox("Saisonnalité hebdomadaire", value=False)
    daily = st.sidebar.checkbox("Saisonnalité journalière", value=False)
    hourly = st.sidebar.checkbox("Saisonnalité horaire", value=False)

# Validation croisée temporelle
n_splits = st.sidebar.slider("Nombre de folds CV", 2, 10, 5)
metrics = {'MAE': [], 'RMSE': [], 'R2': []}

tscv = TimeSeriesSplit(n_splits=n_splits)
for train_idx, test_idx in tscv.split(df):
    df_train = df.iloc[train_idx]
    df_test = df.iloc[test_idx]
    if use_prophet:
        # Préparer pour Prophet
        prop_train = df_train.rename(columns={'Date':'ds','Value':'y'})[['ds','y']]
        m = Prophet(
            yearly_seasonality=yearly,
            weekly_seasonality=weekly,
            daily_seasonality=daily
        )
        if hourly:
            m.add_seasonality(name='hourly', period=24, fourier_order=5)
        m.fit(prop_train)
        future = m.make_future_dataframe(periods=len(df_test), freq='D')
        forecast = m.predict(future)
        y_pred = forecast['yhat'].iloc[-len(df_test):].values
        y_true = df_test['Value'].values
    else:
        # Régression linéaire simple
        model = LinearRegression()
        X_train = df_train[['Days']]
        y_train = df_train['Value']
        X_test = df_test[['Days']]
        y_true = df_test['Value'].values
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    metrics['MAE'].append(mean_absolute_error(y_true, y_pred))
    metrics['RMSE'].append(np.sqrt(mean_squared_error(y_true, y_pred)))
    metrics['R2'].append(r2_score(y_true, y_pred))

st.subheader("Métriques de validation croisée")
st.write(pd.DataFrame(metrics))

# Entraînement complet et prévisions
horizon = st.sidebar.number_input("Horizon de prévision (jours)", 30, 365, 183)

if use_prophet:
    prop_full = df.rename(columns={'Date':'ds','Value':'y'})[['ds','y']]
    m = Prophet(
        yearly_seasonality=yearly,
        weekly_seasonality=weekly,
        daily_seasonality=daily
    )
    if hourly:
        m.add_seasonality(name='hourly', period=24, fourier_order=5)
    m.fit(prop_full)
    df['Fit'] = m.predict(prop_full)['yhat']
    future_full = m.make_future_dataframe(periods=horizon, freq='D')
    forecast = m.predict(future_full)
    future = (
        forecast[['ds','yhat']]
        .tail(horizon)
        .rename(columns={'ds':'Date','yhat':'Prediction'})
    )
else:
    model = LinearRegression()
    model.fit(X, y)
    df['Fit'] = model.predict(X)
    last_date = df['Date'].max()
    future_dates = last_date + pd.to_timedelta(np.arange(1, horizon+1), unit='D')
    future_days = (
        (future_dates - df['Date'].min()) / np.timedelta64(1, 'D')
    ).values.reshape(-1, 1)
    preds = model.predict(future_days)
    future = pd.DataFrame({'Date': future_dates, 'Prediction': preds})

# Visualisation interactive
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Value'], mode='lines+markers', name='Historique'
))
fig.add_trace(go.Scatter(
    x=df['Date'], y=df['Fit'], mode='lines', name='Fit'
))
fig.add_trace(go.Scatter(
    x=future['Date'], y=future['Prediction'], mode='lines', name='Prévisions futures'
))
fig.update_layout(
    title="Prévision du temps de fonctionnement",
    xaxis_title="Date",
    yaxis_title="Valeur",
    hovermode='x unified',
    template='plotly_white'
)

st.subheader("Visualisation des résultats")
st.plotly_chart(fig, use_container_width=True)
