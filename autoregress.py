
#===================================================#
# Eric Daudrix - Lycée Monnerville Cahors - CMQE IF #
#===================================================#

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go

# Titre de l'application
st.title("Prévision du temps de fonctionnement")
st.sidebar.header("Paramètres")

# Upload du fichier CSV
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV", type=["csv"] )
if uploaded_file:
    # Chargement et préparation des données
    df = pd.read_csv(uploaded_file, sep=';')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    st.subheader("Aperçu des données")
    st.write(df)

    # Transformation de la date en nombre de jours
    df['Days'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Days']]
    y = df['Value']

    # Sélection du modèle
    model_name = st.sidebar.selectbox("Modèle de régression", ["LinearRegression", "Ridge", "Lasso"] )
    if model_name == "LinearRegression":
        model = LinearRegression()
    elif model_name == "Ridge":
        model = Ridge()
    else:
        model = Lasso()

    # Validation croisée temporelle
    n_splits = st.sidebar.slider("Nombre de folds CV", min_value=2, max_value=10, value=5)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, r2s = [], [], []
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

    # Affichage des métriques
    st.subheader("Métriques de validation croisée")
    st.write(pd.DataFrame({
        "MAE": maes,
        "RMSE": rmses,
        "R²": r2s
    }))

    # Entraînement sur l'ensemble des données
    model.fit(X, y)
    df['Prévision'] = model.predict(X)

    # Choix de l'horizon de prévision
    horizon = st.sidebar.number_input("Horizon de prévision (jours)", min_value=30, max_value=365, value=183)
    last_day = df['Days'].max()
    future_days = np.arange(last_day + 1, last_day + horizon + 1).reshape(-1, 1)
    future_dates = df['Date'].max() + pd.to_timedelta(np.arange(1, horizon + 1), unit='D')
    future_pred = model.predict(future_days)

    # Visualisation interactive
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Value'], mode='lines+markers', name='Historique'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Prévision'], mode='lines', name='Fit'))
    fig.add_trace(go.Scatter(x=future_dates, y=future_pred, mode='lines', name='Prévisions futures'))
    fig.update_layout(
        title="Prévision du temps de fonctionnement",
        xaxis_title="Date",
        yaxis_title="Temps de fonctionnement",
        hovermode='x unified',
        template='plotly_white'
    )
    st.subheader("Visualisation des résultats")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Veuillez uploader un fichier CSV avec les colonnes `Date` et `Value` séparées par `;`.")
