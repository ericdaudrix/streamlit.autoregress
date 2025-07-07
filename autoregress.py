import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go

# --- Application Streamlit ---
st.title("Prévision du temps de fonctionnement")
st.sidebar.header("Configuration")

# Upload des données
uploaded_file = st.sidebar.file_uploader("Uploader un fichier CSV (Date;Value)", type=["csv"])
if not uploaded_file:
    st.info("Merci d'uploader un CSV avec les colonnes `Date` et `Value` séparées par `;`.")
    st.stop()

# Préparation des données
df = pd.read_csv(uploaded_file, sep=';')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
st.subheader("Aperçu des données")
st.write(df)

df['Days'] = (df['Date'] - df['Date'].min()).dt.days
X = df[['Days']]
y = df['Value']

# Choix du modèle
model_type = st.sidebar.selectbox("Modèle", ["LinearRegression", "MLPRegressor"])
if model_type == "LinearRegression":
    model = LinearRegression()
else:
    # Paramètres MLP
    n_layers = st.sidebar.slider("Nombre de couches cachées", 1, 5, 2)
    size_layers = st.sidebar.slider("Neurones par couche", 10, 200, 50)
    hidden_layer_sizes = tuple([size_layers] * n_layers)
    activation = st.sidebar.selectbox("Fonction d'activation", ["relu", "tanh", "logistic"])
    alpha = st.sidebar.slider("Alpha (L2 penalty)", 0.0001, 1.0, 0.0001, format="%f")
    max_iter = st.sidebar.number_input("Max iterations", 100, 1000, 200)
    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        max_iter=max_iter,
        random_state=42
    )

# Validation croisée temporelle
n_splits = st.sidebar.slider("Nombre de folds CV", 2, 10, 5)
tscv = TimeSeriesSplit(n_splits=n_splits)
maes, rmses, r2s = [], [], []
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    maes.append(mean_absolute_error(y_test, y_pred))
    rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2s.append(r2_score(y_test, y_pred))

st.subheader("Métriques de validation croisée")
st.write(pd.DataFrame({
    "MAE": maes,
    "RMSE": rmses,
    "R²": r2s
}))

# Entraînement final
model.fit(X, y)
df['Fit'] = model.predict(X)

# Prévision future
horizon = st.sidebar.number_input("Horizon de prévision (jours)", 30, 365, 183)
future_dates = df['Date'].max() + pd.to_timedelta(np.arange(1, horizon+1), unit='D')
        # Calcul du nombre de jours pour les dates futures
        future_days = ((future_dates - df['Date'].min()) / np.timedelta64(1, 'D')).values.reshape(-1, 1)
