import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import ta

st.set_page_config(page_title="Bitcoin Predictor Pro", page_icon="₿", layout="wide")

API_URL = "https://KinSushi-bitcoin-predictor-pro.hf.space"

def load_recent_data():
    """Récupère les 7 derniers jours de bougies 4h, avec fallback synthétique."""
    try:
        btc = yf.download("BTC-USD", period="7d", interval="4h", progress=False)
        if not btc.empty:
            btc.columns = ['_'.join(col).strip().lower() for col in btc.columns]
            btc.rename(columns={
                'open_btc-usd': 'open',
                'high_btc-usd': 'high',
                'low_btc-usd': 'low',
                'close_btc-usd': 'close',
                'volume_btc-usd': 'volume'
            }, inplace=True)
            return btc, None
    except Exception as e:
        pass
    
    # Fallback synthétique
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=24, freq='4h')
    close_prices = 50000 + np.random.normal(0, 200, 24).cumsum()
    df = pd.DataFrame({
        'open': np.roll(close_prices, 1) + np.random.normal(0, 50, 24),
        'high': close_prices + np.abs(np.random.normal(0, 100, 24)),
        'low': close_prices - np.abs(np.random.normal(0, 100, 24)),
        'close': close_prices,
        'volume': np.random.exponential(100, 24)
    }, index=dates)
    return df, "Données simulées (source de marché indisponible)."

def get_prediction():
    try:
        resp = requests.get(f"{API_URL}/predict-live", timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except:
        return None

st.title("₿ Bitcoin Predictor Pro")
st.markdown("Prédiction de direction sur 4 heures basée sur un modèle XGBoost")

st.sidebar.header("À propos")
st.sidebar.info(
    "Ce dashboard interroge une API FastAPI déployée sur Hugging Face Spaces. "
    "Le modèle utilise 15 indicateurs techniques et un classifieur XGBoost."
)
st.sidebar.markdown(f"**API endpoint** : `{API_URL}`")

data, warning = load_recent_data()
if warning:
    st.warning(warning)

if data is None or data.empty:
    st.error("Impossible de charger les données de marché.")
    st.stop()

current_price = data['close'].iloc[-1]
prev_close = data['close'].iloc[-2]
change = current_price - prev_close
change_pct = (change / prev_close) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Prix actuel (USD)", f"${current_price:,.2f}")
col2.metric("Variation (4h)", f"{change:+,.2f}", f"{change_pct:+.2f}%")

pred = get_prediction()
if pred:
    pred_class = pred['label']
    proba = pred['probability_up']
    api_warning = pred.get('warning', None)
    emoji = "🟢" if pred_class == "UP" else "🔴"
    col3.metric("Prédiction (4h)", f"{emoji} {pred_class}", f"Confiance : {proba:.1%}")
    if api_warning:
        st.warning(api_warning)

    recent_returns = data['close'].pct_change().dropna()
    avg_move = recent_returns.mean()
    direction = 1 if pred_class == "UP" else -1
    projected_price = current_price * (1 + direction * abs(avg_move))

    st.subheader("📈 Projection de prix")
    proj_col1, proj_col2 = st.columns(2)
    proj_col1.metric("Prix projeté dans 4h", f"${projected_price:,.2f}", f"{direction * abs(avg_move):+.2%}")
    proj_col2.metric("Variation estimée", f"{projected_price - current_price:+,.2f}")

    fig = go.Figure()
    plot_data = data.iloc[-12:]
    fig.add_trace(go.Candlestick(
        x=plot_data.index,
        open=plot_data['open'],
        high=plot_data['high'],
        low=plot_data['low'],
        close=plot_data['close'],
        name="BTC/USD"
    ))
    last_time = plot_data.index[-1]
    next_time = last_time + timedelta(hours=4)
    fig.add_trace(go.Scatter(
        x=[last_time, next_time],
        y=[current_price, projected_price],
        mode='lines+markers',
        line=dict(color='orange', width=2, dash='dot'),
        marker=dict(size=8, color='orange'),
        name="Projection"
    ))
    fig.update_layout(
        title="Évolution du BTC/USD (4h) et projection",
        xaxis_title="Date",
        yaxis_title="Prix (USD)",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Indicateurs techniques récents"):
        df = data.copy()
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        df['macd'] = ta.trend.MACD(close=df['close']).macd()
        df['macd_signal'] = ta.trend.MACD(close=df['close']).macd_signal()
        last_indicators = df.iloc[-1][['rsi', 'macd', 'macd_signal']]
        st.dataframe(last_indicators.to_frame().T)
else:
    st.error("Impossible d'obtenir une prédiction. Vérifiez que l'API est en ligne.")
