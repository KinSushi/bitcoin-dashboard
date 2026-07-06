import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import ta

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Bitcoin Predictor Pro", page_icon="₿", layout="wide")
API_URL = "https://KinSushi-bitcoin-predictor-pro.hf.space"

# =========================
# STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []
if "equity" not in st.session_state:
    st.session_state.equity = [1000]
if "prev_price" not in st.session_state:
    st.session_state.prev_price = None


# =========================
# DATA
# =========================
@st.cache_data(ttl=60)
def load_recent_data():
    try:
        df = yf.download("BTC-USD", period="7d", interval="4h", progress=False)
        if df.empty:
            raise ValueError()
        df = df.rename(columns=str.lower)[["open","high","low","close","volume"]]
        return df, None, "Yahoo Finance"
    except Exception:
        np.random.seed(42)
        dates = pd.date_range(end=datetime.utcnow(), periods=42, freq="4h")
        close = 50000 + np.random.normal(0, 200, 42).cumsum()
        df = pd.DataFrame({
            "open": np.roll(close, 1),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.random.exponential(100, 42)
        }, index=dates)
        return df, "fallback actif", "SIMULATION"


# =========================
# REGIME
# =========================
def market_regime(df):
    r = df["close"].pct_change().dropna()
    vol = r.std() if len(r) else 0
    drift = r.mean() if len(r) else 0

    if vol < 0.008:
        regime = "LOW_VOL"
    elif vol < 0.02:
        regime = "MID_VOL"
    else:
        regime = "HIGH_VOL"

    return regime, float(vol), float(drift)


# =========================
# API
# =========================
def safe_predict():
    try:
        r = requests.get(f"{API_URL}/predict-live", timeout=8)
        if r.status_code != 200:
            return None
        j = r.json()
        return j if "label" in j else None
    except:
        return None


# =========================
# LOAD
# =========================
data, warn, source = load_recent_data()
if warn:
    st.warning(warn)

current = float(data["close"].iloc[-1])
prev = float(data["close"].iloc[-2])

# =========================
# UI TOP
# =========================
c1, c2, c3 = st.columns(3)
c1.metric("BTC", f"${current:,.2f}")
c2.metric("Δ 4H", f"{current-prev:+.2f}", f"{(current-prev)/prev:+.2%}")


# =========================
# REGIME
# =========================
regime, vol, drift = market_regime(data)

st.subheader("Market State")

r1, r2, r3 = st.columns(3)
r1.metric("Regime", regime)
r2.metric("Volatility", f"{vol:.5f}")
r3.metric("Drift", f"{drift:.6f}")


# =========================
# PREDICTION
# =========================
pred = safe_predict()
raw_proba = 0.5
label = "HOLD"

if pred:
    label = pred.get("label", "HOLD")
    raw_proba = float(pred.get("probability_up", 0.5))


adjust = max(0.6, 1 - vol * 10) if regime != "LOW_VOL" else 1.0
proba = np.clip(raw_proba * adjust, 0, 1)


# =========================
# SIGNAL SCORE (stable)
# =========================
regime_factor = {"LOW_VOL":1.05,"MID_VOL":1.0,"HIGH_VOL":0.85}[regime]

drift_norm = np.tanh(drift * 100)

signal_score = (raw_proba - 0.5) * regime_factor + 0.3 * drift_norm

if signal_score > 0.2:
    decision = "BUY"
elif signal_score < -0.2:
    decision = "SELL"
else:
    decision = "HOLD"


# =========================
# PROJECTION (FIXED SIGN BUG)
# =========================
returns = data["close"].pct_change().dropna()

if len(returns):
    expected_move = returns.mean() + returns.std() * (1 if signal_score > 0 else -1)
else:
    expected_move = 0

projected = current * (1 + expected_move)


# =========================
# BACKTEST (stable)
# =========================
equity = st.session_state.equity[-1]
exposure = 0.1

if st.session_state.prev_price is None:
    st.session_state.prev_price = prev

price_delta = (current - st.session_state.prev_price) / st.session_state.prev_price

pnl = 0
if decision == "BUY":
    pnl = equity * exposure * price_delta
elif decision == "SELL":
    pnl = -equity * exposure * price_delta

equity += pnl
st.session_state.equity.append(equity)
st.session_state.prev_price = current


# =========================
# UI PREDICTION (NO HTML)
# =========================
st.subheader("Prediction")

colA, colB, colC = st.columns(3)

colA.metric("Signal", label)
colB.progress(int(proba * 100))
colC.metric("Confidence", f"{proba:.1%}")


# =========================
# DECISION
# =========================
st.subheader("Decision Engine")

d1, d2, d3 = st.columns(3)
d1.metric("Signal Score", f"{signal_score:.3f}")
d2.metric("Action", decision)
d3.metric("Projection", f"${projected:,.2f}")


# =========================
# CHART
# =========================
plot = data.tail(24)

fig = make_subplots(rows=2, cols=1, shared_xaxes=True)

fig.add_trace(go.Candlestick(
    x=plot.index,
    open=plot["open"],
    high=plot["high"],
    low=plot["low"],
    close=plot["close"]
), row=1, col=1)

fig.add_trace(go.Bar(
    x=plot.index,
    y=plot["volume"]
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=[plot.index[-1], plot.index[-1] + timedelta(hours=4)],
    y=[current, projected],
    mode="lines"
), row=1, col=1)

fig.update_layout(height=600, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)


# =========================
# EQUITY
# =========================
st.subheader("Equity")
st.line_chart(st.session_state.equity)


# =========================
# HISTORY SAFE
# =========================
st.session_state.history.append({
    "time": datetime.now().strftime("%H:%M:%S"),
    "label": label,
    "confidence": float(proba),
    "price": current,
    "regime": regime
})

st.dataframe(pd.DataFrame(st.session_state.history[-10:]))


# =========================
# INDICATORS
# =========================
with st.expander("Indicators"):
    tmp = data.copy()
    tmp["rsi"] = ta.momentum.RSIIndicator(tmp["close"]).rsi()
    macd = ta.trend.MACD(tmp["close"])
    tmp["macd"] = macd.macd()

    st.dataframe(tmp.tail(1))