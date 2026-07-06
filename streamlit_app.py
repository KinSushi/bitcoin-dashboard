import streamlit as st
import pandas as pd
import numpy as np
import requests
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

# =========================
# BINANCE DATA (FIXED)
# =========================
@st.cache_data(ttl=60)
def load_recent_data():
    url = "https://api.binance.com/api/v3/klines"

    params = {
        "symbol": "BTCUSDT",
        "interval": "4h",
        "limit": 42
    }

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()

        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError("Empty response")

        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","qv","n","tbbv","tbqv","ignore"
        ])

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")

        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.set_index("open_time")[["open","high","low","close","volume"]]
        df = df.dropna()

        if df.empty:
            raise ValueError("Parsed empty dataframe")

        return df, None

    except Exception as e:
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

        return df, "⚠️ Binance indisponible → fallback actif"

# =========================
# API SAFE
# =========================
def safe_predict():
    try:
        r = requests.get(f"{API_URL}/predict-live", timeout=10)
        if r.status_code != 200:
            return None

        data = r.json()

        if "label" not in data or "probability_up" not in data:
            return None

        return data

    except Exception:
        return None

# =========================
# LOAD
# =========================
data, warning = load_recent_data()

if warning:
    st.warning(warning)

# =========================
# METRICS
# =========================
current = float(data["close"].iloc[-1])
prev = float(data["close"].iloc[-2])

change = current - prev
pct = change / prev

c1, c2, c3 = st.columns(3)

c1.metric("BTC", f"${current:,.2f}")
c2.metric("4H Δ", f"{change:+.2f}", f"{pct:+.2%}")

# =========================
# PREDICTION
# =========================
pred = safe_predict()

if pred:
    label = pred["label"]
    proba = float(pred["probability_up"])

    st.session_state.history = st.session_state.history[-9:] + [{
        "time": datetime.now().strftime("%H:%M:%S"),
        "label": label,
        "confidence": proba
    }]

    color = "#00c853" if label == "UP" else "#ff1744"

    # SAFE UI (NO RAW DIV DOM COMPLEX)
    with c3:
        st.markdown("### 🔮 Prediction")

        st.markdown(
            f"**{label}**",
            unsafe_allow_html=False
        )

        st.progress(int(proba * 100))

        st.write(f"Confidence: {proba:.1%}")

else:
    label = "DOWN"
    proba = 0.0
    c3.error("API offline")

# =========================
# PROJECTION (SAFE)
# =========================
returns = data["close"].pct_change().dropna()

drift = float(returns.mean()) if len(returns) > 0 else 0.0
direction = 1 if label == "UP" else -1

projected = current * (1 + direction * abs(drift))

st.subheader("Projection 4H")
st.metric("Target", f"${projected:,.2f}")

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

colors = np.where(plot["close"] >= plot["open"], "green", "red")

fig.add_trace(go.Bar(
    x=plot.index,
    y=plot["volume"],
    marker_color=colors
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=[plot.index[-1], plot.index[-1] + timedelta(hours=4)],
    y=[current, projected],
    mode="lines+markers",
    line=dict(color="orange", dash="dot")
), row=1, col=1)

fig.update_layout(height=600, template="plotly_white", showlegend=False)

st.plotly_chart(fig, use_container_width=True)

# =========================
# HISTORY
# =========================
if st.session_state.history:
    st.subheader("History")
    st.dataframe(pd.DataFrame(st.session_state.history))

# =========================
# INDICATORS
# =========================
with st.expander("Indicators"):
    tmp = data.copy()

    tmp["rsi"] = ta.momentum.RSIIndicator(tmp["close"]).rsi()
    macd = ta.trend.MACD(tmp["close"])

    tmp["macd"] = macd.macd()
    tmp["signal"] = macd.macd_signal()

    st.dataframe(tmp.tail(1)[["rsi","macd","signal"]])