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
if "equity" not in st.session_state:
    st.session_state.equity = [1000]
if "prev_price" not in st.session_state:
    st.session_state.prev_price = None

# =========================
# BINANCE DATA – réel uniquement
# =========================
@st.cache_data(ttl=60)
def load_recent_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "4h", "limit": 42}
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
            raise ValueError("Empty DataFrame")
        return df, None
    except Exception as e:
        st.error(f"❌ Impossible de récupérer les données réelles depuis Binance : {e}")
        st.stop()

# =========================
# MARKET REGIME
# =========================
def market_regime(df):
    r = df["close"].pct_change().dropna()
    if len(r) < 5:
        return "UNKNOWN", 0.0, 0.0
    vol = float(r.std())
    drift = float(r.mean())
    if vol < 0.008:
        regime = "LOW_VOL"
    elif vol < 0.02:
        regime = "MID_VOL"
    else:
        regime = "HIGH_VOL"
    return regime, vol, drift

# =========================
# API SAFE CALL
# =========================
def safe_predict():
    try:
        r = requests.get(f"{API_URL}/predict-live", timeout=10)
        if r.status_code != 200:
            return None
        j = r.json()
        if "label" not in j or "probability_up" not in j:
            return None
        return j
    except:
        return None

# =========================
# LOAD DATA
# =========================
data, warning = load_recent_data()
if warning:
    st.warning(warning)

current = float(data["close"].iloc[-1])
prev = float(data["close"].iloc[-2])

# =========================
# METRICS
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("💰 BTC", f"${current:,.2f}")
col2.metric("📈 4H Δ", f"{current-prev:+.2f}", f"{(current-prev)/prev:+.2%}")

# =========================
# MARKET STATE
# =========================
regime, vol, drift = market_regime(data)
regime_label = {
    "LOW_VOL": "🟢 LOW VOL",
    "MID_VOL": "🟠 MID VOL",
    "HIGH_VOL": "🔴 HIGH VOL",
    "UNKNOWN": "⚪ UNKNOWN"
}.get(regime, "⚪ UNKNOWN")

st.subheader("🧭 Market State")
cr1, cr2, cr3 = st.columns(3)
cr1.metric("Regime", regime_label)
cr2.metric("Volatility", f"{vol:.5f}")
cr3.metric("Drift", f"{drift:.6f}")

# =========================
# PREDICTION (safe init)
# =========================
pred = safe_predict()
raw_proba = 0.5
label = "HOLD"

if pred:
    label = pred.get("label", "HOLD")
    raw_proba = float(pred.get("probability_up", 0.5))

if regime in ("MID_VOL", "HIGH_VOL"):
    adjustment = max(0.6, 1 - vol * 10)
    proba = raw_proba * adjustment
else:
    proba = raw_proba
proba = min(max(proba, 0.0), 1.0)

st.session_state.history.append({
    "time": datetime.now().strftime("%H:%M:%S"),
    "label": label,
    "confidence": round(proba, 3),
    "price": current,
    "regime": regime
})
st.session_state.history = st.session_state.history[-10:]

with col3:
    st.markdown("### 🔮 Prediction")
    color = "#00c853" if label == "UP" else "#ff1744"
    st.markdown(f"**<span style='color:{color};font-size:1.4em;'>{label}</span>**", unsafe_allow_html=True)
    progress_val = min(max(int(proba * 100), 0), 100)
    st.progress(progress_val)
    st.caption(f"Confidence: {proba:.1%}")

# =========================
# SIGNAL SCORE & DECISION
# =========================
regime_factor = {"LOW_VOL": 1.05, "MID_VOL": 1.0, "HIGH_VOL": 0.85}.get(regime, 1.0)
drift_norm = np.tanh(drift * 50)
signal_score = (raw_proba - 0.5) * 2 * 0.7 * regime_factor + drift_norm * 0.3

if signal_score > 0.25:
    decision = "BUY"
elif signal_score < -0.25:
    decision = "SELL"
else:
    decision = "HOLD"

# =========================
# PROJECTION
# =========================
returns = data["close"].pct_change().dropna()
if len(returns) > 0:
    drift_proj = returns.mean()
    vol_proj = returns.std()
    expected_move = drift_proj + vol_proj * np.sign(signal_score)
else:
    expected_move = 0.0

projected = current * (1 + expected_move)

st.subheader("📈 Projection 4H")
st.metric("Prix projeté", f"${projected:,.2f}")

# =========================
# BACKTEST (exposition 10%)
# =========================
prev_equity = st.session_state.equity[-1]
if st.session_state.prev_price is None:
    st.session_state.prev_price = prev

price_change = (current - st.session_state.prev_price) / st.session_state.prev_price

pnl = 0
if decision == "BUY":
    pnl = prev_equity * 0.1 * price_change
elif decision == "SELL":
    pnl = -prev_equity * 0.1 * price_change

new_equity = prev_equity + pnl
st.session_state.equity.append(new_equity)
st.session_state.equity = st.session_state.equity[-100:]

st.session_state.prev_price = current

# =========================
# DECISION PANEL
# =========================
st.subheader("🧭 Decision Engine")
d1, d2, d3 = st.columns(3)
d1.metric("Regime", regime)
d2.metric("Signal Score", f"{signal_score:.3f}")
d3.metric("Action", decision)

# =========================
# CHART
# =========================
plot = data.tail(24)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.7, 0.3], vertical_spacing=0.05)

fig.add_trace(go.Candlestick(
    x=plot.index,
    open=plot["open"],
    high=plot["high"],
    low=plot["low"],
    close=plot["close"],
    name="BTC/USD"
), row=1, col=1)

colors = np.where(plot["close"] >= plot["open"], "green", "red")
fig.add_trace(go.Bar(
    x=plot.index,
    y=plot["volume"],
    marker_color=colors,
    name="Volume"
), row=2, col=1)

fig.add_trace(go.Scatter(
    x=[plot.index[-1], plot.index[-1] + timedelta(hours=4)],
    y=[current, projected],
    mode="lines+markers",
    line=dict(color="orange", dash="dot"),
    name="Projection"
), row=1, col=1)

fig.update_layout(height=600, template="plotly_white", showlegend=False)
st.plotly_chart(fig, use_container_width=True)

# =========================
# EQUITY & RISK
# =========================
eq = np.array(st.session_state.equity)
peak = np.maximum.accumulate(eq)
drawdown = np.zeros_like(eq)
valid = peak != 0
drawdown[valid] = (eq[valid] - peak[valid]) / peak[valid]
max_dd = np.min(drawdown[valid]) if np.any(valid) else 0.0

col_eq1, col_eq2 = st.columns(2)
with col_eq1:
    st.subheader("📈 Equity Curve")
    fig_eq = go.Figure()
    fig_eq.add_trace(go.Scatter(y=eq, mode="lines", line=dict(color="blue")))
    fig_eq.update_layout(height=300, template="plotly_white", showlegend=False)
    st.plotly_chart(fig_eq, use_container_width=True)

with col_eq2:
    st.subheader("📉 Risk Metrics")
    st.metric("Max Drawdown", f"{max_dd:.2%}")
    st.metric("Current Equity", f"${eq[-1]:.2f}")

# =========================
# HISTORY
# =========================
if st.session_state.history:
    st.subheader("📋 History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

# =========================
# INDICATORS
# =========================
with st.expander("📊 Indicators"):
    tmp = data.copy()
    if not tmp.empty:
        tmp["rsi"] = ta.momentum.RSIIndicator(tmp["close"]).rsi()
        macd = ta.trend.MACD(tmp["close"])
        tmp["macd"] = macd.macd()
        tmp["signal"] = macd.macd_signal()
        st.dataframe(tmp.tail(1)[["rsi","macd","signal"]])
    else:
        st.write("Données insuffisantes")