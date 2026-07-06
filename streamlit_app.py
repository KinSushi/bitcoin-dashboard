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
# MARKET STATE (amélioré)
# =========================
regime, vol, drift = market_regime(data)

# Jauge de volatilité (échelle 0 à 0.03 max)
vol_pct = min(vol / 0.03, 1.0)  # normalisé pour la barre

if regime == "LOW_VOL":
    regime_emoji = "🟢"
    regime_text = "faible"
    vol_color = "#00c853"
elif regime == "MID_VOL":
    regime_emoji = "🟠"
    regime_text = "moyenne"
    vol_color = "#ff9800"
else:
    regime_emoji = "🔴"
    regime_text = "élevée"
    vol_color = "#ff1744"

# Drift formatté
drift_display = f"{drift:+.6f}"  # garde le signe +/-
drift_emoji = "📈" if drift > 0 else "📉" if drift < 0 else "➖"

st.subheader("🧭 Market State")

colA, colB, colC = st.columns(3)

with colA:
    st.metric("Régime", f"{regime_emoji} {regime_text}")
    st.caption("Volatilité normalisée")
    st.progress(vol_pct)

with colB:
    st.metric("Volatilité", f"{vol:.5f}")
    st.caption("Écart-type des rendements 4h")

with colC:
    st.metric("Tendance (drift)", f"{drift_emoji} {drift_display}")
    st.caption("Rendement moyen sur la période")

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

# ----- Calcul précoce du signal et de la projection -----
regime_factor = {"LOW_VOL": 1.05, "MID_VOL": 1.0, "HIGH_VOL": 0.85}.get(regime, 1.0)
drift_norm = np.tanh(drift * 50)
signal_score = (raw_proba - 0.5) * 2 * 0.7 * regime_factor + drift_norm * 0.3

returns = data["close"].pct_change().dropna()
if len(returns) > 0:
    drift_proj = returns.mean()
    vol_proj = returns.std()
    expected_move = drift_proj + vol_proj * np.sign(signal_score)
else:
    expected_move = 0.0
projected = current * (1 + expected_move)
# --------------------------------------------------------

# Historique
st.session_state.history.append({
    "time": datetime.now().strftime("%H:%M:%S"),
    "label": label,
    "confidence": round(proba, 3),
    "price": current,
    "regime": regime
})
st.session_state.history = st.session_state.history[-10:]

# Bloc prédiction + prix projeté dans la colonne 3
with col3:
    st.markdown("### 🔮 Prediction")
    color = "#00c853" if label == "UP" else "#ff1744"
    st.markdown(f"**<span style='color:{color};font-size:1.4em;'>{label}</span>**", unsafe_allow_html=True)
    progress_val = min(max(int(proba * 100), 0), 100)
    st.progress(progress_val)
    st.caption(f"Confiance : {proba:.1%}")
    # ---- AJOUT DU PRIX PROJETÉ ----
    st.metric("🎯 Prix projeté", f"${projected:,.2f}")

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
# DECISION PANEL (amélioré)
# =========================
st.subheader("🧭 Decision Engine")

# Score visuel (normalisé entre 0 et 1 pour la jauge)
score_display = (signal_score + 1) / 2  # signal_score entre -1 et 1 → 0 à 1
score_color = "#00c853" if signal_score > 0.25 else "#ff9800" if signal_score > -0.25 else "#ff1744"

d1, d2, d3 = st.columns(3)

with d1:
    st.metric("Signal Score", f"{signal_score:.3f}")
    st.caption("Composite (ML + tendance)")
    st.progress(min(max(score_display, 0.0), 1.0))

with d2:
    if decision == "BUY":
        decision_emoji = "🟢"
        decision_color = "#00c853"
    elif decision == "SELL":
        decision_emoji = "🔴"
        decision_color = "#ff1744"
    else:
        decision_emoji = "⚪"
        decision_color = "#9e9e9e"

    st.metric("Action", f"{decision_emoji} {decision}")
    st.caption(f"Confiance ML : {raw_proba:.1%}")

with d3:
    # Affichage du régime (rappel, mais avec style)
    st.metric("Régime", f"{regime_emoji} {regime_text}")
    st.caption(f"Volatilité : {vol:.5f}")

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