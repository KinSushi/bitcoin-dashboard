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
# MULTI-SOURCE DATA (robuste)
# =========================
@st.cache_data(ttl=60)
def load_recent_data():
    """
    Essaye de récupérer les bougies 4h BTC/USD depuis plusieurs sources.
    Ordre : Binance → CoinGecko → Kraken → yfinance → Bybit → Bitstamp.
    En cas d'échec total, utilise un fallback synthétique avec avertissement.
    Retourne (DataFrame, warning_message, source_name).
    """
    # ---------- 1. Binance ----------
    try:
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": "BTCUSDT", "interval": "4h", "limit": 42}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError("Empty Binance response")
        df = pd.DataFrame(raw, columns=[
            "open_time","open","high","low","close","volume",
            "close_time","quote_volume","count","taker_buy_base","taker_buy_quote","ignore"
        ])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("open_time")[["open","high","low","close","volume"]]
        df = df.dropna()
        if not df.empty:
            return df, None, "Binance ✅"
    except Exception as e:
        st.caption(f"Binance indisponible : {e}")

    # ---------- 2. CoinGecko ----------
    try:
        url = "https://api.coingecko.com/api/v3/coins/bitcoin/ohlc"
        params = {"vs_currency": "usd", "days": 7}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if not isinstance(raw, list) or len(raw) == 0:
            raise ValueError("Empty CoinGecko response")
        df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = 100.0  # CoinGecko ne fournit pas le volume
        df = df.set_index("timestamp")[["open","high","low","close","volume"]]
        if not df.empty:
            return df, None, "CoinGecko ✅"
    except Exception as e:
        st.caption(f"CoinGecko indisponible : {e}")

    # ---------- 3. Kraken ----------
    try:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {"pair": "XBTUSD", "interval": 240}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if "result" not in raw or "XBTUSD" not in raw["result"]:
            raise ValueError("Invalid Kraken response")
        ohlc = raw["result"]["XBTUSD"]
        if not ohlc:
            raise ValueError("Empty Kraken data")
        df = pd.DataFrame(ohlc, columns=["time","open","high","low","close","vwap","volume","count"])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("time")[["open","high","low","close","volume"]]
        if not df.empty:
            return df, None, "Kraken ✅"
    except Exception as e:
        st.caption(f"Kraken indisponible : {e}")

    # ---------- 4. yfinance ----------
    try:
        ticker = yf.download("BTC-USD", period="7d", interval="4h", progress=False)
        if not ticker.empty:
            ticker.columns = [c.lower() for c in ticker.columns]
            df = ticker[["open","high","low","close","volume"]].copy()
            df.index.name = "timestamp"
            if not df.empty:
                return df, None, "Yahoo Finance ✅"
    except Exception as e:
        st.caption(f"yfinance indisponible : {e}")

    # ---------- 5. Bybit ----------
    try:
        url = "https://api.bybit.com/v5/market/kline"
        params = {
            "category": "spot",
            "symbol": "BTCUSDT",
            "interval": "240",
            "limit": 42
        }
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if raw.get("retCode") != 0 or "result" not in raw or "list" not in raw["result"]:
            raise ValueError("Invalid Bybit response")
        ohlc = raw["result"]["list"]
        if not ohlc:
            raise ValueError("Empty Bybit data")
        df = pd.DataFrame(ohlc, columns=["timestamp","open","high","low","close","volume","turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="ms")
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("timestamp")[["open","high","low","close","volume"]].sort_index()
        if not df.empty:
            return df, None, "Bybit ✅"
    except Exception as e:
        st.caption(f"Bybit indisponible : {e}")

    # ---------- 6. Bitstamp ----------
    try:
        url = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
        params = {"step": 14400, "limit": 42}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if "data" not in raw or "ohlc" not in raw["data"]:
            raise ValueError("Invalid Bitstamp response")
        ohlc = raw["data"]["ohlc"]
        if not ohlc:
            raise ValueError("Empty Bitstamp data")
        df = pd.DataFrame(ohlc, columns=["timestamp","open","high","low","close","volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(float), unit="s")
        for col in ["open","high","low","close","volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.set_index("timestamp")[["open","high","low","close","volume"]].sort_index()
        if not df.empty:
            return df, None, "Bitstamp ✅"
    except Exception as e:
        st.caption(f"Bitstamp indisponible : {e}")

    # ---------- Fallback synthétique ----------
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
    return df, "⚠️ Données simulées – Aucune source de données réelle disponible", "Fallback 🔄"

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
data, warning, source_name = load_recent_data()
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

# Afficher la source des données dans la colonne 3 (à côté de la prédiction plus tard)
# Pour l'instant, on garde la place pour la prédiction

# =========================
# MARKET STATE (amélioré)
# =========================
regime, vol, drift = market_regime(data)

# Jauge de volatilité (échelle 0 à 0.03 max)
vol_pct = min(vol / 0.03, 1.0)

if regime == "LOW_VOL":
    regime_emoji = "🟢"
    regime_text = "faible"
elif regime == "MID_VOL":
    regime_emoji = "🟠"
    regime_text = "moyenne"
else:
    regime_emoji = "🔴"
    regime_text = "élevée"

drift_display = f"{drift:+.6f}"
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
# PREDICTION (avec calculs unifiés)
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

# ----- Calcul du signal score et de la projection (une seule fois) -----
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

# Décision
if signal_score > 0.25:
    decision = "BUY"
elif signal_score < -0.25:
    decision = "SELL"
else:
    decision = "HOLD"

# Historique
st.session_state.history.append({
    "time": datetime.now().strftime("%H:%M:%S"),
    "label": label,
    "confidence": round(proba, 3),
    "price": current,
    "regime": regime,
    "source": source_name
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
    st.metric("🎯 Prix projeté", f"${projected:,.2f}")
    # Source des données
    st.caption(f"Source : {source_name}")

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

score_display = (signal_score + 1) / 2

d1, d2, d3 = st.columns(3)

with d1:
    st.metric("Signal Score", f"{signal_score:.3f}")
    st.caption("Composite (ML + tendance)")
    st.progress(min(max(score_display, 0.0), 1.0))

with d2:
    if decision == "BUY":
        decision_emoji = "🟢"
    elif decision == "SELL":
        decision_emoji = "🔴"
    else:
        decision_emoji = "⚪"
    st.metric("Action", f"{decision_emoji} {decision}")
    st.caption(f"Confiance ML : {raw_proba:.1%}")

with d3:
    st.metric("Régime", f"{regime_emoji} {regime_text}")
    st.caption(f"Volatilité : {vol:.5f}")

# =========================
# CHART (chandeliers + volume + projection)
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
# HISTORY (enrichi)
# =========================
if st.session_state.history:
    st.subheader("📋 Dernières prédictions")
    hist_df = pd.DataFrame(st.session_state.history)
    # Mise en forme conditionnelle (couleurs)
    def color_label(val):
        color = 'green' if val == 'UP' else 'red' if val == 'DOWN' else 'gray'
        return f'color: {color}; font-weight: bold'
    st.dataframe(hist_df.style.applymap(color_label, subset=['label']), use_container_width=True)

# =========================
# INDICATORS (ajout ATR, StochRSI, etc.)
# =========================
with st.expander("📊 Indicateurs techniques avancés"):
    tmp = data.copy()
    if not tmp.empty:
        # RSI
        tmp["rsi"] = ta.momentum.RSIIndicator(tmp["close"]).rsi()
        # MACD
        macd = ta.trend.MACD(tmp["close"])
        tmp["macd"] = macd.macd()
        tmp["macd_signal"] = macd.macd_signal()
        tmp["macd_diff"] = macd.macd_diff()
        # ATR
        tmp["atr"] = ta.volatility.AverageTrueRange(tmp["high"], tmp["low"], tmp["close"], window=14).average_true_range()
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(tmp["close"], window=20, window_dev=2)
        tmp["bb_high"] = bb.bollinger_hband()
        tmp["bb_low"] = bb.bollinger_lband()
        # StochRSI
        stochrsi = ta.momentum.StochRSIIndicator(tmp["close"])
        tmp["stochrsi"] = stochrsi.stochrsi()

        latest = tmp.iloc[-1]
        indicators = {
            "RSI": latest["rsi"],
            "MACD": latest["macd"],
            "Signal MACD": latest["macd_signal"],
            "MACD Diff": latest["macd_diff"],
            "ATR": latest["atr"],
            "BB High": latest["bb_high"],
            "BB Low": latest["bb_low"],
            "StochRSI": latest["stochrsi"]
        }
        st.dataframe(pd.DataFrame(indicators, index=["Valeur"]).T.style.format("{:.4f}"))
    else:
        st.write("Données insuffisantes")

# =========================
# STATUT API ET RAFRAÎCHISSEMENT
# =========================
with st.sidebar:
    st.markdown("---")
    st.subheader("🔌 Statut")
    st.write(f"API : {'🟢 connectée' if pred else '🔴 hors ligne'}")
    st.write(f"Source données : {source_name}")
    st.write(f"Dernière mise à jour : {datetime.utcnow().strftime('%H:%M:%S UTC')}")
    if st.button("🔄 Rafraîchir les données"):
        st.cache_data.clear()
        st.rerun()