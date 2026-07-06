---

## 📝 README

```
---
title: Bitcoin Dashboard
emoji: 🚀
colorFrom: yellow
colorTo: green
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---

# ₿ Bitcoin Predictor Pro

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://streamlit.io)
[![Python 3.11+](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-Space-blue)](https://huggingface.co/spaces/KinSushi/bitcoin-dashboard)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**Tableau de bord quantitatif de prédiction du Bitcoin (BTC/USD) à horizon 4 heures.**
Combine un modèle XGBoost entraîné sur 2 ans de données avec des indicateurs techniques et un moteur de décision (BUY/SELL/HOLD). Les données sont récupérées en temps réel depuis plusieurs sources redondantes.

## ✨ Fonctionnalités

- **Prédiction en direct** : direction UP/DOWN avec probabilité corrigée du régime de marché.
- **Projection de prix à 4h** : estimation réaliste basée sur la volatilité et le drift.
- **Moteur de décision** : BUY / SELL / HOLD avec score composite.
- **Indicateurs techniques avancés** : RSI, MACD, ATR, Bandes de Bollinger, StochRSI.
- **Multi‑source résiliente** : CoinGecko → Kraken → Yahoo Finance → Bybit → Bitstamp → Binance + fallback synthétique.
- **Backtest simulé** : equity curve, drawdown, exposition 10 %.
- **Historique des 10 dernières prédictions** (avec code couleur).
- **Statut API & source de données** visibles dans la barre latérale.
- **Rafraîchissement manuel** et cache de 60 secondes.

## 🖼️ Aperçu

![screenshot](https://ibb.co/jPk1vTWc)

## 🧱 Architecture

```
┌─────────────────┐
│  Streamlit App  │
│ (Hugging Face)  │
└────────┬────────┘
         │  /predict-live
         ▼
┌─────────────────┐
│   FastAPI (ML)  │  ← XGBoost model
│ (autre Space HF) │
└─────────────────┘
         ▲
         │  données 4h
┌────────┴────────┐
│ Multi‑source     │
│ CoinGecko/Kraken │
│ yfinance/Bybit.. │
└─────────────────┘
```

## 🛠️ Stack

- [Streamlit](https://streamlit.io) – interface web interactive
- [FastAPI](https://fastapi.tiangolo.com) – API de prédiction
- [XGBoost](https://xgboost.readthedocs.io) – modèle de classification
- [scikit‑learn](https://scikit-learn.org) – pipeline & préprocessing
- [pandas](https://pandas.pydata.org), [numpy](https://numpy.org) – manipulation des données
- [yfinance](https://github.com/ranaroussi/yfinance), `requests` – récupération multi‑source
- [ta](https://technical-analysis-library-in-python.readthedocs.io) – indicateurs techniques
- [Plotly](https://plotly.com) – graphiques interactifs

## 🚀 Utilisation

### Dashboard public

Rendez‑vous sur **[Hugging Face Space](https://huggingface.co/spaces/KinSushi/bitcoin-dashboard)**.

### API (backend ML)

L’API est accessible séparément :  
`https://KinSushi-bitcoin-predictor-pro.hf.space`

**Endpoints principaux :**

| Méthode | Chemin         | Description                          |
|---------|----------------|--------------------------------------|
| GET     | `/health`      | État du modèle                       |
| GET     | `/predict-live`| Prédiction en direct                 |
| POST    | `/predict`     | Prédiction à partir de features      |
| GET     | `/model-info`  | Informations sur le modèle           |

Exemple d’appel :

```bash
curl https://KinSushi-bitcoin-predictor-pro.hf.space/predict-live
```

## 💻 Installation locale

```bash
git clone https://github.com/KinSushi/bitcoin-dashboard.git
cd bitcoin-dashboard
pip install -r requirements.txt
streamlit run streamlit_app.py
```

> **Note** : L’API doit être en ligne pour recevoir les prédictions. Vous pouvez aussi lancer l’API localement (voir le dépôt `bitcoin-predictor-pro`).

## 📊 Sources de données

Par ordre de priorité :
1. CoinGecko (prix réel, volume estimé)
2. Kraken
3. Yahoo Finance
4. Bybit
5. Bitstamp
6. Binance (souvent bloqué)
7. Fallback synthétique si aucune source n’est disponible

Un message discret indique la source utilisée.

## 🗺️ Roadmap

- [ ] Authentification par token
- [ ] Stockage des prédictions (SQLite)
- [ ] Indicateur de force du signal (Sharpe, win rate)
- [ ] Backtest walk‑forward avec paramètres optimisés
- [ ] Interface de comparaison multi‑timeframes

## 👤 Auteur

Développé par **KinSushi** — [GitHub](https://github.com/KinSushi)

## 📄 Licence

APACHE 2.0 © 2026 KinSushi
```

```powershell
git add README.md
git commit -m "Correction en-tête YAML et badges"
git push origin main
