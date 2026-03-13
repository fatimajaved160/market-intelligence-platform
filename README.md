# 📈 Market Intelligence Platform

A real-time financial market intelligence platform powered by Google BigQuery, 
FinBERT NLP, and LSTM deep learning. Built as a portfolio project demonstrating 
end-to-end data engineering, NLP, and machine learning skills.

## 🔗 Live Dashboard
**[market-intelligence-platform-2026.streamlit.app](https://market-intelligence-platform-2026.streamlit.app)**

## 🏗️ Architecture
```
Data Sources          Cloud Database        ML Models          Dashboard
────────────          ──────────────        ─────────          ─────────
Yahoo Finance   ───►  Google BigQuery  ───► LSTM Forecast ───► Streamlit
NewsAPI         ───►  prices table     ───► FinBERT NLP   ───► Plotly Charts
                      news_sentiment                           Live URL
                      predictions
```

## 📊 Live Results

| Asset | LSTM Accuracy | Current Sentiment |
|---|---|---|
| S&P 500 | 98.81% | 🔴 Negative |
| Oil | 97.79% | 🔴 Negative |
| Bitcoin | 96.87% | 🟡 Neutral |
| Gold | 91.12% | 🟢 Positive |

## 🔍 Project Overview

This platform automatically:
1. Pulls daily prices for Oil, Gold, S&P500 and Bitcoin via Yahoo Finance
2. Pulls financial news headlines via NewsAPI
3. Stores everything in Google BigQuery cloud database
4. Runs FinBERT sentiment analysis on news headlines
5. Forecasts 30-day price movements using LSTM neural networks
6. Displays everything on a live interactive Streamlit dashboard

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data Collection | Python, yfinance, NewsAPI |
| Cloud Database | Google BigQuery |
| NLP | FinBERT (ProsusAI/finbert) |
| Deep Learning | PyTorch LSTM |
| Dashboard | Streamlit, Plotly |
| Deployment | Streamlit Cloud |
| Version Control | Git, GitHub |

## 📁 Project Structure
```
market-intelligence-platform/
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_news_sentiment.ipynb
│   ├── 03_analysis.ipynb
│   └── 04_lstm_forecasting.ipynb
├── dashboard/
│   └── app.py
├── reports/
│   ├── correlation_heatmap.png
│   ├── sentiment_vs_returns.png
│   └── lstm_all_assets.png
├── requirements.txt
└── README.md
```

## 🔑 Key Findings

**Asset Correlations:**
- Bitcoin and S&P500 most correlated (r=0.313) — both risk-on assets
- Gold nearly uncorrelated with everything — true safe haven
- Oil moves independently — driven by geopolitical supply/demand

**Sentiment Analysis (March 2026):**
- S&P500 most negative sentiment (-0.263) — bearish market mood
- Gold most positive (0.106) — safe haven demand rising
- 383 headlines analysed across 4 assets using FinBERT

**LSTM Forecasts:**
- S&P500 most predictable (98.81% accuracy)
- Gold hardest to predict (91.12%) — reacts to unpredictable macro events
- All models trained on 3 years of daily price data (3,568 rows)

## ☁️ Google BigQuery Schema

**prices table** — 3,568 rows
- date, asset, ticker, open, high, low, close, volume, daily_return, volatility_30

**news_sentiment table** — 383 rows
- date, asset, headline, source, sentiment_label, sentiment_score, sentiment_numeric

**predictions table** — 120 rows
- date, asset, predicted_price, model, accuracy

## 🚀 Run Locally
```bash
git clone https://github.com/fatimajaved160/market-intelligence-platform.git
cd market-intelligence-platform
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

*Note: Requires Google Cloud credentials with BigQuery access*

---

*Project by Fatima Javed | Data Science MSc*  
*Tools: Python · BigQuery · PyTorch · FinBERT · Streamlit · Plotly*
