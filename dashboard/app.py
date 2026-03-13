import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from google.cloud import bigquery
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Market Intelligence Platform",
    page_icon="📈",
    layout="wide"
)

# ============================================================
# BIGQUERY CONNECTION
# ============================================================

@st.cache_resource
def get_client():
    import json
    from google.oauth2.credentials import Credentials
    
    creds_dict = {
        "client_id": st.secrets["gcp_credentials"]["client_id"],
        "client_secret": st.secrets["gcp_credentials"]["client_secret"],
        "refresh_token": st.secrets["gcp_credentials"]["refresh_token"],
        "token_uri": "https://oauth2.googleapis.com/token",
        "quota_project_id": st.secrets["gcp_credentials"]["quota_project_id"]
    }
    
    credentials = Credentials(
        token=None,
        refresh_token=creds_dict["refresh_token"],
        token_uri=creds_dict["token_uri"],
        client_id=creds_dict["client_id"],
        client_secret=creds_dict["client_secret"],
        quota_project_id=creds_dict["quota_project_id"]
    )
    
    return bigquery.Client(
        project="project-a13303d6-d497-41fa-ab8",
        credentials=credentials
    )
client = get_client()

# ============================================================
# LOAD DATA FROM BIGQUERY
# ============================================================

@st.cache_data(ttl=3600)
def load_prices():
    return client.query("""
        SELECT * FROM `project-a13303d6-d497-41fa-ab8.market_intelligence.prices`
        ORDER BY asset, date
    """).to_dataframe()

@st.cache_data(ttl=3600)
def load_sentiment():
    return client.query("""
        SELECT * FROM `project-a13303d6-d497-41fa-ab8.market_intelligence.news_sentiment`
        ORDER BY asset, date
    """).to_dataframe()

@st.cache_data(ttl=3600)
def load_predictions():
    return client.query("""
        SELECT * FROM `project-a13303d6-d497-41fa-ab8.market_intelligence.predictions`
        ORDER BY asset, date
    """).to_dataframe()

# Load all data
prices = load_prices()
sentiment = load_sentiment()
predictions = load_predictions()

prices["date"] = pd.to_datetime(prices["date"])
sentiment["date"] = pd.to_datetime(sentiment["date"])
predictions["date"] = pd.to_datetime(predictions["date"])

# ============================================================
# HEADER
# ============================================================

st.title("📈 Market Intelligence Platform")
st.markdown("*Real-time financial market analysis powered by BigQuery, FinBERT & LSTM*")
st.divider()

# ============================================================
# ROW 1 — KEY METRICS
# ============================================================

st.subheader("🔢 Current Market Snapshot")

col1, col2, col3, col4 = st.columns(4)

assets_info = {
    "Oil": {"emoji": "🛢️", "color": "steelblue"},
    "Gold": {"emoji": "🥇", "color": "gold"},
    "SP500": {"emoji": "📊", "color": "green"},
    "Bitcoin": {"emoji": "₿", "color": "orange"}
}

for col, asset in zip([col1, col2, col3, col4], ["Oil", "Gold", "SP500", "Bitcoin"]):
    asset_prices = prices[prices["asset"] == asset].sort_values("date")
    current_price = asset_prices["close"].iloc[-1]
    prev_price = asset_prices["close"].iloc[-2]
    change_pct = ((current_price - prev_price) / prev_price) * 100
    arrow = "▲" if change_pct > 0 else "▼"
    
    # Get sentiment
    asset_sent = sentiment[sentiment["asset"] == asset]
    recent_sent = asset_sent.sort_values("date").tail(20)
    avg_sent = recent_sent["sentiment_numeric"].mean()
    sent_label = "🟢 Positive" if avg_sent > 0.1 else "🔴 Negative" if avg_sent < -0.1 else "🟡 Neutral"
    
    col.metric(
        label=f"{assets_info[asset]['emoji']} {asset}",
        value=f"${current_price:,.2f}",
        delta=f"{arrow} {abs(change_pct):.2f}%"
    )
    col.caption(f"Sentiment: {sent_label}")

st.divider()

# ============================================================
# ROW 2 — PRICE CHARTS
# ============================================================

st.subheader("📉 Price History")

selected_asset = st.selectbox("Select Asset", ["Oil", "Gold", "SP500", "Bitcoin"])

asset_data = prices[prices["asset"] == selected_asset].sort_values("date")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=asset_data["date"],
    y=asset_data["close"],
    name="Price",
    line=dict(color="steelblue", width=1.5)
))

# Add 30 day MA
asset_data["MA30"] = asset_data["close"].rolling(30).mean()
fig.add_trace(go.Scatter(
    x=asset_data["date"],
    y=asset_data["MA30"],
    name="30-Day MA",
    line=dict(color="orange", width=1.5, dash="dash")
))

fig.update_layout(
    title=f"{selected_asset} Price History",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    height=400,
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ============================================================
# ROW 3 — CORRELATION & SENTIMENT
# ============================================================

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🔗 Asset Correlation")
    prices_pivot = prices.pivot_table(
        index="date", columns="asset", values="close"
    ).pct_change().corr()
    
    fig_corr = px.imshow(
        prices_pivot,
        color_continuous_scale="RdYlGn",
        zmin=-1, zmax=1,
        text_auto=".2f",
        title="Return Correlation Matrix"
    )
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

with col_right:
    st.subheader("📰 News Sentiment")
    sent_summary = sentiment.groupby("asset")["sentiment_numeric"].mean().reset_index()
    sent_summary.columns = ["Asset", "Avg Sentiment"]
    sent_summary["Sentiment"] = sent_summary["Avg Sentiment"].apply(
        lambda x: "🟢 Positive" if x > 0.1 else "🔴 Negative" if x < -0.1 else "🟡 Neutral"
    )
    
    fig_sent = px.bar(
        sent_summary,
        x="Asset",
        y="Avg Sentiment",
        color="Avg Sentiment",
        color_continuous_scale="RdYlGn",
        title="Average News Sentiment by Asset",
        text="Sentiment"
    )
    fig_sent.update_layout(height=400)
    st.plotly_chart(fig_sent, use_container_width=True)

st.divider()

# ============================================================
# ROW 4 — LSTM FORECASTS
# ============================================================

st.subheader("🔮 30-Day LSTM Price Forecasts")

cols = st.columns(4)

for col, asset in zip(cols, ["Oil", "Gold", "SP500", "Bitcoin"]):
    asset_pred = predictions[predictions["asset"] == asset].sort_values("date")
    asset_prices_curr = prices[prices["asset"] == asset].sort_values("date")
    
    current = asset_prices_curr["close"].iloc[-1]
    forecast = asset_pred["predicted_price"].iloc[-1]
    change = ((forecast - current) / current) * 100
    accuracy = asset_pred["accuracy"].iloc[-1]
    direction = "▲" if change > 0 else "▼"
    
    col.metric(
        label=f"{assets_info[asset]['emoji']} {asset} (30d)",
        value=f"${forecast:,.2f}",
        delta=f"{direction} {abs(change):.1f}% from current"
    )
    col.caption(f"Model accuracy: {accuracy:.1f}%")

st.divider()

# ============================================================
# ROW 5 — FORECAST CHART
# ============================================================

st.subheader("📊 Forecast vs History")

forecast_asset = st.selectbox("Select Asset for Forecast", 
                               ["Oil", "Gold", "SP500", "Bitcoin"],
                               key="forecast_select")

hist_data = prices[prices["asset"] == forecast_asset].sort_values("date").tail(90)
pred_data = predictions[predictions["asset"] == forecast_asset].sort_values("date")

fig_forecast = go.Figure()

fig_forecast.add_trace(go.Scatter(
    x=hist_data["date"],
    y=hist_data["close"],
    name="Historical",
    line=dict(color="steelblue", width=2)
))

fig_forecast.add_trace(go.Scatter(
    x=pred_data["date"],
    y=pred_data["predicted_price"],
    name="LSTM Forecast",
    line=dict(color="red", width=2, dash="dash")
))

fig_forecast.update_layout(
    title=f"{forecast_asset} — Last 90 Days + 30 Day Forecast",
    xaxis_title="Date",
    yaxis_title="Price (USD)",
    height=400,
    hovermode="x unified"
)

st.plotly_chart(fig_forecast, use_container_width=True)

st.divider()

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
*Data sources: Yahoo Finance (prices), NewsAPI (headlines)*  
*Models: FinBERT (sentiment), LSTM (forecasting)*  
*Storage: Google BigQuery*  
*Built by Fatima Javed — Data Science MSc Portfolio Project*
""")


