# app.py — simple dashboard to display daily_recommendations.csv
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

st.set_page_config(page_title="Pro Trader Picks Dashboard", layout="wide")

st.markdown(
    """
    <style>
    .header {background: linear-gradient(90deg,#0b1220,#081827); padding:18px; border-radius:10px;}
    .title {font-size:28px; color:#a8e3ff; font-weight:700;}
    .sub {color:#cfefff; opacity:0.9;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="header"><div class="title">Pro Trader Picks — Daily Recommendations</div>'
    '<div class="sub">Data precomputed by nightly scan (120-day ML model)</div></div>',
    unsafe_allow_html=True
)

# ---------------------- load data ----------------------
try:
    df = pd.read_csv("daily_recommendations.csv")
except Exception:
    st.error("No daily_recommendations.csv found — run scan_all.py first.")
    st.stop()

if df.empty:
    st.warning("No data available.")
    st.stop()

# ---------------------- top picks ----------------------
df["Potential %"] = (df["Target"] / df["Entry"] - 1) * 100
df["Risk %"] = (1 - df["Stop"] / df["Entry"]) * 100
df["Proba"] = df["Proba"].map("{:.1%}".format)
df["CV"] = df["CV"].map("{:.1%}".format)
for col in ["Entry", "Target", "Stop"]:
    df[col] = df[col].map(lambda x: f"₹{x:.2f}")

top5 = df.head(5)

st.success(f"✅ Showing {len(top5)} BUY recommendations from latest scan.")
st.dataframe(top5, use_container_width=True)

# ---------------------- chart viewer ----------------------
sym = st.selectbox("View chart for", top5["Stock"])
if sym:
    data = yf.Ticker(sym + ".NS").history(period="120d")
    if not data.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["Open"], high=data["High"], low=data["Low"], close=data["Close"],
            name="Price"
        ))
        data["ma21"] = data["Close"].rolling(21).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data["ma21"], name="MA21"))
        fig.update_layout(height=450, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("⚠️ Educational use only — not financial advice.")