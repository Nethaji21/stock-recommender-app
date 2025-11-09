import streamlit as st
import pandas as pd
from generate_analysis import generate_trade_report

# ------------------------------
# Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="Pro Trader Picks", layout="wide")

st.title("💹 Pro Trader Picks — AI-Powered Stock Analysis")
st.caption("Daily AI-driven market analysis with professional explanations")

# ------------------------------
# Load Data
# ------------------------------
try:
    df = pd.read_csv("daily_recommendations.csv")
except Exception as e:
    st.error(f"⚠️ Could not read daily_recommendations.csv — {e}")
    st.stop()

if df.empty:
    st.warning("No data found in daily_recommendations.csv. Please run scan_all.py first.")
    st.stop()

# ------------------------------
# Display Top Picks
# ------------------------------
top = df.head(10)
st.success(f"Showing Top {len(top)} AI picks")

for _, row in top.iterrows():
    # --- Safe fallback for missing columns ---
    trend = row["Trend"] if "Trend" in row else (
        "Bullish" if "Target" in row and "Entry" in row and row["Target"] > row["Entry"] else "Bearish"
    )
    entry = row["Entry"] if "Entry" in row else 0
    stop = row["Stop"] if "Stop" in row else 0
    target = row["Target"] if "Target" in row else 0
    proba = row["Proba"] if "Proba" in row else 0.5

    # --- Generate AI Report ---
    report = generate_trade_report(
        row["Stock"], trend, entry, stop, target, proba
    )

    # --- Display Report ---
    with st.expander(f"📈 {row['Stock']} — {trend} Setup"):
        st.write(report)
        st.markdown(
            f"**Entry:** ₹{entry:.2f} | **Stop:** ₹{stop:.2f} | **Target:** ₹{target:.2f}"
        )
        st.progress(min(proba, 1.0))

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("⚙️ Data sourced via yfinance & ML models. Educational use only.")
