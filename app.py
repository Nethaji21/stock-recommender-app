import streamlit as st
import pandas as pd
from generate_analysis import generate_trade_report

st.set_page_config(page_title="Pro Trader Picks", layout="wide")

st.title("💹 Pro Trader Picks — AI-Powered Stock Analysis")
st.caption("Daily AI-driven market analysis with professional explanations")

try:
    df = pd.read_csv("daily_recommendations.csv")
except Exception:
    st.error("daily_recommendations.csv not found. Please run scan_all.py or wait for daily update.")
    st.stop()

top = df.head(10)
st.success(f"Showing Top {len(top)} AI picks")

for _, row in top.iterrows():
    report = generate_trade_report(
        row["Stock"], row["Trend"], row["Entry"], row["Stop"], row["Target"], row["Proba"]
    )
    with st.expander(f"📈 {row['Stock']} — {row['Trend']} setup"):
        st.write(report)
        st.markdown(f"**Entry:** ₹{row['Entry']} | **Stop:** ₹{row['Stop']} | **Target:** ₹{row['Target']}")
        st.progress(min(row["Proba"],1.0))
