import streamlit as st
import pandas as pd

# ------------------------------
# PAGE CONFIG
# ------------------------------
st.set_page_config(
    page_title="Pro Trader Picks",
    layout="wide"
)

# ------------------------------
# HEADER
# ------------------------------
st.title("💹 Pro Trader Picks — Advanced AI Scanner")
st.caption("Advanced ML-based stock recommendation dashboard")

# ------------------------------
# LOAD DATA
# ------------------------------
try:
    df = pd.read_csv("daily_recommendations.csv")
except Exception as e:
    st.error(f"Could not load daily_recommendations.csv — {e}")
    st.stop()

if df.empty:
    st.warning("No recommendations found")
    st.stop()

# ------------------------------
# TOP PICKS
# ------------------------------
top = df.head(10)

st.success(f"Showing Top {len(top)} AI Picks")

# ------------------------------
# DISPLAY
# ------------------------------
for _, row in top.iterrows():

    stock = row.get("Stock", "Unknown")
    trend = row.get("Trend", "Neutral")

    entry = row.get("Entry", 0)
    target = row.get("Target", 0)
    stop = row.get("Stop", 0)

    profit_pct = row.get("Profit_%", 0)
    loss_pct = row.get("Loss_%", 0)

    rr = row.get("Risk_Reward", 0)
    confidence = row.get("Confidence", 0)
    qty = row.get("Position_Size", 0)

    commentary = row.get("Commentary", "No commentary available")

    with st.expander(f"📈 {stock} — {trend} Setup"):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Entry", f"₹{entry}")
            st.metric("Target", f"₹{target}")

        with col2:
            st.metric("Stop Loss", f"₹{stop}")
            st.metric("Profit %", f"{profit_pct}%")

        with col3:
            st.metric("Risk : Reward", f"1 : {rr}")
            st.metric("Confidence", f"{confidence}%")

        st.progress(min(confidence / 100, 1.0))

        st.markdown("---")

        st.markdown(f"### 📊 Trade Commentary")
        st.write(commentary)

        st.markdown("---")

        st.markdown(f"### 🧮 Position Sizing")
        st.write(
            f"Recommended Quantity based on ₹1,00,000 capital and 2% risk per trade: **{qty} shares**"
        )

        st.markdown("---")

        st.markdown("### ⚡ Trade Summary")

        st.write(
            f"This setup has a projected upside/downside move of {profit_pct}% "
            f"with estimated risk of {loss_pct}%. "
            f"Current model confidence is {confidence}% with a risk-reward ratio of 1:{rr}."
        )

# ------------------------------
# RAW TABLE
# ------------------------------
st.markdown("---")

st.markdown("## 📋 Full Recommendation Table")

st.dataframe(df.head(20), use_container_width=True)

# ------------------------------
# FOOTER
# ------------------------------
st.markdown("---")
st.caption("Educational use only. Not financial advice.")
