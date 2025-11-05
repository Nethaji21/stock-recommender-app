# app.py — Pro Trader Picks (final stable version)
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from datetime import timedelta
import ta
import concurrent.futures
import time

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Pro Trader Picks — Daily Top 5", layout="wide")

# ----------------- Styling -----------------
st.markdown(
    """
    <style>
    .header {background: linear-gradient(90deg,#0b1220,#081827); padding:18px; border-radius:10px;}
    .title {font-size:28px; color:#a8e3ff; font-weight:700;}
    .sub {color:#cfefff; opacity:0.9;}
    .card {background:#06111a; padding:12px; border-radius:8px; border:1px solid rgba(255,255,255,0.03);}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="header"><div class="title">Pro Trader Picks</div>'
    '<div class="sub">Daily Top 5 BUY Recommendations — Ensemble ML Model (Educational Use Only)</div></div>',
    unsafe_allow_html=True
)

# ----------------- Constants -----------------
MIN_HISTORY_DAYS = 120
DEFAULT_TIMEFRAME = "365d"
DEFAULT_TOPN = 200
MIN_PICKS = 5
DEFAULT_TTL = 3600  # cache 1h

# ----------------- Utilities -----------------
@st.cache_data(ttl=DEFAULT_TTL)
def load_stock_list(path="stocks_list.csv"):
    """Loads ticker list and ensures .NS suffix for NSE stocks."""
    try:
        df = pd.read_csv(path)
        if "SYMBOL" in df.columns:
            syms = df["SYMBOL"].astype(str).tolist()
        else:
            syms = df.iloc[:, 0].astype(str).tolist()
        # auto-append .NS if missing
        syms = [
            s.strip() + ".NS" if not s.strip().endswith(".NS") else s.strip()
            for s in syms
            if s.strip() != ""
        ]
        print(f"✅ Loaded {len(syms)} tickers")
        return syms
    except Exception as e:
        print("❌ Error loading stock list:", e)
        return []


@st.cache_data(ttl=DEFAULT_TTL)
def download_history(symbol, period):
    try:
        data = yf.Ticker(symbol).history(period=period, interval="1d")
        if data is None or data.empty:
            print(f"⚠️ Empty data for {symbol}")
            return None
        return data.dropna()
    except Exception as e:
        print(f"❌ Error downloading {symbol}: {e}")
        return None


def add_technical_features(df):
    """Add technical indicators using TA-lib"""
    df = df.copy()
    df["ret1"] = df["Close"].pct_change(1)
    df["ret5"] = df["Close"].pct_change(5)
    df["ma7"] = df["Close"].rolling(7).mean()
    df["ma21"] = df["Close"].rolling(21).mean()
    df["vol_ma14"] = df["Volume"].rolling(14).mean()
    df["vol_spike"] = df["Volume"] / (df["vol_ma14"] + 1e-9)
    try:
        df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
        macd = ta.trend.MACD(df["Close"])
        df["macd_diff"] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(df["Close"])
        df["bb_h"] = bb.bollinger_hband()
        df["bb_l"] = bb.bollinger_lband()
        df["atr"] = ta.volatility.AverageTrueRange(
            df["High"], df["Low"], df["Close"]
        ).average_true_range()
    except Exception:
        pass
    return df.dropna()


def prepare_ml_data(hist):
    df = hist.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    feats = ["rsi", "macd_diff", "vol_spike", "atr", "ret1", "ret5", "ma7", "ma21"]
    avail = [f for f in feats if f in df.columns]
    if len(avail) < 5:
        return None, None, None
    X = df[avail].iloc[:-1]
    y = df["target"].iloc[:-1]
    return X, y, avail


def build_model():
    rf = Pipeline(
        [("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=120, random_state=42))]
    )
    lr = Pipeline(
        [("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=400, random_state=42))]
    )
    return VotingClassifier(estimators=[("rf", rf), ("lr", lr)], voting="soft")


def evaluate_symbol(sym, period):
    hist = download_history(sym, period)
    if hist is None or len(hist) < MIN_HISTORY_DAYS:
        return None
    hist = add_technical_features(hist)
    X, y, feats = prepare_ml_data(hist)
    if X is None or y is None:
        return None

    # 🛠 Fix: skip if only one target class
    if y.nunique() < 2:
        print(f"⚠️ Skipped {sym}: target has only one class ({y.unique()})")
        return None

    model = build_model()
    try:
        cv = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        cv_acc = float(np.mean(cv))
    except Exception:
        cv_acc = np.nan

    model.fit(X, y)
    proba = float(model.predict_proba(X.iloc[-1:].to_numpy())[0][1])
    last_close = hist["Close"].iloc[-1]
    atr = hist["atr"].iloc[-1] if "atr" in hist.columns else last_close * 0.03
    target = max(last_close * 1.10, last_close + atr * 2)
    stop = last_close * 0.95
    print(f"✅ Scanned {sym}: proba={proba:.2f}")
    return {
        "Stock": sym.replace(".NS", ""),
        "Proba": proba,
        "CV": cv_acc,
        "Entry": last_close,
        "Target": target,
        "Stop": stop,
        "ATR": atr,
    }


def run_scan(universe, period, topn=100):
    to_scan = universe[:topn]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(evaluate_symbol, s, period) for s in to_scan]
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            if res:
                results.append(res)
    return results


# ----------------- Sidebar Controls -----------------
st.sidebar.header("Settings")
timeframe = st.sidebar.selectbox("History timeframe", ["180d", "365d", "730d"], index=1)
topn = st.sidebar.slider("Top N stocks to scan", 10, 500, 100, 10)
run_btn = st.sidebar.button("Run Daily Scan")

# ----------------- Main Logic -----------------
stock_list = load_stock_list()
if not stock_list:
    st.error("No valid tickers found. Please upload a stocks_list.csv with tickers like 'TCS.NS'.")
    st.stop()

if run_btn:
    st.info("Scanning stocks... please wait 1–2 minutes.")
    t0 = time.time()
    data = run_scan(stock_list, timeframe, topn)
    if len(data) == 0:
        st.error("No recommendations found. Verify your ticker list or internet connectivity.")
        st.stop()

    df = pd.DataFrame(data)
    df["Score"] = df["Proba"] * (0.6 + 0.4 * df["CV"].fillna(0.5))
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    picks = df.head(max(MIN_PICKS, 5))

    # Format
    picks["Potential %"] = (picks["Target"] / picks["Entry"] - 1) * 100
    picks["Risk %"] = (1 - picks["Stop"] / picks["Entry"]) * 100
    picks_display = picks.copy()
    picks_display["Entry"] = picks_display["Entry"].map(lambda x: f"₹{x:.2f}")
    picks_display["Target"] = picks_display["Target"].map(lambda x: f"₹{x:.2f}")
    picks_display["Stop"] = picks_display["Stop"].map(lambda x: f"₹{x:.2f}")
    picks_display["Proba"] = picks_display["Proba"].map("{:.1%}".format)
    picks_display["CV"] = picks_display["CV"].map("{:.1%}".format)
    picks_display["Potential %"] = picks_display["Potential %"].map("{:.1f}%".format)
    picks_display["Risk %"] = picks_display["Risk %"].map("{:.1f}%".format)

    st.success(f"✅ Found {len(picks_display)} top BUY recommendations!")
    st.dataframe(picks_display, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download all recommendations (CSV)", csv, "recommendations.csv", "text/csv")

    choice = st.selectbox("Deep dive chart", picks["Stock"])
    if choice:
        sym = choice + ".NS"
        hist = download_history(sym, timeframe)
        if hist is not None:
            fig = go.Figure()
            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name="Price",
                )
            )
            if "ma21" in hist.columns:
                hist["ma21"] = hist["Close"].rolling(21).mean()
                fig.add_trace(go.Scatter(x=hist.index, y=hist["ma21"], name="MA21"))
            fig.update_layout(height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

else:
    st.markdown("### 👈 Click **Run Daily Scan** to get today's top 5 BUY recommendations.")
    st.markdown("Ensure your `stocks_list.csv` file is uploaded and has NSE tickers (e.g., `TCS.NS`).")

st.markdown("---")
st.caption("⚠️ For education only. No financial advice.")
