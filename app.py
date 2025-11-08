# app.py — Pro Trader Picks (Full-Universe 120d Version)
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
import ta
import concurrent.futures
import time
import os, json, traceback
from pathlib import Path

# ----------------- Streamlit Page Config -----------------
st.set_page_config(page_title="Pro Trader Picks — Full Universe", layout="wide")

# ----------------- Styling -----------------
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
    '<div class="header"><div class="title">Pro Trader Picks — Full Universe</div>'
    '<div class="sub">Scans all stocks (~12k) over 120 days for top 5 BUY setups (Educational Use Only)</div></div>',
    unsafe_allow_html=True
)

# ----------------- Constants -----------------
MIN_HISTORY_DAYS = 60
PERIOD = "120d"           # Fixed timeframe
BATCH_SIZE = 200          # tickers per batch
SLEEP_BETWEEN = 1.0       # seconds between batches
MIN_PICKS = 5
CACHE_DIR = Path("yf_cache")
CACHE_DIR.mkdir(exist_ok=True)

# ----------------- Utility functions -----------------
@st.cache_data(ttl=3600)
def load_stock_list(path="stocks_list.csv"):
    try:
        df = pd.read_csv(path)
        if "SYMBOL" in df.columns:
            syms = df["SYMBOL"].astype(str).tolist()
        else:
            syms = df.iloc[:, 0].astype(str).tolist()
        syms = [
            s.strip() + ".NS" if not s.strip().endswith(".NS") else s.strip()
            for s in syms if s.strip() != ""
        ]
        print(f"✅ Loaded {len(syms)} tickers")
        return syms
    except Exception as e:
        print("❌ Error loading stock list:", e)
        return []

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def cache_save(symbol, payload):
    try:
        with open(CACHE_DIR / f"{symbol}.json", "w") as f:
            json.dump(payload, f)
    except Exception:
        pass

def cache_load(symbol):
    p = CACHE_DIR / f"{symbol}.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None

def add_technical_features(df):
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
    rf = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=120, random_state=42))])
    lr = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=400, random_state=42))])
    return VotingClassifier(estimators=[("rf", rf), ("lr", lr)], voting="soft")

# ----------------- Main scanning logic -----------------
def run_scan_all(universe, period=PERIOD, batch_size=BATCH_SIZE, sleep_between_batches=SLEEP_BETWEEN):
    results = []
    total = len(universe)
    processed = 0

    for batch_idx, batch in enumerate(chunk_list(universe, batch_size), start=1):
        try:
            raw = yf.download(tickers=batch, period=period, interval="1d", group_by="ticker", threads=True, progress=False)
        except Exception as e:
            print(f"⚠️ Batch {batch_idx} failed: {e}")
            raw = None

        for sym in batch:
            try:
                cached = cache_load(sym.replace(".NS", ""))
                if cached:
                    results.append(cached)
                    processed += 1
                    continue

                df = None
                if raw is not None and isinstance(raw.columns, pd.MultiIndex):
                    if sym in raw.columns.get_level_values(0):
                        df = raw[sym].dropna(how="all")
                        if df.empty:
                            df = None
                if df is None or len(df) < MIN_HISTORY_DAYS:
                    try:
                        df = yf.Ticker(sym).history(period=period, interval="1d")
                        if df is None or df.empty:
                            df = None
                    except Exception:
                        df = None
                if df is None or len(df) < MIN_HISTORY_DAYS:
                    processed += 1
                    continue

                hist = add_technical_features(df)
                X, y, feats = prepare_ml_data(hist)
                if X is None or y is None or y.nunique() < 2:
                    processed += 1
                    continue

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

                rec = {
                    "Stock": sym.replace(".NS", ""),
                    "Proba": proba,
                    "CV": cv_acc,
                    "Entry": last_close,
                    "Target": target,
                    "Stop": stop,
                    "ATR": atr,
                }
                results.append(rec)
                cache_save(sym.replace(".NS", ""), rec)
            except Exception as e:
                print(f"Error processing {sym}: {e}\n{traceback.format_exc()}")
            finally:
                processed += 1
        print(f"Batch {batch_idx} done: {processed}/{total}")
        time.sleep(sleep_between_batches)

    return results

# ----------------- App UI -----------------
stock_list = load_stock_list()
if not stock_list:
    st.error("No valid tickers found in stocks_list.csv. Ensure tickers like 'TCS.NS'.")
    st.stop()

if st.button("Run Full Universe Scan (120 days)"):
    st.info("Scanning all tickers… please wait. This may take a long time for 12k stocks.")
    t0 = time.time()
    data = run_scan_all(stock_list)
    duration = time.time() - t0

    if len(data) == 0:
        st.error("No valid recommendations found.")
        st.stop()

    df = pd.DataFrame(data)
    df["Score"] = df["Proba"] * (0.6 + 0.4 * df["CV"].fillna(0.5))
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    picks = df.head(max(MIN_PICKS, 5))

    picks["Potential %"] = (picks["Target"] / picks["Entry"] - 1) * 100
    picks["Risk %"] = (1 - picks["Stop"] / picks["Entry"]) * 100

    for col in ["Entry", "Target", "Stop"]:
        picks[col] = picks[col].map(lambda x: f"₹{x:.2f}")
    picks["Proba"] = picks["Proba"].map("{:.1%}".format)
    picks["CV"] = picks["CV"].map("{:.1%}".format)
    picks["Potential %"] = picks["Potential %"].map("{:.1f}%".format)
    picks["Risk %"] = picks["Risk %"].map("{:.1f}%".format)

    st.success(f"✅ Scan finished in {duration/60:.1f} minutes — showing top {len(picks)} BUY setups.")
    st.dataframe(picks, use_container_width=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download all results (CSV)", csv, "recommendations.csv", "text/csv")

    sym = st.selectbox("View chart for", picks["Stock"])
    if sym:
        hist = yf.Ticker(sym + ".NS").history(period=PERIOD)
        if not hist.empty:
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=hist.index,
                open=hist["Open"], high=hist["High"],
                low=hist["Low"], close=hist["Close"], name="Price"
            ))
            hist["ma21"] = hist["Close"].rolling(21).mean()
            fig.add_trace(go.Scatter(x=hist.index, y=hist["ma21"], name="MA21"))
            fig.update_layout(height=450, xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.markdown("### 👈 Press **Run Full Universe Scan (120 days)** to get today's top 5 BUY recommendations.")

st.markdown("---")
st.caption("⚠️ For education only. No financial advice.")