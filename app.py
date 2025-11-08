import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta, time, random, json, traceback
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from yfinance.exceptions import YFRateLimitError
import plotly.graph_objects as go

# =================== CONFIG ===================
PERIOD = "120d"              # scanning window
MIN_HISTORY = 60             # skip stocks with less data
BATCH_SIZE = 100             # symbols per batch
SLEEP_BETWEEN_BATCHES = 1.0  # delay between batches
CACHE_DIR = Path("yf_cache")
CACHE_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Pro Trader Picks", layout="wide")

# =================== STYLES ===================
st.markdown("""
<style>
h1, h2, h3, h4 {color: #a8e3ff;}
.reportview-container {background-color: #0b1220;}
.sidebar .sidebar-content {background-color: #081827;}
.block-container {padding-top: 1rem;}
</style>
""", unsafe_allow_html=True)

st.title("💹 Pro Trader Picks — Daily ML Scanner")
st.caption("Auto-loads stock list, scans with ensemble ML model, and shows top 5 BUY picks. (Educational use only)")

# =================== HELPERS ===================
def safe_download(sym, period=PERIOD, tries=4, sleep_base=5):
    """Download Yahoo Finance data with rate-limit protection."""
    for attempt in range(tries):
        try:
            data = yf.Ticker(sym).history(period=period, interval="1d")
            if not data.empty:
                return data
        except YFRateLimitError:
            wait = sleep_base * (attempt + 1) + random.uniform(0, 2)
            st.info(f"⏳ Yahoo rate limit for {sym}, retrying in {wait:.1f}s...")
            time.sleep(wait)
        except Exception as e:
            st.write(f"⚠️ {sym} failed: {e}")
            break
    return None

def add_features(df):
    """Add technical indicators."""
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
        df["atr"] = ta.volatility.AverageTrueRange(df["High"], df["Low"], df["Close"]).average_true_range()
    except Exception:
        pass
    return df.dropna()

def prepare_data(hist):
    """Prepare ML-ready dataset."""
    df = hist.copy()
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    feats = ["rsi", "macd_diff", "vol_spike", "atr", "ret1", "ret5", "ma7", "ma21"]
    avail = [f for f in feats if f in df.columns]
    if len(avail) < 5:
        return None, None
    X = df[avail].iloc[:-1]
    y = df["target"].iloc[:-1]
    return X, y

def model_build():
    """Ensemble ML model."""
    rf = Pipeline([("scaler", StandardScaler()),
                   ("rf", RandomForestClassifier(n_estimators=120, random_state=42))])
    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=400, random_state=42))])
    return VotingClassifier(estimators=[("rf", rf), ("lr", lr)], voting="soft")

def evaluate_symbol(sym):
    """Compute ML-based BUY confidence for a symbol."""
    cache_file = CACHE_DIR / f"{sym.replace('.NS','')}.json"
    if cache_file.exists():
        try:
            return json.load(open(cache_file))
        except Exception:
            pass

    df = safe_download(sym)
    if df is None or len(df) < MIN_HISTORY:
        return None
    hist = add_features(df)
    X, y = prepare_data(hist)
    if X is None or y.nunique() < 2:
        return None

    model = model_build()
    try:
        cv = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        cv_acc = float(np.mean(cv))
    except Exception:
        cv_acc = np.nan

    model.fit(X, y)
    prob = float(model.predict_proba(X.iloc[-1:])[0][1])
    last_close = hist["Close"].iloc[-1]
    atr = hist["atr"].iloc[-1] if "atr" in hist.columns else last_close * 0.03
    rec = {
        "Stock": sym.replace(".NS", ""),
        "Proba": prob,
        "CV": cv_acc,
        "Entry": last_close,
        "Target": max(last_close * 1.10, last_close + atr * 2),
        "Stop": last_close * 0.95,
        "ATR": atr,
    }
    json.dump(rec, open(cache_file, "w"))
    return rec

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# =================== LOAD STOCK LIST ===================
try:
    stocks_df = pd.read_csv("stocks_list.csv")
    tickers = stocks_df.iloc[:,0].astype(str).tolist()
    tickers = [s.strip() + ".NS" if not s.strip().endswith(".NS") else s.strip() for s in tickers if s.strip()]
    st.write(f"✅ Loaded {len(tickers)} tickers from stocks_list.csv")
except Exception as e:
    st.error(f"❌ Could not read stocks_list.csv — {e}")
    st.stop()

# =================== RUN SCAN ===================
start_btn = st.button("🚀 Run Daily Scan")

if start_btn:
    results = []
    progress = st.progress(0)
    total = len(tickers)
    scanned = 0
    start_time = time.time()

    for batch in chunk(tickers, BATCH_SIZE):
        for sym in batch:
            try:
                res = evaluate_symbol(sym)
                if res:
                    results.append(res)
            except Exception as e:
                st.write(f"⚠️ {sym} error: {e}")
            scanned += 1
            progress.progress(min(scanned/total, 1.0))
        time.sleep(SLEEP_BETWEEN_BATCHES)

    if results:
        df = pd.DataFrame(results)
        df["Score"] = df["Proba"] * (0.6 + 0.4 * df["CV"].fillna(0.5))
        df["Potential %"] = (df["Target"] / df["Entry"] - 1) * 100
        df = df.sort_values("Score", ascending=False).reset_index(drop=True)
        top5 = df.head(5)

        runtime = (time.time() - start_time) / 60
        st.success(f"✅ Scan complete in {runtime:.1f} minutes. {len(df)} valid results found.")
        st.dataframe(top5, use_container_width=True)

        st.download_button("⬇️ Download All Results (CSV)", df.to_csv(index=False), "daily_recommendations.csv")

        # ====== Chart viewer ======
        sym = st.selectbox("📊 View chart for", top5["Stock"])
        if sym:
            data = yf.Ticker(sym + ".NS").history(period="120d")
            if not data.empty:
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=data.index,
                    open=data["Open"], high=data["High"],
                    low=data["Low"], close=data["Close"],
                    name="Price"
                ))
                data["MA21"] = data["Close"].rolling(21).mean()
                fig.add_trace(go.Scatter(x=data.index, y=data["MA21"], name="MA21"))
                fig.update_layout(height=450, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ No valid recommendations found. Try again later.")

else:
    st.info("👆 Click 'Run Daily Scan' to start scanning automatically.")
