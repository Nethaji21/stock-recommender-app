"""
scan_all.py
------------
Nightly batch scanner for all tickers in stocks_list.csv.
Fetches 120 days of data, handles Yahoo rate limits, caches, and writes daily_recommendations.csv.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import time, os, json, random, traceback
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from yfinance.exceptions import YFRateLimitError

PERIOD = "120d"
MIN_HISTORY_DAYS = 60
BATCH_SIZE = 150
SLEEP_BETWEEN_BATCHES = 1.5
CACHE_DIR = Path("yf_cache")
OUTPUT_FILE = Path("daily_recommendations.csv")
CACHE_DIR.mkdir(exist_ok=True)

# --------------------- helpers ---------------------
def safe_download(sym, period=PERIOD, tries=4, sleep_base=5):
    for attempt in range(tries):
        try:
            df = yf.Ticker(sym).history(period=period, interval="1d")
            if not df.empty:
                return df
        except YFRateLimitError:
            wait = sleep_base * (attempt + 1) + random.uniform(0, 3)
            print(f"⏳ Rate limit for {sym}, retry in {wait:.1f}s")
            time.sleep(wait)
        except Exception as e:
            print(f"⚠️ {sym} failed: {e}")
            break
    return None

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def cache_save(sym, data):
    try:
        with open(CACHE_DIR / f"{sym}.json", "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def cache_load(sym):
    p = CACHE_DIR / f"{sym}.json"
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
    rf = Pipeline([("scaler", StandardScaler()),
                   ("rf", RandomForestClassifier(n_estimators=120, random_state=42))])
    lr = Pipeline([("scaler", StandardScaler()),
                   ("lr", LogisticRegression(max_iter=400, random_state=42))])
    return VotingClassifier(estimators=[("rf", rf), ("lr", lr)], voting="soft")

def evaluate_symbol(sym, period=PERIOD):
    cached = cache_load(sym.replace(".NS", ""))
    if cached:
        return cached
    df = safe_download(sym, period)
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return None
    hist = add_technical_features(df)
    X, y, feats = prepare_ml_data(hist)
    if X is None or y is None or y.nunique() < 2:
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
    rec = {
        "Stock": sym.replace(".NS", ""),
        "Proba": proba,
        "CV": cv_acc,
        "Entry": last_close,
        "Target": target,
        "Stop": stop,
        "ATR": atr,
    }
    cache_save(sym.replace(".NS", ""), rec)
    return rec

def run_scan_all(universe):
    results = []
    total = len(universe)
    processed = 0
    for batch_i, batch in enumerate(chunk_list(universe, BATCH_SIZE), start=1):
        for sym in batch:
            try:
                r = evaluate_symbol(sym)
                if r:
                    results.append(r)
            except Exception as e:
                print(f"Error {sym}: {e}\n{traceback.format_exc()}")
            processed += 1
        print(f"Batch {batch_i} done — {processed}/{total}")
        time.sleep(SLEEP_BETWEEN_BATCHES)
    return results

# --------------------- main ---------------------
def main():
    print("📊 Starting full-universe scan...")
    if not Path("stocks_list.csv").exists():
        print("❌ stocks_list.csv not found.")
        return
    df = pd.read_csv("stocks_list.csv")
    syms = df.iloc[:, 0].astype(str).tolist()
    syms = [s.strip() + ".NS" if not s.strip().endswith(".NS") else s.strip()
            for s in syms if s.strip()]
    print(f"Loaded {len(syms)} tickers")

    start = time.time()
    data = run_scan_all(syms)
    elapsed = (time.time() - start) / 60
    print(f"✅ Completed scan in {elapsed:.1f} min with {len(data)} results")

    if not data:
        print("⚠️ No results.")
        return

    df = pd.DataFrame(data)
    df["Score"] = df["Proba"] * (0.6 + 0.4 * df["CV"].fillna(0.5))
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"💾 Saved {len(df)} results to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()