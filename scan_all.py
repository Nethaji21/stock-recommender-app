"""
scan_all.py — Automated daily stock scanner
-------------------------------------------
Runs a simplified ML-based scan for top stock picks and saves
results as daily_recommendations.csv.

Designed for GitHub Actions or offline batch runs.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ta
import time, json, random, traceback
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from yfinance.exceptions import YFRateLimitError

# ================= CONFIG ==================
PERIOD = "120d"          # how many days of data to fetch
MIN_HISTORY = 60         # skip if fewer rows
BATCH_SIZE = 100         # how many symbols before a short pause
SLEEP_BETWEEN_BATCHES = 1.5
CACHE_DIR = Path("yf_cache")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = Path("daily_recommendations.csv")

# ===========================================

def safe_download(sym, period=PERIOD, tries=4, sleep_base=5):
    """Download with retry and backoff."""
    for attempt in range(tries):
        try:
            df = yf.Ticker(sym).history(period=period, interval="1d")
            if not df.empty:
                return df
        except YFRateLimitError:
            wait = sleep_base * (attempt + 1) + random.uniform(0, 3)
            print(f"[RateLimit] {sym}: sleeping {wait:.1f}s...")
            time.sleep(wait)
        except Exception as e:
            print(f"[Error] {sym}: {e}")
            break
    return None


def add_features(df):
    """Add technical indicators used in the ML model."""
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


def prepare_data(df):
    """Prepare ML inputs."""
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    feats = ["rsi", "macd_diff", "vol_spike", "atr", "ret1", "ret5", "ma7", "ma21"]
    X = df[feats].iloc[:-1].fillna(0)
    y = df["target"].iloc[:-1]
    return X, y


def build_model():
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=120, random_state=42))
    ])
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=400, random_state=42))
    ])
    return VotingClassifier(estimators=[("rf", rf), ("lr", lr)], voting="soft")


def evaluate_symbol(sym):
    cache_file = CACHE_DIR / f"{sym.replace('.NS','')}.json"
    if cache_file.exists():
        try:
            return json.load(open(cache_file))
        except Exception:
            pass

    df = safe_download(sym)
    if df is None or len(df) < MIN_HISTORY:
        return None
    df = add_features(df)
    if len(df) < 30:
        return None
    X, y = prepare_data(df)
    if y.nunique() < 2:
        return None

    model = build_model()
    try:
        cv = cross_val_score(model, X, y, cv=3, scoring="accuracy")
        cv_acc = float(np.mean(cv))
    except Exception:
        cv_acc = np.nan
    model.fit(X, y)
    prob = float(model.predict_proba(X.iloc[-1:].to_numpy())[0][1])
    last_close = df["Close"].iloc[-1]
    atr = df["atr"].iloc[-1] if "atr" in df.columns else last_close * 0.03
    rec = {
        "Stock": sym.replace(".NS",""),
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


def main():
    print("🚀 Starting daily scan...")
    if not Path("stocks_list.csv").exists():
        print("❌ Missing stocks_list.csv")
        return

    tickers = pd.read_csv("stocks_list.csv").iloc[:, 0].astype(str).tolist()
    tickers = [t.strip() + ".NS" if not t.strip().endswith((".NS",".BO")) else t.strip()
               for t in tickers if t.strip()]
    print(f"Loaded {len(tickers)} tickers")

    results = []
    total = len(tickers)
    processed = 0
    start = time.time()

    for batch in chunk(tickers, BATCH_SIZE):
        for sym in batch:
            try:
                res = evaluate_symbol(sym)
                if res:
                    results.append(res)
            except Exception as e:
                print(f"[{sym}] {e}")
            processed += 1
            if processed % 20 == 0:
                print(f"→ {processed}/{total} processed, {len(results)} valid")
        time.sleep(SLEEP_BETWEEN_BATCHES)

    if not results:
        print("⚠️ No valid results found.")
        return

    df = pd.DataFrame(results)
    df["Score"] = df["Proba"] * (0.6 + 0.4 * df["CV"].fillna(0.5))
    df["Potential %"] = (df["Target"] / df["Entry"] - 1) * 100
    df = df.sort_values("Score", ascending=False).reset_index(drop=True)
    df.to_csv(OUTPUT_FILE, index=False)
    top5 = df.head(5)
    print("✅ Saved:", OUTPUT_FILE)
    print("\nTop 5 picks:")
    print(top5[["Stock","Entry","Target","Proba","CV"]])
    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")

if __name__ == "__main__":
    main()
