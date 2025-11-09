import pandas as pd
import numpy as np
import yfinance as yf
import ta, time, json, random, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from yfinance.exceptions import YFRateLimitError

PERIOD = "120d"
MIN_HISTORY = 60
CACHE_DIR = Path("yf_cache")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = Path("daily_recommendations.csv")
THREADS = 8

def safe_download(sym, tries=4, sleep_base=5):
    for attempt in range(tries):
        try:
            df = yf.Ticker(sym).history(period=PERIOD, interval="1d")
            if not df.empty:
                return df
        except YFRateLimitError:
            wait = sleep_base * (attempt + 1) + random.uniform(0, 2)
            time.sleep(wait)
        except Exception:
            pass
    return None

def add_features(df):
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
    df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    feats = ["rsi","macd_diff","vol_spike","atr","ret1","ret5","ma7","ma21"]
    X = df[feats].iloc[:-1].fillna(0)
    y = df["target"].iloc[:-1]
    return X, y

def build_model():
    rf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=80, random_state=42))
    ])
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=300, random_state=42))
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
    trend = "Bullish" if prob > 0.55 else "Bearish"
    rec = {
        "Stock": sym.replace(".NS",""),
        "Trend": trend,
        "Proba": prob,
        "CV": cv_acc,
        "Entry": round(last_close * (1.005 if trend=="Bullish" else 0.995), 2),
        "Target": round(last_close * (1.1 if trend=="Bullish" else 0.9), 2),
        "Stop": round(last_close * (0.95 if trend=="Bullish" else 1.05), 2),
        "ATR": round(atr, 2)
    }
    json.dump(rec, open(cache_file, "w"))
    return rec

def main():
    df = pd.read_csv("stocks_list.csv")
    tickers = [s.strip() + ".NS" if not s.strip().endswith(".NS") else s.strip()
               for s in df.iloc[:,0].astype(str) if s.strip()]
    print(f"Scanning {len(tickers)} tickers using {THREADS} threads...")
    results=[]
    with ThreadPoolExecutor(max_workers=THREADS) as ex:
        futures={ex.submit(evaluate_symbol,t):t for t in tickers}
        for i,f in enumerate(as_completed(futures)):
            r=f.result()
            if r: results.append(r)
            if i%50==0:
                print(f"{i}/{len(tickers)} done, {len(results)} valid")
    if results:
        df=pd.DataFrame(results)
        df["Score"]=df["Proba"]*(0.6+0.4*df["CV"].fillna(0.5))
        df=df.sort_values("Score",ascending=False).reset_index(drop=True)
        df.to_csv(OUTPUT_FILE,index=False)
        print(f"✅ Saved {len(df)} results to {OUTPUT_FILE}")
    else:
        print("⚠️ No results found.")

if __name__=="__main__":
    main()
