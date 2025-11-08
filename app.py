# app.py — All-in-one optimized scanner + Streamlit UI
# - Batched yf.download prefilter
# - Threaded ML evaluation on filtered tickers
# - Incremental checkpointing (results_checkpoint.csv)
# - Optional LightGBM (if installed) in ensemble
# - Position sizing helper using ATR-based stop
#
# Requirements (add to requirements.txt):
# streamlit,pandas,numpy,yfinance,ta,scikit-learn,plotly
# Optional: lightgbm

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import time, random, json, traceback, math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, f_classif
import plotly.graph_objects as go
from yfinance.exceptions import YFRateLimitError

# Try to import lightgbm if available
try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

# ----------------- Config / defaults -----------------
PERIOD = "120d"                 # timeframe for indicators
MIN_HISTORY = 60                # skip stocks with less history
DEFAULT_BATCH = 200             # how many tickers per big yf.download batch
EVAL_THREADS = 8                # threads for per-symbol ML evaluation
CACHE_DIR = Path("yf_cache"); CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT = Path("results_checkpoint.csv")

st.set_page_config(page_title="Pro Trader Picks — Optimized", layout="wide")
st.title("🔧 Pro Trader Picks — Optimized All-in-One Scanner")
st.caption("Batched, cached, ML ensemble + trading helpers. Educational only — not financial advice.")

# ----------------- Sidebar controls -----------------
st.sidebar.header("Scan controls")
fast_test = st.sidebar.checkbox("Fast test mode (first N tickers)", value=True)
fast_n = st.sidebar.number_input("Fast test N", min_value=20, max_value=2000, value=200, step=10)
batch_size = st.sidebar.number_input("yf.download batch size", min_value=50, max_value=1000, value=DEFAULT_BATCH, step=10)
threads = st.sidebar.number_input("Per-symbol threads", min_value=1, max_value=32, value=EVAL_THREADS, step=1)
sleep_between_batches = st.sidebar.slider("Sleep between yf batches (s)", 0.5, 5.0, 1.0, 0.1)
use_lgb = st.sidebar.checkbox("Use LightGBM in ensemble if available", value=False)
risk_per_trade_pct = st.sidebar.slider("Risk per trade (% of capital)", 0.1, 5.0, 1.0, 0.1)
account_size = st.sidebar.number_input("Capital (₹)", min_value=1000.0, value=100000.0, step=1000.0)

st.sidebar.markdown("---")
st.sidebar.write("Checks & utilities")
clear_cache_btn = st.sidebar.button("Clear yf cache")
if clear_cache_btn:
    for p in CACHE_DIR.glob("*.json"):
        p.unlink(missing_ok=True)
    st.sidebar.success("Cache cleared")

# ----------------- Helper functions -----------------
def read_stock_list(path="stocks_list.csv"):
    p = Path(path)
    if not p.exists():
        st.error("stocks_list.csv not found in repo root. Please add it.")
        st.stop()
    df = pd.read_csv(path)
    # Accept first column as symbol
    syms = df.iloc[:,0].astype(str).tolist()
    syms = [s.strip() for s in syms if s.strip()!='']
    # append .NS if ok (optional)
    # If user tickers already have suffix, we keep them
    normalized = []
    for s in syms:
        # assume NSE if no dot in symbol and looks alphabetic and uppercase
        if '.' not in s and s.isupper():
            normalized.append(s + ".NS")
        else:
            normalized.append(s)
    return normalized

def checkpoint_save(results_df):
    try:
        results_df.to_csv(CHECKPOINT, index=False)
    except Exception as e:
        st.warning(f"Could not save checkpoint: {e}")

def checkpoint_load():
    if CHECKPOINT.exists():
        try:
            return pd.read_csv(CHECKPOINT)
        except Exception:
            return None
    return None

def safe_yf_download(tickers, period=PERIOD, group_by='ticker', tries=3):
    """Call yf.download with retry/backoff"""
    wait_base = 1.0
    for attempt in range(tries):
        try:
            df = yf.download(tickers=tickers, period=period, interval='1d', group_by=group_by, threads=True, progress=False)
            return df
        except YFRateLimitError as e:
            wait = wait_base * (attempt+1) + random.random()*2
            st.write(f"Rate limited by Yahoo — sleeping {wait:.1f}s before retry")
            time.sleep(wait)
        except Exception as e:
            # sometimes connection error
            wait = wait_base * (attempt+1)
            st.write(f"yf.download exception: {e} — retry {attempt+1} after {wait}s")
            time.sleep(wait)
    return None

def safe_single_history(sym, period=PERIOD, tries=4):
    """Fallback to per-ticker history with retries"""
    for attempt in range(tries):
        try:
            df = yf.Ticker(sym).history(period=period, interval='1d')
            if df is None or df.empty:
                time.sleep(1 + attempt)
                continue
            return df
        except YFRateLimitError:
            time.sleep(2 + attempt*1.5)
        except Exception:
            time.sleep(1)
    return None

def add_technical_features(df):
    df = df.copy()
    df['ret1'] = df['Close'].pct_change(1)
    df['ret5'] = df['Close'].pct_change(5)
    df['ma7'] = df['Close'].rolling(7).mean()
    df['ma21'] = df['Close'].rolling(21).mean()
    df['vol_ma14'] = df['Volume'].rolling(14).mean()
    df['vol_spike'] = df['Volume'] / (df['vol_ma14'] + 1e-9)
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['macd_diff'] = macd.macd_diff()
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    except Exception:
        pass
    return df.dropna()

def prepare_ml_dataset(hist):
    hist = hist.copy()
    hist['target'] = (hist['Close'].shift(-1) > hist['Close']).astype(int)
    features = ['rsi','macd_diff','vol_spike','atr','ret1','ret5','ma7','ma21','obv']
    avail = [f for f in features if f in hist.columns]
    if len(avail) < 6:
        return None, None, None
    X = hist[avail].iloc[:-1]
    y = hist['target'].iloc[:-1]
    return X, y, avail

def build_ensemble(use_lightgbm=False):
    estimators = []
    rf = ("rf", Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=120, random_state=42))]))
    estimators.append(rf)
    lr = ("lr", Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=400, random_state=42))]))
    estimators.append(lr)
    if use_lightgbm and HAS_LGB:
        lgbm = ("lgb", Pipeline([('s', StandardScaler()), ('lgb', lgb.LGBMClassifier(n_estimators=200, random_state=42))]))
        estimators.append(lgbm)
    ensemble = VotingClassifier(estimators=estimators, voting='soft')
    return ensemble

# Per-symbol evaluation (heavy)
def evaluate_symbol_full(sym, use_lightgbm=False):
    """Returns dict with results or None"""
    # check cache file
    cache_file = CACHE_DIR / f"{sym.replace('.NS','')}.json"
    if cache_file.exists():
        try:
            return json.loads(open(cache_file).read())
        except Exception:
            pass

    # download full history
    hist = safe_single_history(sym, period=PERIOD)
    if hist is None or len(hist) < MIN_HISTORY:
        return None
    hist = add_technical_features(hist)
    X, y, feats = prepare_ml_dataset(hist)
    if X is None or y.nunique() < 2:
        return None

    # optional feature selection for speed
    try:
        selector = SelectKBest(f_classif, k=min(8, X.shape[1]))
        Xsel = selector.fit_transform(X.fillna(0), y)
        selected_cols = [X.columns[i] for i in range(X.shape[1]) if selector.get_support()[i]]
        X = pd.DataFrame(Xsel, columns=selected_cols, index=X.index)
    except Exception:
        pass

    model = build_ensemble(use_lightgbm)
    try:
        cv = cross_val_score(model, X.fillna(0), y, cv=3, scoring='accuracy')
        cv_acc = float(np.mean(cv))
    except Exception:
        cv_acc = np.nan

    # prevent training errors if label is single class
    if y.nunique() < 2:
        return None

    model.fit(X.fillna(0), y)
    try:
        proba = float(model.predict_proba(X.iloc[-1:].fillna(0))[0][1])
    except Exception:
        proba = float(model.predict_proba(X.iloc[-1:].fillna(0).to_numpy())[0][1])

    last_close = hist['Close'].iloc[-1]
    atr = hist['atr'].iloc[-1] if 'atr' in hist.columns else last_close * 0.03
    target_atr = last_close + 2*atr
    fixed_target = last_close * 1.10
    target = max(target_atr, fixed_target)
    stop = last_close * 0.95

    rec = {
        'Stock': sym.replace('.NS',''),
        'SymFull': sym,
        'Proba': proba,
        'CV': cv_acc,
        'Entry': float(last_close),
        'Target': float(target),
        'Stop': float(stop),
        'ATR': float(atr)
    }

    # save cache file
    try:
        open(cache_file, 'w').write(json.dumps(rec))
    except Exception:
        pass

    return rec

# ----------------- Fast prefilter step (batched) -----------------
def batch_prefilter(universe, batch_size=BATCH_SIZE, period='30d'):
    """
    Use yf.download in batches to fetch Close & Volume quickly and compute simple filters:
    - momentum_5d = pct change over 5 days
    - avg volume > 0
    Returns list of tickers that pass cheap filters (momentum positive OR high vol spike).
    """
    filtered = []
    total = len(universe)
    for i in range(0, total, batch_size):
        batch = universe[i:i+batch_size]
        raw = safe_yf_download(batch, period=period)
        if raw is None:
            # fallback: attempt per-symbol but skip heavy processing
            for s in batch:
                try:
                    df = safe_single_history(s, period=period)
                    if df is None or df.empty: continue
                    # compute 5d momentum
                    if 'Close' in df.columns and len(df)>=6:
                        m5 = df['Close'].pct_change(5).iloc[-1]
                        if m5 is not None and not np.isnan(m5) and m5 > 0.0:
                            filtered.append(s)
                except Exception:
                    continue
            time.sleep(sleep_between_batches)
            continue

        # raw is multiindex (ticker, attr) when multiple tickers
        if isinstance(raw.columns, pd.MultiIndex):
            for s in batch:
                try:
                    if s not in raw.columns.get_level_values(0): continue
                    df = raw[s].dropna(how='all')
                    if df.empty: continue
                    if 'Close' not in df.columns: continue
                    if len(df) < 6: continue
                    m5 = df['Close'].pct_change(5).iloc[-1]
                    vol = df['Volume'].iloc[-1] if 'Volume' in df.columns else 0
                    if (m5 is not None and not np.isnan(m5) and m5 > 0.0) or (vol > 0):
                        filtered.append(s)
                except Exception:
                    continue
        else:
            # single-ticker or odd format: handle simply
            for s in batch:
                try:
                    df = raw.dropna()
                    if df.empty: continue
                    if 'Close' not in df.columns: continue
                    if len(df) < 6: continue
                    m5 = df['Close'].pct_change(5).iloc[-1]
                    if (m5 is not None and not np.isnan(m5) and m5 > 0.0):
                        filtered.append(s)
                except Exception:
                    continue

        time.sleep(sleep_between_batches)
    return filtered

# ----------------- Main UI / flow -----------------
# Load stock list from repo root
symbols = read_stock_list()
st.write(f"Universe: {len(symbols)} tickers loaded.")

if fast_test:
    symbols = symbols[:int(fast_n)]
    st.info(f"Fast test enabled — scanning first {len(symbols)} tickers only.")

if st.button("Start full scan (batched + prefilter + ML)"):
    start_time = time.time()
    # load checkpoint if exists and resume
    checkpoint_df = checkpoint_load()
    processed_symbols = set()
    results = []
    if checkpoint_df is not None and not checkpoint_df.empty:
        processed_symbols = set(checkpoint_df['Stock'].astype(str).tolist())
        results = checkpoint_df.to_dict('records')
        st.info(f"Resumed from checkpoint: {len(processed_symbols)} symbols already processed.")

    # Step 1: cheap prefilter using 30d data to drop many tickers quickly
    st.write("1) Running cheap prefilter (30d slices) to drop low-probability tickers...")
    filtered = batch_prefilter(symbols, batch_size=batch_size, period='30d')
    st.write(f"Prefilter kept {len(filtered)} tickers (out of {len(symbols)}).")

    # Step 2: run ML eval on filtered tickers (multi-threaded)
    to_eval = [s for s in filtered if s.replace('.NS','') not in processed_symbols]
    st.write(f"2) Evaluating {len(to_eval)} tickers with ML (threads={threads}) — this may take time.")
    progress_bar = st.progress(0)
    total = len(to_eval)
    completed = 0
    batch_counter = 0

    # incremental save frequency
    SAVE_FREQ = max(1, int(max(1, len(to_eval) / 20)))  # save ~20 times throughout run
    with ThreadPoolExecutor(max_workers=threads) as ex:
        futures = {ex.submit(evaluate_symbol_full, sym, use_lightgbm and HAS_LGB): sym for sym in to_eval}
        for fut in as_completed(futures):
            sym = futures[fut]
            batch_counter += 1
            try:
                res = fut.result()
                if res:
                    results.append(res)
            except Exception as e:
                st.write(f"Error evaluating {sym}: {e}")
            completed += 1
            progress_bar.progress(completed / total if total>0 else 1.0)

            # periodic checkpoint save
            if completed % SAVE_FREQ == 0 or completed == total:
                try:
                    df_checkpoint = pd.DataFrame(results)
                    checkpoint_save(df_checkpoint)
                except Exception:
                    pass

    # final checkpoint save
    df_results = pd.DataFrame(results)
    checkpoint_save(df_results)

    elapsed = time.time() - start_time
    st.success(f"Scan complete — processed {len(results)} valid tickers in {elapsed/60:.1f} minutes.")

    if df_results.empty:
        st.warning("No results to show. Try increasing time window or reducing batch size.")
    else:
        # compute score and EV metric
        df_results['PotentialPct'] = (df_results['Target'] / df_results['Entry'] - 1) * 100
        df_results['RiskPct'] = (1 - df_results['Stop'] / df_results['Entry']) * 100
        # Expected value (approx) = prob * reward - (1-prob)*risk
        df_results['EV'] = df_results['Proba'] * (df_results['Target'] - df_results['Entry']) - (1 - df_results['Proba']) * (df_results['Entry'] - df_results['Stop'])
        # Suggest position sizing using risk per trade: qty = (risk_pct_of_capital) / (stop_distance)
        df_results['StopDistance'] = df_results['Entry'] - df_results['Stop']
        df_results['PositionSizeValue'] = (risk_per_trade_pct / 100.0) * account_size
        df_results['SuggestedQty'] = (df_results['PositionSizeValue'] / df_results['StopDistance']).apply(lambda x: max(0, math.floor(x)) if pd.notna(x) and x>0 else 0)

        # Ranking
        df_results['Score'] = df_results['Proba'] * (0.6 + 0.4 * df_results['CV'].fillna(0.5)) + 0.2 * (df_results['EV'] / (df_results['Entry']+1e-9))
        df_results = df_results.sort_values('Score', ascending=False).reset_index(drop=True)

        # Show top 10, but highlight top 5
        top5 = df_results.head(5)
        st.markdown("### 🔝 Top 5 BUY suggestions (ranked)")
        display_cols = ['Stock','Entry','Target','Stop','PotentialPct','RiskPct','Proba','CV','EV','SuggestedQty']
        df_display = top5.copy()
        df_display['Entry'] = df_display['Entry'].map(lambda x: f"₹{x:.2f}")
        df_display['Target'] = df_display['Target'].map(lambda x: f"₹{x:.2f}")
        df_display['Stop'] = df_display['Stop'].map(lambda x: f"₹{x:.2f}")
        df_display['PotentialPct'] = df_display['PotentialPct'].map(lambda x: f"{x:.1f}%")
        df_display['RiskPct'] = df_display['RiskPct'].map(lambda x: f"{x:.1f}%")
        df_display['Proba'] = df_display['Proba'].map(lambda x: f"{x:.1%}")
        df_display['CV'] = df_display['CV'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else "N/A")
        st.dataframe(df_display[display_cols], use_container_width=True)

        st.markdown("### 📥 Download full results")
        st.download_button("Download results CSV", df_results.to_csv(index=False).encode('utf-8'), "results_full.csv", "text/csv")

        st.markdown("### 📈 Chart viewer for a pick")
        pick = st.selectbox("Pick stock", df_results['Stock'].head(50).tolist())
        if pick:
            try:
                hist = yf.Ticker(pick + ".NS").history(period=PERIOD)
                if hist is not None and not hist.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'],
                                                low=hist['Low'], close=hist['Close'], name='Price'))
                    hist['ma21'] = hist['Close'].rolling(21).mean()
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['ma21'], name='MA21'))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.write(f"Chart error: {e}")

# End main button flow

st.markdown("---")
st.caption("Notes: This tool is educational. Large scans may take long and may hit Yahoo limits; the script uses prefiltering, caching, backoff and checkpoints to be robust. Consider running on a dedicated VM for full 12k universe runs.")
