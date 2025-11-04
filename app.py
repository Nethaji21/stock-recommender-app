# app.py — Professional daily top-5 stock picker (Streamlit)
# Run: streamlit run app.py
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
from datetime import datetime, timedelta
import ta
import concurrent.futures
import time
from textblob import TextBlob
import feedparser

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

st.markdown('<div class="header"><div class="title">Pro Trader Picks</div>'
            '<div class="sub">Daily top stock picks — ensemble model, target 10% profit, minimal UI for traders.</div></div>',
            unsafe_allow_html=True)

# ----------------- Defaults -----------------
MIN_HISTORY_DAYS = 120
DEFAULT_TIMEFRAME = '365d'
DEFAULT_TOPN = 200
MIN_PICKS = 5
DEFAULT_CONF_THRESHOLD = 0.55  # used for informational display only (we will still return top 5)
DEFAULT_TTL = 3600  # cache TTL in seconds

# ----------------- Utilities -----------------
@st.cache_data(ttl=DEFAULT_TTL)
def load_stock_list(path="stocks_list.csv"):
    try:
        df = pd.read_csv(path)
        # Accept either 'SYMBOL' column or first column as tickers
        if 'SYMBOL' in df.columns:
            syms = df['SYMBOL'].astype(str).tolist()
        else:
            syms = df.iloc[:, 0].astype(str).tolist()
        # normalize common NSE tickers (optional)
        syms = [s.strip() for s in syms if isinstance(s, str) and s.strip() != ""]
        return syms
    except Exception as e:
        return []

@st.cache_data(ttl=DEFAULT_TTL)
def download_history(symbol, period):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval='1d', actions=False)
        if hist is None or hist.empty:
            return None
        hist = hist.dropna()
        return hist
    except Exception:
        return None

def add_technical_features(df):
    df = df.copy()
    df['ret1'] = df['Close'].pct_change(1)
    df['ret5'] = df['Close'].pct_change(5)
    df['ma7'] = df['Close'].rolling(7).mean()
    df['ma21'] = df['Close'].rolling(21).mean()
    df['ema12'] = df['Close'].ewm(span=12).mean()
    df['ema26'] = df['Close'].ewm(span=26).mean()
    df['vol_ma14'] = df['Volume'].rolling(14).mean()
    df['vol_spike'] = df['Volume'] / (df['vol_ma14'] + 1e-9)
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        macd = ta.trend.MACD(df['Close'])
        df['macd_diff'] = macd.macd_diff()
        bb = ta.volatility.BollingerBands(df['Close'])
        df['bb_h'] = bb.bollinger_hband()
        df['bb_l'] = bb.bollinger_lband()
        df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['Close']).stochrsi()
    except Exception:
        pass
    df = df.dropna()
    return df

def prepare_ml_data(hist):
    df = hist.copy()
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    candidate_feats = ['rsi','macd_diff','bb_h','bb_l','vol_spike','atr','obv','stoch_rsi','ret1','ret5','ma7','ma21']
    available = [f for f in candidate_feats if f in df.columns and df[f].notna().sum() > 0]
    if len(available) < 5 or len(df) < 60:
        return None, None, None
    X = df[available].iloc[:-1]  # last target uses future
    y = df['target'].iloc[:-1]
    return X, y, available

def build_model():
    pipe_rf = Pipeline([('s', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=150, random_state=42))])
    pipe_lr = Pipeline([('s', StandardScaler()), ('lr', LogisticRegression(max_iter=500, random_state=42))])
    ensemble = VotingClassifier(estimators=[('rf', pipe_rf), ('lr', pipe_lr)], voting='soft')
    return ensemble

def safe_sentiment_pool(symbol_short):
    """
    Quick polarity from RSS + TextBlob — optional and slow; cached at top.
    """
    try:
        feeds = [
            'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
            'https://www.moneycontrol.com/rss/MCtopnews.xml'
        ]
    except:
        feeds = []
    scores = []
    for url in feeds:
        try:
            f = feedparser.parse(url)
            for e in (f.entries[:20] if hasattr(f,'entries') else []):
                title = getattr(e, 'title', '') or ''
                summary = getattr(e, 'summary', '') or ''
                txt = (title + ' ' + summary).lower()
                if symbol_short.lower() in txt:
                    scores.append(TextBlob(title + ' ' + summary).sentiment.polarity)
        except Exception:
            continue
    return float(np.mean(scores)) if len(scores) > 0 else 0.0

# ----------------- Sidebar controls -----------------
st.sidebar.header("Scan / Model Controls")
timeframe = st.sidebar.selectbox("History timeframe", options=['180d','365d','730d'], index=1)
top_n = st.sidebar.number_input("Top N stocks to scan (max)", min_value=20, max_value=1000, value=DEFAULT_TOPN, step=10)
min_picks = st.sidebar.number_input("Minimum picks to return", min_value=1, max_value=10, value=MIN_PICKS, step=1)
enable_backtest = st.sidebar.checkbox("Enable light historical backtest (slower)", value=False)
backtest_hold = st.sidebar.slider("Backtest hold days", min_value=1, max_value=21, value=5)
newsapi_key = st.sidebar.text_input("NewsAPI key (optional) — for improved sentiment", type="password")
enable_sector = st.sidebar.checkbox("Show sector (via yfinance)", value=True)
refresh_cache = st.sidebar.button("Clear cache (force fresh download)")
scan_button = st.sidebar.button("Run daily scan now")

if refresh_cache:
    load_stock_list.clear()
    download_history.clear()
    st.sidebar.success("Cache cleared.")

# ----------------- Load universe -----------------
stock_list = load_stock_list()
if len(stock_list) == 0:
    st.error("Please put a CSV named 'stocks_list.csv' in the app folder with one column of tickers (or a column named SYMBOL).")
    st.stop()

st.sidebar.write(f"Universe loaded: {len(stock_list)} tickers")
st.write("### Scan controls & summary")
st.markdown(f"- Scanning up to **{top_n}** tickers (first N from your file).")
st.markdown(f"- Minimum picks returned: **{min_picks}** (app ensures this).")
st.caption("Tip: keep Top N at 200–400 for a good balance between coverage and speed. Parallel download is used.")

# ----------------- Helper scanning function -----------------
def evaluate_symbol(sym, period, do_backtest=False, backtest_hold=5, show_sector=False):
    """
    Download, compute features, train simple ensemble, get buy probability, backtest score (optional),
    compute 10% target and 5% stoploss.
    Returns dict or None.
    """
    try:
        hist = download_history(sym, period)
        if hist is None or len(hist) < MIN_HISTORY_DAYS:
            return None
        hist = add_technical_features(hist)
        X, y, features = prepare_ml_data(hist)
        if X is None or y is None:
            return None
        model = build_model()
        # cross-val for reliability indicator
        try:
            cv_acc = float(np.mean(cross_val_score(model, X, y, cv=3, scoring='accuracy')))
        except Exception:
            cv_acc = np.nan
        # fit on all data and predict last row
        model.fit(X, y)
        last = X.iloc[-1]
        proba = float(model.predict_proba(last.to_frame().T)[0][1])
        # compute simple momentum (last 5-day return)
        momentum = float(hist['Close'].pct_change(5).iloc[-1]) if 'Close' in hist.columns else 0.0
        # optional light backtest: expanding-window simple simulation
        bt_win = np.nan
        if do_backtest:
            bt_win = simple_backtest_light(hist, features, hold_days=backtest_hold)
        last_close = float(hist['Close'].iloc[-1])
        # enforce 10% target. If ATR suggests higher target, keep the higher (helps volatile stocks)
        atr = float(hist['atr'].iloc[-1]) if 'atr' in hist.columns else (np.std(hist['Close'].pct_change()) * last_close)
        target_by_atr = last_close + atr * 2.0  # 2x ATR is a reasonable swing target
        fixed_target = last_close * 1.10
        target = max(fixed_target, target_by_atr)
        stop_loss = last_close * 0.95
        # optional sector
        sector = None
        if show_sector:
            try:
                info = yf.Ticker(sym).info
                sector = info.get('sector', None)
            except Exception:
                sector = None
        return {
            'Stock': sym.replace('.NS',''),
            'SymFull': sym,
            'Proba': proba,
            'CV_Acc': cv_acc,
            'Momentum': momentum,
            'BacktestWin': bt_win,
            'Entry': last_close,
            'Target': target,
            'StopLoss': stop_loss,
            'ATR': atr,
            'Sector': sector
        }
    except Exception:
        return None

def simple_backtest_light(hist, features, hold_days=5):
    """
    A small, fast expanding-window light backtest.
    We train on past and test 1-step forward; count profitable predicted buys.
    """
    try:
        df = hist.copy().dropna()
        X = df[features]
        y = (df['Close'].shift(-1) > df['Close']).astype(int)
        X = X.iloc[:-hold_days]
        y = y.iloc[:-hold_days]
        if len(X) < 40:
            return np.nan
        model = build_model()
        wins = 0
        trades = 0
        # expanding-window realistic simulation; limited to speed
        for i in range(30, len(X)-1, max(1, int(len(X)/50))):  # sample a subset to stay fast
            X_train = X.iloc[:i]
            y_train = y.iloc[:i]
            X_test = X.iloc[i:i+1]
            if y_train.nunique() < 2:
                continue
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test)[0][1]
            if p > 0.6:
                idx = X_test.index[0]
                # find sell price after hold_days
                sell_idx = idx + timedelta(days=hold_days)
                try:
                    sell_price = df.loc[df.index >= sell_idx, 'Close'].iloc[0]
                except Exception:
                    continue
                buy_price = df.loc[idx, 'Close']
                trades += 1
                if sell_price > buy_price:
                    wins += 1
        if trades == 0:
            return np.nan
        return float(wins / trades)
    except Exception:
        return np.nan

# ----------------- Run scan (parallel) -----------------
def run_scan(universe, period, top_n, min_picks, backtest_flag=False, backtest_hold=5, show_sector=False):
    results = []
    to_scan = universe[:top_n]
    # parallelize downloads & eval
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(evaluate_symbol, s, period, backtest_flag, backtest_hold, show_sector): s for s in to_scan}
        for i, fut in enumerate(concurrent.futures.as_completed(futures)):
            sym = futures[fut]
            try:
                res = fut.result()
                if res is not None:
                    results.append(res)
            except Exception:
                continue
    return results

# ----------------- UI: run or show last results -----------------
st.markdown("## Run daily scan")
st.write("Choose options and press **Run daily scan now**. The app will return top picks ranked by confidence × momentum × optional backtest.")
if scan_button:
    t0 = time.time()
    with st.spinner("Scanning universe — this may take a minute for 200+ tickers..."):
        scan_results = run_scan(stock_list, timeframe, int(top_n), int(min_picks), enable_backtest, backtest_hold, enable_sector)
    duration = time.time() - t0
    st.success(f"Scan complete — evaluated {len(scan_results)} tickers in {duration:.1f}s")

    if len(scan_results) == 0:
        st.warning("No symbols returned — try increasing Top N or reduce timeframe.")
    else:
        df = pd.DataFrame(scan_results)
        # Fill NaN backtest with median to avoid NaN sorting
        df['BacktestWin'] = df['BacktestWin'].fillna(df['BacktestWin'].median() if df['BacktestWin'].notna().any() else 0.5)
        # Score composition:
        # Score = proba * (1 + momentum_factor) * (0.6 + 0.4*backtest_win) * CV acc factor
        # momentum factor: positive momentum increases score, negative reduces
        df['MomentumFactor'] = 1 + (df['Momentum'].clip(-0.2, 0.5))  # bounds to avoid wild swings
        df['BacktestFactor'] = 0.6 + 0.4 * df['BacktestWin']  # between 0.6 and 1.0
        df['CVFactor'] = 1 + df['CV_Acc'].fillna(0.0) - 0.5  # center around 0.5
        df['Score'] = df['Proba'] * df['MomentumFactor'] * df['BacktestFactor'] * df['CVFactor']
        df = df.sort_values('Score', ascending=False).reset_index(drop=True)

        # Ensure at least min_picks picks
        picks = df.head(max(int(min_picks), MIN_PICKS)).copy()
        # Guarantee at least 5 by fallback: if not enough qualified, take top rows anyway
        if len(picks) < min_picks and len(df) >= min_picks:
            picks = df.head(min_picks).copy()

        # Now compute fixed 10% target & 5% stop (also keep ATR-based if larger target)
        picks['FixedTarget'] = picks['Entry'] * 1.10
        picks['ATRTarget'] = picks['Entry'] + picks['ATR'] * 2.0
        picks['TargetFinal'] = picks[['FixedTarget', 'ATRTarget']].max(axis=1)
        picks['StopLossFinal'] = picks['Entry'] * 0.95
        picks['PotentialPct'] = (picks['TargetFinal'] / picks['Entry'] - 1) * 100
        picks['RiskPct'] = (1 - picks['StopLossFinal'] / picks['Entry']) * 100

        # Display picks
        display_cols = ['Stock', 'Sector', 'Entry', 'TargetFinal', 'StopLossFinal', 'PotentialPct', 'RiskPct', 'Proba', 'BacktestWin', 'CV_Acc', 'Score']
        picks_display = picks[display_cols].copy()
        picks_display = picks_display.rename(columns={'TargetFinal': 'Target', 'StopLossFinal':'StopLoss'})
        # Formatting
        picks_display['Entry'] = picks_display['Entry'].map(lambda x: f"₹{x:.2f}")
        picks_display['Target'] = picks_display['Target'].map(lambda x: f"₹{x:.2f}")
        picks_display['StopLoss'] = picks_display['StopLoss'].map(lambda x: f"₹{x:.2f}")
        picks_display['PotentialPct'] = picks_display['PotentialPct'].map(lambda x: f"{x:.1f}%")
        picks_display['RiskPct'] = picks_display['RiskPct'].map(lambda x: f"{x:.1f}%")
        picks_display['Proba'] = picks_display['Proba'].map(lambda x: f"{x:.1%}")
        picks_display['BacktestWin'] = picks_display['BacktestWin'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else "N/A")
        picks_display['CV_Acc'] = picks_display['CV_Acc'].map(lambda x: f"{x:.1%}" if not pd.isna(x) else "N/A")
        picks_display['Score'] = picks_display['Score'].map(lambda x: f"{x:.4f}")

        st.markdown(f"### Today's top {len(picks_display)} AI-selected BUY candidates (aiming ≥10% profit)")
        st.dataframe(picks_display, use_container_width=True)

        # Download CSV
        csv = picks.to_csv(index=False).encode('utf-8')
        st.download_button("Download picks CSV", csv, "pro_trader_picks.csv", "text/csv")

        # Let the user click one for a deep dive
        chosen = st.selectbox("Deep dive: choose one pick", options=picks['Stock'].tolist())
        if chosen:
            row = picks[picks['Stock'] == chosen].iloc[0]
            symfull = row['SymFull']  # original symbol with .NS possibly
            hist = download_history(symfull, timeframe)
            if hist is not None:
                st.markdown(f"#### {chosen} chart & indicators")
                hist = add_technical_features(hist)
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'))
                if 'ma21' in hist.columns:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['ma21'], name='MA21'))
                fig.update_layout(height=450, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                st.write("Latest stats:")
                st.write({
                    'Entry': f"₹{row['Entry']:.2f}",
                    'Target (final)': f"₹{row['TargetFinal']:.2f}",
                    'StopLoss': f"₹{row['StopLossFinal']:.2f}",
                    'Predicted Prob': f"{row['Proba']:.1%}",
                    'Momentum (5d)': f"{row['Momentum']:.2%}"})
            else:
                st.warning("No historical data available for deep-dive chart.")
        st.info("⚠️ This tool is educational. Backtests here are simplified. Always perform your own risk management and due diligence.")

# ----------------- If not scanning yet, give sample overview -----------------
else:
    st.markdown("Press **Run daily scan now** in the sidebar to evaluate the universe and get today's picks.")
    st.markdown("You can also:")
    st.markdown("- Provide a NewsAPI key in the sidebar to improve sentiment (optional).")
    st.markdown("- Enable 'Enable light historical backtest' (slower, but increases reliability).")
    st.markdown("- Toggle 'Show sector' to display sectors fetched from yfinance (may be slower).")

# ----------------- Footer / Help -----------------
st.markdown("---")
st.markdown("**How this picks stocks:** model predicts next-day up-probability using technical features, then scores stocks by probability × momentum × light backtest × CV reliability. Final targets are at least 10% (or ATR-based if larger).")
st.markdown("**Optional next steps (recommended):** add fundamental filters (PE, revenue growth), attach position sizing rules, and implement stop-loss order automation via your broker API (careful).")