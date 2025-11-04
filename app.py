# app.py (enhanced)
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
from sklearn.metrics import accuracy_score, precision_score, recall_score
import ta
import feedparser
from textblob import TextBlob
from datetime import timedelta

st.set_page_config(page_title="Smart Stocks — Enhanced", layout="wide")

# ---- Styling ----
st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(180deg,#071021,#071023 40%); color:#cfd8e3; }
    .header {background: linear-gradient(90deg,#0f1724,#07203a); padding:18px; border-radius:12px; box-shadow: 0 4px 20px rgba(0,0,0,0.6);}
    .big-title {font-size:28px; color:#58a6ff; margin:0; font-weight:700;}
    .sub {color:#9fb4d7; margin-top:4px; margin-bottom:0;}
    .card {background:#071024; border: 1px solid rgba(255,255,255,0.04); padding:12px; border-radius:10px;}
    .metric {font-size:20px; color:#d7e8ff; font-weight:600;}
    .stButton>button {background-color:#238636; color:white; border-radius:8px; height:3em;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Header
with st.container():
    st.markdown('<div class="header"><p class="big-title">Smart Stocks — AI Recommendations</p>'
                '<p class="sub">Improved model, better features, backtest & cleaner UI — educational only.</p></div>',
                unsafe_allow_html=True)

# ---- Constants / Defaults ----
DEFAULT_TIMEFRAME = '180d'   # default lookback for features
MIN_HISTORY_DAYS = 60        # minimum history needed
DEFAULT_TOPN_SCAN = 100
DEFAULT_CONF_THRESHOLD = 0.62
DEFAULT_HOLD_DAYS = 5

# ---- Utilities & Caching ----
@st.cache_data(ttl=3600)
def fetch_stock_list():
    try:
        df = pd.read_csv("stocks_list.csv")
        if 'SYMBOL' in df.columns:
            return df['SYMBOL'].astype(str).tolist()
        else:
            # try first column
            return df.iloc[:,0].astype(str).tolist()
    except Exception as e:
        return []

def safe_parse(feed_url):
    try:
        return feedparser.parse(feed_url)
    except:
        return None

@st.cache_data(ttl=900)
def get_sentiment_score(symbol_short):
    """Basic sentiment from two feeds + TextBlob polarity aggregated."""
    scores = []
    news_feeds = [
        'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
        'https://feeds.feedburner.com/ndtvprofit-latest'
    ]
    for url in news_feeds:
        feed = safe_parse(url)
        if feed and hasattr(feed, 'entries'):
            for entry in feed.entries[:12]:
                title = getattr(entry, 'title', '') or ''
                summary = getattr(entry, 'summary', '') or ''
                txt = (title + ' ' + summary).lower()
                if symbol_short.lower() in txt:
                    scores.append(TextBlob(title + ' ' + summary).sentiment.polarity)
    return float(np.mean(scores)) if len(scores) > 0 else 0.0

@st.cache_data(ttl=1800)
def download_history(symbol, period):
    try:
        t = yf.Ticker(symbol)
        hist = t.history(period=period, interval='1d', actions=False)
        if hist is None or hist.empty:
            return None
        hist = hist.dropna()
        return hist
    except Exception as e:
        return None

# ---- Feature engineering ----
def add_technical_features(df):
    df = df.copy()
    # returns
    df['ret1'] = df['Close'].pct_change(1)
    df['ret5'] = df['Close'].pct_change(5)
    df['ma7'] = df['Close'].rolling(7).mean()
    df['ma21'] = df['Close'].rolling(21).mean()
    df['ema12'] = df['Close'].ewm(span=12).mean()
    df['ema26'] = df['Close'].ewm(span=26).mean()
    df['vol_ma14'] = df['Volume'].rolling(14).mean()
    df['vol_spike'] = df['Volume'] / (df['vol_ma14'] + 1e-9)
    # TA library features
    try:
        df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
        df['macd_diff'] = ta.trend.MACD(df['Close']).macd_diff()
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

# ---- Modeling helpers ----
def prepare_ml_data(hist):
    # expects historical dataframe with features added
    df = hist.copy()
    # target: next-day close > today close
    df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df = df.dropna()
    # use a set of robust features
    feats = ['rsi','macd_diff','bb_h','bb_l','vol_spike','atr','obv','stoch_rsi','ret1','ret5','ma7','ma21']
    available = [f for f in feats if f in df.columns and df[f].notna().sum() > 0]
    if len(available) < 5:
        return None, None, None
    X = df[available].iloc[:-1]  # drop last because target uses shift(-1)
    y = df['target'].iloc[:-1]
    return X, y, available

def build_model():
    # pipeline for both models
    pipe1 = Pipeline([('scaler', StandardScaler()), ('rf', RandomForestClassifier(n_estimators=120, random_state=42))])
    pipe2 = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression(max_iter=400, random_state=42))])
    ensemble = VotingClassifier(estimators=[('rf', pipe1), ('lr', pipe2)], voting='soft')
    return ensemble

def model_predict_proba(model, X_train, y_train, X_last):
    model.fit(X_train, y_train)
    try:
        proba = model.predict_proba(X_last.reshape(1, -1))[0][1]
    except:
        proba = model.predict_proba(X_last.to_frame().T)[0][1]
    return float(proba)

def cross_validate_model(model, X, y):
    try:
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        return float(np.mean(scores))
    except:
        return None

# ---- Simple backtest simulate buy next day and hold N days ----
def simple_backtest(hist, features, hold_days=DEFAULT_HOLD_DAYS):
    """
    For each row i in hist where we can compute features and target,
    simulate: if model predicted up, buy next open and sell after hold_days close.
    Here we do a simplified historical simulation using a model trained in-expanding manner.
    We'll compute historical accuracy: proportion of trades that were profitable.
    """
    df = hist.copy().dropna()
    X = df[features]
    y = (df['Close'].shift(-1) > df['Close']).astype(int)
    X = X.iloc[:-hold_days]
    y = y.iloc[:-hold_days]
    if len(X) < 30:
        return None
    model = build_model()
    wins = 0
    trades = 0
    # expanding-window training for realistic test
    for i in range(30, len(X)-1):
        X_train = X.iloc[:i]
        y_train = y.iloc[:i]
        X_test_row = X.iloc[i:i+1]
        if y_train.nunique() < 2:
            continue
        try:
            model.fit(X_train, y_train)
            p = model.predict_proba(X_test_row)[0][1]
            if p > 0.6:  # historically take predicted buys only
                buy_index = X_test_row.index[0]
                sell_index = buy_index + timedelta(days=hold_days)
                # find closest sell index in df (index is date)
                try:
                    sell_price = df.loc[df.index >= sell_index, 'Close'].iloc[0]
                except:
                    # if not enough days left, skip
                    continue
                buy_price = df.loc[buy_index, 'Close']
                trades += 1
                if sell_price > buy_price:
                    wins += 1
        except Exception:
            continue
    if trades == 0:
        return None
    return float(wins / trades)

# ---- Sidebar controls ----
st.sidebar.header("Scan & Model Settings")
timeframe = st.sidebar.selectbox("Feature timeframe (history)", options=['90d','180d','365d','730d'], index=1)
top_n = st.sidebar.number_input("Top N stocks to scan (performance sensitive)", min_value=10, max_value=500, value=DEFAULT_TOPN_SCAN, step=10)
confidence_threshold = st.sidebar.slider("Confidence threshold for BUY", min_value=0.51, max_value=0.9, value=float(DEFAULT_CONF_THRESHOLD), step=0.01)
hold_days = st.sidebar.slider("Hypothetical hold days for target", min_value=1, max_value=21, value=int(DEFAULT_HOLD_DAYS), step=1)
min_backtest_win = st.sidebar.slider("Minimum historical backtest win-rate (for recommendation)", min_value=0.0, max_value=1.0, value=0.55, step=0.01)
run_scan = st.sidebar.button("Run Scan")

# ---- Load stocks ----
stock_list = fetch_stock_list()
if not stock_list:
    st.error("No stocks list found. Make sure stocks_list.csv exists and has SYMBOL column or tickers in first column.")
    st.stop()

st.markdown(f"**Stocks loaded:** {len(stock_list)} · analyzing top {top_n} (first N from file)")
st.caption("Note: this is experimental and educational — backtest results are simplified estimates, not trading advice.")

# ---- Single-stock detailed view on the left ----
col1, col2 = st.columns([2,1])

with col2:
    st.markdown("### Quick Analyzer")
    selected_stock = st.selectbox("Select stock for deep analysis", options=stock_list, index=0)
    if selected_stock:
        hist = download_history(selected_stock, period=timeframe)
        if hist is None or len(hist) < MIN_HISTORY_DAYS:
            st.warning("Not enough history for this symbol.")
        else:
            hist = add_technical_features(hist)
            X, y, features = prepare_ml_data(hist)
            if X is None:
                st.warning("Insufficient features computed for ML.")
            else:
                model = build_model()
                cv_acc = cross_validate_model(model, X, y) or 0.0
                # fit and predict last
                model.fit(X, y)
                last_row = X.iloc[-1]
                proba = float(model.predict_proba(last_row.to_frame().T)[0][1])
                sentiment = get_sentiment_score(selected_stock.replace('.NS',''))
                last_close = hist['Close'].iloc[-1]
                # targets based on ATR
                atr = hist['atr'].iloc[-1] if 'atr' in hist.columns else (hist['Close'].pct_change().std()*last_close)
                target = last_close + atr * 1.0
                stop_loss = last_close - atr * 0.8

                st.markdown(f"#### {selected_stock} — Latest snapshot")
                st.metric("Model BUY probability", f"{proba:.2%}", delta=None)
                st.metric("CV accuracy (3-fold)", f"{cv_acc:.2%}")
                st.metric("Sentiment (news polarity)", f"{sentiment:.3f}")

                # Price chart with overlays
                fig = go.Figure()
                fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close'], name='Price'))
                if 'ma21' in hist.columns:
                    fig.add_trace(go.Scatter(x=hist.index, y=hist['ma21'], name='MA21', mode='lines', line=dict(width=1)))
                if 'rsi' in hist.columns:
                    fig.update_layout(yaxis2=dict(title='RSI', overlaying='y', side='right', range=[0,100]))
                fig.update_layout(title=f"{selected_stock} Price", height=420, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Feature importances / details"):
                    # simple RF feature importance
                    try:
                        rf = RandomForestClassifier(n_estimators=80, random_state=42)
                        rf.fit(X, y)
                        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                        st.write(importances.to_frame("importance").head(12))
                    except Exception:
                        st.write("Feature importance not available.")

# ---- Bulk scan & recommendations ----
with col1:
    st.markdown("## Scan & Recommendations")

    if run_scan:
        progress = st.progress(0)
        recommendations = []
        scanned = 0
        total_to_scan = min(len(stock_list), top_n)
        for i, sym in enumerate(stock_list[:total_to_scan]):
            scanned += 1
            progress.progress(int(scanned/total_to_scan * 100))
            hist = download_history(sym, period=timeframe)
            if hist is None or len(hist) < MIN_HISTORY_DAYS:
                continue
            hist = add_technical_features(hist)
            X, y, features = prepare_ml_data(hist)
            if X is None or len(X) < 40:
                continue
            model = build_model()
            try:
                # cross-val accuracy estimate
                cv_acc = cross_validate_model(model, X, y) or 0.0
                model.fit(X, y)
                last_row = X.iloc[-1]
                proba = float(model.predict_proba(last_row.to_frame().T)[0][1])
                # quick historical backtest (lightweight)
                backtest_win = simple_backtest(hist, features, hold_days=hold_days) or 0.0
                last_close = hist['Close'].iloc[-1]
                atr = hist['atr'].iloc[-1] if 'atr' in hist.columns else (hist['Close'].pct_change().std()*last_close)
                target = last_close + atr * 1.0
                # final filter
                if proba >= confidence_threshold and backtest_win >= min_backtest_win:
                    recommendations.append({
                        'Stock': sym.replace('.NS',''),
                        'Proba': proba,
                        'CV Acc': cv_acc,
                        'Backtest Win': backtest_win,
                        'Entry': last_close,
                        'Target': target,
                        'ATR': atr
                    })
            except Exception:
                continue

        progress.empty()
        if len(recommendations) == 0:
            st.info("No recommendations found with current filters. Try lowering thresholds or increasing Top N.")
        else:
            rec_df = pd.DataFrame(recommendations).sort_values(by='Proba', ascending=False).reset_index(drop=True)
            rec_df_display = rec_df.copy()
            rec_df_display['Proba'] = rec_df_display['Proba'].map("{:.2%}".format)
            rec_df_display['CV Acc'] = rec_df_display['CV Acc'].map(lambda x: f"{x:.2%}" if not pd.isna(x) else "N/A")
            rec_df_display['Backtest Win'] = rec_df_display['Backtest Win'].map("{:.2%}".format)
            rec_df_display['Entry'] = rec_df_display['Entry'].map(lambda x: f"₹{x:.2f}")
            rec_df_display['Target'] = rec_df_display['Target'].map(lambda x: f"₹{x:.2f}")
            st.write(f"### Top recommendations ({len(rec_df)} found)")
            st.dataframe(rec_df_display)

            # Download button
            csv = rec_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download recommendations CSV", csv, "recs.csv", "text/csv")

            # allow pick one for deep dive
            chosen = st.selectbox("Deep dive into recommendation", options=rec_df['Stock'])
            if chosen:
                chosen_sym = chosen + '.NS' if not chosen.endswith('.NS') else chosen
                hist = download_history(chosen_sym, period=timeframe)
                hist = add_technical_features(hist)
                X, y, features = prepare_ml_data(hist)
                if X is not None:
                    model = build_model()
                    model.fit(X, y)
                    last_row = X.iloc[-1]
                    proba = float(model.predict_proba(last_row.to_frame().T)[0][1])
                    sentiment = get_sentiment_score(chosen)
                    st.markdown(f"### {chosen} — Detailed report")
                    st.write(f"Model probability: {proba:.2%}")
                    st.write(f"Sentiment: {sentiment:.3f}")
                    st.write("Feature set used:", features)
                    st.line_chart(hist[['Close']].tail(120))
                    # show feature importance if RF available
                    try:
                        rf = RandomForestClassifier(n_estimators=80, random_state=42)
                        rf.fit(X, y)
                        importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
                        st.write(importances.head(10))
                    except:
                        pass

# Footer / Disclaimer
st.markdown("---")
st.caption("Disclaimer: This tool is for educational/demonstration purposes only. Not financial advice. Always DYOR before trading.")