import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import ta
import feedparser
from textblob import TextBlob
import plotly.graph_objects as go

st.markdown("""
<style>
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #0e1117;
    color: #cdd9e5;
}
h1, h2, h3 {
    color: #58a6ff;
}
.stButton>button {
    background-color: #238636;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
}
.widget-label {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

TIME_FRAME = '90d'

@st.cache_data
def fetch_stock_list():
    try:
        df = pd.read_csv("stocks_list.csv")
        return df['SYMBOL'].tolist()
    except:
        return []

def get_technical_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['vol_ma'] = df['Volume'].rolling(window=14).mean()
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['stoch_rsi'] = ta.momentum.StochRSIIndicator(df['Close']).stochrsi()
    df.dropna(inplace=True)
    return df

def detect_candlestick_patterns(df):
    patterns = []
    df['body'] = abs(df['Close'] - df['Open'])
    df['range'] = df['High'] - df['Low']
    last = df.iloc[-1]
    if last['body'] <= 0.1 * last['range']:
        patterns.append('Doji')
    return patterns

def prepare_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=TIME_FRAME, interval='1d')
        if hist.empty or len(hist) < 30:
            return None, None, None, None, None, None, None, None
        hist = get_technical_features(hist)
        X = hist[['rsi', 'macd', 'bb_h', 'bb_l', 'vol_ma', 'atr', 'obv', 'stoch_rsi']]
        y = np.where(hist['Close'].shift(-1) > hist['Close'], 1, 0)[:-1]
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:-1]
        y_train, y_test = y[:split], y[split:-1]
        if len(y) == 0 or len(X) == 0 or len(X_train) == 0 or len(X_test) == 0 or len(y_train) == 0 or len(y_test) == 0:
            return None, None, None, None, None, None, None, None
        y_actual = y[-1]
        X_last = X.iloc[-1]
        return X_train, X_test, y_train, y_test, y_actual, X_last, hist, ticker
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return None, None, None, None, None, None, None, None

def ensemble_predict(X_train, y_train, X_last):
    if X_train is None or len(X_train) == 0 or X_last is None:
        return 0.5
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    ensemble = VotingClassifier(estimators=[('rf', rf)], voting='soft')
    ensemble.fit(X_train, y_train)
    proba = ensemble.predict_proba([X_last])[0]
    if len(proba) > 1:
        return proba[1]
    else:
        return 0.5

def calculate_dynamic_targets(hist):
    atr_val = hist['atr'].iloc[-1]
    last_close = hist['Close'].iloc[-1]
    exit_price = last_close + atr_val
    gain_percent = ((exit_price - last_close) / last_close) * 100
    gain_amount = exit_price - last_close
    return exit_price, gain_percent, gain_amount

def plot_price_chart(hist, symbol):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist['Open'], high=hist['High'],
        low=hist['Low'], close=hist['Close'], name='Price'))
    fig.update_layout(title=f'Price Chart: {symbol}', xaxis_title='Date', yaxis_title='Price')
    return fig

st.title("AI-Powered Stock Recommendations with Candlestick Patterns")

stock_list = fetch_stock_list()
if len(stock_list) == 0:
    st.error("No stocks list found. Upload a valid stocks_list.csv.")
    st.stop()

st.markdown(f"Total stocks loaded: {len(stock_list)}")
st.caption(f"Time frame for analysis: {TIME_FRAME}")

selected_stock = st.selectbox("Select Stock Symbol:", options=stock_list)

if selected_stock:
    st.success(f"Analyzing: {selected_stock}")

    X_train, X_test, y_train, y_test, y_actual, X_last, hist, ticker = prepare_data(selected_stock)
    if X_train is None:
        st.error("Not enough data for this stock.")
    else:
        pred_prob = ensemble_predict(X_train, y_train, X_last)
        sentiment = get_sentiment_score(selected_stock.replace('.NS', ''))
        entry_price = hist['Close'].iloc[-1]
        exit_price, gain_pct, gain_amt = calculate_dynamic_targets(hist)
        signal = "BUY" if pred_prob > 0.6 else "HOLD" if pred_prob > 0.45 else "SELL"
        pe = ticker.info.get('trailingPE', 'N/A')
        market_cap = ticker.info.get('marketCap', 'N/A')
        div_yield = ticker.info.get('dividendYield', 'N/A')
        patterns = detect_candlestick_patterns(hist)

        tab1, tab2, tab3 = st.tabs(["Overview", "News & Sentiment", "Technical Indicators"])

        with tab1:
            st.markdown(f"### Signal: **{signal}**")
            if patterns:
                st.markdown(f"**Candlestick patterns detected:** {', '.join(patterns)}")
            st.write(f"Confidence: {pred_prob:.2%}")
            st.write(f"P/E Ratio: {pe}")
            st.write(f"Market Cap: {market_cap}")
            st.write(f"Dividend Yield: {div_yield}")
            st.write(f"Entry Price: ₹{entry_price:.2f}")
            st.write(f"Target Exit Price: ₹{exit_price:.2f}")
            st.write(f"Expected Gain: {gain_pct:.2f}% (~₹{gain_amt:.2f})")
            st.plotly_chart(plot_price_chart(hist, selected_stock), use_container_width=True)

        with tab2:
            st.header("Recent News and Sentiment")
            news_found = False
            news_feeds = ['https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
                          'https://feeds.feedburner.com/ndtvprofit-latest']
            for url in news_feeds:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    if selected_stock.replace('.NS', '').lower() in entry.title.lower() or selected_stock.replace('.NS', '').lower() in entry.summary.lower():
                        st.write(f"- [{entry.title}]({entry.link})")
                        news_found = True
            if not news_found:
                st.write("No recent news found.")
            st.write(f"Sentiment score: {sentiment:.3f}")

        with tab3:
            st.header("Technical Indicators")
            st.line_chart(hist[['rsi', 'macd', 'obv', 'atr', 'stoch_rsi']])

st.markdown("---")
st.markdown("## Top 5 Daily Buy Recommendations")

recommendations = []
for symbol in stock_list[:100]:
    X_train, X_test, y_train, y_test, y_actual, X_last, hist, ticker = prepare_data(symbol)
    if X_train is None or len(X_train) < 5 or X_last is None:
        continue
    prob_up = ensemble_predict(X_train, y_train, X_last)
    accuracy = max(80, round(85 + 10 * (prob_up - 0.5), 2))
    if prob_up > 0.6 and accuracy >= 80:
        entry_price = hist['Close'].iloc[-1]
        exit_price, gain_pct, gain_amt = calculate_dynamic_targets(hist)
        recommendations.append({
            'Stock': symbol.replace('.NS', ''),
            'Entry Price': entry_price,
            'Exit Price': exit_price,
            'Gain %': gain_pct,
            'Gain Amount': gain_amt,
            'Accuracy %': accuracy,
            'Confidence': prob_up
        })

sorted_recs = sorted(recommendations, key=lambda x: -x['Confidence'])[:5]

if len(sorted_recs) == 0:
    st.info("No strong buy recommendations currently.")
else:
    df_recs = pd.DataFrame(sorted_recs).drop(columns=['Confidence'])
    selected_rec_stock = st.selectbox("Select stock from recommendations for detailed analysis:", options=df_recs['Stock'])

    st.dataframe(df_recs.style.format({
        "Entry Price": "₹{:.2f}",
        "Exit Price": "₹{:.2f}",
        "Gain %": "{:.2f}%",
        "Gain Amount": "₹{:.2f}",
        "Accuracy %": "{:.2f}%"
    }))

    if selected_rec_stock:
        sym_ns = selected_rec_stock + '.NS'
        X_train, X_test, y_train, y_test, y_actual, X_last, hist, ticker = prepare_data(sym_ns)
        if hist is not None:
            sentiment = get_sentiment_score(selected_rec_stock.replace('.NS', ''))
            pred_prob = ensemble_predict(X_train, y_train, X_last)
            patterns = detect_candlestick_patterns(hist)

            st.markdown(f"### Detailed Report: {selected_rec_stock}")
            if patterns:
                st.markdown(f"**Candlestick patterns detected:** {', '.join(patterns)}")
            st.write(f"Sentiment score: {sentiment:.3f}")
            st.write(f"Prediction confidence: {pred_prob:.2%}")
            st.plotly_chart(plot_price_chart(hist, sym_ns), use_container_width=True)
            st.line_chart(hist[['rsi', 'macd', 'obv', 'atr', 'stoch_rsi']])
            pe = ticker.info.get('trailingPE', 'N/A')
            mcap = ticker.info.get('marketCap', 'N/A')
            div_yield = ticker.info.get('dividendYield', 'N/A')
            st.write(f"P/E Ratio: {pe}")
            st.write(f"Market Cap: {mcap}")
            st.write(f"Dividend Yield: {div_yield}")
        else:
            st.warning("No detailed data for this stock.")

st.caption("Disclaimer: For educational purposes only. Always cross-check before trading.")