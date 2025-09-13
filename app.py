import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import ta
import feedparser
from textblob import TextBlob
import datetime

# --- Constants and stock lists ---

# Example NSE multi-cap stocks; replace with full NSE + BSE large/mid/small cap lists as needed
STOCKS = {
    'TATAMOTORS.NS': 'Tata Motors',
    'HDFCBANK.NS': 'HDFC Bank',
    'RELIANCE.NS': 'Reliance Industries',
    'INFY.NS': 'Infosys',
    'SBIN.NS': 'SBI',
    'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'PNB.NS': 'Punjab National Bank'
}

LOOKBACK_PERIOD = '1y'  # 1 year historical data for model training

NEWS_FEEDS = [
    'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
    'https://feeds.feedburner.com/ndtvprofit-latest'
]

# --- Feature engineering functions ---

def get_technical_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd_diff()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['vol_ma'] = df['Volume'].rolling(window=14).mean()
    df.dropna(inplace=True)
    return df

def get_sentiment_score(stock_name):
    scores = []
    for url in NEWS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:20]:  # latest 20 news items
            if stock_name.lower() in entry.title.lower() or stock_name.lower() in entry.summary.lower():
                txt = entry.title + ' ' + entry.summary
                score = TextBlob(txt).sentiment.polarity
                scores.append(score)
    return np.mean(scores) if scores else 0

def prepare_data(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=LOOKBACK_PERIOD, interval='1d')
    if hist.empty:
        return None, None, None, None, None, None, None
    hist = get_technical_features(hist)
    X = hist[['rsi', 'macd', 'bb_h', 'bb_l', 'vol_ma']]
    y = np.where(hist['Close'].shift(-1) > hist['Close'], 1, 0)[:-1]  # 1 day next price up or not
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:-1]
    y_train, y_test = y[:split], y[split:-1]
    return X_train, X_test, y_train, y_test, y[-1], X.iloc[-1], hist

def ensemble_predict(X_train, y_train, X_last):
    if X_train is None:
        return 0.5  # neutral when no data
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    ensemble = VotingClassifier(estimators=[('rf', rf)], voting='soft')
    ensemble.fit(X_train, y_train)
    prob = ensemble.predict_proba([X_last])[0][1]
    return prob

def project_returns(hist):
    # Simple projection using historical returns mean & std dev annualized
    daily_returns = hist['Close'].pct_change().dropna()
    mean_return = daily_returns.mean() * 252  # trading days approx
    std_return = daily_returns.std() * np.sqrt(252)
    return mean_return, std_return

# --- Streamlit app UI and logic ---

st.set_page_config(page_title="AI Stock & Portfolio Advisor", layout="wide")
st.title("AI-Powered Indian Stock & Portfolio Recommendation Tool")

st.markdown("### Stock Swing Trading Recommendations")

col1, col2 = st.columns([3, 1])

with col1:
    selected_stock = st.selectbox(
        "Select stock for detailed recommendation",
        options=list(STOCKS.keys()),
        format_func=lambda x: STOCKS[x]
    )

    st.markdown("### Portfolio Analysis")
    portfolio_input = st.text_area(
        "Enter your owned stock codes (comma separated, e.g. TATAMOTORS.NS,HDFCBANK.NS)"
    )

with col2:
    st.markdown("### Fetch Recommendations")
    if st.button("Get Trading Signal & Projections"):
        with st.spinner("Analyzing stock..."):
            X_train, X_test, y_train, y_test, y_actual, X_last, hist = prepare_data(selected_stock)
            if X_train is None:
                st.error("No historical data found for this stock.")
            else:
                prob_up = ensemble_predict(X_train, y_train, X_last)
                sentiment = get_sentiment_score(STOCKS[selected_stock])
                mean_return, std_return = project_returns(hist)
                st.write(f"**Prediction Probability UP:** {prob_up:.2%}")
                st.write(f"**Sentiment Score:** {sentiment:.3f}")
                st.write(f"**Projected Annual Return:** {(mean_return * 100):.2f}% ± {(std_return * 100):.2f}%")
                signal = "BUY" if prob_up > 0.6 else "HOLD" if prob_up > 0.45 else "SELL"
                st.markdown(f"### Signal: **{signal}**")

                # Plot historical close and technical indicators
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=hist.index, open=hist['Open'], high=hist['High'],
                    low=hist['Low'], close=hist['Close'], name='Price'
                ))
                fig.add_trace(go.Scatter(x=hist.index, y=hist['rsi'], mode='lines', name='RSI'))
                st.plotly_chart(fig, use_container_width=True)

if portfolio_input:
    st.markdown("## Portfolio Sell/Hold Analysis")
    portfolio_codes = [code.strip().upper() for code in portfolio_input.split(',') if code.strip()]
    portfolio_results = []
    for code in portfolio_codes:
        X_train, X_test, y_train, y_test, y_actual, X_last, hist = prepare_data(code)
        if X_train is None:
            portfolio_results.append({'Stock': code, 'Error': 'No data'})
            continue
        prob_up = ensemble_predict(X_train, y_train, X_last)
        mean_return, std_return = project_returns(hist)
        # Simple hold if predicted UP prob > 0.5, else sell
        advice = "Hold" if prob_up > 0.5 else "Sell"
        portfolio_results.append({
            'Stock': code,
            'Predicted UP Prob': f"{prob_up:.2%}",
            'Projected Annual Return %': f"{mean_return * 100:.2f}%",
            'Advice': advice
        })
    df_portfolio = pd.DataFrame(portfolio_results)
    st.dataframe(df_portfolio)

st.markdown("---")
st.caption("Note: This tool uses historical price data, technical analysis, and simple AI models for informational purposes only. Always cross-check before trading.")