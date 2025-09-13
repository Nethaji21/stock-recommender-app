import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import ta
import feedparser
from textblob import TextBlob

SYMBOLS = ['TATAMOTORS.NS', 'HDFCBANK.NS', 'RELIANCE.NS', 'INFY.NS', 'SBIN.NS']
LOOKBACK = '6mo'
NEWS_FEEDS = [
    'https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms',
    'https://feeds.feedburner.com/ndtvprofit-latest'
]

def get_technical_features(df):
    df['rsi'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    df['macd'] = ta.trend.MACD(df['Close']).macd()
    bb = ta.volatility.BollingerBands(df['Close'])
    df['bb_h'] = bb.bollinger_hband()
    df['bb_l'] = bb.bollinger_lband()
    df['vol_ma'] = df['Volume'].rolling(15).mean()
    return df

def get_sentiment_score(stock_name):
    scores = []
    for url in NEWS_FEEDS:
        feed = feedparser.parse(url)
        for entry in feed.entries[:10]:
            if stock_name.lower() in entry.title.lower() or stock_name.lower() in entry.summary.lower():
                txt = entry.title + ' ' + entry.summary
                score = TextBlob(txt).sentiment.polarity
                scores.append(score)
    avg = np.mean(scores) if scores else 0
    return avg

def prepare_data(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period=LOOKBACK)
    hist = get_technical_features(hist)
    hist = hist.dropna()
    X = hist[['rsi', 'macd', 'bb_h', 'bb_l', 'vol_ma']]
    y = np.where(hist['Close'].shift(-1) > hist['Close'], 1, 0)[:-1]
    split = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:split], X[split:-1], y[:split], y[split:-1]
    return X_train, X_test, y_train, y_test, y[-1], X.iloc[-1], hist

def ensemble_predict(X_train, y_train, X_last):
    rf = RandomForestClassifier(n_estimators=50)
    rf.fit(X_train, y_train)
    ensemble = VotingClassifier(estimators=[('rf', rf)], voting='soft')
    ensemble.fit(X_train, y_train)
    pred = ensemble.predict_proba([X_last])
    return pred[0][1]

def smart_recommend():
    results = []
    for symbol in SYMBOLS:
        X_train
