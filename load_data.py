import yfinance as yf
import pandas as pd
import streamlit as st

# Downloads 1 years worth of historical data for a list of tickers using yfinance.
@st.cache_data(ttl=3600)
def load_historical_data(tickers):
    data = {}
    for ticker in tickers:
        try:
            # Fetch 1-year daily data for each ticker
            crypto_data = yf.download(ticker, period='1y', interval='1d')
            if not crypto_data.empty:
                data[ticker] = crypto_data
        except Exception as e:
            st.error(f"Error loading data for {ticker}: {e}")
    combined_df = pd.concat(data, axis=1)
    combined_df.to_csv('historical_crypto_data.csv')
    return combined_df
