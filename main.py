import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import r2_score
import yfinance as yf
from load_data import load_historical_data
from preprocessing import preprocess_data
from ML_models import (
    perform_pca,
    kmeans_clustering,
    fit_arima_model,
    fit_lstm_model,
    fit_svm_model,
    fit_random_forest_model,
    fit_prophet_model,
)
from EDA import (
    calculate_profit,
    generate_signals,
    display_predictive_times,
    display_best_times_to_trade,
    display_market_state,
    display_graph,
    display_decision_support,
)

# Streamlit GUI Setup
st.set_page_config(page_title="Coin Trading", layout="wide", initial_sidebar_state="expanded")
st.sidebar.title("Solent Intelligence Coin Trading Platform 🪙")

# Define tickers
tickers = {
    'BTC-USD': 'Bitcoin',
    'ETH-USD': 'Ethereum',
    'USDT-USD': 'Tether USDt',
    'BNB-USD': 'BNB',
    'SOL-USD': 'Solana',
    'USDC-USD': 'USD Coin',
    'XRP-USD': 'XRP',
    'DOGE-USD': 'Dogecoin',
    'AVAX-USD': 'Avalanche',
    'SHIB-USD': 'Shiba Inu',
    'WBTC-USD': 'Wrapped Bitcoin',
    'BCH-USD': 'Bitcoin Cash',
    'LINK-USD': 'Chainlink',
    'NEAR-USD': 'NEAR Protocol',
    'LTC-USD': 'Litecoin',
    'PEPE24478-USD': 'Pepe',
    'DAI-USD': 'Dai',
    'ETC-USD': 'Ethereum Classic',
    'FDUSD-USD': 'FDUSD',
    'ARB11841-USD': 'Arbitrum',
    'OP-USD': 'Optimism',
    'WIF-USD': 'dogwifhat',
    'FLOKI-USD': 'FLOKI',
    'BONK-USD': 'Bonk',
    'GALA-USD': 'GALA',
    'VBNB-USD': 'VBNB',
    'BOME-USD': 'BOOK OF MEME',
    'VBTC-USD': 'VBTC',
    'WETH-USD': 'WETH',
    'SOL16116-USD': 'Wrapped Solana'
}

# Pre-process data
historical_data = load_historical_data(tickers)
adj_close_df = preprocess_data(historical_data)

# Perform PCA
reduced_data = perform_pca(adj_close_df)
reduced_df = pd.DataFrame(reduced_data, index=adj_close_df.columns.get_level_values(0))

# K-means clustering
clusters = kmeans_clustering(reduced_data)
reduced_df['Cluster Group'] = clusters

# Select predefined coins
selected_coins = ['BTC-USD', 'ETH-USD', 'SHIB-USD', 'VBTC-USD']

# Use the predefined coins in the sidebar selection
ticker_option = st.sidebar.selectbox("Select Ticker", selected_coins)

# Page selection
page_option = st.sidebar.selectbox("Select Option", [
    "Homepage", "PCA", "Correlation", "EDA", "ARIMA", "LSTM",
    "Random Forest", "SVM", "Prophet Forecast", "Decision Support", "What-If Scenario"
])

# Homepage
if page_option == "Homepage":
    st.title("Solent Intelligence Coin Trading Platform 🪙")
    st.write("""
     Welcome to the Solent Intelligence Coin Trading Platform. This tool allows you to:
        - Perform Exploratory Data Analysis (EDA)
        - Analyze correlations between cryptocurrencies
        - Predict and forecast cryptocurrency prices using various machine learning models
        - Generate trading signals based on forecasts
        - Make informed trading decisions based on desired profit margins
        """)
    st.write("Use the sidebar to navigate to different sections of the tool.")

# PCA
elif page_option == "PCA":
    st.title("PCA Results")
    st.dataframe(reduced_df)

# Correlation
elif page_option == "Correlation":
    st.title("Correlation Analysis for Selected Coins")
    correlations = adj_close_df.corr()

    for coin in selected_coins:
        st.subheader(f"Top correlations for {coin}")
        top_positive = correlations[coin].sort_values(ascending=False).head(5)
        top_negative = correlations[coin].sort_values().head(4)

        st.write("Most positively correlated:")
        st.write(top_positive)
        st.write("Most negatively correlated:")
        st.write(top_negative)

# EDA
elif page_option == "EDA":
    st.title(f"Exploratory Data Analysis (EDA) for {ticker_option}")

    close_prices = adj_close_df[ticker_option]
    st.write(f"Temporal structure for {ticker_option}")
    st.line_chart(close_prices)

    st.write(f"Rolling mean for {ticker_option}")
    rolling_mean = close_prices.rolling(window=30).mean()
    st.line_chart(rolling_mean)

    st.write(f"Distribution of observations for {ticker_option}")
    st.bar_chart(close_prices.value_counts())

    st.write(f"Histogram of adjusted close prices for {ticker_option}")
    st.bar_chart(np.histogram(close_prices, bins=30)[0])
    plt.xticks(rotation=45)

    st.write(f"Rolling statistics for {ticker_option}")
    rolling_std = close_prices.rolling(window=30).std()
    st.line_chart(rolling_std)

    st.write(f"Seasonal decomposition for {ticker_option}")
    decomposition = seasonal_decompose(close_prices, period=30)
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 8))
    decomposition.observed.plot(ax=ax1)
    decomposition.trend.plot(ax=ax2)
    decomposition.seasonal.plot(ax=ax3)
    decomposition.resid.plot(ax=ax4)
    ax1.set_title('Observed')
    ax2.set_title('Trend')
    ax3.set_title('Seasonal')
    ax4.set_title('Residual')
    plt.tight_layout()
    st.pyplot(fig)

# ARIMA Model
elif page_option == "ARIMA":
    st.title(f"ARIMA Model for {ticker_option}")

    close_prices = adj_close_df[ticker_option]
    train_size = int(len(close_prices) * 0.8)
    train_data, test_data = close_prices[:train_size], close_prices[train_size:]

    arima_model_fit = fit_arima_model(train_data, order=(5, 1, 0))
    arima_forecast = arima_model_fit.forecast(steps=len(test_data))
    arima_signals = generate_signals(arima_forecast, test_data)
    st.line_chart(arima_forecast)
    st.write("Forecast signal:", arima_signals[-1])

    # Accuracy
    arima_accuracy = r2_score(test_data, arima_forecast)
    st.write(f"ARIMA Model R² Accuracy: {arima_accuracy:.2f}")

    # Validate prediction with today's data
    try:
        latest_data = yf.download(ticker_option, period='1d', interval='1d')
        if not latest_data.empty:
            actual_price_today = latest_data['Close'].values[-1]
            predicted_price_yesterday = arima_forecast[-1]
            st.write(f"Actual Price Today: {actual_price_today:.2f}")
            st.write(f"Predicted Price Yesterday: {predicted_price_yesterday:.2f}")
            prediction_accuracy = 100 - abs((actual_price_today - predicted_price_yesterday) / actual_price_today) * 100
            st.write(f"Prediction Accuracy: {prediction_accuracy:.2f}%")
    except Exception as e:
        st.write(f"Error fetching today's data: {e}")

    # Additional analysis and visuals
    display_predictive_times(close_prices, arima_forecast, "ARIMA")
    display_best_times_to_trade(close_prices, arima_forecast, "ARIMA")
    display_market_state(close_prices, "ARIMA")
    display_graph(close_prices, arima_forecast, "ARIMA")

# LSTM Model
elif page_option == "LSTM":
    st.title(f"LSTM Model for {ticker_option}")

    close_prices = adj_close_df[ticker_option]
    train_size = int(len(close_prices) * 0.8)
    train_data, test_data = close_prices[:train_size], close_prices[train_size:]

    lstm_model, scaler = fit_lstm_model(train_data)

    scaled_test_data = scaler.transform(test_data.values.reshape(-1, 1))
    lstm_forecast = lstm_model.predict(
        np.reshape(scaled_test_data, (scaled_test_data.shape[0], scaled_test_data.shape[1], 1)))
    lstm_forecast = scaler.inverse_transform(lstm_forecast).flatten()
    lstm_signals = generate_signals(lstm_forecast, test_data)
    st.line_chart(lstm_forecast)
    st.write("Forecast signal:", lstm_signals[-1])

    # Accuracy
    lstm_accuracy = r2_score(test_data, lstm_forecast)
    st.write(f"LSTM Model R² Accuracy: {lstm_accuracy:.2f}")

    # Validate prediction with today's data
    try:
        latest_data = yf.download(ticker_option, period='1d', interval='1d')
        if not latest_data.empty:
            actual_price_today = latest_data['Close'].values[-1]
            predicted_price_yesterday = lstm_forecast[-1]
            st.write(f"Actual Price Today: {actual_price_today:.2f}")
            st.write(f"Predicted Price Yesterday: {predicted_price_yesterday:.2f}")
            prediction_accuracy = 100 - abs((actual_price_today - predicted_price_yesterday) / actual_price_today) * 100
            st.write(f"Prediction Accuracy: {prediction_accuracy:.2f}%")
    except Exception as e:
        st.write(f"Error fetching today's data: {e}")

    # Additional analysis and visuals
    display_predictive_times(close_prices, lstm_forecast, "LSTM")
    display_best_times_to_trade(close_prices, lstm_forecast, "LSTM")
    display_market_state(close_prices, "LSTM")
    display_graph(close_prices, lstm_forecast, "LSTM")

# Random Forest Model
elif page_option == "Random Forest":
    st.title(f"Random Forest Model for {ticker_option}")

    close_prices = adj_close_df[ticker_option]
    train_size = int(len(close_prices) * 0.8)
    train_data, test_data = close_prices[:train_size], close_prices[train_size:]

    rf_model = fit_random_forest_model(train_data)
    rf_forecast = rf_model.predict(np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1))
    rf_signals = generate_signals(rf_forecast, test_data)
    st.line_chart(rf_forecast)
    st.write("Forecast signal:", rf_signals[-1])

    # Accuracy
    rf_accuracy = r2_score(test_data, rf_forecast)
    st.write(f"Random Forest Model R² Accuracy: {rf_accuracy:.2f}")

    # Validate prediction with today's data
    try:
        latest_data = yf.download(ticker_option, period='1d', interval='1d')
        if not latest_data.empty:
            actual_price_today = latest_data['Close'].values[-1]
            predicted_price_yesterday = rf_forecast[-1]
            st.write(f"Actual Price Today: {actual_price_today:.2f}")
            st.write(f"Predicted Price Yesterday: {predicted_price_yesterday:.2f}")
            prediction_accuracy = 100 - abs((actual_price_today - predicted_price_yesterday) / actual_price_today) * 100
            st.write(f"Prediction Accuracy: {prediction_accuracy:.2f}%")
    except Exception as e:
        st.write(f"Error fetching today's data: {e}")

    # Additional analysis and visuals
    display_predictive_times(close_prices, rf_forecast, "Random Forest")
    display_best_times_to_trade(close_prices, rf_forecast, "Random Forest")
    display_market_state(close_prices, "Random Forest")
    display_graph(close_prices, rf_forecast, "Random Forest")

# SVM Model
elif page_option == "SVM":
    st.title(f"SVM Model for {ticker_option}")

    close_prices = adj_close_df[ticker_option]
    train_size = int(len(close_prices) * 0.8)
    train_data, test_data = close_prices[:train_size], close_prices[train_size:]

    svm_model = fit_svm_model(train_data)
    svm_forecast = svm_model.predict(np.arange(len(train_data), len(train_data) + len(test_data)).reshape(-1, 1))
    svm_signals = generate_signals(svm_forecast, test_data)
    st.line_chart(svm_forecast)
    st.write("Forecast signal:", svm_signals[-1])

    # Accuracy
    svm_accuracy = r2_score(test_data, svm_forecast)
    st.write(f"SVM Model R² Accuracy: {svm_accuracy:.2f}")

    # Validate prediction with today's data
    try:
        latest_data = yf.download(ticker_option, period='1d', interval='1d')
        if not latest_data.empty:
            actual_price_today = latest_data['Close'].values[-1]
            predicted_price_yesterday = svm_forecast[-1]
            st.write(f"Actual Price Today: {actual_price_today:.2f}")
            st.write(f"Predicted Price Yesterday: {predicted_price_yesterday:.2f}")
            prediction_accuracy = 100 - abs((actual_price_today - predicted_price_yesterday) / actual_price_today) * 100
            st.write(f"Prediction Accuracy: {prediction_accuracy:.2f}%")
    except Exception as e:
        st.write(f"Error fetching today's data: {e}")

    # Additional analysis and visuals
    display_predictive_times(close_prices, svm_forecast, "SVM")
    display_best_times_to_trade(close_prices, svm_forecast, "SVM")
    display_market_state(close_prices, "SVM")
    display_graph(close_prices, svm_forecast, "SVM")

# Prophet Forecast
elif page_option == "Prophet Forecast":
    st.title(f"Facebook Prophet Model for {ticker_option}")

    close_prices = adj_close_df[ticker_option]
    train_size = int(len(close_prices) * 0.8)
    train_data, test_data = close_prices[:train_size], close_prices[train_size:]

    prophet_model = fit_prophet_model(train_data)
    future = prophet_model.make_future_dataframe(periods=len(test_data))
    prophet_forecast = prophet_model.predict(future)['yhat'].iloc[-len(test_data):].values
    prophet_signals = generate_signals(prophet_forecast, test_data)
    st.line_chart(prophet_forecast)
    st.write("Forecast signal:", prophet_signals[-1])

    # Accuracy
    prophet_accuracy = r2_score(test_data, prophet_forecast)
    st.write(f"Prophet Model R² Accuracy: {prophet_accuracy:.2f}")

    # Validate prediction with today's data
    try:
        latest_data = yf.download(ticker_option, period='1d', interval='1d')
        if not latest_data.empty:
            actual_price_today = latest_data['Close'].values[-1]
            predicted_price_yesterday = prophet_forecast[-1]
            st.write(f"Actual Price Today: {actual_price_today:.2f}")
            st.write(f"Predicted Price Yesterday: {predicted_price_yesterday:.2f}")
            prediction_accuracy = 100 - abs((actual_price_today - predicted_price_yesterday) / actual_price_today) * 100
            st.write(f"Prediction Accuracy: {prediction_accuracy:.2f}%")
    except Exception as e:
        st.write(f"Error fetching today's data: {e}")

    # Additional analysis and visuals
    display_predictive_times(close_prices, prophet_forecast, "Prophet")
    display_best_times_to_trade(close_prices, prophet_forecast, "Prophet")
    display_market_state(close_prices, "Prophet")
    display_graph(close_prices, prophet_forecast, "Prophet")

# Decision Support
elif page_option == "Decision Support":
    # Re-fit Prophet model to ensure we're using the most up-to-date data
    close_prices = adj_close_df[ticker_option]
    train_size = int(len(close_prices) * 0.8)
    train_data, test_data = close_prices[:train_size], close_prices[train_size:]

    prophet_model = fit_prophet_model(train_data)
    future = prophet_model.make_future_dataframe(periods=len(test_data))
    prophet_forecast = prophet_model.predict(future)['yhat'].iloc[-len(test_data):].values

    display_decision_support(close_prices, prophet_forecast, ticker_option)

# What-If Scenario
elif page_option == "What-If Scenario":
    st.title("What-If Scenario Analysis")

    close_prices = adj_close_df[ticker_option]
    current_price = close_prices.iloc[-1]

    st.write(f"Current price of {ticker_option}: ${current_price:.2f}")
    buy_price = st.number_input("Buy Price ($):", min_value=0.0, step=0.01, value=current_price)
    sell_price = st.number_input("Sell Price ($):", min_value=0.0, step=0.01, value=current_price)
    quantity = st.number_input("Quantity:", min_value=0.0, step=0.01, value=1.0)

    profit = calculate_profit(buy_price, sell_price, quantity)
    st.write(f"Potential Profit: ${profit:.2f}")

    prophet_model = fit_prophet_model(close_prices)
    future = prophet_model.make_future_dataframe(periods=30)
    prophet_forecast = prophet_model.predict(future)

    st.write("### Forecast for the next 30 days")
    st.line_chart(prophet_forecast['yhat'])
