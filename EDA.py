import numpy as np
import streamlit as st
import plotly.graph_objs as go
from sklearn.metrics import r2_score


# Calculate profit given buy and sell prices and quantity.
def calculate_profit(buy_price, sell_price, quantity):
    return (sell_price - buy_price) * quantity

# Generate buy, sell, or hold signals based on forecast and actual prices.
def generate_signals(forecast, actual, threshold=0.01):
    signals = []
    for f, a in zip(forecast, actual):
        if f > a * (1 + threshold):
            signals.append("Buy")
        elif f < a * (1 - threshold):
            signals.append("Sell")
        else:
            signals.append("Hold")
    return signals

# Display predictive times for high and low prices.
def display_predictive_times(data, forecast, model_name):
    daily_forecast = forecast[:1]
    weekly_forecast = forecast[:7]
    monthly_forecast = forecast[:30]

    daily_high = np.max(daily_forecast)
    daily_low = np.min(daily_forecast)

    weekly_high = np.max(weekly_forecast)
    weekly_low = np.min(weekly_forecast)

    monthly_high = np.max(monthly_forecast)
    monthly_low = np.min(monthly_forecast)

    st.write(f"### Predictive Times for {model_name} Model")
    st.write(f"Daily - High: {daily_high:.2f}, Low: {daily_low:.2f}")
    st.write(f"Weekly - High: {weekly_high:.2f}, Low: {weekly_low:.2f}")
    st.write(f"Monthly - High: {monthly_high:.2f}, Low: {monthly_low:.2f}")

# Display best times to trade with profit information.
def display_best_times_to_trade(data, forecast, model_name):
    buy_price = data[-1]
    sell_price = forecast[-1]
    profit = calculate_profit(buy_price, sell_price, 1)  # Assuming 1 unit for simplicity

    st.write(f"### Best Times to Trade with {model_name} Model")
    st.write(f"Buy at: {buy_price:.2f}")
    st.write(f"Sell at: {sell_price:.2f}")
    st.write(f"Anticipated Profit: {sell_price - buy_price:.2f}")

# Display market state with daily, weekly, and monthly changes.
def display_market_state(data, model_name):
    daily_change = data[-1] - data[-2]
    weekly_change = data[-1] - data[-7]
    monthly_change = data[-1] - data[-30]

    st.write(f"### Market State for {model_name} Model")
    st.write(f"Daily Change: {daily_change:.2f}")
    st.write(f"Weekly Change: {weekly_change:.2f}")
    st.write(f"Monthly Change: {monthly_change:.2f}")

# Display a graph of actual vs forecast prices.
def display_graph(data, forecast, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=data.index[-len(forecast):], y=forecast,
                             mode='lines', name='Forecast'))
    fig.update_layout(title=f'{model_name} Model - Actual vs Forecast',
                      xaxis_title='Date', yaxis_title='Price')
    st.plotly_chart(fig)

# Calculate moving average.
def calculate_moving_average(data, window):
    return data.rolling(window=window).mean()

# Function to assist users in making data driven decisions.
def display_decision_support(close_prices, prophet_forecast, ticker_option):
    st.title("Decision Support System")
    st.write("Specify your desired profit or interval for trading recommendations.")

    # Input for desired profit and interval
    profit_option = st.number_input("Desired Profit (£):", min_value=0.0, step=0.01)
    interval_option = st.selectbox("Interval:", ["1 day", "1 week", "1 month", "1 quarter"])

    # Calculate buy and sell recommendations
    buy_price = close_prices.iloc[-1]  # Current price
    sell_price = prophet_forecast[-1]  # Last predicted price

    anticipated_profit = calculate_profit(buy_price, sell_price, 1)  # Assuming 1 unit for simplicity

    st.write(f"Based on Prophet forecast, buy {ticker_option} at {buy_price:.2f} and sell at {sell_price:.2f}.")
    st.write(f"Anticipated profit for 1 unit: £{anticipated_profit:.2f}")

    # Confidence level using R^2 score
    confidence_level = r2_score(close_prices[-len(prophet_forecast):], prophet_forecast)
    st.write(f"Confidence level of prediction (R^2): {confidence_level:.2f}")

    # Display market state
    display_market_state(close_prices, "Decision Support")

