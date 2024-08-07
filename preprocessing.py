import pandas as pd

# Extracts and preprocess the adjusted close prices from historical data.
def preprocess_data(historical_data):
    adj_close_df = historical_data.xs('Adj Close', level=1, axis=1)
    adj_close_df.ffill(inplace=True)
    adj_close_df.bfill(inplace=True)
    return adj_close_df
