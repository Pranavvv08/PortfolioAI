import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os

# Load existing price data (we'll use this to align dates)
price_df = pd.read_csv("data/price_matrix.csv", index_col=0, parse_dates=True)

# Use the same symbol list as before (with .NS)
symbols = price_df.columns.tolist()

# Date range
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

# Prepare feature DataFrame
liquidity_features = pd.DataFrame(index=price_df.index)

for symbol in symbols:
    try:
        print(f"Volume for {symbol}")
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
        df = df[["Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index)
        df = df.reindex(price_df.index)  # align with existing dates

        # Compute dollar volume
        dollar_volume = df["Close"] * df["Volume"]
        liquidity_features[f"{symbol}_dollar_vol"] = dollar_volume

        for window in [5, 10, 20]:
            liquidity_features[f"{symbol}_avg_vol_{window}d"] = df["Volume"].rolling(window).mean()
            liquidity_features[f"{symbol}_volume_change_{window}d"] = df["Volume"].pct_change(window)
            liquidity_features[f"{symbol}_rolling_dollar_vol_{window}d"] = dollar_volume.rolling(window).mean()
    except Exception as e:
        print(f"Skipped {symbol}: {e}")

# Save features
liquidity_features.dropna(inplace=True)
liquidity_features.to_csv("data/liquidity_features.csv")
print("💧 Saved liquidity features to data/liquidity_features.csv")
