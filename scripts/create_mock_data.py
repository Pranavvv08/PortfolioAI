#!/usr/bin/env python3
"""
Create mock data for testing sentiment analysis system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# NIFTY 50 symbols
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "ITC.NS", "LT.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS",
    "AXISBANK.NS", "HCLTECH.NS", "MARUTI.NS", "ASIANPAINT.NS", "WIPRO.NS",
    "ULTRACEMCO.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "NTPC.NS", "TITAN.NS",
    "SUNPHARMA.NS", "TECHM.NS", "INDUSINDBK.NS", "POWERGRID.NS", "JSWSTEEL.NS",
    "ADANIENT.NS", "BAJAJFINSV.NS", "CIPLA.NS", "COALINDIA.NS", "EICHERMOT.NS",
    "HINDALCO.NS", "ONGC.NS", "TATACONSUM.NS", "NESTLEIND.NS", "DRREDDY.NS",
    "M&M.NS", "GRASIM.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "APOLLOHOSP.NS",
    "DIVISLAB.NS", "BPCL.NS", "SBILIFE.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS", "UPL.NS", "TATASTEEL.NS", "ADANIPORTS.NS", "IOC.NS"
]

# Generate mock price data
end_date = datetime.today()
start_date = end_date - timedelta(days=180)
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

# Filter to only business days
business_days = [d for d in date_range if d.weekday() < 5]

print(f"Creating mock data for {len(business_days)} business days...")

# Create price matrix with realistic price movements
np.random.seed(42)  # For reproducible results

price_data = {}
for symbol in NIFTY50_SYMBOLS:
    # Start with a random base price
    base_price = np.random.uniform(100, 5000)
    prices = [base_price]
    
    # Generate realistic price movements
    for i in range(1, len(business_days)):
        # Daily return with some trend and volatility
        daily_return = np.random.normal(0.0005, 0.02)  # Slight positive trend with 2% daily vol
        new_price = prices[-1] * (1 + daily_return)
        prices.append(max(new_price, 1))  # Ensure positive prices
    
    price_data[symbol] = prices

# Create DataFrame
price_df = pd.DataFrame(price_data, index=business_days)

# Save price matrix
price_df.to_csv("data/price_matrix.csv")
print("Saved mock price data to data/price_matrix.csv")

# Create log returns
log_returns = np.log1p(price_df.pct_change())
log_returns.to_csv("data/log_returns.csv")
print("Saved log returns to data/log_returns.csv")

# Create mock liquidity features too
print("Creating mock liquidity features...")

liquidity_features = pd.DataFrame(index=business_days)

for symbol in NIFTY50_SYMBOLS:
    # Mock volume data
    base_volume = np.random.uniform(1000000, 10000000)  # Base daily volume
    volumes = []
    
    for i in range(len(business_days)):
        # Volume with some randomness
        volume = base_volume * np.random.uniform(0.5, 2.0)
        volumes.append(volume)
    
    # Calculate liquidity features
    volume_series = pd.Series(volumes, index=business_days)
    price_series = price_df[symbol]
    dollar_volume = price_series * volume_series
    
    # Basic liquidity features
    liquidity_features[f"{symbol}_dollar_vol"] = dollar_volume
    
    for window in [5, 10, 20]:
        liquidity_features[f"{symbol}_avg_vol_{window}d"] = volume_series.rolling(window).mean()
        liquidity_features[f"{symbol}_volume_change_{window}d"] = volume_series.pct_change(window)
        liquidity_features[f"{symbol}_rolling_dollar_vol_{window}d"] = dollar_volume.rolling(window).mean()

# Drop rows with NaN values
liquidity_features.dropna(inplace=True)

# Save liquidity features
liquidity_features.to_csv("data/liquidity_features.csv")
print("Saved mock liquidity features to data/liquidity_features.csv")

# Preview
print("\nPrice data preview:")
print(price_df.tail())
print(f"\nShape: {price_df.shape}")

print("\nLiquidity features preview:")
print(liquidity_features.tail())
print(f"\nLiquidity features shape: {liquidity_features.shape}")