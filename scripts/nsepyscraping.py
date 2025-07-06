import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Output folder
os.makedirs("data", exist_ok=True)

# Define date range
end_date = datetime.today()
start_date = end_date - timedelta(days=180)

# NIFTY 50 symbols with .NS suffix for NSE
nifty50_symbols = [
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

print("Downloading NIFTY 50 stock data...")
data = yf.download(
    tickers=nifty50_symbols,
    start=start_date,
    end=end_date,
    group_by='ticker',   # so each symbol has its own sub-data
    auto_adjust=True,    # get adjusted close by default
    threads=True
)

# Build a price matrix
price_df = pd.DataFrame()

for symbol in nifty50_symbols:
    try:
        symbol_data = data[symbol]["Close"]
        price_df[symbol] = symbol_data
    except KeyError:
        print(f"Missing data for {symbol}")

# Drop rows with all NaNs
price_df.dropna(how='all', inplace=True)

# Save to CSV
price_df.to_csv("data/price_matrix.csv")
print("Saved data/price_matrix.csv")

# Compute log returns
log_returns = np.log1p(price_df.pct_change().dropna())
log_returns.to_csv("data/log_returns.csv")
print("Saved data/log_returns.csv")

# Preview
print("\nPreview:")
print(price_df.tail())
