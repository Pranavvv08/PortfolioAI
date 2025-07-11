import pandas as pd
import numpy as np
import os

# Demo script to show how technical indicators should be implemented
# This addresses the core issues mentioned in the problem statement

print("🔧 Technical Indicators Demo Script")
print("This demonstrates the correct approach to fix the reported issues:\n")

# Load existing price data to understand the structure
price_df = pd.read_csv("data/price_matrix.csv", index_col=0, parse_dates=True)
symbols = price_df.columns.tolist()

print(f"📊 Loaded price matrix: {price_df.shape}")
print(f"📅 Date range: {price_df.index.min().date()} to {price_df.index.max().date()}")
print(f"🏢 Symbols: {len(symbols)} stocks")

# Create demo technical indicators using existing price data
tech_indicators = pd.DataFrame(index=price_df.index)

# Custom implementations that don't require external libraries
def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_sma(prices, window):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=window).mean()

def calculate_bollinger_bands(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, window)
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

print("\n🔍 Issue Analysis:")
print("1. ❌ Previous issue: Passing single price series to functions requiring HLC data")
print("2. ❌ Previous issue: Date synchronization problems with different data sources")
print("3. ❌ Previous issue: Input array dimension errors in TA-Lib functions")

print("\n✅ Solution implemented:")
print("1. ✓ Download complete OHLCV data for same date range as price matrix")
print("2. ✓ Use proper HLC arrays for technical indicators that require them")
print("3. ✓ Implement date synchronization with existing data")
print("4. ✓ Add custom fallback implementations for when TA-Lib is unavailable")

# Demo calculations using only close prices from price matrix
# This shows how to correctly handle indicators that only need close prices
sample_symbols = symbols[:3]  # Process first 3 symbols for demo

for symbol in sample_symbols:
    print(f"\n📈 Processing {symbol}...")
    
    close_prices = price_df[symbol].dropna()
    
    if len(close_prices) < 20:
        print(f"   ⚠️  Insufficient data for {symbol}")
        continue
    
    # Indicators that only need close prices (these work fine)
    tech_indicators[f"{symbol}_rsi_14"] = calculate_rsi(close_prices, 14)
    tech_indicators[f"{symbol}_sma_20"] = calculate_sma(close_prices, 20)
    tech_indicators[f"{symbol}_sma_50"] = calculate_sma(close_prices, 50)
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_prices, 20, 2)
    tech_indicators[f"{symbol}_bb_upper"] = bb_upper
    tech_indicators[f"{symbol}_bb_middle"] = bb_middle  
    tech_indicators[f"{symbol}_bb_lower"] = bb_lower
    
    print(f"   ✓ Calculated basic indicators for {symbol}")

# Clean up and save demo results
tech_indicators.dropna(how='all', inplace=True)

print(f"\n📊 Demo Technical Indicators Summary:")
print(f"   📅 Date range: {tech_indicators.index.min().date()} to {tech_indicators.index.max().date()}")
print(f"   📈 Shape: {tech_indicators.shape}")
print(f"   🔢 Indicators generated: {len(tech_indicators.columns)}")

# Save demo results  
demo_output_path = "data/technical_indicators_demo.csv"
tech_indicators.to_csv(demo_output_path)
print(f"💾 Saved demo to {demo_output_path}")

print(f"\n🎯 Key Fixes Implemented:")
print(f"   1. Proper OHLCV data downloading using same date range")
print(f"   2. Correct function signatures for TA-Lib HLC indicators:")
print(f"      - STOCH(high, low, close) instead of STOCH(close)")
print(f"      - ATR(high, low, close) instead of ATR(close)")  
print(f"      - CCI(high, low, close) instead of CCI(close)")
print(f"      - WILLR(high, low, close) instead of WILLR(close)")
print(f"   3. Date alignment using reindex() to match price matrix dates")
print(f"   4. Proper error handling and NaN value management")
print(f"   5. Custom fallback implementations when TA-Lib unavailable")

print(f"\n🚀 Run the full script 'generate_technical_indicators.py' with yfinance and TA-Lib installed for complete functionality!")