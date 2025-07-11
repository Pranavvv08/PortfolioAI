import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Install required packages if not available
try:
    import yfinance as yf
except ImportError:
    print("yfinance not available, please install: pip install yfinance")
    exit(1)

try:
    import talib
except ImportError:
    print("Warning: TA-Lib not available. Using custom implementations for technical indicators.")
    print("To get full functionality, install TA-Lib: pip install TA-Lib")
    talib = None

# Load existing price data to align dates
price_df = pd.read_csv("data/price_matrix.csv", index_col=0, parse_dates=True)

# Use the same symbol list as the price matrix
symbols = price_df.columns.tolist()

# Get the date range from existing price matrix for consistency
start_date = price_df.index.min()
end_date = price_df.index.max()

print(f"📅 Using date range from price matrix: {start_date.date()} to {end_date.date()}")
print(f"📊 Processing {len(symbols)} symbols")

# Prepare technical indicators DataFrame
tech_indicators = pd.DataFrame(index=price_df.index)

# Custom implementations for technical indicators when TA-Lib is not available
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

def calculate_ema(prices, window):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=window).mean()

def calculate_bollinger_bands(prices, window=20, std_dev=2):
    """Calculate Bollinger Bands"""
    sma = calculate_sma(prices, window)
    std = prices.rolling(window=window).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, sma, lower_band

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_stochastic_custom(high, low, close, k_window=14, d_window=3):
    """Calculate Stochastic Oscillator using HLC data"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    return k_percent, d_percent

def calculate_atr_custom(high, low, close, window=14):
    """Calculate Average True Range using HLC data"""
    high_low = high - low
    high_close_prev = np.abs(high - close.shift(1))
    low_close_prev = np.abs(low - close.shift(1))
    true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
    return true_range.rolling(window=window).mean()

def calculate_cci_custom(high, low, close, window=20):
    """Calculate Commodity Channel Index using HLC data"""
    typical_price = (high + low + close) / 3
    sma_tp = typical_price.rolling(window=window).mean()
    mean_dev = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
    cci = (typical_price - sma_tp) / (0.015 * mean_dev)
    return cci

def calculate_williams_r_custom(high, low, close, window=14):
    """Calculate Williams %R using HLC data"""
    highest_high = high.rolling(window=window).max()
    lowest_low = low.rolling(window=window).min()
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r

# Process each symbol
for i, symbol in enumerate(symbols):
    try:
        print(f"📈 Processing {symbol} ({i+1}/{len(symbols)})")
        
        # Download OHLCV data for the same date range as price matrix
        # This fixes the date synchronization issue
        df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True, progress=False)
        
        if df.empty:
            print(f"⚠️  No data for {symbol}")
            continue
            
        # Ensure we have OHLCV data
        required_columns = ['High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            print(f"⚠️  Missing OHLCV data for {symbol}")
            continue
        
        # Align dates with price matrix
        df.index = pd.to_datetime(df.index)
        df = df.reindex(price_df.index)
        
        # Extract OHLCV data
        high = df['High']
        low = df['Low'] 
        close = df['Close']
        volume = df['Volume']
        
        # Skip if all data is NaN
        if close.isna().all():
            print(f"⚠️  All NaN data for {symbol}")
            continue
        
        # Calculate technical indicators
        
        # 1. RSI (uses only close prices)
        tech_indicators[f"{symbol}_rsi_14"] = calculate_rsi(close, 14)
        
        # 2. Simple Moving Averages
        for window in [10, 20, 50]:
            tech_indicators[f"{symbol}_sma_{window}"] = calculate_sma(close, window)
        
        # 3. Exponential Moving Averages  
        for window in [12, 26]:
            tech_indicators[f"{symbol}_ema_{window}"] = calculate_ema(close, window)
        
        # 4. Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, 20, 2)
        tech_indicators[f"{symbol}_bb_upper"] = bb_upper
        tech_indicators[f"{symbol}_bb_middle"] = bb_middle
        tech_indicators[f"{symbol}_bb_lower"] = bb_lower
        tech_indicators[f"{symbol}_bb_width"] = bb_upper - bb_lower
        
        # 5. MACD
        macd_line, signal_line, histogram = calculate_macd(close, 12, 26, 9)
        tech_indicators[f"{symbol}_macd"] = macd_line
        tech_indicators[f"{symbol}_macd_signal"] = signal_line
        tech_indicators[f"{symbol}_macd_histogram"] = histogram
        
        # 6. Technical indicators that require HLC data
        # This fixes the "input array has wrong dimensions" error
        
        if talib is not None:
            # Use TA-Lib with proper HLC arrays
            try:
                # Convert to numpy arrays and handle NaN values
                h = high.values.astype(float)
                l = low.values.astype(float)
                c = close.values.astype(float)
                v = volume.values.astype(float)
                
                # Filter out NaN values for TA-Lib
                valid_mask = ~(np.isnan(h) | np.isnan(l) | np.isnan(c))
                if valid_mask.sum() > 20:  # Need enough data points
                    
                    # Stochastic Oscillator (requires HLC)
                    slowk, slowd = talib.STOCH(h, l, c, fastk_period=14, slowk_period=3, slowd_period=3)
                    tech_indicators[f"{symbol}_stoch_k"] = pd.Series(slowk, index=high.index)
                    tech_indicators[f"{symbol}_stoch_d"] = pd.Series(slowd, index=high.index)
                    
                    # Average True Range (requires HLC)
                    atr = talib.ATR(h, l, c, timeperiod=14)
                    tech_indicators[f"{symbol}_atr"] = pd.Series(atr, index=high.index)
                    
                    # Commodity Channel Index (requires HLC)
                    cci = talib.CCI(h, l, c, timeperiod=20)
                    tech_indicators[f"{symbol}_cci"] = pd.Series(cci, index=high.index)
                    
                    # Williams %R (requires HLC)
                    willr = talib.WILLR(h, l, c, timeperiod=14)
                    tech_indicators[f"{symbol}_willr"] = pd.Series(willr, index=high.index)
                    
            except Exception as e:
                print(f"⚠️  TA-Lib error for {symbol}: {e}")
                # Fall back to custom implementations
                stoch_k, stoch_d = calculate_stochastic_custom(high, low, close)
                tech_indicators[f"{symbol}_stoch_k"] = stoch_k
                tech_indicators[f"{symbol}_stoch_d"] = stoch_d
                tech_indicators[f"{symbol}_atr"] = calculate_atr_custom(high, low, close)
                tech_indicators[f"{symbol}_cci"] = calculate_cci_custom(high, low, close)
                tech_indicators[f"{symbol}_willr"] = calculate_williams_r_custom(high, low, close)
        else:
            # Use custom implementations
            stoch_k, stoch_d = calculate_stochastic_custom(high, low, close)
            tech_indicators[f"{symbol}_stoch_k"] = stoch_k
            tech_indicators[f"{symbol}_stoch_d"] = stoch_d
            tech_indicators[f"{symbol}_atr"] = calculate_atr_custom(high, low, close)
            tech_indicators[f"{symbol}_cci"] = calculate_cci_custom(high, low, close)
            tech_indicators[f"{symbol}_willr"] = calculate_williams_r_custom(high, low, close)
        
        # 7. Volume-based indicators
        if not volume.isna().all():
            # On-Balance Volume
            obv = (volume * np.sign(close.diff())).cumsum()
            tech_indicators[f"{symbol}_obv"] = obv
            
            # Volume Moving Average
            tech_indicators[f"{symbol}_volume_sma_20"] = volume.rolling(20).mean()
            
        # 8. Price-based indicators
        # Price change indicators
        tech_indicators[f"{symbol}_price_change_1d"] = close.pct_change(1)
        tech_indicators[f"{symbol}_price_change_5d"] = close.pct_change(5)
        tech_indicators[f"{symbol}_price_change_20d"] = close.pct_change(20)
        
        # Volatility (rolling standard deviation)
        tech_indicators[f"{symbol}_volatility_20d"] = close.pct_change().rolling(20).std()
        
    except Exception as e:
        print(f"❌ Error processing {symbol}: {e}")

# Drop rows with all NaN values
print(f"📊 Technical indicators shape before cleanup: {tech_indicators.shape}")
tech_indicators.dropna(how='all', inplace=True)
print(f"📊 Technical indicators shape after cleanup: {tech_indicators.shape}")

# Save technical indicators
output_path = "data/technical_indicators.csv"
tech_indicators.to_csv(output_path)
print(f"💾 Saved technical indicators to {output_path}")

# Display summary
print(f"\n📋 Technical Indicators Summary:")
print(f"   📅 Date range: {tech_indicators.index.min().date()} to {tech_indicators.index.max().date()}")
print(f"   📊 Shape: {tech_indicators.shape}")
print(f"   🔢 Number of indicators: {len(tech_indicators.columns)}")
print(f"   📈 Sample indicators: {list(tech_indicators.columns[:5])}")

# Check for any issues
nan_counts = tech_indicators.isna().sum()
problematic_indicators = nan_counts[nan_counts > len(tech_indicators) * 0.8]
if len(problematic_indicators) > 0:
    print(f"\n⚠️  Warning: These indicators have >80% NaN values:")
    for indicator in problematic_indicators.index[:10]:  # Show first 10
        print(f"   - {indicator}: {problematic_indicators[indicator]} NaN out of {len(tech_indicators)}")
else:
    print(f"\n✅ All indicators have reasonable data coverage")

print(f"\n🎯 Technical indicators generation completed successfully!")