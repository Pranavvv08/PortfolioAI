import csv
import math
from typing import List, Dict, Any
from collections import defaultdict

def read_csv_to_dict(filepath: str) -> Dict[str, Any]:
    """Read CSV file and return as dictionary with dates as keys."""
    data = {}
    with open(filepath, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = row.pop('Date')
            # Convert string prices to float, handle missing values
            for symbol in row:
                try:
                    row[symbol] = float(row[symbol]) if row[symbol] else None
                except ValueError:
                    row[symbol] = None
            data[date] = row
    return data

def calculate_sma(prices: List[float], window: int) -> List[float]:
    """Calculate Simple Moving Average."""
    sma = []
    for i in range(len(prices)):
        if i < window - 1:
            sma.append(None)
        else:
            window_prices = [p for p in prices[i-window+1:i+1] if p is not None]
            if len(window_prices) == window:
                sma.append(sum(window_prices) / window)
            else:
                sma.append(None)
    return sma

def calculate_ema(prices: List[float], window: int) -> List[float]:
    """Calculate Exponential Moving Average."""
    ema = []
    multiplier = 2 / (window + 1)
    
    for i, price in enumerate(prices):
        if price is None:
            ema.append(None)
            continue
            
        if i == 0:
            ema.append(price)
        elif ema[i-1] is not None:
            ema.append((price * multiplier) + (ema[i-1] * (1 - multiplier)))
        else:
            ema.append(price)
    return ema

def calculate_rsi(prices: List[float], window: int = 14) -> List[float]:
    """Calculate Relative Strength Index."""
    if len(prices) < window + 1:
        return [None] * len(prices)
    
    gains = []
    losses = []
    rsi = []
    
    # Calculate price changes
    for i in range(1, len(prices)):
        if prices[i] is None or prices[i-1] is None:
            gains.append(0)
            losses.append(0)
        else:
            change = prices[i] - prices[i-1]
            gains.append(max(change, 0))
            losses.append(max(-change, 0))
    
    # First RSI value
    for i in range(len(prices)):
        if i < window:
            rsi.append(None)
        elif i == window:
            avg_gain = sum(gains[i-window:i]) / window
            avg_loss = sum(losses[i-window:i]) / window
            if avg_loss == 0:
                rsi.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100 - (100 / (1 + rs)))
        else:
            # Use Wilder's smoothing
            prev_avg_gain = (rsi[i-1] if rsi[i-1] else 50) * (window - 1) / window + gains[i-1] / window
            prev_avg_loss = (100 - (rsi[i-1] if rsi[i-1] else 50)) * (window - 1) / window + losses[i-1] / window
            
            if prev_avg_loss == 0:
                rsi.append(100)
            else:
                rs = prev_avg_gain / prev_avg_loss
                rsi.append(100 - (100 / (1 + rs)))
    
    return rsi

def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD line, signal line, and histogram."""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    # MACD line
    macd_line = []
    for i in range(len(prices)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line.append(ema_fast[i] - ema_slow[i])
        else:
            macd_line.append(None)
    
    # Signal line (EMA of MACD line)
    signal_line = calculate_ema(macd_line, signal)
    
    # Histogram
    histogram = []
    for i in range(len(macd_line)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram.append(macd_line[i] - signal_line[i])
        else:
            histogram.append(None)
    
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(prices: List[float], window: int = 20, std_dev: float = 2) -> tuple:
    """Calculate Bollinger Bands."""
    sma = calculate_sma(prices, window)
    upper_band = []
    lower_band = []
    
    for i in range(len(prices)):
        if i < window - 1:
            upper_band.append(None)
            lower_band.append(None)
        else:
            window_prices = [p for p in prices[i-window+1:i+1] if p is not None]
            if len(window_prices) == window and sma[i] is not None:
                variance = sum([(p - sma[i]) ** 2 for p in window_prices]) / window
                std = math.sqrt(variance)
                upper_band.append(sma[i] + (std_dev * std))
                lower_band.append(sma[i] - (std_dev * std))
            else:
                upper_band.append(None)
                lower_band.append(None)
    
    return upper_band, sma, lower_band

def calculate_atr(high: List[float], low: List[float], close: List[float], window: int = 14) -> List[float]:
    """Calculate Average True Range."""
    true_ranges = []
    
    for i in range(len(close)):
        if i == 0:
            true_ranges.append(high[i] - low[i] if high[i] and low[i] else None)
        else:
            if all(x is not None for x in [high[i], low[i], close[i-1]]):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i-1])
                tr3 = abs(low[i] - close[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            else:
                true_ranges.append(None)
    
    return calculate_sma(true_ranges, window)

def calculate_williams_r(high: List[float], low: List[float], close: List[float], window: int = 14) -> List[float]:
    """Calculate Williams %R."""
    williams_r = []
    
    for i in range(len(close)):
        if i < window - 1:
            williams_r.append(None)
        else:
            highest_high = max([h for h in high[i-window+1:i+1] if h is not None])
            lowest_low = min([l for l in low[i-window+1:i+1] if l is not None])
            
            if close[i] is not None and highest_high != lowest_low:
                wr = ((highest_high - close[i]) / (highest_high - lowest_low)) * -100
                williams_r.append(wr)
            else:
                williams_r.append(None)
    
    return williams_r

def calculate_roc(prices: List[float], window: int) -> List[float]:
    """Calculate Rate of Change."""
    roc = []
    
    for i in range(len(prices)):
        if i < window:
            roc.append(None)
        elif prices[i] is not None and prices[i-window] is not None and prices[i-window] != 0:
            roc.append(((prices[i] - prices[i-window]) / prices[i-window]) * 100)
        else:
            roc.append(None)
    
    return roc

def generate_technical_features():
    """Generate comprehensive technical features for all NIFTY 50 stocks."""
    
    # Load price matrix
    print("Loading price data...")
    price_data = read_csv_to_dict("data/price_matrix.csv")
    
    if not price_data:
        raise FileNotFoundError("data/price_matrix.csv not found or empty")
    
    # Get all dates and symbols
    dates = list(price_data.keys())
    symbols = list(price_data[dates[0]].keys())
    
    print(f"Processing {len(symbols)} symbols for {len(dates)} dates...")
    
    # Prepare feature storage
    features = defaultdict(dict)
    
    # Process each symbol
    for symbol in symbols:
        print(f"Processing {symbol}...")
        
        # Extract price series for this symbol
        prices = [price_data[date][symbol] for date in dates]
        
        # For simplicity, use closing prices for high/low approximations
        # In real implementation, you'd want separate OHLC data
        high_prices = prices  # Approximation
        low_prices = prices   # Approximation
        
        # Multiple timeframes
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # Momentum Indicators
            rsi = calculate_rsi(prices, window)
            roc = calculate_roc(prices, window)
            
            # Moving Averages
            sma = calculate_sma(prices, window)
            ema = calculate_ema(prices, window)
            
            # Volatility (rolling standard deviation approximation)
            volatility = []
            for i in range(len(prices)):
                if i < window - 1:
                    volatility.append(None)
                else:
                    window_prices = [p for p in prices[i-window+1:i+1] if p is not None]
                    if len(window_prices) >= 2:
                        mean_price = sum(window_prices) / len(window_prices)
                        variance = sum([(p - mean_price) ** 2 for p in window_prices]) / len(window_prices)
                        volatility.append(math.sqrt(variance))
                    else:
                        volatility.append(None)
            
            # Store features for each date
            for i, date in enumerate(dates):
                if date not in features:
                    features[date] = {}
                
                # Momentum features
                features[date][f"{symbol}_rsi_{window}d"] = rsi[i]
                features[date][f"{symbol}_roc_{window}d"] = roc[i]
                
                # Trend features
                features[date][f"{symbol}_sma_{window}d"] = sma[i]
                features[date][f"{symbol}_ema_{window}d"] = ema[i]
                
                # Volatility features
                features[date][f"{symbol}_volatility_{window}d"] = volatility[i]
                
                # Price relative to moving average
                if prices[i] is not None and sma[i] is not None:
                    features[date][f"{symbol}_price_to_sma_{window}d"] = prices[i] / sma[i]
                
                # Moving average slope (trend strength)
                if i >= 1 and sma[i] is not None and sma[i-1] is not None:
                    features[date][f"{symbol}_sma_slope_{window}d"] = sma[i] - sma[i-1]
        
        # Single timeframe indicators (using 14-day default)
        macd_line, macd_signal, macd_hist = calculate_macd(prices)
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(prices)
        atr = calculate_atr(high_prices, low_prices, prices)
        williams_r = calculate_williams_r(high_prices, low_prices, prices)
        
        for i, date in enumerate(dates):
            # MACD features
            features[date][f"{symbol}_macd_line"] = macd_line[i]
            features[date][f"{symbol}_macd_signal"] = macd_signal[i]
            features[date][f"{symbol}_macd_histogram"] = macd_hist[i]
            
            # Bollinger Bands
            features[date][f"{symbol}_bb_upper"] = bb_upper[i]
            features[date][f"{symbol}_bb_middle"] = bb_middle[i]
            features[date][f"{symbol}_bb_lower"] = bb_lower[i]
            
            # Bollinger Band width and position
            if bb_upper[i] is not None and bb_lower[i] is not None:
                features[date][f"{symbol}_bb_width"] = bb_upper[i] - bb_lower[i]
                if prices[i] is not None and bb_upper[i] != bb_lower[i]:
                    features[date][f"{symbol}_bb_position"] = (prices[i] - bb_lower[i]) / (bb_upper[i] - bb_lower[i])
            
            # ATR
            features[date][f"{symbol}_atr"] = atr[i]
            
            # Williams %R
            features[date][f"{symbol}_williams_r"] = williams_r[i]
            
            # Moving Average Crossovers (SMA 20/50)
            sma_20 = calculate_sma(prices, 20)
            sma_50 = calculate_sma(prices, 50)
            if i > 0 and all(x is not None for x in [sma_20[i], sma_20[i-1], sma_50[i], sma_50[i-1]]):
                # Golden cross / death cross signal
                prev_diff = sma_20[i-1] - sma_50[i-1]
                curr_diff = sma_20[i] - sma_50[i]
                features[date][f"{symbol}_ma_crossover_signal"] = 1 if prev_diff <= 0 and curr_diff > 0 else (-1 if prev_diff >= 0 and curr_diff < 0 else 0)
    
    # Write features to CSV
    print("Writing technical features to CSV...")
    
    # Get all feature names
    all_feature_names = set()
    for date_features in features.values():
        all_feature_names.update(date_features.keys())
    
    feature_names = sorted(list(all_feature_names))
    
    with open("data/technical_features.csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['Date'] + feature_names)
        
        # Write data
        for date in sorted(dates):
            row = [date]
            for feature_name in feature_names:
                value = features[date].get(feature_name)
                row.append(value if value is not None else '')
            writer.writerow(row)
    
    print("✨ Technical features saved to data/technical_features.csv")
    print(f"Generated {len(feature_names)} features for {len(dates)} dates")

if __name__ == "__main__":
    generate_technical_features()