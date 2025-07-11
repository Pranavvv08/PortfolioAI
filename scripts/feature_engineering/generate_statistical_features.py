import pandas as pd
import numpy as np

# Load price matrix
df = pd.read_csv("data/price_matrix.csv", index_col=0, parse_dates=True)

# Compute log returns
log_returns = np.log1p(df.pct_change())

# Feature window sizes
windows = [5, 10, 20]

# Prepare DataFrame to store features
features = pd.DataFrame(index=log_returns.index)

# For each stock
for symbol in df.columns:
    for window in windows:
        # Momentum: Rolling mean of returns
        features[f"{symbol}_mom_{window}d"] = log_returns[symbol].rolling(window=window).mean()

        # Volatility: Rolling std of returns
        features[f"{symbol}_vol_{window}d"] = log_returns[symbol].rolling(window=window).std()

        # Sharpe Ratio approximation
        features[f"{symbol}_sharpe_{window}d"] = (
            features[f"{symbol}_mom_{window}d"] /
            features[f"{symbol}_vol_{window}d"].replace(0, np.nan)
        )

# Drop rows with all NaNs
features.dropna(inplace=True)

# Save features
features.to_csv("data/statistical_features.csv")
print("Saved feature matrix to data/features.csv")
