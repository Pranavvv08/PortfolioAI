import pandas as pd
import os

# Paths
features_path = "data/statistical_features.csv"
liquidity_path = "data/liquidity_features.csv"
output_path = "data/final_features.csv"

# Check if files exist
if not os.path.exists(features_path):
    raise FileNotFoundError(f"{features_path} not found. Run your momentum/volatility script first.")
if not os.path.exists(liquidity_path):
    raise FileNotFoundError(f"{liquidity_path} not found. Run your liquidity feature script first.")

# Load feature sets
print("Loading feature files...")
df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
df_liquidity = pd.read_csv(liquidity_path, index_col=0, parse_dates=True)

# Align both on date index
df_combined = pd.concat([df_features, df_liquidity], axis=1)

# Drop rows with missing values in any feature
df_combined.dropna(inplace=True)

# Save final feature matrix
df_combined.to_csv(output_path)
print(f"All features merged and saved to {output_path}")

# Preview
print("\nPreview of final feature matrix:")
print(df_combined.tail())
