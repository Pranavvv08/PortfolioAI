import pandas as pd
import os

# Paths
features_path = "data/statistical_features.csv"
liquidity_path = "data/liquidity_features.csv"
technical_path = "data/technical_features.csv"
output_path = "data/final_features.csv"

# Check if files exist
if not os.path.exists(features_path):
    raise FileNotFoundError(f"{features_path} not found. Run your momentum/volatility script first.")
if not os.path.exists(liquidity_path):
    raise FileNotFoundError(f"{liquidity_path} not found. Run your liquidity feature script first.")
if not os.path.exists(technical_path):
    raise FileNotFoundError(f"{technical_path} not found. Run the technical features script first.")

# Load feature sets
print("Loading feature files...")
df_features = pd.read_csv(features_path, index_col=0, parse_dates=True)
df_liquidity = pd.read_csv(liquidity_path, index_col=0, parse_dates=True)
df_technical = pd.read_csv(technical_path, index_col=0, parse_dates=True)

# Align all feature sets on date index
print("Merging feature sets...")
df_combined = pd.concat([df_features, df_liquidity, df_technical], axis=1)

# Drop rows with missing values in any feature
print("Cleaning data...")
initial_rows = len(df_combined)
df_combined.dropna(inplace=True)
final_rows = len(df_combined)

print(f"Dropped {initial_rows - final_rows} rows with missing values")
print(f"Final dataset has {final_rows} rows and {len(df_combined.columns)} features")

# Save final feature matrix
df_combined.to_csv(output_path)
print(f"All features merged and saved to {output_path}")

# Preview
print("\nPreview of final feature matrix:")
print(df_combined.tail())
print(f"\nFeature breakdown:")
print(f"- Statistical features: {len([col for col in df_combined.columns if any(feat in col for feat in ['mom_', 'vol_', 'sharpe_'])])}")
print(f"- Liquidity features: {len([col for col in df_combined.columns if any(feat in col for feat in ['dollar_vol', 'avg_vol', 'volume_change'])])}")
print(f"- Technical features: {len([col for col in df_combined.columns if any(feat in col for feat in ['rsi_', 'macd_', 'roc_', 'sma_', 'ema_', 'bb_', 'atr', 'williams_r'])])}")
print(f"- Total features: {len(df_combined.columns)}")
