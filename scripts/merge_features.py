import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
features_path = "data/statistical_features.csv"
liquidity_path = "data/liquidity_features.csv"
sentiment_path = "data/sentiment_features.csv"
output_path = "data/final_features.csv"

def load_feature_file(file_path: str, feature_type: str) -> pd.DataFrame:
    """Load a feature file with error handling."""
    if not os.path.exists(file_path):
        logger.warning(f"{feature_type} file {file_path} not found. Skipping.")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Normalize datetime index to date only (remove time component)
        df.index = df.index.date
        df.index = pd.to_datetime(df.index)
        
        logger.info(f"Loaded {feature_type} features: {df.shape}")
        logger.info(f"{feature_type} date range: {df.index.min()} to {df.index.max()}")
        return df
    except Exception as e:
        logger.error(f"Error loading {feature_type} features from {file_path}: {e}")
        return pd.DataFrame()

def align_features(*dataframes) -> pd.DataFrame:
    """Align multiple DataFrames on their datetime index."""
    # Filter out empty DataFrames
    valid_dfs = [df for df in dataframes if not df.empty]
    
    if not valid_dfs:
        logger.warning("No valid feature DataFrames to merge")
        return pd.DataFrame()
    
    if len(valid_dfs) == 1:
        return valid_dfs[0]
    
    # Find common date range
    start_date = max(df.index.min() for df in valid_dfs)
    end_date = min(df.index.max() for df in valid_dfs)
    
    logger.info(f"Aligning features to date range: {start_date} to {end_date}")
    
    # Align all DataFrames to common date range
    aligned_dfs = []
    for df in valid_dfs:
        aligned_df = df.loc[start_date:end_date]
        aligned_dfs.append(aligned_df)
    
    # Concatenate along columns
    combined_df = pd.concat(aligned_dfs, axis=1)
    
    return combined_df

def clean_merged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare merged features."""
    if df.empty:
        return df
    
    logger.info(f"Initial merged features shape: {df.shape}")
    
    # Remove duplicate columns (if any)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Calculate missing data percentage for each column
    missing_pct = df.isnull().sum() / len(df)
    
    # Drop columns with more than 50% missing data
    cols_to_drop = missing_pct[missing_pct > 0.5].index
    if len(cols_to_drop) > 0:
        logger.info(f"Dropping {len(cols_to_drop)} columns with >50% missing data")
        df = df.drop(columns=cols_to_drop)
    
    # Drop rows with more than 30% missing data
    row_missing_threshold = int(0.3 * df.shape[1])
    df = df.dropna(thresh=df.shape[1] - row_missing_threshold)
    
    # Forward fill remaining missing values (appropriate for time series)
    df = df.fillna(method='ffill')
    
    # Drop any remaining rows with NaN values
    df = df.dropna()
    
    logger.info(f"Final cleaned features shape: {df.shape}")
    
    return df

def add_feature_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Add summary statistics across all features."""
    if df.empty:
        return df
    
    logger.info("Adding cross-feature statistics...")
    
    # Select only numeric columns for statistics
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    if len(numeric_cols) == 0:
        return df
    
    # Add rolling correlation features (if we have enough features)
    if len(numeric_cols) > 10:
        # Calculate rolling correlation of first 10 features
        sample_features = numeric_cols[:10]
        correlation_matrix = df[sample_features].rolling(window=5).corr()
        
        # Extract mean correlation for each feature
        for feature in sample_features[:5]:  # Limit to avoid too many features
            corr_series = correlation_matrix.loc[(slice(None), feature), :]
            mean_corr = corr_series.groupby(level=0).mean().mean(axis=1)
            df[f"{feature}_mean_correlation"] = mean_corr
    
    # Add feature volatility indicators
    for window in [5, 10]:
        # Rolling standard deviation of returns for sentiment features
        sentiment_cols = [col for col in numeric_cols if 'sentiment' in col.lower()]
        for col in sentiment_cols[:5]:  # Limit number of features
            df[f"{col}_volatility_{window}d"] = df[col].rolling(window=window).std()
    
    return df

# Main execution
logger.info("Starting feature merging process...")

# Load all feature files
print("Loading feature files...")
df_features = load_feature_file(features_path, "Statistical")
df_liquidity = load_feature_file(liquidity_path, "Liquidity")
df_sentiment = load_feature_file(sentiment_path, "Sentiment")

# Align and merge features
print("Aligning and merging features...")
df_combined = align_features(df_features, df_liquidity, df_sentiment)

if df_combined.empty:
    logger.error("No features to merge. Please check that feature files exist and contain data.")
    exit(1)

# Clean merged features
print("Cleaning merged features...")
df_combined = clean_merged_features(df_combined)

# Add additional feature statistics
print("Adding feature statistics...")
df_combined = add_feature_statistics(df_combined)

# Save final feature matrix
df_combined.to_csv(output_path)
logger.info(f"All features merged and saved to {output_path}")

# Generate comprehensive summary
print(f"\n{'='*50}")
print("FEATURE MERGING SUMMARY")
print(f"{'='*50}")

print(f"Final feature matrix shape: {df_combined.shape}")
print(f"Date range: {df_combined.index.min()} to {df_combined.index.max()}")
print(f"Total features: {df_combined.shape[1]}")

# Feature type breakdown
feature_types = {
    'statistical': len([col for col in df_combined.columns if any(x in col.lower() for x in ['mom', 'vol', 'sharpe'])]),
    'liquidity': len([col for col in df_combined.columns if any(x in col.lower() for x in ['volume', 'dollar'])]),
    'sentiment': len([col for col in df_combined.columns if 'sentiment' in col.lower()]),
    'other': 0
}
feature_types['other'] = df_combined.shape[1] - sum(feature_types.values())

print(f"\nFeature breakdown:")
for feat_type, count in feature_types.items():
    if count > 0:
        print(f"  {feat_type.capitalize()}: {count} features")

# Check for missing values
missing_values = df_combined.isnull().sum().sum()
print(f"\nMissing values: {missing_values}")

# Preview
print(f"\n{'='*50}")
print("PREVIEW OF FINAL FEATURE MATRIX")
print(f"{'='*50}")
print(df_combined.tail())

# Sample feature names
print(f"\nSample feature names:")
sample_features = list(df_combined.columns[:10])
for i, feature in enumerate(sample_features, 1):
    print(f"  {i}. {feature}")

if df_combined.shape[1] > 10:
    print(f"  ... and {df_combined.shape[1] - 10} more features")

print(f"\n✅ Feature merging completed successfully!")
print(f"📁 Output saved to: {output_path}")
