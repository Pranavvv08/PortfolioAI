import pandas as pd
import os

def merge_all_features():
    """Merge technical, liquidity, and sentiment features"""
    
    # Load all feature files
    features_to_merge = []
    feature_files = [
        ("data/statistical_features.csv", "Technical Features"),
        ("data/liquidity_features.csv", "Liquidity Features"),
        ("data/sentiment_features.csv", "Sentiment Features")
    ]
    
    print("Merging all feature sets...")
    
    for file_path, feature_type in feature_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            features_to_merge.append(df)
            print(f"Loaded {feature_type}: {df.shape}")
        else:
            print(f"{feature_type} not found at {file_path}")
    
    if not features_to_merge:
        raise FileNotFoundError("No feature files found!")
    
    # For sentiment features, we need to broadcast to match technical features timeline
    if len(features_to_merge) > 1:
        # Use technical features as the base timeline
        base_df = features_to_merge[0]  # Assuming this is technical features
        
        # Merge other features
        for i, df in enumerate(features_to_merge[1:], 1):
            if df.shape[0] == 1:  # Sentiment features (single row)
                # Broadcast sentiment features to match the timeline
                sentiment_broadcasted = pd.DataFrame(
                    index=base_df.index,
                    columns=df.columns
                )
                # Fill all rows with the same sentiment values
                for col in df.columns:
                    sentiment_broadcasted[col] = df[col].iloc[0]
                
                base_df = pd.concat([base_df, sentiment_broadcasted], axis=1)
            else:
                # Normal merge for time-series features
                base_df = pd.concat([base_df, df], axis=1)
        
        combined_df = base_df
    else:
        combined_df = features_to_merge[0]
    
    # Remove rows with too many missing values
    combined_df = combined_df.dropna(thresh=len(combined_df.columns) * 0.8)
    
    # Save final feature matrix
    output_path = "data/final_features_with_sentiment.csv"
    combined_df.to_csv(output_path)
    
    print(f"All features merged and saved to {output_path}")
    print(f"Final feature matrix shape: {combined_df.shape}")
    
    return combined_df

if __name__ == "__main__":
    merge_all_features()