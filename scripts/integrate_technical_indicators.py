#!/usr/bin/env python3
"""
Integration script showing how to use technical indicators with the portfolio pipeline
"""

import pandas as pd
import numpy as np
import os

def integrate_technical_indicators():
    """Integrate technical indicators with existing features"""
    
    print("🔗 Technical Indicators Integration")
    print("=" * 50)
    
    # Load existing feature files
    files_to_check = [
        ("data/price_matrix.csv", "Price Matrix"),
        ("data/log_returns.csv", "Log Returns"),
        ("data/statistical_features.csv", "Statistical Features"),
        ("data/liquidity_features.csv", "Liquidity Features"),
        ("data/technical_indicators_demo.csv", "Technical Indicators (Demo)")
    ]
    
    available_data = {}
    
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                available_data[description] = df
                print(f"✅ {description}: {df.shape}")
            except Exception as e:
                print(f"❌ Error loading {description}: {e}")
        else:
            print(f"⚠️  {description}: File not found")
    
    if len(available_data) == 0:
        print("❌ No data files available for integration")
        return False
    
    # Find common date range across all datasets
    print(f"\n📅 Finding common date range...")
    
    all_indices = [df.index for df in available_data.values()]
    common_start = max(idx.min() for idx in all_indices)
    common_end = min(idx.max() for idx in all_indices)
    
    print(f"   Common date range: {common_start.date()} to {common_end.date()}")
    
    # Align all datasets to common date range
    print(f"\n🔄 Aligning datasets...")
    aligned_data = {}
    
    for name, df in available_data.items():
        # Filter to common date range
        aligned_df = df.loc[common_start:common_end]
        aligned_data[name] = aligned_df
        print(f"   {name}: {df.shape} -> {aligned_df.shape}")
    
    # Merge all features
    print(f"\n🔀 Merging features...")
    
    # Start with the first dataset
    first_key = list(aligned_data.keys())[0]
    merged_features = aligned_data[first_key].copy()
    print(f"   Starting with {first_key}: {merged_features.shape}")
    
    # Add other datasets
    for name, df in list(aligned_data.items())[1:]:
        # Merge on index (dates)
        before_shape = merged_features.shape
        merged_features = pd.concat([merged_features, df], axis=1, join='inner')
        print(f"   Added {name}: {before_shape} -> {merged_features.shape}")
    
    # Clean up merged data
    print(f"\n🧹 Cleaning merged data...")
    initial_shape = merged_features.shape
    
    # Remove columns with too many NaN values (>80%)
    nan_threshold = len(merged_features) * 0.8
    good_columns = merged_features.columns[merged_features.isna().sum() < nan_threshold]
    merged_features = merged_features[good_columns]
    
    # Remove rows with too many NaN values (>50% of columns)
    nan_threshold_rows = len(merged_features.columns) * 0.5
    good_rows = merged_features.isna().sum(axis=1) < nan_threshold_rows
    merged_features = merged_features[good_rows]
    
    print(f"   Before cleanup: {initial_shape}")
    print(f"   After cleanup: {merged_features.shape}")
    
    # Save integrated features
    output_path = "data/integrated_features_with_technical.csv"
    merged_features.to_csv(output_path)
    print(f"\n💾 Saved integrated features to {output_path}")
    
    # Display summary
    print(f"\n📊 Integration Summary:")
    print(f"   📅 Date range: {merged_features.index.min().date()} to {merged_features.index.max().date()}")
    print(f"   📈 Shape: {merged_features.shape}")
    print(f"   🏢 Time periods: {len(merged_features)}")
    print(f"   📊 Features: {len(merged_features.columns)}")
    
    # Show feature categories
    print(f"\n🏷️  Feature Categories:")
    
    feature_categories = {
        'Technical Indicators': 0,
        'Statistical Features': 0,
        'Liquidity Features': 0,
        'Price/Return Features': 0,
        'Other': 0
    }
    
    for col in merged_features.columns:
        if any(indicator in col.lower() for indicator in ['rsi', 'sma', 'ema', 'bb_', 'macd', 'stoch', 'atr', 'cci', 'willr', 'obv']):
            feature_categories['Technical Indicators'] += 1
        elif any(stat in col.lower() for stat in ['mom_', 'vol_', 'sharpe_']):
            feature_categories['Statistical Features'] += 1
        elif any(liq in col.lower() for liq in ['volume', 'dollar_vol', 'liquidity']):
            feature_categories['Liquidity Features'] += 1
        elif any(price in col.lower() for price in ['return', 'price', 'close']):
            feature_categories['Price/Return Features'] += 1
        else:
            feature_categories['Other'] += 1
    
    for category, count in feature_categories.items():
        if count > 0:
            print(f"   {category}: {count} features")
    
    # Data quality check
    print(f"\n🔍 Data Quality Check:")
    total_values = merged_features.size
    nan_values = merged_features.isna().sum().sum()
    coverage = (total_values - nan_values) / total_values * 100
    print(f"   Data coverage: {coverage:.1f}%")
    
    # Show sample data
    print(f"\n📋 Sample Features (first 5):")
    for col in merged_features.columns[:5]:
        print(f"   {col}")
    
    print(f"\n✅ Integration completed successfully!")
    print(f"📈 Ready for machine learning pipeline with {merged_features.shape[1]} features")
    
    return True

if __name__ == "__main__":
    success = integrate_technical_indicators()
    exit(0 if success else 1)