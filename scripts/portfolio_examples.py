"""
Example usage of the Portfolio Neural Network

This script demonstrates how to use the portfolio optimization neural network
with different configurations and provides a simplified interface.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio_neural_network import PortfolioNeuralNetwork
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_basic_example():
    """Run a basic example with default settings"""
    print("Running Basic Portfolio Neural Network Example")
    print("=" * 50)
    
    # Initialize with basic settings
    pnn = PortfolioNeuralNetwork(
        lookback_period=5,
        prediction_horizon=1,
        random_state=42
    )
    
    # Load data
    pnn.load_data()
    
    # Prepare features and targets using risk-adjusted returns
    features, targets = pnn.prepare_features_and_targets(
        target_method='risk_adjusted_returns'
    )
    
    # Train with reasonable settings for quick demonstration
    history = pnn.train_model(
        features, targets,
        epochs=30,  # Fewer epochs for quick testing
        batch_size=16,
        early_stopping_patience=5
    )
    
    # Run backtest
    backtest_results = pnn.backtest_portfolio()
    
    # Save model
    pnn.save_model('outputs/basic_portfolio_model.h5')
    
    return pnn, backtest_results

def run_advanced_example():
    """Run an advanced example with custom settings"""
    print("\nRunning Advanced Portfolio Neural Network Example")
    print("=" * 50)
    
    # Initialize with custom settings
    pnn = PortfolioNeuralNetwork(
        lookback_period=3,
        prediction_horizon=1,
        random_state=123
    )
    
    # Load data
    pnn.load_data()
    
    # Try different target method
    features, targets = pnn.prepare_features_and_targets(
        target_method='simple_returns'
    )
    
    # Custom neural network architecture
    def custom_build_model(self, input_dim, output_dim):
        import tensorflow as tf
        
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, input_dim=input_dim, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(output_dim, activation='linear'),
            tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x))
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    # Replace the build_model method temporarily
    original_build_model = pnn.build_model
    pnn.build_model = lambda input_dim, output_dim, **kwargs: custom_build_model(pnn, input_dim, output_dim)
    
    # Train model
    history = pnn.train_model(
        features, targets,
        epochs=25,
        batch_size=8
    )
    
    # Restore original method
    pnn.build_model = original_build_model
    
    # Run backtest
    backtest_results = pnn.backtest_portfolio()
    
    # Save model with different name
    pnn.save_model('outputs/advanced_portfolio_model.h5')
    
    return pnn, backtest_results

def compare_models():
    """Compare different model configurations"""
    print("\nComparing Different Model Configurations")
    print("=" * 50)
    
    results = {}
    
    # Configuration 1: Risk-adjusted returns
    pnn1 = PortfolioNeuralNetwork(random_state=42)
    pnn1.load_data()
    features1, targets1 = pnn1.prepare_features_and_targets('risk_adjusted_returns')
    pnn1.train_model(features1, targets1, epochs=20, verbose=0)
    results['Risk-Adjusted'] = pnn1.backtest_portfolio()
    
    # Configuration 2: Simple returns
    pnn2 = PortfolioNeuralNetwork(random_state=42)
    pnn2.load_data()
    features2, targets2 = pnn2.prepare_features_and_targets('simple_returns')
    pnn2.train_model(features2, targets2, epochs=20, verbose=0)
    results['Simple Returns'] = pnn2.backtest_portfolio()
    
    # Print comparison
    print("\nModel Comparison Results:")
    print("-" * 40)
    for name, result in results.items():
        metrics = result['metrics']
        print(f"{name}:")
        print(f"  Total Return: {metrics['portfolio_total_return']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['portfolio_sharpe']:.3f}")
        print(f"  Outperformance: {metrics['portfolio_total_return'] - metrics['equal_weight_total_return']:.2f}%")
        print()

def generate_predictions_for_new_data():
    """Example of how to use a trained model for new predictions"""
    print("\nGenerating Predictions for New Data")
    print("=" * 40)
    
    # Load a trained model
    try:
        pnn = PortfolioNeuralNetwork()
        pnn.load_model('outputs/basic_portfolio_model.h5')
        
        # Load current data
        pnn.load_data()
        
        # Get latest features (last row)
        latest_features = pd.read_csv('data/final_features.csv')
        feature_cols = [col for col in latest_features.columns if col != 'Date']
        latest_features_array = latest_features[feature_cols].iloc[-1:].values
        
        # Predict portfolio weights
        predicted_weights = pnn.predict_portfolio_weights(latest_features_array)
        
        print(f"Predicted portfolio weights for latest date:")
        print(f"Date: {latest_features['Date'].iloc[-1]}")
        
        # Show top 10 holdings
        weights_df = pd.DataFrame({
            'Stock': pnn.stock_names,
            'Weight': predicted_weights[0]
        }).sort_values('Weight', ascending=False)
        
        print("\nTop 10 Holdings:")
        print(weights_df.head(10).to_string(index=False))
        
        print(f"\nTotal weight sum: {predicted_weights[0].sum():.6f}")
        
    except FileNotFoundError:
        print("No trained model found. Please run basic example first.")

def main():
    """Run all examples"""
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    print("Portfolio Neural Network Examples")
    print("=" * 60)
    
    # Run basic example
    pnn_basic, results_basic = run_basic_example()
    
    # Run advanced example
    pnn_advanced, results_advanced = run_advanced_example()
    
    # Compare models
    compare_models()
    
    # Generate predictions
    generate_predictions_for_new_data()
    
    print("\nAll examples completed successfully!")
    print("\nSaved models:")
    print("- outputs/basic_portfolio_model.h5")
    print("- outputs/advanced_portfolio_model.h5")
    print("- outputs/portfolio_neural_network.h5")

if __name__ == "__main__":
    main()