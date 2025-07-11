"""
Simple test script for the Portfolio Neural Network
"""

import sys
import os
sys.path.append('scripts')

from portfolio_neural_network import PortfolioNeuralNetwork

def test_basic_functionality():
    """Test basic functionality of the neural network"""
    print("Testing Portfolio Neural Network - Basic Functionality")
    print("=" * 55)
    
    # Initialize
    pnn = PortfolioNeuralNetwork(random_state=42)
    
    # Load data
    print("1. Loading data...")
    pnn.load_data()
    print(f"   Loaded {len(pnn.stock_names)} stocks")
    
    # Prepare features and targets
    print("2. Preparing features and targets...")
    features, targets = pnn.prepare_features_and_targets(target_method='risk_adjusted_returns')
    print(f"   Features shape: {features.shape}")
    print(f"   Targets shape: {targets.shape}")
    
    # Train model (quick training for testing)
    print("3. Training model...")
    history = pnn.train_model(features, targets, epochs=5, verbose=0)
    print("   Model trained successfully")
    
    # Run backtest
    print("4. Running backtest...")
    results = pnn.backtest_portfolio()
    print("   Backtest completed")
    
    # Print key results
    metrics = results['metrics']
    print(f"\n5. Results Summary:")
    print(f"   Portfolio Return: {metrics['portfolio_total_return']:.2f}%")
    print(f"   Benchmark Return: {metrics['equal_weight_total_return']:.2f}%")
    print(f"   Portfolio Sharpe: {metrics['portfolio_sharpe']:.3f}")
    print(f"   Benchmark Sharpe: {metrics['equal_weight_sharpe']:.3f}")
    
    # Save model
    print("6. Saving model...")
    pnn.save_model('outputs/test_model.h5')
    print("   Model saved successfully")
    
    # Test prediction on new data
    print("7. Testing prediction...")
    sample_features = features[-1:] # Last sample
    weights = pnn.predict_portfolio_weights(sample_features)
    print(f"   Predicted weights sum: {weights[0].sum():.6f}")
    print(f"   Top 3 stocks: {sorted(zip(pnn.stock_names, weights[0]), key=lambda x: x[1], reverse=True)[:3]}")
    
    print("\n✅ All tests passed successfully!")

if __name__ == "__main__":
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    test_basic_functionality()