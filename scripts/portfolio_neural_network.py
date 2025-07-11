"""
Deep Neural Network for Portfolio Optimization

This script implements a neural network that takes engineered features (statistical, liquidity, correlation)
and predicts optimal portfolio weights for NIFTY 50 stocks.

Features:
- Multi-layer neural network with TensorFlow/Keras
- Target variables based on risk-adjusted returns
- Proper train/validation/test splits
- Regularization (dropout, L2 regularization)
- Portfolio weights that sum to 1
- Backtesting functionality
- Model saving for future predictions
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

class PortfolioNeuralNetwork:
    def __init__(self, lookback_period=5, prediction_horizon=1, random_state=42):
        """
        Initialize the Portfolio Neural Network
        
        Args:
            lookback_period (int): Number of periods to look back for features
            prediction_horizon (int): Number of periods ahead to predict
            random_state (int): Random seed for reproducibility
        """
        self.lookback_period = lookback_period
        self.prediction_horizon = prediction_horizon
        self.random_state = random_state
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_targets = StandardScaler()
        self.stock_names = None
        self.feature_names = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
    def load_data(self, features_path='data/final_features.csv', returns_path='data/log_returns.csv'):
        """Load and prepare data"""
        print("Loading data...")
        
        # Load features and returns
        self.features_df = pd.read_csv(features_path)
        self.returns_df = pd.read_csv(returns_path)
        
        # Parse dates
        self.features_df['Date'] = pd.to_datetime(self.features_df['Date'])
        self.returns_df['Date'] = pd.to_datetime(self.returns_df['Date'])
        
        # Sort by date
        self.features_df = self.features_df.sort_values('Date').reset_index(drop=True)
        self.returns_df = self.returns_df.sort_values('Date').reset_index(drop=True)
        
        # Extract stock names from returns columns
        self.stock_names = [col for col in self.returns_df.columns if col != 'Date']
        print(f"Loaded data for {len(self.stock_names)} stocks")
        print(f"Features shape: {self.features_df.shape}")
        print(f"Returns shape: {self.returns_df.shape}")
        
    def create_targets(self, method='risk_adjusted_returns'):
        """
        Create target variables for portfolio optimization
        
        Args:
            method (str): Method for creating targets
                - 'simple_returns': Next period returns
                - 'risk_adjusted_returns': Sharpe ratio based targets
                - 'mean_reversion': Mean reversion based targets
        """
        print(f"Creating targets using method: {method}")
        
        # Ensure data is aligned by date
        merged_data = pd.merge(self.features_df, self.returns_df, on='Date', how='inner')
        
        if method == 'simple_returns':
            # Simple next-period returns
            target_data = []
            for i in range(len(merged_data) - self.prediction_horizon):
                future_returns = merged_data.iloc[i + self.prediction_horizon][self.stock_names].values
                # Ensure it's numeric and handle any NaN values
                future_returns = pd.to_numeric(future_returns, errors='coerce')
                target_data.append(future_returns)
            
        elif method == 'risk_adjusted_returns':
            # Risk-adjusted returns (Sharpe-like metric)
            target_data = []
            for i in range(len(merged_data) - self.prediction_horizon - 4):  # Need extra data for std calculation
                # Get future returns for next 5 periods
                future_slice = merged_data.iloc[i + self.prediction_horizon:i + self.prediction_horizon + 5]
                
                # Calculate mean and std of future returns
                mean_returns = future_slice[self.stock_names].mean().values
                std_returns = future_slice[self.stock_names].std().values + 1e-6  # Add small epsilon
                
                # Risk-adjusted score (Sharpe-like)
                risk_adjusted = mean_returns / std_returns
                target_data.append(risk_adjusted)
                
        elif method == 'mean_reversion':
            # Mean reversion based targets
            target_data = []
            lookback_window = 10
            
            for i in range(lookback_window, len(merged_data) - self.prediction_horizon):
                # Historical mean
                hist_slice = merged_data.iloc[i - lookback_window:i]
                hist_mean = hist_slice[self.stock_names].mean().values
                
                # Current returns
                current_returns = merged_data.iloc[i][self.stock_names].values
                
                # Mean reversion signal
                reversion_signal = hist_mean - current_returns
                target_data.append(reversion_signal)
        
        return np.array(target_data)
    
    def prepare_features_and_targets(self, target_method='risk_adjusted_returns'):
        """Prepare features and targets for training"""
        print("Preparing features and targets...")
        
        # Merge data by date
        merged_data = pd.merge(self.features_df, self.returns_df, on='Date', how='inner')
        
        # Extract feature columns (exclude Date and stock return columns)
        feature_cols = [col for col in merged_data.columns 
                       if col != 'Date' and col not in self.stock_names]
        self.feature_names = feature_cols
        
        # Create targets
        targets = self.create_targets(method=target_method)
        
        # Prepare features (align with targets)
        if target_method == 'risk_adjusted_returns':
            start_idx = 0
            end_idx = len(targets)
        elif target_method == 'mean_reversion':
            start_idx = 10  # lookback_window
            end_idx = start_idx + len(targets)
        else:
            start_idx = 0
            end_idx = len(targets)
            
        features = merged_data.iloc[start_idx:end_idx][feature_cols].values
        
        # Remove any NaN or infinite values
        features_clean = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0).values
        targets_clean = pd.DataFrame(targets).replace([np.inf, -np.inf], np.nan).fillna(0).values
        
        mask = np.isfinite(features_clean).all(axis=1) & np.isfinite(targets_clean).all(axis=1)
        features = features_clean[mask]
        targets = targets_clean[mask]
        
        print(f"Final dataset shape - Features: {features.shape}, Targets: {targets.shape}")
        
        return features, targets
    
    def build_model(self, input_dim, output_dim, hidden_layers=[512, 256, 128], 
                   dropout_rate=0.3, l2_reg=0.001):
        """
        Build the neural network model
        
        Args:
            input_dim (int): Number of input features
            output_dim (int): Number of output stocks
            hidden_layers (list): Hidden layer sizes
            dropout_rate (float): Dropout rate for regularization
            l2_reg (float): L2 regularization strength
        """
        print("Building neural network model...")
        
        model = tf.keras.Sequential()
        
        # Input layer
        model.add(tf.keras.layers.Dense(
            hidden_layers[0], 
            input_dim=input_dim,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Hidden layers
        for layer_size in hidden_layers[1:]:
            model.add(tf.keras.layers.Dense(
                layer_size,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l2(l2_reg)
            ))
            model.add(tf.keras.layers.Dropout(dropout_rate))
        
        # Output layer (raw predictions)
        model.add(tf.keras.layers.Dense(output_dim, activation='linear'))
        
        # Portfolio weight normalization layer
        model.add(tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x)))
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        print("Model architecture:")
        model.summary()
        
        return model
    
    def train_model(self, features, targets, test_size=0.2, val_size=0.2, 
                   epochs=100, batch_size=32, early_stopping_patience=10, verbose=1):
        """Train the neural network model"""
        print("Training the model...")
        
        # Split into train/validation/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, targets, test_size=test_size, random_state=self.random_state, shuffle=False
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=self.random_state, shuffle=False
        )
        
        print(f"Train set: {X_train.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        
        # Scale features
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        X_val_scaled = self.scaler_features.transform(X_val)
        X_test_scaled = self.scaler_features.transform(X_test)
        
        # Scale targets
        y_train_scaled = self.scaler_targets.fit_transform(y_train)
        y_val_scaled = self.scaler_targets.transform(y_val)
        y_test_scaled = self.scaler_targets.transform(y_test)
        
        # Build model
        self.model = self.build_model(
            input_dim=X_train_scaled.shape[1],
            output_dim=y_train_scaled.shape[1]
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=(X_val_scaled, y_val_scaled),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Evaluate on test set
        test_loss, test_mae = self.model.evaluate(X_test_scaled, y_test_scaled, verbose=0)
        if verbose > 0:
            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test MAE: {test_mae:.4f}")
        
        # Store test data for backtesting
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.history = history
        
        return history
    
    def predict_portfolio_weights(self, features):
        """Predict portfolio weights for given features"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Scale features
        features_scaled = self.scaler_features.transform(features)
        
        # Predict weights (already normalized by softmax)
        weights = self.model.predict(features_scaled)
        
        return weights
    
    def backtest_portfolio(self, start_date=None, end_date=None, initial_capital=100000):
        """
        Backtest the portfolio strategy
        
        Args:
            start_date (str): Start date for backtesting
            end_date (str): End date for backtesting  
            initial_capital (float): Initial portfolio value
        """
        print("Running backtest...")
        
        if self.X_test is None:
            raise ValueError("No test data available. Train the model first.")
        
        # Predict weights on test set
        predicted_weights = self.model.predict(self.X_test)
        
        # Get corresponding return data for test period
        # Note: This is simplified - in practice you'd need to align dates properly
        test_returns = self.y_test
        
        # Calculate portfolio returns
        portfolio_returns = []
        equal_weight_returns = []
        equal_weights = np.ones(len(self.stock_names)) / len(self.stock_names)
        
        for i in range(len(predicted_weights)):
            # Portfolio return using predicted weights
            port_return = np.sum(predicted_weights[i] * test_returns[i])
            portfolio_returns.append(port_return)
            
            # Equal weight benchmark
            eq_return = np.sum(equal_weights * test_returns[i])
            equal_weight_returns.append(eq_return)
        
        portfolio_returns = np.array(portfolio_returns)
        equal_weight_returns = np.array(equal_weight_returns)
        
        # Calculate cumulative returns
        portfolio_cum_returns = np.cumprod(1 + portfolio_returns) * initial_capital
        equal_weight_cum_returns = np.cumprod(1 + equal_weight_returns) * initial_capital
        
        # Calculate performance metrics
        portfolio_total_return = (portfolio_cum_returns[-1] / initial_capital - 1) * 100
        equal_weight_total_return = (equal_weight_cum_returns[-1] / initial_capital - 1) * 100
        
        portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252) * 100
        equal_weight_volatility = np.std(equal_weight_returns) * np.sqrt(252) * 100
        
        portfolio_sharpe = (np.mean(portfolio_returns) / np.std(portfolio_returns)) * np.sqrt(252)
        equal_weight_sharpe = (np.mean(equal_weight_returns) / np.std(equal_weight_returns)) * np.sqrt(252)
        
        # Print results
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Portfolio Strategy:")
        print(f"  Total Return: {portfolio_total_return:.2f}%")
        print(f"  Volatility: {portfolio_volatility:.2f}%")
        print(f"  Sharpe Ratio: {portfolio_sharpe:.3f}")
        print(f"\nEqual Weight Benchmark:")
        print(f"  Total Return: {equal_weight_total_return:.2f}%")
        print(f"  Volatility: {equal_weight_volatility:.2f}%")
        print(f"  Sharpe Ratio: {equal_weight_sharpe:.3f}")
        print(f"\nOutperformance: {portfolio_total_return - equal_weight_total_return:.2f}%")
        
        # Store results
        self.backtest_results = {
            'portfolio_returns': portfolio_returns,
            'equal_weight_returns': equal_weight_returns,
            'portfolio_cum_returns': portfolio_cum_returns,
            'equal_weight_cum_returns': equal_weight_cum_returns,
            'predicted_weights': predicted_weights,
            'metrics': {
                'portfolio_total_return': portfolio_total_return,
                'equal_weight_total_return': equal_weight_total_return,
                'portfolio_volatility': portfolio_volatility,
                'equal_weight_volatility': equal_weight_volatility,
                'portfolio_sharpe': portfolio_sharpe,
                'equal_weight_sharpe': equal_weight_sharpe
            }
        }
        
        return self.backtest_results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        # Set matplotlib backend for headless environments
        import matplotlib
        matplotlib.use('Agg')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Plot MAE
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training history plot saved to outputs/training_history.png")
    
    def plot_backtest_results(self):
        """Plot backtest results"""
        if not hasattr(self, 'backtest_results'):
            print("No backtest results available")
            return
            
        # Set matplotlib backend for headless environments
        import matplotlib
        matplotlib.use('Agg')
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Cumulative returns
        ax1.plot(self.backtest_results['portfolio_cum_returns'], label='Neural Network Portfolio')
        ax1.plot(self.backtest_results['equal_weight_cum_returns'], label='Equal Weight Benchmark')
        ax1.set_title('Cumulative Returns')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        
        # Daily returns distribution
        ax2.hist(self.backtest_results['portfolio_returns'], alpha=0.7, label='Portfolio', bins=30)
        ax2.hist(self.backtest_results['equal_weight_returns'], alpha=0.7, label='Equal Weight', bins=30)
        ax2.set_title('Daily Returns Distribution')
        ax2.set_xlabel('Daily Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # Portfolio weights heatmap (sample)
        weights_sample = self.backtest_results['predicted_weights'][:20]  # First 20 predictions
        sns.heatmap(weights_sample.T, ax=ax3, cmap='viridis', cbar=True)
        ax3.set_title('Portfolio Weights Over Time (Sample)')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Stocks')
        
        # Performance metrics comparison
        metrics = self.backtest_results['metrics']
        categories = ['Total Return (%)', 'Volatility (%)', 'Sharpe Ratio']
        portfolio_values = [metrics['portfolio_total_return'], 
                           metrics['portfolio_volatility'], 
                           metrics['portfolio_sharpe']]
        benchmark_values = [metrics['equal_weight_total_return'], 
                           metrics['equal_weight_volatility'], 
                           metrics['equal_weight_sharpe']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, portfolio_values, width, label='Neural Network Portfolio')
        ax4.bar(x + width/2, benchmark_values, width, label='Equal Weight Benchmark')
        ax4.set_title('Performance Metrics Comparison')
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Value')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig('outputs/backtest_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Backtest results plot saved to outputs/backtest_results.png")
    
    def save_model(self, model_path='outputs/portfolio_neural_network.h5'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save!")
            
        # Create outputs directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        
        # Save scalers and metadata
        import pickle
        metadata = {
            'scaler_features': self.scaler_features,
            'scaler_targets': self.scaler_targets,
            'stock_names': self.stock_names,
            'feature_names': self.feature_names,
            'lookback_period': self.lookback_period,
            'prediction_horizon': self.prediction_horizon
        }
        
        with open(model_path.replace('.h5', '_metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Model saved to {model_path}")
        print(f"Metadata saved to {model_path.replace('.h5', '_metadata.pkl')}")
    
    def load_model(self, model_path='outputs/portfolio_neural_network.h5'):
        """Load a trained model"""
        import pickle
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        
        # Load metadata
        with open(model_path.replace('.h5', '_metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        self.scaler_features = metadata['scaler_features']
        self.scaler_targets = metadata['scaler_targets']
        self.stock_names = metadata['stock_names']
        self.feature_names = metadata['feature_names']
        self.lookback_period = metadata['lookback_period']
        self.prediction_horizon = metadata['prediction_horizon']
        
        print(f"Model loaded from {model_path}")


def main():
    """Example usage of the Portfolio Neural Network"""
    print("Portfolio Neural Network for NIFTY 50 Optimization")
    print("="*60)
    
    # Initialize the model
    pnn = PortfolioNeuralNetwork(lookback_period=5, prediction_horizon=1)
    
    # Load data
    pnn.load_data()
    
    # Prepare features and targets
    features, targets = pnn.prepare_features_and_targets(target_method='risk_adjusted_returns')
    
    # Train the model
    history = pnn.train_model(features, targets, epochs=50, batch_size=16)
    
    # Plot training history
    pnn.plot_training_history()
    
    # Run backtest
    backtest_results = pnn.backtest_portfolio()
    
    # Plot backtest results
    pnn.plot_backtest_results()
    
    # Save the model
    pnn.save_model()
    
    print("\nTraining and evaluation complete!")


if __name__ == "__main__":
    main()