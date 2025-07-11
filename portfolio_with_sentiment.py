#!/usr/bin/env python3
"""
Portfolio optimization neural network with sentiment analysis integration.

This demonstrates how to use the enhanced feature set (including sentiment)
for portfolio optimization using a simple neural network.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, ElasticNet

# Deep learning (optional)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """Portfolio optimization with sentiment-enhanced features."""
    
    def __init__(self):
        """Initialize portfolio optimizer."""
        self.scaler = RobustScaler()
        self.models = {}
        self.feature_importance = {}
        
        # NIFTY 50 symbols
        self.nifty50_symbols = [
            "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
            "ITC.NS", "LT.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS",
            "AXISBANK.NS", "HCLTECH.NS", "MARUTI.NS", "ASIANPAINT.NS", "WIPRO.NS",
            "ULTRACEMCO.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "NTPC.NS", "TITAN.NS",
            "SUNPHARMA.NS", "TECHM.NS", "INDUSINDBK.NS", "POWERGRID.NS", "JSWSTEEL.NS",
            "ADANIENT.NS", "BAJAJFINSV.NS", "CIPLA.NS", "COALINDIA.NS", "EICHERMOT.NS",
            "HINDALCO.NS", "ONGC.NS", "TATACONSUM.NS", "NESTLEIND.NS", "DRREDDY.NS",
            "M&M.NS", "GRASIM.NS", "HDFCLIFE.NS", "BRITANNIA.NS", "APOLLOHOSP.NS",
            "DIVISLAB.NS", "BPCL.NS", "SBILIFE.NS", "TATAMOTORS.NS", "HEROMOTOCO.NS",
            "BAJAJ-AUTO.NS", "UPL.NS", "TATASTEEL.NS", "ADANIPORTS.NS", "IOC.NS"
        ]
    
    def load_and_prepare_data(self, features_file: str = "data/final_features.csv") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for training."""
        logger.info("Loading and preparing data...")
        
        # Load features
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        features_df = pd.read_csv(features_file, index_col=0, parse_dates=True)
        logger.info(f"Loaded features: {features_df.shape}")
        
        # Load price data for targets
        price_file = "data/price_matrix.csv"
        if not os.path.exists(price_file):
            raise FileNotFoundError(f"Price file not found: {price_file}")
        
        price_df = pd.read_csv(price_file, index_col=0, parse_dates=True)
        
        # Normalize price data index to match features
        price_df.index = price_df.index.date
        price_df.index = pd.to_datetime(price_df.index)
        
        logger.info(f"Loaded prices: {price_df.shape}")
        
        # Calculate returns as targets
        returns_df = price_df.pct_change().dropna()
        
        # Align features and returns
        common_dates = features_df.index.intersection(returns_df.index)
        
        if len(common_dates) == 0:
            raise ValueError("No common dates between features and returns")
        
        features_aligned = features_df.loc[common_dates]
        returns_aligned = returns_df.loc[common_dates]
        
        logger.info(f"Aligned data shape - Features: {features_aligned.shape}, Returns: {returns_aligned.shape}")
        logger.info(f"Date range: {common_dates.min()} to {common_dates.max()}")
        
        return features_aligned, returns_aligned
    
    def create_targets(self, returns_df: pd.DataFrame, target_type: str = "next_day") -> pd.DataFrame:
        """Create prediction targets from returns."""
        if target_type == "next_day":
            # Predict next day returns
            targets = returns_df.shift(-1).dropna()
        elif target_type == "weekly":
            # Predict weekly returns
            targets = returns_df.rolling(window=5).mean().shift(-5).dropna()
        elif target_type == "sharpe":
            # Predict Sharpe ratio
            mean_returns = returns_df.rolling(window=20).mean()
            vol_returns = returns_df.rolling(window=20).std()
            targets = (mean_returns / vol_returns).shift(-1).dropna()
        else:
            raise ValueError(f"Unknown target type: {target_type}")
        
        return targets
    
    def analyze_feature_importance(self, features_df: pd.DataFrame) -> Dict:
        """Analyze feature importance by category."""
        logger.info("Analyzing feature importance by category...")
        
        feature_categories = {
            'statistical': [col for col in features_df.columns if any(x in col.lower() for x in ['mom', 'vol', 'sharpe'])],
            'liquidity': [col for col in features_df.columns if any(x in col.lower() for x in ['volume', 'dollar'])],
            'sentiment': [col for col in features_df.columns if 'sentiment' in col.lower()],
            'market_sentiment': [col for col in features_df.columns if 'market_sentiment' in col.lower()],
            'other': []
        }
        
        # Calculate correlations with a sample stock
        sample_returns = pd.read_csv("data/log_returns.csv", index_col=0, parse_dates=True)
        sample_returns.index = sample_returns.index.date
        sample_returns.index = pd.to_datetime(sample_returns.index)
        
        # Use first available stock
        sample_stock = sample_returns.columns[0]
        stock_returns = sample_returns[sample_stock].dropna()
        
        # Align with features
        common_dates = features_df.index.intersection(stock_returns.index)
        if len(common_dates) > 0:
            aligned_features = features_df.loc[common_dates]
            aligned_returns = stock_returns.loc[common_dates]
            
            # Calculate correlations
            correlations = {}
            for category, cols in feature_categories.items():
                if cols:
                    category_features = aligned_features[cols]
                    corr_values = category_features.corrwith(aligned_returns).abs()
                    correlations[category] = {
                        'mean_correlation': corr_values.mean(),
                        'max_correlation': corr_values.max(),
                        'top_features': corr_values.nlargest(3).to_dict()
                    }
        
        analysis = {
            'feature_counts': {cat: len(cols) for cat, cols in feature_categories.items()},
            'correlations': correlations if 'correlations' in locals() else {},
            'total_features': features_df.shape[1]
        }
        
        return analysis
    
    def train_portfolio_models(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> Dict:
        """Train multiple portfolio optimization models."""
        logger.info("Training portfolio optimization models...")
        
        # Create targets (next day returns for simplicity)
        targets = self.create_targets(returns_df, "next_day")
        
        # Align features and targets
        common_dates = features_df.index.intersection(targets.index)
        X = features_df.loc[common_dates]
        y = targets.loc[common_dates]
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        logger.info(f"Training data shape: X{X.shape}, y{y.shape}")
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        models_to_train = {
            'ridge': Ridge(alpha=1.0),
            'elastic_net': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        }
        
        results = {}
        
        for model_name, model in models_to_train.items():
            logger.info(f"Training {model_name}...")
            
            fold_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Train on first stock for demonstration
                stock_col = y.columns[0]
                y_train_stock = y_train[stock_col]
                y_val_stock = y_val[stock_col]
                
                # Train model
                model.fit(X_train_scaled, y_train_stock)
                
                # Predict
                y_pred = model.predict(X_val_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_val_stock, y_pred)
                r2 = r2_score(y_val_stock, y_pred)
                
                fold_scores.append({'mse': mse, 'r2': r2})
            
            # Average scores across folds
            avg_mse = np.mean([score['mse'] for score in fold_scores])
            avg_r2 = np.mean([score['r2'] for score in fold_scores])
            
            results[model_name] = {
                'avg_mse': avg_mse,
                'avg_r2': avg_r2,
                'fold_scores': fold_scores
            }
            
            self.models[model_name] = model
            
            logger.info(f"{model_name} - MSE: {avg_mse:.6f}, R²: {avg_r2:.3f}")
        
        return results
    
    def create_simple_portfolio(self, features_df: pd.DataFrame, returns_df: pd.DataFrame) -> Dict:
        """Create a simple sentiment-weighted portfolio."""
        logger.info("Creating sentiment-weighted portfolio...")
        
        # Extract sentiment features
        sentiment_cols = [col for col in features_df.columns if 'composite_sentiment' in col]
        
        if not sentiment_cols:
            logger.warning("No sentiment features found, using equal weights")
            # Equal weight portfolio
            n_stocks = len(self.nifty50_symbols)
            weights = np.ones(n_stocks) / n_stocks
            portfolio_returns = returns_df.mean(axis=1)
        else:
            # Get latest sentiment scores
            latest_sentiment = features_df[sentiment_cols].iloc[-1]
            
            # Extract stock symbols from column names
            stock_sentiments = {}
            for col in sentiment_cols:
                stock = col.replace('_composite_sentiment', '')
                if stock in returns_df.columns:
                    stock_sentiments[stock] = latest_sentiment[col]
            
            if stock_sentiments:
                # Create weights based on sentiment (higher sentiment = higher weight)
                sentiments = np.array(list(stock_sentiments.values()))
                
                # Normalize sentiment to positive values
                min_sentiment = sentiments.min()
                if min_sentiment < 0:
                    sentiments = sentiments - min_sentiment + 0.1
                
                # Normalize to sum to 1
                weights = sentiments / sentiments.sum()
                
                # Calculate portfolio returns
                stock_names = list(stock_sentiments.keys())
                portfolio_returns = (returns_df[stock_names] * weights).sum(axis=1)
            else:
                # Fallback to equal weights
                portfolio_returns = returns_df.mean(axis=1)
                weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
        
        # Calculate portfolio metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        portfolio_stats = {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'total_return': (1 + portfolio_returns).cumprod().iloc[-1] - 1,
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'weights': weights if isinstance(weights, list) else weights.tolist()
        }
        
        return portfolio_stats
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()


def main():
    """Main function to demonstrate portfolio optimization with sentiment."""
    print("🚀 Portfolio Optimization with Sentiment Analysis")
    print("=" * 60)
    
    try:
        # Initialize optimizer
        optimizer = PortfolioOptimizer()
        
        # Load and prepare data
        features_df, returns_df = optimizer.load_and_prepare_data()
        
        # Analyze features
        feature_analysis = optimizer.analyze_feature_importance(features_df)
        
        print("\n📊 FEATURE ANALYSIS")
        print("-" * 40)
        for category, count in feature_analysis['feature_counts'].items():
            if count > 0:
                print(f"{category.capitalize()}: {count} features")
        
        if 'correlations' in feature_analysis and feature_analysis['correlations']:
            print(f"\n📈 CORRELATION ANALYSIS")
            print("-" * 40)
            for category, stats in feature_analysis['correlations'].items():
                if 'mean_correlation' in stats:
                    print(f"{category.capitalize()}: avg correlation = {stats['mean_correlation']:.3f}")
        
        # Train models
        model_results = optimizer.train_portfolio_models(features_df, returns_df)
        
        print(f"\n🤖 MODEL PERFORMANCE")
        print("-" * 40)
        for model_name, results in model_results.items():
            print(f"{model_name.capitalize()}:")
            print(f"  MSE: {results['avg_mse']:.6f}")
            print(f"  R²: {results['avg_r2']:.3f}")
        
        # Create sentiment-based portfolio
        portfolio_stats = optimizer.create_simple_portfolio(features_df, returns_df)
        
        print(f"\n💼 SENTIMENT-ENHANCED PORTFOLIO")
        print("-" * 40)
        print(f"Annual Return: {portfolio_stats['annual_return']:.2%}")
        print(f"Annual Volatility: {portfolio_stats['annual_volatility']:.2%}")
        print(f"Sharpe Ratio: {portfolio_stats['sharpe_ratio']:.3f}")
        print(f"Total Return: {portfolio_stats['total_return']:.2%}")
        print(f"Max Drawdown: {portfolio_stats['max_drawdown']:.2%}")
        
        # Summary
        print(f"\n✅ INTEGRATION SUCCESS")
        print("-" * 40)
        print(f"✓ Successfully integrated {feature_analysis['feature_counts']['sentiment']} sentiment features")
        print(f"✓ Total features available: {feature_analysis['total_features']}")
        print(f"✓ Portfolio optimization models trained and evaluated")
        print(f"✓ Sentiment-weighted portfolio created and analyzed")
        
        print(f"\n🎯 NEXT STEPS")
        print("-" * 40)
        print("• Fine-tune model hyperparameters")
        print("• Implement more sophisticated portfolio optimization algorithms")
        print("• Add risk constraints and transaction costs")
        print("• Backtest strategies over longer periods")
        print("• Integrate real-time news feeds when available")
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        print(f"\n❌ Error: {e}")
        print("Please ensure all data files are generated before running this script.")


if __name__ == "__main__":
    main()