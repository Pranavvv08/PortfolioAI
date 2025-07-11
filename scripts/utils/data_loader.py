"""
Data loading and preprocessing utilities for Portfolio Optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Load and preprocess portfolio data"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self._price_data = None
        self._statistical_features = None
        self._log_returns = None
    
    def load_price_data(self):
        """Load historical price data"""
        if self._price_data is None:
            price_file = self.data_dir / "price_matrix.csv"
            self._price_data = pd.read_csv(price_file, index_col=0, parse_dates=True)
        return self._price_data
    
    def load_statistical_features(self):
        """Load statistical features (momentum, volatility, Sharpe ratios)"""
        if self._statistical_features is None:
            features_file = self.data_dir / "statistical_features.csv"
            self._statistical_features = pd.read_csv(features_file, index_col=0, parse_dates=True)
        return self._statistical_features
    
    def get_log_returns(self):
        """Calculate log returns from price data"""
        if self._log_returns is None:
            prices = self.load_price_data()
            self._log_returns = np.log(prices / prices.shift(1)).dropna()
        return self._log_returns
    
    def get_simple_returns(self):
        """Calculate simple returns from price data"""
        prices = self.load_price_data()
        return prices.pct_change().dropna()
    
    def get_stock_list(self):
        """Get list of available stocks"""
        prices = self.load_price_data()
        return list(prices.columns)
    
    def get_feature_data(self, stocks=None, start_date=None, end_date=None):
        """Get feature data for specified stocks and date range"""
        features = self.load_statistical_features()
        
        if start_date:
            features = features[features.index >= start_date]
        if end_date:
            features = features[features.index <= end_date]
            
        if stocks:
            # Filter features for specified stocks
            stock_features = []
            for stock in stocks:
                stock_cols = [col for col in features.columns if col.startswith(f"{stock}_")]
                stock_features.extend(stock_cols)
            features = features[stock_features]
            
        return features
    
    def get_momentum_signals(self, window='5d', stocks=None):
        """Extract momentum signals for specified window"""
        features = self.load_statistical_features()
        stocks = stocks or self.get_stock_list()
        
        momentum_cols = [f"{stock}_mom_{window}" for stock in stocks 
                        if f"{stock}_mom_{window}" in features.columns]
        return features[momentum_cols]
    
    def get_volatility_signals(self, window='5d', stocks=None):
        """Extract volatility signals for specified window"""
        features = self.load_statistical_features()
        stocks = stocks or self.get_stock_list()
        
        vol_cols = [f"{stock}_vol_{window}" for stock in stocks 
                   if f"{stock}_vol_{window}" in features.columns]
        return features[vol_cols]
    
    def get_sharpe_signals(self, window='5d', stocks=None):
        """Extract Sharpe ratio signals for specified window"""
        features = self.load_statistical_features()
        stocks = stocks or self.get_stock_list()
        
        sharpe_cols = [f"{stock}_sharpe_{window}" for stock in stocks 
                      if f"{stock}_sharpe_{window}" in features.columns]
        return features[sharpe_cols]
    
    def get_aligned_data(self, stocks=None, start_date=None, end_date=None):
        """Get aligned price, returns, and feature data"""
        # Get all data
        prices = self.load_price_data()
        returns = self.get_log_returns()
        features = self.load_statistical_features()
        
        # Filter stocks
        if stocks:
            available_stocks = [s for s in stocks if s in prices.columns]
            prices = prices[available_stocks]
            returns = returns[available_stocks]
        else:
            available_stocks = list(prices.columns)
        
        # Get features for selected stocks
        feature_cols = []
        for stock in available_stocks:
            stock_features = [col for col in features.columns if col.startswith(f"{stock}_")]
            feature_cols.extend(stock_features)
        features = features[feature_cols]
        
        # Align dates - use intersection of all date ranges
        common_dates = prices.index.intersection(returns.index).intersection(features.index)
        
        if start_date:
            common_dates = common_dates[common_dates >= start_date]
        if end_date:
            common_dates = common_dates[common_dates <= end_date]
        
        # Filter all datasets to common dates
        prices_aligned = prices.loc[common_dates]
        returns_aligned = returns.loc[common_dates]
        features_aligned = features.loc[common_dates]
        
        return {
            'prices': prices_aligned,
            'returns': returns_aligned,
            'features': features_aligned,
            'stocks': available_stocks,
            'dates': common_dates
        }
    
    def get_covariance_matrix(self, stocks=None, window=None, method='sample'):
        """Calculate covariance matrix of returns"""
        returns = self.get_log_returns()
        
        if stocks:
            available_stocks = [s for s in stocks if s in returns.columns]
            returns = returns[available_stocks]
        
        if window:
            returns = returns.tail(window)
        
        if method == 'sample':
            return returns.cov()
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage estimator
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf()
            cov_shrink = lw.fit(returns.fillna(0)).covariance_
            return pd.DataFrame(cov_shrink, index=returns.columns, columns=returns.columns)
        else:
            raise ValueError(f"Unknown covariance method: {method}")
    
    def get_correlation_matrix(self, stocks=None, window=None):
        """Calculate correlation matrix of returns"""
        returns = self.get_log_returns()
        
        if stocks:
            available_stocks = [s for s in stocks if s in returns.columns]
            returns = returns[available_stocks]
        
        if window:
            returns = returns.tail(window)
            
        return returns.corr()