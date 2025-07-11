"""
Portfolio strategies implementation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from risk_models import RiskModel, MinimumVarianceModel, MaximumSharpeModel


class PortfolioStrategy:
    """Base class for portfolio strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signals(self, data: Dict) -> np.ndarray:
        """Generate strategy signals for each asset"""
        raise NotImplementedError
    
    def calculate_weights(self, signals: np.ndarray, data: Dict, 
                         constraints=None) -> np.ndarray:
        """Calculate portfolio weights from signals"""
        raise NotImplementedError


class MeanReversionStrategy(PortfolioStrategy):
    """Mean reversion strategy using momentum indicators"""
    
    def __init__(self, lookback_window: str = '20d', momentum_threshold: float = 0.0,
                 volatility_weight: bool = True):
        super().__init__("Mean Reversion Strategy")
        self.lookback_window = lookback_window
        self.momentum_threshold = momentum_threshold
        self.volatility_weight = volatility_weight
    
    def generate_signals(self, data: Dict) -> np.ndarray:
        """Generate mean reversion signals"""
        features = data['features']
        stocks = data['stocks']
        
        # Get momentum signals for the specified window
        momentum_cols = [f"{stock}_mom_{self.lookback_window}" for stock in stocks]
        momentum_data = features[[col for col in momentum_cols if col in features.columns]]
        
        if momentum_data.empty:
            return np.zeros(len(stocks))
        
        # Get latest momentum values
        latest_momentum = momentum_data.iloc[-1].values
        
        # Mean reversion signal: negative momentum indicates oversold conditions
        # Stronger negative momentum = stronger buy signal
        signals = -latest_momentum  # Invert momentum for mean reversion
        
        # Apply threshold
        signals = np.where(latest_momentum < self.momentum_threshold, signals, 0)
        
        # Normalize signals
        if np.sum(np.abs(signals)) > 0:
            signals = signals / np.sum(np.abs(signals))
        else:
            signals = np.ones(len(stocks)) / len(stocks)
        
        return signals
    
    def calculate_weights(self, signals: np.ndarray, data: Dict, 
                         constraints=None) -> np.ndarray:
        """Calculate portfolio weights from mean reversion signals"""
        if self.volatility_weight:
            # Weight by inverse volatility to reduce risk
            features = data['features']
            stocks = data['stocks']
            
            vol_cols = [f"{stock}_vol_{self.lookback_window}" for stock in stocks]
            vol_data = features[[col for col in vol_cols if col in features.columns]]
            
            if not vol_data.empty:
                latest_vol = vol_data.iloc[-1].values
                latest_vol = np.where(latest_vol > 0, latest_vol, np.median(latest_vol))
                vol_weights = 1 / latest_vol
                vol_weights = vol_weights / np.sum(vol_weights)
                
                # Combine signals with volatility weighting
                weights = signals * vol_weights
            else:
                weights = signals
        else:
            weights = signals
        
        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights


class MomentumStrategy(PortfolioStrategy):
    """Momentum strategy following trending stocks"""
    
    def __init__(self, lookback_window: str = '10d', momentum_threshold: float = 0.0,
                 top_n: Optional[int] = None):
        super().__init__("Momentum Strategy")
        self.lookback_window = lookback_window
        self.momentum_threshold = momentum_threshold
        self.top_n = top_n
    
    def generate_signals(self, data: Dict) -> np.ndarray:
        """Generate momentum signals"""
        features = data['features']
        stocks = data['stocks']
        
        # Get momentum signals for the specified window
        momentum_cols = [f"{stock}_mom_{self.lookback_window}" for stock in stocks]
        momentum_data = features[[col for col in momentum_cols if col in features.columns]]
        
        if momentum_data.empty:
            return np.zeros(len(stocks))
        
        # Get latest momentum values
        latest_momentum = momentum_data.iloc[-1].values
        
        # Momentum signal: positive momentum indicates upward trend
        signals = latest_momentum.copy()
        
        # Apply threshold - only positive momentum above threshold
        signals = np.where(signals > self.momentum_threshold, signals, 0)
        
        # Select top N stocks if specified
        if self.top_n and self.top_n < len(signals):
            top_indices = np.argsort(signals)[-self.top_n:]
            filtered_signals = np.zeros_like(signals)
            filtered_signals[top_indices] = signals[top_indices]
            signals = filtered_signals
        
        # Normalize signals
        if np.sum(signals) > 0:
            signals = signals / np.sum(signals)
        else:
            signals = np.ones(len(stocks)) / len(stocks)
        
        return signals
    
    def calculate_weights(self, signals: np.ndarray, data: Dict, 
                         constraints=None) -> np.ndarray:
        """Calculate portfolio weights from momentum signals"""
        # Use signals directly as weights for momentum strategy
        weights = signals.copy()
        
        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights


class LowVolatilityStrategy(PortfolioStrategy):
    """Low volatility strategy focusing on low-risk stocks"""
    
    def __init__(self, lookback_window: str = '20d', use_min_variance: bool = True):
        super().__init__("Low Volatility Strategy")
        self.lookback_window = lookback_window
        self.use_min_variance = use_min_variance
    
    def generate_signals(self, data: Dict) -> np.ndarray:
        """Generate low volatility signals"""
        features = data['features']
        stocks = data['stocks']
        
        # Get volatility signals for the specified window
        vol_cols = [f"{stock}_vol_{self.lookback_window}" for stock in stocks]
        vol_data = features[[col for col in vol_cols if col in features.columns]]
        
        if vol_data.empty:
            return np.ones(len(stocks)) / len(stocks)
        
        # Get latest volatility values
        latest_vol = vol_data.iloc[-1].values
        
        # Low volatility signal: inverse of volatility
        # Higher weight for lower volatility stocks
        signals = 1 / (latest_vol + 1e-8)  # Add small constant to avoid division by zero
        
        # Normalize signals
        signals = signals / np.sum(signals)
        
        return signals
    
    def calculate_weights(self, signals: np.ndarray, data: Dict, 
                         constraints=None) -> np.ndarray:
        """Calculate portfolio weights for low volatility strategy"""
        if self.use_min_variance:
            # Use minimum variance optimization
            returns = data['returns']
            min_var_model = MinimumVarianceModel(returns)
            weights = min_var_model.optimize_weights(constraints)
        else:
            # Use inverse volatility weighting
            weights = signals.copy()
            weights = weights / np.sum(weights)
        
        return weights


class RiskParityStrategy(PortfolioStrategy):
    """Risk parity strategy with equal risk contribution"""
    
    def __init__(self, lookback_window: int = 60):
        super().__init__("Risk Parity Strategy")
        self.lookback_window = lookback_window
    
    def generate_signals(self, data: Dict) -> np.ndarray:
        """Generate equal risk signals (equal for all assets)"""
        stocks = data['stocks']
        return np.ones(len(stocks)) / len(stocks)
    
    def calculate_weights(self, signals: np.ndarray, data: Dict, 
                         constraints=None) -> np.ndarray:
        """Calculate risk parity portfolio weights"""
        from risk_models import RiskBudgetingModel
        
        returns = data['returns']
        
        # Use recent data for covariance estimation
        if len(returns) > self.lookback_window:
            recent_returns = returns.tail(self.lookback_window)
        else:
            recent_returns = returns
        
        # Equal risk budgets for all assets
        n_assets = len(data['stocks'])
        equal_risk_budgets = np.ones(n_assets) / n_assets
        
        risk_parity_model = RiskBudgetingModel(recent_returns, equal_risk_budgets)
        weights = risk_parity_model.optimize_weights(constraints)
        
        return weights


class SharpeOptimizationStrategy(PortfolioStrategy):
    """Sharpe ratio optimization strategy"""
    
    def __init__(self, lookback_window: int = 60, risk_free_rate: float = 0.0):
        super().__init__("Sharpe Optimization Strategy")
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
    
    def generate_signals(self, data: Dict) -> np.ndarray:
        """Generate signals based on Sharpe ratios"""
        features = data['features']
        stocks = data['stocks']
        
        # Get Sharpe ratio signals
        sharpe_cols = [f"{stock}_sharpe_20d" for stock in stocks]  # Use 20d Sharpe ratios
        sharpe_data = features[[col for col in sharpe_cols if col in features.columns]]
        
        if sharpe_data.empty:
            return np.ones(len(stocks)) / len(stocks)
        
        # Get latest Sharpe ratios
        latest_sharpe = sharpe_data.iloc[-1].values
        
        # Use positive Sharpe ratios as signals
        signals = np.maximum(latest_sharpe, 0)
        
        # Normalize signals
        if np.sum(signals) > 0:
            signals = signals / np.sum(signals)
        else:
            signals = np.ones(len(stocks)) / len(stocks)
        
        return signals
    
    def calculate_weights(self, signals: np.ndarray, data: Dict, 
                         constraints=None) -> np.ndarray:
        """Calculate portfolio weights using Sharpe optimization"""
        returns = data['returns']
        
        # Use recent data for optimization
        if len(returns) > self.lookback_window:
            recent_returns = returns.tail(self.lookback_window)
        else:
            recent_returns = returns
        
        max_sharpe_model = MaximumSharpeModel(recent_returns, self.risk_free_rate)
        weights = max_sharpe_model.optimize_weights(constraints)
        
        return weights


class AdaptiveStrategy(PortfolioStrategy):
    """Adaptive strategy that combines multiple strategies"""
    
    def __init__(self, strategies: List[PortfolioStrategy], 
                 adaptation_window: int = 30):
        super().__init__("Adaptive Strategy")
        self.strategies = strategies
        self.adaptation_window = adaptation_window
        self.strategy_weights = np.ones(len(strategies)) / len(strategies)
    
    def _evaluate_strategy_performance(self, strategy: PortfolioStrategy, 
                                     data: Dict) -> float:
        """Evaluate recent performance of a strategy"""
        returns = data['returns']
        
        if len(returns) < self.adaptation_window:
            return 0.0
        
        # Get recent data
        recent_data = {
            'returns': returns.tail(self.adaptation_window),
            'features': data['features'].tail(self.adaptation_window),
            'stocks': data['stocks']
        }
        
        # Calculate strategy weights and returns
        signals = strategy.generate_signals(recent_data)
        weights = strategy.calculate_weights(signals, recent_data)
        
        strategy_returns = (recent_data['returns'] * weights).sum(axis=1)
        
        # Return Sharpe ratio as performance metric
        if strategy_returns.std() > 0:
            return strategy_returns.mean() / strategy_returns.std()
        else:
            return 0.0
    
    def update_strategy_weights(self, data: Dict):
        """Update weights for each sub-strategy based on recent performance"""
        performances = []
        
        for strategy in self.strategies:
            perf = self._evaluate_strategy_performance(strategy, data)
            performances.append(max(perf, 0))  # Only positive performances
        
        performances = np.array(performances)
        
        # Update strategy weights based on performance
        if np.sum(performances) > 0:
            self.strategy_weights = performances / np.sum(performances)
        else:
            self.strategy_weights = np.ones(len(self.strategies)) / len(self.strategies)
    
    def generate_signals(self, data: Dict) -> np.ndarray:
        """Generate adaptive signals by combining strategies"""
        self.update_strategy_weights(data)
        
        # Combine signals from all strategies
        combined_signals = np.zeros(len(data['stocks']))
        
        for i, strategy in enumerate(self.strategies):
            strategy_signals = strategy.generate_signals(data)
            combined_signals += self.strategy_weights[i] * strategy_signals
        
        return combined_signals
    
    def calculate_weights(self, signals: np.ndarray, data: Dict, 
                         constraints=None) -> np.ndarray:
        """Calculate adaptive portfolio weights"""
        # Use combined signals as weights
        weights = signals.copy()
        
        # Normalize to sum to 1
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights


def create_strategy_ensemble() -> List[PortfolioStrategy]:
    """Create a default ensemble of strategies"""
    strategies = [
        MeanReversionStrategy(lookback_window='20d'),
        MomentumStrategy(lookback_window='10d'),
        LowVolatilityStrategy(lookback_window='20d'),
        RiskParityStrategy(lookback_window=60),
        SharpeOptimizationStrategy(lookback_window=60)
    ]
    return strategies


def backtest_strategy(strategy: PortfolioStrategy, data: Dict, 
                     rebalance_freq: int = 5, constraints=None) -> pd.DataFrame:
    """Backtest a strategy over historical data"""
    returns = data['returns']
    features = data['features']
    
    # Align dates
    common_dates = returns.index.intersection(features.index)
    backtest_dates = common_dates[::rebalance_freq]  # Rebalance every N days
    
    results = []
    
    for i, date in enumerate(backtest_dates[1:], 1):  # Start from second date
        # Get data up to current date
        historical_data = {
            'returns': returns.loc[:date],
            'features': features.loc[:date],
            'stocks': data['stocks']
        }
        
        # Generate signals and weights
        signals = strategy.generate_signals(historical_data)
        weights = strategy.calculate_weights(signals, historical_data, constraints)
        
        # Calculate next period return
        next_date_idx = common_dates.get_loc(date) + 1
        if next_date_idx < len(common_dates):
            next_date = common_dates[next_date_idx]
            period_returns = returns.loc[next_date]
            portfolio_return = np.sum(weights * period_returns)
            
            results.append({
                'date': next_date,
                'portfolio_return': portfolio_return,
                'weights': weights.copy()
            })
    
    return pd.DataFrame(results)