"""
Portfolio performance analytics and backtesting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import json


class PerformanceAnalytics:
    """Portfolio performance analysis and metrics calculation"""
    
    def __init__(self, portfolio_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None,
                 risk_free_rate: float = 0.0):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns
        self.risk_free_rate = risk_free_rate
        self.freq = self._infer_frequency()
    
    def _infer_frequency(self) -> int:
        """Infer the frequency of returns data"""
        # Assume daily returns (252 trading days per year)
        return 252
    
    def calculate_cumulative_returns(self) -> pd.Series:
        """Calculate cumulative returns"""
        return (1 + self.portfolio_returns).cumprod()
    
    def calculate_annualized_return(self) -> float:
        """Calculate annualized return"""
        total_return = self.calculate_cumulative_returns().iloc[-1]
        n_periods = len(self.portfolio_returns)
        return (total_return ** (self.freq / n_periods)) - 1
    
    def calculate_annualized_volatility(self) -> float:
        """Calculate annualized volatility"""
        return self.portfolio_returns.std() * np.sqrt(self.freq)
    
    def calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        excess_return = self.portfolio_returns.mean() - self.risk_free_rate / self.freq
        return excess_return / self.portfolio_returns.std() * np.sqrt(self.freq)
    
    def calculate_max_drawdown(self) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics"""
        cumulative = self.calculate_cumulative_returns()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        max_dd = drawdown.min()
        max_dd_start = rolling_max.idxmax()
        max_dd_end = drawdown.idxmin()
        
        # Calculate recovery time
        recovery_date = None
        if max_dd_end < cumulative.index[-1]:
            post_dd = cumulative[cumulative.index > max_dd_end]
            recovery_idx = post_dd[post_dd >= rolling_max.loc[max_dd_end]].index
            if len(recovery_idx) > 0:
                recovery_date = recovery_idx[0]
        
        return {
            'max_drawdown': max_dd,
            'max_dd_start': max_dd_start,
            'max_dd_end': max_dd_end,
            'recovery_date': recovery_date,
            'drawdown_duration': (max_dd_end - max_dd_start).days if max_dd_start else None
        }
    
    def calculate_var_cvar(self, confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional VaR"""
        var = np.percentile(self.portfolio_returns, confidence_level * 100)
        cvar = self.portfolio_returns[self.portfolio_returns <= var].mean()
        
        return {
            'var': var,
            'cvar': cvar,
            'var_annualized': var * np.sqrt(self.freq),
            'cvar_annualized': cvar * np.sqrt(self.freq)
        }
    
    def calculate_sortino_ratio(self, target_return: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        excess_return = self.portfolio_returns.mean() - self.risk_free_rate / self.freq
        downside_returns = self.portfolio_returns[self.portfolio_returns < target_return]
        downside_deviation = downside_returns.std()
        
        if downside_deviation == 0:
            return np.inf if excess_return > 0 else 0
        
        return excess_return / downside_deviation * np.sqrt(self.freq)
    
    def calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio"""
        annual_return = self.calculate_annualized_return()
        max_dd = abs(self.calculate_max_drawdown()['max_drawdown'])
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / max_dd
    
    def calculate_information_ratio(self) -> Optional[float]:
        """Calculate information ratio vs benchmark"""
        if self.benchmark_returns is None:
            return None
        
        active_returns = self.portfolio_returns - self.benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return active_returns.mean() / tracking_error * np.sqrt(self.freq)
    
    def calculate_beta(self) -> Optional[float]:
        """Calculate portfolio beta vs benchmark"""
        if self.benchmark_returns is None:
            return None
        
        covariance = np.cov(self.portfolio_returns, self.benchmark_returns)[0, 1]
        benchmark_variance = self.benchmark_returns.var()
        
        return covariance / benchmark_variance
    
    def calculate_alpha(self) -> Optional[float]:
        """Calculate portfolio alpha vs benchmark"""
        if self.benchmark_returns is None:
            return None
        
        beta = self.calculate_beta()
        portfolio_return = self.calculate_annualized_return()
        benchmark_return = (1 + self.benchmark_returns).prod() ** (self.freq / len(self.benchmark_returns)) - 1
        
        return portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
    
    def calculate_upside_capture(self) -> Optional[float]:
        """Calculate upside capture ratio"""
        if self.benchmark_returns is None:
            return None
        
        up_market = self.benchmark_returns > 0
        if up_market.sum() == 0:
            return None
        
        portfolio_up = self.portfolio_returns[up_market].mean()
        benchmark_up = self.benchmark_returns[up_market].mean()
        
        return portfolio_up / benchmark_up if benchmark_up != 0 else None
    
    def calculate_downside_capture(self) -> Optional[float]:
        """Calculate downside capture ratio"""
        if self.benchmark_returns is None:
            return None
        
        down_market = self.benchmark_returns < 0
        if down_market.sum() == 0:
            return None
        
        portfolio_down = self.portfolio_returns[down_market].mean()
        benchmark_down = self.benchmark_returns[down_market].mean()
        
        return portfolio_down / benchmark_down if benchmark_down != 0 else None
    
    def calculate_win_rate(self) -> float:
        """Calculate percentage of positive return periods"""
        return (self.portfolio_returns > 0).mean()
    
    def calculate_comprehensive_metrics(self) -> Dict:
        """Calculate all performance metrics"""
        metrics = {
            'annualized_return': self.calculate_annualized_return(),
            'annualized_volatility': self.calculate_annualized_volatility(),
            'sharpe_ratio': self.calculate_sharpe_ratio(),
            'sortino_ratio': self.calculate_sortino_ratio(),
            'calmar_ratio': self.calculate_calmar_ratio(),
            'win_rate': self.calculate_win_rate(),
            'skewness': self.portfolio_returns.skew(),
            'kurtosis': self.portfolio_returns.kurtosis(),
        }
        
        # Add drawdown metrics
        dd_metrics = self.calculate_max_drawdown()
        metrics.update(dd_metrics)
        
        # Add VaR/CVaR metrics
        var_metrics = self.calculate_var_cvar()
        metrics.update(var_metrics)
        
        # Add benchmark-relative metrics if available
        if self.benchmark_returns is not None:
            metrics.update({
                'information_ratio': self.calculate_information_ratio(),
                'beta': self.calculate_beta(),
                'alpha': self.calculate_alpha(),
                'upside_capture': self.calculate_upside_capture(),
                'downside_capture': self.calculate_downside_capture()
            })
        
        return metrics


class Backtester:
    """Portfolio backtesting framework"""
    
    def __init__(self, data_loader, strategy, constraints=None):
        self.data_loader = data_loader
        self.strategy = strategy
        self.constraints = constraints
        self.results = None
    
    def run_backtest(self, start_date: Optional[str] = None, end_date: Optional[str] = None,
                    rebalance_freq: int = 5, initial_capital: float = 100000,
                    transaction_cost: float = 0.001) -> Dict:
        """Run portfolio backtest"""
        # Get aligned data
        aligned_data = self.data_loader.get_aligned_data(start_date=start_date, end_date=end_date)
        
        returns = aligned_data['returns']
        features = aligned_data['features']
        dates = aligned_data['dates']
        stocks = aligned_data['stocks']
        
        # Rebalancing dates
        rebalance_dates = dates[::rebalance_freq]
        
        # Initialize tracking variables
        portfolio_values = [initial_capital]
        portfolio_returns = []
        weights_history = []
        transaction_costs = []
        
        current_weights = np.zeros(len(stocks))
        
        for i, rebal_date in enumerate(rebalance_dates[:-1]):
            # Get data up to rebalancing date
            historical_data = {
                'returns': returns.loc[:rebal_date],
                'features': features.loc[:rebal_date],
                'stocks': stocks
            }
            
            # Generate new weights
            signals = self.strategy.generate_signals(historical_data)
            new_weights = self.strategy.calculate_weights(signals, historical_data, self.constraints)
            
            # Calculate transaction costs
            weight_changes = np.sum(np.abs(new_weights - current_weights))
            transaction_cost_amount = weight_changes * transaction_cost * portfolio_values[-1]
            transaction_costs.append(transaction_cost_amount)
            
            # Update current weights
            current_weights = new_weights.copy()
            weights_history.append({
                'date': rebal_date,
                'weights': current_weights.copy()
            })
            
            # Calculate returns until next rebalancing
            next_rebal_date = rebalance_dates[i + 1]
            period_dates = dates[(dates > rebal_date) & (dates <= next_rebal_date)]
            
            for date in period_dates:
                if date in returns.index:
                    daily_returns = returns.loc[date].values
                    portfolio_return = np.sum(current_weights * daily_returns)
                    portfolio_returns.append(portfolio_return)
                    
                    # Update portfolio value
                    new_value = portfolio_values[-1] * (1 + portfolio_return)
                    portfolio_values.append(new_value)
        
        # Create results DataFrame
        portfolio_returns = pd.Series(portfolio_returns, 
                                    index=dates[1:len(portfolio_returns)+1])
        
        # Calculate benchmark (equal weight) returns for comparison
        n_stocks = len(stocks)
        equal_weights = np.ones(n_stocks) / n_stocks
        benchmark_returns = (returns * equal_weights).sum(axis=1)
        
        # Align benchmark with portfolio returns
        benchmark_returns = benchmark_returns.loc[portfolio_returns.index]
        
        self.results = {
            'portfolio_returns': portfolio_returns,
            'benchmark_returns': benchmark_returns,
            'portfolio_values': pd.Series(portfolio_values[1:], index=portfolio_returns.index),
            'weights_history': pd.DataFrame(weights_history),
            'transaction_costs': np.sum(transaction_costs),
            'final_value': portfolio_values[-1],
            'total_return': (portfolio_values[-1] / initial_capital) - 1
        }
        
        return self.results
    
    def get_performance_analytics(self) -> PerformanceAnalytics:
        """Get performance analytics object"""
        if self.results is None:
            raise ValueError("Run backtest first")
        
        return PerformanceAnalytics(
            self.results['portfolio_returns'],
            self.results['benchmark_returns']
        )
    
    def generate_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if self.results is None:
            raise ValueError("Run backtest first")
        
        analytics = self.get_performance_analytics()
        metrics = analytics.calculate_comprehensive_metrics()
        
        # Add backtest-specific metrics
        metrics.update({
            'strategy_name': self.strategy.name,
            'total_return': self.results['total_return'],
            'final_value': self.results['final_value'],
            'transaction_costs': self.results['transaction_costs'],
            'number_of_rebalances': len(self.results['weights_history']),
            'average_position_size': self.results['weights_history']['weights'].apply(
                lambda x: np.mean(x[x > 0.001])
            ).mean(),
            'average_number_of_positions': self.results['weights_history']['weights'].apply(
                lambda x: np.sum(x > 0.001)
            ).mean()
        })
        
        return metrics


class PortfolioComparison:
    """Compare multiple portfolio strategies"""
    
    def __init__(self):
        self.strategies_results = {}
    
    def add_strategy_results(self, strategy_name: str, backtest_results: Dict):
        """Add results from a strategy backtest"""
        self.strategies_results[strategy_name] = backtest_results
    
    def compare_performance(self) -> pd.DataFrame:
        """Compare performance metrics across strategies"""
        comparison_data = []
        
        for strategy_name, results in self.strategies_results.items():
            analytics = PerformanceAnalytics(
                results['portfolio_returns'],
                results['benchmark_returns']
            )
            
            metrics = analytics.calculate_comprehensive_metrics()
            metrics['strategy'] = strategy_name
            comparison_data.append(metrics)
        
        return pd.DataFrame(comparison_data).set_index('strategy')
    
    def plot_cumulative_returns(self, figsize: Tuple[int, int] = (12, 8)):
        """Plot cumulative returns comparison"""
        plt.figure(figsize=figsize)
        
        for strategy_name, results in self.strategies_results.items():
            cumulative = (1 + results['portfolio_returns']).cumprod()
            plt.plot(cumulative.index, cumulative.values, label=strategy_name)
        
        # Add benchmark
        if len(self.strategies_results) > 0:
            benchmark = list(self.strategies_results.values())[0]['benchmark_returns']
            benchmark_cumulative = (1 + benchmark).cumprod()
            plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                    label='Equal Weight Benchmark', linestyle='--', alpha=0.7)
        
        plt.title('Cumulative Returns Comparison')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_rolling_sharpe(self, window: int = 60, figsize: Tuple[int, int] = (12, 6)):
        """Plot rolling Sharpe ratios"""
        plt.figure(figsize=figsize)
        
        for strategy_name, results in self.strategies_results.items():
            returns = results['portfolio_returns']
            rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
            plt.plot(rolling_sharpe.index, rolling_sharpe.values, label=strategy_name)
        
        plt.title(f'Rolling {window}-Day Sharpe Ratio')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()


def save_results_to_files(backtest_results: Dict, performance_metrics: Dict, 
                         output_dir: str = "results"):
    """Save backtest results and metrics to files"""
    import os
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save portfolio weights
    weights_df = backtest_results['weights_history']
    if not weights_df.empty:
        # Expand weights into separate columns
        weights_expanded = pd.DataFrame(
            weights_df['weights'].tolist(),
            index=weights_df['date']
        )
        weights_expanded.to_csv(output_path / "portfolio_weights.csv")
    
    # Save performance metrics
    # Convert numpy types to Python types for JSON serialization
    metrics_serializable = {}
    for key, value in performance_metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif pd.isna(value):
            metrics_serializable[key] = None
        elif isinstance(value, pd.Timestamp):
            metrics_serializable[key] = value.isoformat()
        else:
            metrics_serializable[key] = value
    
    with open(output_path / "performance_report.json", 'w') as f:
        json.dump(metrics_serializable, f, indent=2, default=str)
    
    # Save backtest results
    backtest_df = pd.DataFrame({
        'date': backtest_results['portfolio_returns'].index,
        'portfolio_return': backtest_results['portfolio_returns'].values,
        'benchmark_return': backtest_results['benchmark_returns'].values,
        'portfolio_value': backtest_results['portfolio_values'].values
    })
    backtest_df.to_csv(output_path / "backtest_results.csv", index=False)
    
    print(f"Results saved to {output_path}")
    return output_path