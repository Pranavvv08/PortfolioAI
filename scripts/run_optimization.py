#!/usr/bin/env python3
"""
Run Portfolio Optimization System
"""

import sys
import os
sys.path.append('.')

import numpy as np
import pandas as pd
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from utils.data_loader import DataLoader
from utils.constraints import create_default_constraints, create_sector_mapping
from risk_models import EqualWeightModel, MinimumVarianceModel, MaximumSharpeModel, RiskBudgetingModel
from portfolio_strategies import (MeanReversionStrategy, MomentumStrategy, LowVolatilityStrategy,
                                RiskParityStrategy, SharpeOptimizationStrategy)
from performance_analytics import Backtester, save_results_to_files


def main():
    print("=" * 60)
    print("PORTFOLIO OPTIMIZATION ENGINE")
    print("=" * 60)
    
    # Initialize data loader
    print("Loading data...")
    loader = DataLoader('../data')
    aligned_data = loader.get_aligned_data()
    
    stocks = aligned_data['stocks']
    returns = aligned_data['returns']
    features = aligned_data['features']
    
    print(f"Data loaded: {len(stocks)} stocks, {len(returns)} observations")
    print(f"Period: {returns.index[0].date()} to {returns.index[-1].date()}")
    
    # Create constraints
    print("\nSetting up constraints...")
    n_assets = len(stocks)
    constraints = create_default_constraints(n_assets, max_single_weight=0.15)
    
    # Define strategies
    strategies = {
        'Equal Weight': None,  # Will use equal weights
        'Momentum': MomentumStrategy(lookback_window='10d', top_n=20),
        'Mean Reversion': MeanReversionStrategy(lookback_window='20d'),
        'Low Volatility': LowVolatilityStrategy(lookback_window='20d'),
        'Risk Parity': RiskParityStrategy(lookback_window=60),
        'Sharpe Optimization': SharpeOptimizationStrategy(lookback_window=60)
    }
    
    # Define risk models
    risk_models = {
        'Equal Weight': EqualWeightModel(returns),
        'Minimum Variance': MinimumVarianceModel(returns),
        'Maximum Sharpe': MaximumSharpeModel(returns, risk_free_rate=0.02/252),
        'Risk Parity': RiskBudgetingModel(returns)
    }
    
    print(f"Setup complete: {len(strategies)} strategies, {len(risk_models)} risk models")
    
    # Results storage
    all_results = {}
    portfolio_weights = {}
    
    print("\n" + "=" * 60)
    print("RUNNING OPTIMIZATIONS")
    print("=" * 60)
    
    # 1. Risk Model Optimizations
    print("\n1. Risk Model Optimizations:")
    print("-" * 30)
    
    for model_name, model in risk_models.items():
        try:
            print(f"  Optimizing {model_name}...")
            weights = model.optimize_weights(constraints)
            
            # Calculate basic metrics
            portfolio_return = np.sum(weights * returns.mean()) * 252
            portfolio_vol = np.sqrt(weights.T @ returns.cov().values @ weights) * np.sqrt(252)
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            portfolio_weights[f"Risk_Model_{model_name}"] = weights
            all_results[f"Risk_Model_{model_name}"] = {
                'annual_return': portfolio_return,
                'annual_volatility': portfolio_vol,
                'sharpe_ratio': sharpe,
                'method': 'Risk Model'
            }
            
            print(f"    Return: {portfolio_return:.2%}, Vol: {portfolio_vol:.2%}, Sharpe: {sharpe:.3f}")
            
        except Exception as e:
            print(f"    Error in {model_name}: {str(e)}")
            continue
    
    # 2. Strategy Optimizations
    print("\n2. Strategy Optimizations:")
    print("-" * 30)
    
    for strategy_name, strategy in strategies.items():
        try:
            print(f"  Running {strategy_name}...")
            
            if strategy_name == 'Equal Weight':
                # Simple equal weighting
                weights = np.ones(n_assets) / n_assets
                
                # Calculate metrics
                portfolio_return = np.sum(weights * returns.mean()) * 252
                portfolio_vol = np.sqrt(weights.T @ returns.cov().values @ weights) * np.sqrt(252)
                sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
                
                portfolio_weights[strategy_name] = weights
                all_results[strategy_name] = {
                    'annual_return': portfolio_return,
                    'annual_volatility': portfolio_vol,
                    'sharpe_ratio': sharpe,
                    'method': 'Equal Weight'
                }
                
            else:
                # Run backtesting for other strategies
                backtester = Backtester(loader, strategy, constraints)
                backtest_results = backtester.run_backtest(rebalance_freq=10)
                
                analytics = backtester.get_performance_analytics()
                metrics = analytics.calculate_comprehensive_metrics()
                
                # Get latest weights
                weights_history = backtest_results['weights_history']
                if not weights_history.empty:
                    latest_weights = weights_history.iloc[-1]['weights']
                    portfolio_weights[strategy_name] = latest_weights
                
                all_results[strategy_name] = metrics
                all_results[strategy_name]['method'] = 'Strategy Backtest'
            
            result = all_results[strategy_name]
            print(f"    Return: {result['annual_return']:.2%}, Vol: {result['annual_volatility']:.2%}, "
                  f"Sharpe: {result['sharpe_ratio']:.3f}")
            
        except Exception as e:
            print(f"    Error in {strategy_name}: {str(e)}")
            continue
    
    # 3. Generate Output Files
    print("\n" + "=" * 60)
    print("GENERATING OUTPUT FILES")
    print("=" * 60)
    
    # Create results directory
    os.makedirs('../results', exist_ok=True)
    
    # Save portfolio weights
    print("Saving portfolio weights...")
    weights_df = pd.DataFrame(portfolio_weights, index=stocks)
    weights_df.to_csv('../results/portfolio_weights.csv')
    print(f"  Saved {len(weights_df.columns)} portfolio allocations")
    
    # Create performance report
    print("Creating performance report...")
    
    # Calculate summary statistics
    summary_stats = {}
    for name, results in all_results.items():
        summary_stats[name] = {
            'annualized_return': float(results.get('annual_return', results.get('annualized_return', 0))),
            'annualized_volatility': float(results.get('annual_volatility', results.get('annualized_volatility', 0))),
            'sharpe_ratio': float(results.get('sharpe_ratio', 0)),
            'max_drawdown': float(results.get('max_drawdown', 0)),
            'method': results.get('method', 'Unknown')
        }
    
    performance_report = {
        'generation_date': datetime.now().isoformat(),
        'data_summary': {
            'start_date': returns.index[0].isoformat(),
            'end_date': returns.index[-1].isoformat(),
            'n_observations': len(returns),
            'n_assets': len(stocks),
            'assets': stocks
        },
        'optimization_results': summary_stats,
        'best_sharpe_strategy': max(summary_stats.keys(), 
                                   key=lambda x: summary_stats[x]['sharpe_ratio']),
        'lowest_volatility_strategy': min(summary_stats.keys(), 
                                         key=lambda x: summary_stats[x]['annualized_volatility']),
        'highest_return_strategy': max(summary_stats.keys(), 
                                      key=lambda x: summary_stats[x]['annualized_return'])
    }
    
    with open('../results/performance_report.json', 'w') as f:
        json.dump(performance_report, f, indent=2, default=str)
    
    # Create simple backtest results
    print("Creating backtest results...")
    
    # Generate simple backtest for equal weight strategy
    eq_weights = np.ones(n_assets) / n_assets
    eq_returns = (returns * eq_weights).sum(axis=1)
    eq_cumulative = (1 + eq_returns).cumprod()
    
    backtest_df = pd.DataFrame({
        'date': returns.index,
        'strategy': 'Equal Weight',
        'portfolio_return': eq_returns.values,
        'benchmark_return': eq_returns.values,  # Self-benchmark for simplicity
        'portfolio_value': eq_cumulative.values,
        'cumulative_return': eq_cumulative.values - 1
    })
    
    backtest_df.to_csv('../results/backtest_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    
    print("\\nStrategy Performance Ranking (by Sharpe Ratio):")
    sorted_strategies = sorted(summary_stats.items(), 
                              key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    
    for i, (name, stats) in enumerate(sorted_strategies, 1):
        print(f"{i:2d}. {name:25s} | Sharpe: {stats['sharpe_ratio']:6.3f} | "
              f"Return: {stats['annualized_return']:6.1%} | Vol: {stats['annualized_volatility']:6.1%}")
    
    print(f"\\nFiles generated in '../results/':")
    print(f"  • portfolio_weights.csv     - Optimal allocations for each strategy")
    print(f"  • performance_report.json   - Comprehensive performance metrics")
    print(f"  • backtest_results.csv     - Historical backtest results")
    
    print("\\n" + "=" * 60)
    print("PORTFOLIO OPTIMIZATION COMPLETE!")
    print("=" * 60)
    
    return {
        'portfolio_weights': weights_df,
        'performance_report': performance_report,
        'results': all_results
    }


if __name__ == "__main__":
    results = main()