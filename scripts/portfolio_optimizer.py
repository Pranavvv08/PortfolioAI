"""
Main Portfolio Optimization Engine
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
from datetime import datetime

# Import our modules
from utils.data_loader import DataLoader
from utils.constraints import PortfolioConstraints, create_default_constraints, create_sector_mapping
from risk_models import (EqualWeightModel, MinimumVarianceModel, MaximumSharpeModel, 
                        RiskBudgetingModel, RiskMetrics)
from portfolio_strategies import (MeanReversionStrategy, MomentumStrategy, LowVolatilityStrategy,
                                RiskParityStrategy, SharpeOptimizationStrategy, AdaptiveStrategy,
                                create_strategy_ensemble)
from performance_analytics import Backtester, PerformanceAnalytics, PortfolioComparison, save_results_to_files
from utils.visualizations import PortfolioVisualizer, create_strategy_comparison_plots

warnings.filterwarnings('ignore')


class PortfolioOptimizer:
    """Main portfolio optimization engine"""
    
    def __init__(self, data_dir: str = "data", output_dir: str = "results"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize data loader
        self.data_loader = DataLoader(self.data_dir)
        
        # Load data
        self._load_data()
        
        # Initialize strategy and model containers
        self.strategies = {}
        self.risk_models = {}
        self.backtest_results = {}
        
        print(f"Portfolio Optimizer initialized with {len(self.stocks)} stocks")
        print(f"Data period: {self.start_date} to {self.end_date}")
    
    def _load_data(self):
        """Load and prepare all data"""
        self.aligned_data = self.data_loader.get_aligned_data()
        self.stocks = self.aligned_data['stocks']
        self.returns = self.aligned_data['returns']
        self.features = self.aligned_data['features']
        self.prices = self.aligned_data['prices']
        self.dates = self.aligned_data['dates']
        
        self.start_date = self.dates[0]
        self.end_date = self.dates[-1]
        self.n_assets = len(self.stocks)
        
        print(f"Loaded data: {len(self.returns)} return observations, {len(self.features.columns)} features")
    
    def add_strategy(self, name: str, strategy) -> 'PortfolioOptimizer':
        """Add a portfolio strategy"""
        self.strategies[name] = strategy
        return self
    
    def add_risk_model(self, name: str, model) -> 'PortfolioOptimizer':
        """Add a risk model"""
        self.risk_models[name] = model
        return self
    
    def setup_default_strategies(self) -> 'PortfolioOptimizer':
        """Setup default portfolio strategies"""
        self.strategies = {
            'Mean Reversion': MeanReversionStrategy(lookback_window='20d'),
            'Momentum': MomentumStrategy(lookback_window='10d', top_n=20),
            'Low Volatility': LowVolatilityStrategy(lookback_window='20d', use_min_variance=True),
            'Risk Parity': RiskParityStrategy(lookback_window=60),
            'Sharpe Optimization': SharpeOptimizationStrategy(lookback_window=60),
            'Adaptive': AdaptiveStrategy(create_strategy_ensemble())
        }
        
        print(f"Added {len(self.strategies)} default strategies")
        return self
    
    def setup_default_risk_models(self) -> 'PortfolioOptimizer':
        """Setup default risk models"""
        self.risk_models = {
            'Equal Weight': EqualWeightModel(self.returns),
            'Minimum Variance': MinimumVarianceModel(self.returns, shrinkage=True),
            'Maximum Sharpe': MaximumSharpeModel(self.returns, risk_free_rate=0.02/252),
            'Risk Parity': RiskBudgetingModel(self.returns)
        }
        
        print(f"Added {len(self.risk_models)} risk models")
        return self
    
    def create_constraints(self, max_single_weight: float = 0.1, 
                          max_sector_weight: Optional[Dict[str, float]] = None,
                          long_only: bool = True) -> PortfolioConstraints:
        """Create portfolio constraints"""
        constraints = PortfolioConstraints(self.n_assets)
        
        # Basic constraints
        constraints.add_sum_constraint(1.0)
        
        if long_only:
            constraints.add_long_only_constraint()
        
        if max_single_weight < 1.0:
            constraints.add_diversification_constraint(max_single_weight)
        
        # Sector constraints
        if max_sector_weight:
            sector_mapping = create_sector_mapping(self.stocks)
            sector_limits = {sector: (0.0, max_weight) 
                           for sector, max_weight in max_sector_weight.items()}
            constraints.add_sector_constraints(sector_mapping, sector_limits)
        
        return constraints
    
    def optimize_single_strategy(self, strategy_name: str, constraints: Optional[PortfolioConstraints] = None,
                                start_date: Optional[str] = None, end_date: Optional[str] = None,
                                rebalance_freq: int = 5) -> Dict:
        """Optimize a single strategy"""
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        strategy = self.strategies[strategy_name]
        
        # Create default constraints if none provided
        if constraints is None:
            constraints = self.create_constraints()
        
        # Run backtest
        backtester = Backtester(self.data_loader, strategy, constraints)
        results = backtester.run_backtest(
            start_date=start_date, 
            end_date=end_date,
            rebalance_freq=rebalance_freq
        )
        
        # Get performance analytics
        analytics = backtester.get_performance_analytics()
        performance_metrics = analytics.calculate_comprehensive_metrics()
        
        # Store results
        self.backtest_results[strategy_name] = {
            'results': results,
            'performance': performance_metrics,
            'analytics': analytics
        }
        
        print(f"Strategy '{strategy_name}' optimization completed")
        print(f"  Total Return: {performance_metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {performance_metrics['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown: {performance_metrics['max_drawdown']:.2%}")
        
        return self.backtest_results[strategy_name]
    
    def optimize_all_strategies(self, constraints: Optional[PortfolioConstraints] = None,
                               start_date: Optional[str] = None, end_date: Optional[str] = None,
                               rebalance_freq: int = 5) -> Dict:
        """Optimize all strategies"""
        print("Optimizing all strategies...")
        
        for strategy_name in self.strategies.keys():
            try:
                self.optimize_single_strategy(
                    strategy_name, constraints, start_date, end_date, rebalance_freq
                )
            except Exception as e:
                print(f"Error optimizing {strategy_name}: {str(e)}")
                continue
        
        return self.backtest_results
    
    def optimize_risk_models(self, constraints: Optional[PortfolioConstraints] = None) -> Dict:
        """Optimize using different risk models"""
        print("Optimizing risk models...")
        
        if constraints is None:
            constraints = self.create_constraints()
        
        risk_model_results = {}
        
        for model_name, model in self.risk_models.items():
            try:
                # Get optimal weights
                weights = model.optimize_weights(constraints)
                
                # Calculate performance metrics
                metrics = RiskMetrics.calculate_portfolio_metrics(weights, self.returns)
                
                risk_model_results[model_name] = {
                    'weights': weights,
                    'metrics': metrics
                }
                
                print(f"  {model_name}: Sharpe={metrics['sharpe_ratio']:.3f}, "
                      f"Vol={metrics['annualized_volatility']:.2%}")
                
            except Exception as e:
                print(f"Error optimizing {model_name}: {str(e)}")
                continue
        
        return risk_model_results
    
    def generate_optimal_portfolios(self, constraints: Optional[PortfolioConstraints] = None) -> pd.DataFrame:
        """Generate optimal portfolio weights for all strategies"""
        if not self.backtest_results:
            print("No backtest results found. Running optimization...")
            self.optimize_all_strategies(constraints)
        
        # Extract latest weights from each strategy
        portfolio_weights = {}
        
        for strategy_name, results in self.backtest_results.items():
            weights_history = results['results']['weights_history']
            if not weights_history.empty:
                latest_weights = weights_history.iloc[-1]['weights']
                portfolio_weights[strategy_name] = latest_weights
        
        # Create DataFrame
        weights_df = pd.DataFrame(portfolio_weights, index=self.stocks)
        
        return weights_df
    
    def compare_strategies(self) -> PortfolioComparison:
        """Compare all strategy results"""
        if not self.backtest_results:
            raise ValueError("No backtest results to compare. Run optimization first.")
        
        comparison = PortfolioComparison()
        
        for strategy_name, strategy_results in self.backtest_results.items():
            comparison.add_strategy_results(strategy_name, strategy_results['results'])
        
        return comparison
    
    def create_performance_report(self) -> Dict:
        """Create comprehensive performance report"""
        if not self.backtest_results:
            raise ValueError("No results to report. Run optimization first.")
        
        report = {
            'optimization_date': datetime.now().isoformat(),
            'data_period': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'n_observations': len(self.returns)
            },
            'universe': {
                'n_assets': self.n_assets,
                'assets': self.stocks
            },
            'strategies': {}
        }
        
        # Add strategy results
        for strategy_name, results in self.backtest_results.items():
            report['strategies'][strategy_name] = results['performance']
        
        # Add risk model results if available
        if hasattr(self, 'risk_model_results'):
            report['risk_models'] = self.risk_model_results
        
        return report
    
    def save_results(self, include_visualizations: bool = True) -> Path:
        """Save all results to files"""
        print(f"Saving results to {self.output_dir}")
        
        # Create performance report
        report = self.create_performance_report()
        
        # Save portfolio weights
        weights_df = self.generate_optimal_portfolios()
        weights_df.to_csv(self.output_dir / "portfolio_weights.csv")
        
        # Save performance report
        import json
        with open(self.output_dir / "performance_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save detailed backtest results
        all_backtest_results = []
        for strategy_name, results in self.backtest_results.items():
            strategy_results = results['results']
            backtest_df = pd.DataFrame({
                'date': strategy_results['portfolio_returns'].index,
                'strategy': strategy_name,
                'portfolio_return': strategy_results['portfolio_returns'].values,
                'benchmark_return': strategy_results['benchmark_returns'].values,
                'portfolio_value': strategy_results['portfolio_values'].values
            })
            all_backtest_results.append(backtest_df)
        
        if all_backtest_results:
            combined_backtest = pd.concat(all_backtest_results, ignore_index=True)
            combined_backtest.to_csv(self.output_dir / "backtest_results.csv", index=False)
        
        # Create visualizations if requested
        if include_visualizations:
            self._create_visualizations()
        
        print(f"Results saved successfully to {self.output_dir}")
        return self.output_dir
    
    def _create_visualizations(self):
        """Create and save visualization plots"""
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Strategy comparison plots
        if len(self.backtest_results) > 1:
            comparison_data = {name: results['results'] 
                             for name, results in self.backtest_results.items()}
            
            figures = create_strategy_comparison_plots(comparison_data, str(viz_dir))
            
            # Create comparison object for additional plots
            comparison = self.compare_strategies()
            
            # Cumulative returns comparison
            fig = comparison.plot_cumulative_returns()
            fig.savefig(viz_dir / "cumulative_returns_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Rolling Sharpe comparison
            fig = comparison.plot_rolling_sharpe()
            fig.savefig(viz_dir / "rolling_sharpe_comparison.png", dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        # Individual strategy visualizations
        for strategy_name, results in self.backtest_results.items():
            strategy_viz_dir = viz_dir / strategy_name.replace(' ', '_').lower()
            strategy_viz_dir.mkdir(exist_ok=True)
            
            visualizer = PortfolioVisualizer(results['results'])
            
            # Create individual plots
            fig1 = visualizer.plot_cumulative_returns()
            fig1.savefig(strategy_viz_dir / "cumulative_returns.png", dpi=300, bbox_inches='tight')
            plt.close(fig1)
            
            fig2 = visualizer.plot_drawdown()
            fig2.savefig(strategy_viz_dir / "drawdown_analysis.png", dpi=300, bbox_inches='tight')
            plt.close(fig2)
            
            fig3 = visualizer.plot_rolling_metrics()
            fig3.savefig(strategy_viz_dir / "rolling_metrics.png", dpi=300, bbox_inches='tight')
            plt.close(fig3)
            
            fig4 = visualizer.plot_returns_distribution()
            fig4.savefig(strategy_viz_dir / "returns_distribution.png", dpi=300, bbox_inches='tight')
            plt.close(fig4)
            
            # Dashboard
            fig5 = visualizer.create_performance_dashboard()
            fig5.savefig(strategy_viz_dir / "performance_dashboard.png", dpi=300, bbox_inches='tight')
            plt.close(fig5)
    
    def run_full_optimization(self, constraints: Optional[PortfolioConstraints] = None,
                             rebalance_freq: int = 5, save_results: bool = True) -> Dict:
        """Run complete optimization pipeline"""
        print("=" * 60)
        print("PORTFOLIO OPTIMIZATION ENGINE")
        print("=" * 60)
        
        # Setup default strategies and risk models if not already done
        if not self.strategies:
            self.setup_default_strategies()
        
        if not self.risk_models:
            self.setup_default_risk_models()
        
        # Create default constraints if none provided
        if constraints is None:
            constraints = self.create_constraints(max_single_weight=0.15)
        
        # Optimize all strategies
        self.optimize_all_strategies(constraints=constraints, rebalance_freq=rebalance_freq)
        
        # Optimize risk models
        self.risk_model_results = self.optimize_risk_models(constraints)
        
        # Create comparison
        if len(self.backtest_results) > 1:
            comparison = self.compare_strategies()
            comparison_df = comparison.compare_performance()
            
            print("\n" + "=" * 60)
            print("STRATEGY COMPARISON")
            print("=" * 60)
            print(comparison_df.round(4))
        
        # Save results
        if save_results:
            self.save_results(include_visualizations=True)
        
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        
        return {
            'strategies': self.backtest_results,
            'risk_models': self.risk_model_results,
            'comparison': comparison_df if len(self.backtest_results) > 1 else None
        }


def main():
    """Main execution function"""
    # Initialize optimizer
    optimizer = PortfolioOptimizer(data_dir="data", output_dir="results")
    
    # Create constraints
    constraints = optimizer.create_constraints(
        max_single_weight=0.15,  # Max 15% in any single stock
        max_sector_weight={
            'Banking': 0.3,     # Max 30% in banking
            'IT': 0.25,         # Max 25% in IT
            'Energy': 0.2       # Max 20% in energy
        }
    )
    
    # Run full optimization
    results = optimizer.run_full_optimization(
        constraints=constraints,
        rebalance_freq=5,  # Rebalance every 5 days
        save_results=True
    )
    
    return optimizer, results


if __name__ == "__main__":
    optimizer, results = main()