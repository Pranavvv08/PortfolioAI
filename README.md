# Portfolio Optimization Engine

A comprehensive portfolio optimization system that generates optimal portfolios using different strategies and risk models for Indian NSE stocks.

## Overview

This system implements multiple portfolio optimization approaches, risk models, and performance analytics to create and evaluate optimal portfolio allocations. It processes historical price data and statistical features to generate actionable investment strategies with proper risk management and performance tracking.

## Features

### Portfolio Optimization Strategies
- **Mean Reversion Strategy**: Using momentum indicators to identify oversold/overbought stocks
- **Momentum Strategy**: Following trending stocks with positive momentum  
- **Low Volatility Strategy**: Focus on stocks with lower risk profiles
- **Risk Parity Strategy**: Equal risk contribution from each asset
- **Sharpe Ratio Optimization**: Maximize risk-adjusted returns
- **Adaptive Strategy**: Combines multiple strategies based on recent performance

### Risk Models
- **Equal Weight**: Simple 1/N allocation
- **Minimum Variance**: Minimize portfolio volatility  
- **Maximum Sharpe**: Optimize risk-adjusted returns
- **Risk Budgeting**: Control individual stock risk contributions

### Key Capabilities
- Dynamic rebalancing based on statistical features
- Portfolio performance analytics and backtesting
- Risk metrics calculation (VaR, CVaR, drawdown)
- Sector diversification constraints
- Position size limits and turnover constraints
- Performance comparison against benchmarks
- Comprehensive visualization and reporting

## Data

The system uses NSE (National Stock Exchange of India) data for 50 major stocks:

- **Price Data**: `data/price_matrix.csv` - Daily price data for 50 NSE stocks
- **Statistical Features**: `data/statistical_features.csv` - Momentum, volatility, and Sharpe ratios
- **Period**: February 2025 to July 2025 (103 trading days)
- **Universe**: 50 major NSE stocks including RELIANCE, TCS, INFY, HDFCBANK, etc.

## File Structure

```
scripts/
├── portfolio_optimizer.py          # Main optimization engine
├── risk_models.py                  # Risk calculation functions  
├── performance_analytics.py        # Backtesting and metrics
├── portfolio_strategies.py         # Different strategy implementations
├── run_optimization.py            # Main execution script
├── create_visualizations.py       # Visualization generator
└── utils/
    ├── data_loader.py              # Load and preprocess data
    ├── constraints.py              # Portfolio constraints
    └── visualizations.py           # Charts and plots

results/
├── portfolio_weights.csv           # Optimal allocations
├── performance_report.json         # Key metrics and statistics
├── backtest_results.csv           # Historical performance
└── portfolio_analysis.png         # Visualization charts
```

## Installation

1. Install required dependencies:
```bash
pip install pandas numpy scipy matplotlib seaborn scikit-learn cvxpy
```

2. Ensure data files are in the `data/` directory:
   - `price_matrix.csv`
   - `statistical_features.csv`

## Usage

### Quick Start

Run the complete optimization pipeline:

```bash
cd scripts/
python run_optimization.py
```

This will:
1. Load and process the market data
2. Run all optimization strategies and risk models
3. Generate performance analytics and comparisons
4. Create output files in the `results/` directory

### Custom Optimization

```python
from portfolio_optimizer import PortfolioOptimizer

# Initialize optimizer
optimizer = PortfolioOptimizer(data_dir="data", output_dir="results")

# Setup strategies and risk models
optimizer.setup_default_strategies()
optimizer.setup_default_risk_models()

# Create constraints
constraints = optimizer.create_constraints(
    max_single_weight=0.15,  # Max 15% in any single stock
    max_sector_weight={
        'Banking': 0.3,     # Max 30% in banking
        'IT': 0.25,         # Max 25% in IT
        'Energy': 0.2       # Max 20% in energy
    }
)

# Run optimization
results = optimizer.run_full_optimization(constraints=constraints)
```

## Output Files

### portfolio_weights.csv
Contains optimal portfolio weights for each strategy:
- Rows: Individual stocks (50 NSE stocks)
- Columns: Different strategies and risk models
- Values: Portfolio weights (sum to 1.0 for each strategy)

### performance_report.json
Comprehensive performance metrics including:
- Annualized returns and volatility
- Sharpe ratios and risk metrics
- Maximum drawdown and VaR/CVaR
- Strategy rankings and comparisons

### backtest_results.csv
Historical backtest performance data:
- Date-by-date portfolio returns
- Cumulative performance tracking
- Benchmark comparisons

### portfolio_analysis.png
Visualization dashboard showing:
- Top holdings by strategy
- Portfolio concentration analysis
- Sector allocation heatmaps
- Risk-return scatter plots

## Performance Results

Based on the optimization run:

| Strategy | Sharpe Ratio | Annual Return | Annual Volatility |
|----------|-------------|---------------|-------------------|
| Maximum Sharpe Risk Model | 3.505 | 54.4% | 15.5% |
| Risk Parity | 1.600 | 23.0% | 15.1% |
| Momentum | 1.489 | 26.0% | 16.4% |
| Mean Reversion | 1.314 | 25.1% | 18.3% |
| Low Volatility | 1.257 | 19.7% | 15.3% |
| Equal Weight | 1.199 | 18.1% | 15.1% |

## Key Features

### Risk Management
- Position size limits (max 15% per stock)
- Sector concentration limits
- Long-only constraints
- Transaction cost modeling
- Liquidity constraints

### Performance Analytics
- Sharpe ratio, Sortino ratio, Calmar ratio
- Value at Risk (VaR) and Conditional VaR
- Maximum drawdown analysis
- Rolling performance metrics
- Benchmark-relative analysis

### Strategy Implementation
- **Mean Reversion**: Identifies oversold conditions using negative momentum signals
- **Momentum**: Follows positive trending stocks with momentum above threshold  
- **Low Volatility**: Minimizes portfolio risk using minimum variance optimization
- **Risk Parity**: Ensures equal risk contribution from each asset
- **Sharpe Optimization**: Maximizes risk-adjusted returns using historical data

## Technical Specifications

- **Optimization Engine**: scipy.optimize for portfolio optimization
- **Risk Models**: Ledoit-Wolf shrinkage covariance estimation
- **Rebalancing**: Configurable frequency (default: every 5 days)
- **Constraints**: Flexible constraint framework supporting multiple constraint types
- **Performance**: Comprehensive risk and return metrics
- **Visualization**: matplotlib and seaborn for charts and analysis

## Data Requirements

The system expects:
1. `price_matrix.csv`: Daily price data with dates as index and stock symbols as columns
2. `statistical_features.csv`: Statistical features with momentum, volatility, and Sharpe ratios

Feature naming convention: `{SYMBOL}_{metric}_{window}d`
- Metrics: `mom` (momentum), `vol` (volatility), `sharpe` (Sharpe ratio)  
- Windows: `5d`, `10d`, `20d` (5, 10, 20 day periods)

## Extensions

The framework is designed to be extensible:

- **New Strategies**: Implement `PortfolioStrategy` base class
- **Custom Risk Models**: Extend `RiskModel` base class  
- **Additional Constraints**: Use `PortfolioConstraints` framework
- **Alternative Data**: Modify `DataLoader` for new data sources
- **Custom Analytics**: Extend `PerformanceAnalytics` class

## License

This portfolio optimization system is designed for educational and research purposes. Always perform due diligence before making investment decisions.

## Contact

For questions or issues, please refer to the documentation within each module or review the example usage in the main execution scripts.