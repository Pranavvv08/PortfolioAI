#!/usr/bin/env python3
"""
Create simple portfolio visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the portfolio weights
weights_df = pd.read_csv('../results/portfolio_weights.csv', index_col=0)

print("Creating portfolio visualizations...")

# Set up the plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create a figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Portfolio Optimization Results', fontsize=16, fontweight='bold')

# 1. Top Holdings Comparison (Top Left)
# Show top 10 holdings for selected strategies
selected_strategies = ['Risk_Model_Maximum Sharpe', 'Momentum', 'Risk Parity', 'Equal Weight']
top_n = 10

ax1 = axes[0, 0]
strategy_colors = sns.color_palette("husl", len(selected_strategies))

for i, strategy in enumerate(selected_strategies):
    if strategy in weights_df.columns:
        strategy_weights = weights_df[strategy].sort_values(ascending=False).head(top_n)
        x_pos = np.arange(len(strategy_weights)) + i * 0.2
        ax1.bar(x_pos, strategy_weights.values, width=0.18, 
               label=strategy.replace('Risk_Model_', ''), alpha=0.8, color=strategy_colors[i])

ax1.set_title(f'Top {top_n} Holdings by Strategy')
ax1.set_xlabel('Stocks')
ax1.set_ylabel('Weight')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.tick_params(axis='x', rotation=45)

# 2. Portfolio Concentration (Top Right)
ax2 = axes[0, 1]
concentrations = []
strategy_names = []

for strategy in weights_df.columns:
    weights = weights_df[strategy]
    # Calculate Herfindahl index (concentration measure)
    hhi = np.sum(weights ** 2)
    concentrations.append(hhi)
    strategy_names.append(strategy.replace('Risk_Model_', ''))

bars = ax2.bar(range(len(concentrations)), concentrations, color=sns.color_palette("viridis", len(concentrations)))
ax2.set_title('Portfolio Concentration (Herfindahl Index)')
ax2.set_xlabel('Strategy')
ax2.set_ylabel('Concentration Index')
ax2.set_xticks(range(len(strategy_names)))
ax2.set_xticklabels(strategy_names, rotation=45, ha='right')

# Add value labels on bars
for bar, conc in zip(bars, concentrations):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.001,
             f'{conc:.3f}', ha='center', va='bottom', fontsize=8)

# 3. Sector Allocation Heatmap (Bottom Left)
# Create simplified sector mapping
sector_mapping = {
    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS', 'INDUSINDBK.NS'],
    'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'],
    'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'COALINDIA.NS'],
    'FMCG': ['ITC.NS', 'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS'],
    'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'M&M.NS']
}

# Calculate sector weights
sector_weights = {}
for strategy in weights_df.columns:
    sector_weights[strategy] = {}
    for sector, stocks in sector_mapping.items():
        sector_weight = weights_df.loc[weights_df.index.isin(stocks), strategy].sum()
        sector_weights[strategy][sector] = sector_weight

sector_df = pd.DataFrame(sector_weights).T
sector_df = sector_df.fillna(0)

ax3 = axes[1, 0]
im = ax3.imshow(sector_df.values, cmap='YlOrRd', aspect='auto', interpolation='nearest')
ax3.set_title('Sector Allocation Heatmap')
ax3.set_xticks(range(len(sector_df.columns)))
ax3.set_xticklabels(sector_df.columns)
ax3.set_yticks(range(len(sector_df.index)))
ax3.set_yticklabels([name.replace('Risk_Model_', '') for name in sector_df.index])

# Add colorbar
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Sector Weight')

# 4. Risk-Return Scatter (Bottom Right)
# Load performance metrics
import json
with open('../results/performance_report.json', 'r') as f:
    perf_data = json.load(f)

ax4 = axes[1, 1]
returns = []
volatilities = []
names = []
colors = []

color_palette = sns.color_palette("Set2", len(perf_data['optimization_results']))

for i, (name, metrics) in enumerate(perf_data['optimization_results'].items()):
    returns.append(metrics['annualized_return'])
    volatilities.append(metrics['annualized_volatility'])
    names.append(name.replace('Risk_Model_', ''))
    colors.append(color_palette[i])

scatter = ax4.scatter(volatilities, returns, c=colors, s=100, alpha=0.7)
ax4.set_title('Risk-Return Profile')
ax4.set_xlabel('Annualized Volatility')
ax4.set_ylabel('Annualized Return')

# Format axes as percentage
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

# Add strategy labels
for i, name in enumerate(names):
    ax4.annotate(name, (volatilities[i], returns[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('../results/portfolio_analysis.png', dpi=300, bbox_inches='tight')
print("Portfolio analysis chart saved to '../results/portfolio_analysis.png'")

# Create a simple summary table
print("\nCreating summary table...")

# Portfolio weights summary
print("\nTop 5 Holdings by Strategy:")
print("=" * 80)

for strategy in ['Risk_Model_Maximum Sharpe', 'Momentum', 'Risk Parity']:
    if strategy in weights_df.columns:
        top_holdings = weights_df[strategy].sort_values(ascending=False).head(5)
        print(f"\n{strategy}:")
        for stock, weight in top_holdings.items():
            print(f"  {stock:15s}: {weight:6.1%}")

print(f"\nVisualization files created in '../results/':")
print(f"  • portfolio_analysis.png - Comprehensive portfolio analysis charts")

plt.show()