"""
Visualization utilities for portfolio analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PortfolioVisualizer:
    """Portfolio visualization utilities"""
    
    def __init__(self, results: Dict, figsize: Tuple[int, int] = (12, 8)):
        self.results = results
        self.figsize = figsize
        
    def plot_cumulative_returns(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot cumulative returns vs benchmark"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Portfolio cumulative returns
        portfolio_cum = (1 + self.results['portfolio_returns']).cumprod()
        benchmark_cum = (1 + self.results['benchmark_returns']).cumprod()
        
        ax.plot(portfolio_cum.index, portfolio_cum.values, 
                label='Portfolio', linewidth=2, color='navy')
        ax.plot(benchmark_cum.index, benchmark_cum.values, 
                label='Benchmark (Equal Weight)', linewidth=2, 
                color='red', linestyle='--', alpha=0.7)
        
        ax.set_title('Cumulative Returns: Portfolio vs Benchmark', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Return', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add final values as text
        final_portfolio = portfolio_cum.iloc[-1]
        final_benchmark = benchmark_cum.iloc[-1]
        
        ax.text(0.02, 0.98, f'Portfolio Final: {final_portfolio:.3f}\nBenchmark Final: {final_benchmark:.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_drawdown(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot drawdown analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]+2))
        
        # Calculate drawdowns
        portfolio_cum = (1 + self.results['portfolio_returns']).cumprod()
        portfolio_rolling_max = portfolio_cum.expanding().max()
        portfolio_dd = (portfolio_cum - portfolio_rolling_max) / portfolio_rolling_max
        
        benchmark_cum = (1 + self.results['benchmark_returns']).cumprod()
        benchmark_rolling_max = benchmark_cum.expanding().max()
        benchmark_dd = (benchmark_cum - benchmark_rolling_max) / benchmark_rolling_max
        
        # Top plot: Cumulative returns with peaks
        ax1.plot(portfolio_cum.index, portfolio_cum.values, label='Portfolio', color='navy')
        ax1.plot(portfolio_rolling_max.index, portfolio_rolling_max.values, 
                label='Portfolio Peaks', color='darkgreen', alpha=0.7)
        ax1.fill_between(portfolio_cum.index, portfolio_cum.values, portfolio_rolling_max.values,
                        alpha=0.3, color='red', label='Drawdown')
        
        ax1.set_title('Portfolio Value and Drawdown Periods', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Drawdown percentages
        ax2.fill_between(portfolio_dd.index, portfolio_dd.values, 0, 
                        alpha=0.7, color='red', label='Portfolio DD')
        ax2.fill_between(benchmark_dd.index, benchmark_dd.values, 0, 
                        alpha=0.5, color='orange', label='Benchmark DD')
        
        ax2.set_title('Drawdown Comparison', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_rolling_metrics(self, window: int = 60, save_path: Optional[str] = None) -> plt.Figure:
        """Plot rolling performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        portfolio_returns = self.results['portfolio_returns']
        benchmark_returns = self.results['benchmark_returns']
        
        # Rolling Sharpe Ratio
        portfolio_rolling_sharpe = (portfolio_returns.rolling(window).mean() / 
                                   portfolio_returns.rolling(window).std() * np.sqrt(252))
        benchmark_rolling_sharpe = (benchmark_returns.rolling(window).mean() / 
                                   benchmark_returns.rolling(window).std() * np.sqrt(252))
        
        axes[0, 0].plot(portfolio_rolling_sharpe.index, portfolio_rolling_sharpe.values, 
                       label='Portfolio', color='navy')
        axes[0, 0].plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe.values, 
                       label='Benchmark', color='red', alpha=0.7)
        axes[0, 0].set_title(f'Rolling {window}-Day Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Rolling Volatility
        portfolio_rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
        benchmark_rolling_vol = benchmark_returns.rolling(window).std() * np.sqrt(252)
        
        axes[0, 1].plot(portfolio_rolling_vol.index, portfolio_rolling_vol.values, 
                       label='Portfolio', color='navy')
        axes[0, 1].plot(benchmark_rolling_vol.index, benchmark_rolling_vol.values, 
                       label='Benchmark', color='red', alpha=0.7)
        axes[0, 1].set_title(f'Rolling {window}-Day Volatility')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rolling Beta (if benchmark available)
        if len(portfolio_returns) > window:
            rolling_corr = portfolio_returns.rolling(window).corr(benchmark_returns)
            portfolio_rolling_std = portfolio_returns.rolling(window).std()
            benchmark_rolling_std = benchmark_returns.rolling(window).std()
            rolling_beta = rolling_corr * (portfolio_rolling_std / benchmark_rolling_std)
            
            axes[1, 0].plot(rolling_beta.index, rolling_beta.values, color='purple')
            axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Beta = 1')
            axes[1, 0].set_title(f'Rolling {window}-Day Beta')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling Alpha
        rolling_returns_p = portfolio_returns.rolling(window).mean() * 252
        rolling_returns_b = benchmark_returns.rolling(window).mean() * 252
        rolling_alpha = rolling_returns_p - rolling_returns_b
        
        axes[1, 1].plot(rolling_alpha.index, rolling_alpha.values, color='green')
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Alpha = 0')
        axes[1, 1].set_title(f'Rolling {window}-Day Alpha')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_returns_distribution(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot returns distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        portfolio_returns = self.results['portfolio_returns']
        benchmark_returns = self.results['benchmark_returns']
        
        # Histogram
        axes[0, 0].hist(portfolio_returns, bins=50, alpha=0.7, label='Portfolio', color='navy')
        axes[0, 0].hist(benchmark_returns, bins=50, alpha=0.5, label='Benchmark', color='red')
        axes[0, 0].set_title('Returns Distribution')
        axes[0, 0].set_xlabel('Daily Returns')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(portfolio_returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Portfolio vs Normal)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box Plot
        box_data = [portfolio_returns.values, benchmark_returns.values]
        axes[1, 0].boxplot(box_data, labels=['Portfolio', 'Benchmark'])
        axes[1, 0].set_title('Returns Box Plot')
        axes[1, 0].set_ylabel('Daily Returns')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Scatter Plot
        axes[1, 1].scatter(benchmark_returns, portfolio_returns, alpha=0.6, color='purple')
        
        # Add regression line
        slope, intercept, r_value, p_value, std_err = stats.linregress(benchmark_returns, portfolio_returns)
        line = slope * benchmark_returns + intercept
        axes[1, 1].plot(benchmark_returns, line, 'r-', alpha=0.8, 
                       label=f'R² = {r_value**2:.3f}')
        
        axes[1, 1].set_title('Portfolio vs Benchmark Returns')
        axes[1, 1].set_xlabel('Benchmark Returns')
        axes[1, 1].set_ylabel('Portfolio Returns')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_portfolio_weights(self, save_path: Optional[str] = None, top_n: int = 10) -> plt.Figure:
        """Plot portfolio weights over time"""
        weights_df = self.results['weights_history']
        
        if weights_df.empty:
            print("No weights history available")
            return None
        
        # Expand weights into separate columns
        weights_expanded = pd.DataFrame(
            weights_df['weights'].tolist(),
            index=weights_df['date']
        )
        
        # Get stock names (assuming they're available)
        if hasattr(self, 'stock_names'):
            weights_expanded.columns = self.stock_names
        else:
            weights_expanded.columns = [f'Stock_{i}' for i in range(len(weights_expanded.columns))]
        
        # Select top N stocks by average weight
        avg_weights = weights_expanded.mean().sort_values(ascending=False)
        top_stocks = avg_weights.head(top_n).index
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1]+2))
        
        # Stacked area plot for top stocks
        weights_top = weights_expanded[top_stocks]
        weights_top.plot.area(ax=ax1, alpha=0.7)
        
        ax1.set_title(f'Portfolio Weights Over Time (Top {top_n} Holdings)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Weight')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # Heatmap of all weights
        im = ax2.imshow(weights_expanded.T.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        ax2.set_title('All Portfolio Weights Heatmap')
        ax2.set_xlabel('Rebalancing Period')
        ax2.set_ylabel('Assets')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_risk_return_scatter(self, other_strategies: Optional[Dict] = None, 
                                save_path: Optional[str] = None) -> plt.Figure:
        """Plot risk-return scatter plot"""
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Calculate portfolio metrics
        portfolio_returns = self.results['portfolio_returns']
        portfolio_annual_return = portfolio_returns.mean() * 252
        portfolio_annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        benchmark_returns = self.results['benchmark_returns']
        benchmark_annual_return = benchmark_returns.mean() * 252
        benchmark_annual_vol = benchmark_returns.std() * np.sqrt(252)
        
        # Plot main portfolio
        ax.scatter(portfolio_annual_vol, portfolio_annual_return, 
                  s=100, color='navy', label='Portfolio', marker='o')
        
        # Plot benchmark
        ax.scatter(benchmark_annual_vol, benchmark_annual_return, 
                  s=100, color='red', label='Benchmark', marker='s')
        
        # Plot other strategies if provided
        if other_strategies:
            colors = sns.color_palette("husl", len(other_strategies))
            for i, (name, results) in enumerate(other_strategies.items()):
                returns = results['portfolio_returns']
                annual_return = returns.mean() * 252
                annual_vol = returns.std() * np.sqrt(252)
                ax.scatter(annual_vol, annual_return, s=100, 
                          color=colors[i], label=name, marker='^')
        
        ax.set_title('Risk-Return Scatter Plot', fontsize=14, fontweight='bold')
        ax.set_xlabel('Annualized Volatility', fontsize=12)
        ax.set_ylabel('Annualized Return', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format axes as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_performance_dashboard(self, save_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive performance dashboard"""
        fig = plt.figure(figsize=(20, 15))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        portfolio_returns = self.results['portfolio_returns']
        benchmark_returns = self.results['benchmark_returns']
        
        # 1. Cumulative Returns (top left, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        portfolio_cum = (1 + portfolio_returns).cumprod()
        benchmark_cum = (1 + benchmark_returns).cumprod()
        
        ax1.plot(portfolio_cum.index, portfolio_cum.values, label='Portfolio', linewidth=2)
        ax1.plot(benchmark_cum.index, benchmark_cum.values, label='Benchmark', linewidth=2, alpha=0.7)
        ax1.set_title('Cumulative Returns', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns Distribution (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(portfolio_returns, bins=30, alpha=0.7, label='Portfolio')
        ax2.hist(benchmark_returns, bins=30, alpha=0.5, label='Benchmark')
        ax2.set_title('Returns Distribution')
        ax2.legend()
        
        # 3. Drawdown (middle left, spanning 2 columns)
        ax3 = fig.add_subplot(gs[1, :2])
        portfolio_cum = (1 + portfolio_returns).cumprod()
        portfolio_rolling_max = portfolio_cum.expanding().max()
        portfolio_dd = (portfolio_cum - portfolio_rolling_max) / portfolio_rolling_max
        
        ax3.fill_between(portfolio_dd.index, portfolio_dd.values, 0, alpha=0.7, color='red')
        ax3.set_title('Portfolio Drawdown')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 4. Rolling Sharpe (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        rolling_sharpe = (portfolio_returns.rolling(60).mean() / 
                         portfolio_returns.rolling(60).std() * np.sqrt(252))
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax4.set_title('60-Day Rolling Sharpe')
        ax4.grid(True, alpha=0.3)
        
        # 5. Portfolio Weights (bottom left)
        ax5 = fig.add_subplot(gs[2, 0])
        weights_df = self.results['weights_history']
        if not weights_df.empty:
            latest_weights = weights_df.iloc[-1]['weights']
            # Show top 10 weights
            top_indices = np.argsort(latest_weights)[-10:]
            ax5.barh(range(len(top_indices)), latest_weights[top_indices])
            ax5.set_title('Top 10 Current Holdings')
            ax5.set_xlabel('Weight')
        
        # 6. Monthly Returns Heatmap (bottom middle)
        ax6 = fig.add_subplot(gs[2, 1])
        monthly_returns = portfolio_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        if len(monthly_returns) > 1:
            monthly_matrix = monthly_returns.values.reshape(-1, 1)
            im = ax6.imshow(monthly_matrix.T, cmap='RdYlGn', aspect='auto')
            ax6.set_title('Monthly Returns')
            plt.colorbar(im, ax=ax6)
        
        # 7. Key Metrics Table (bottom right)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')
        
        # Calculate key metrics
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol
        max_dd = portfolio_dd.min()
        
        metrics_text = f"""
        Key Metrics:
        
        Annual Return: {annual_return:.2%}
        Annual Volatility: {annual_vol:.2%}
        Sharpe Ratio: {sharpe:.2f}
        Max Drawdown: {max_dd:.2%}
        
        Win Rate: {(portfolio_returns > 0).mean():.1%}
        Best Day: {portfolio_returns.max():.2%}
        Worst Day: {portfolio_returns.min():.2%}
        """
        
        ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def create_strategy_comparison_plots(comparison_results: Dict[str, Dict], 
                                   save_dir: Optional[str] = None) -> List[plt.Figure]:
    """Create comparison plots for multiple strategies"""
    figures = []
    
    # 1. Cumulative Returns Comparison
    fig1, ax = plt.subplots(figsize=(12, 8))
    
    for strategy_name, results in comparison_results.items():
        portfolio_cum = (1 + results['portfolio_returns']).cumprod()
        ax.plot(portfolio_cum.index, portfolio_cum.values, label=strategy_name, linewidth=2)
    
    ax.set_title('Strategy Comparison: Cumulative Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_dir:
        fig1.savefig(f"{save_dir}/strategy_comparison_cumulative.png", dpi=300, bbox_inches='tight')
    
    figures.append(fig1)
    
    # 2. Risk-Return Scatter
    fig2, ax = plt.subplots(figsize=(10, 8))
    
    colors = sns.color_palette("husl", len(comparison_results))
    
    for i, (strategy_name, results) in enumerate(comparison_results.items()):
        returns = results['portfolio_returns']
        annual_return = returns.mean() * 252
        annual_vol = returns.std() * np.sqrt(252)
        
        ax.scatter(annual_vol, annual_return, s=100, color=colors[i], 
                  label=strategy_name, marker='o')
    
    ax.set_title('Risk-Return Profile by Strategy', fontsize=14, fontweight='bold')
    ax.set_xlabel('Annualized Volatility')
    ax.set_ylabel('Annualized Return')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format as percentage
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
    
    if save_dir:
        fig2.savefig(f"{save_dir}/strategy_comparison_risk_return.png", dpi=300, bbox_inches='tight')
    
    figures.append(fig2)
    
    return figures