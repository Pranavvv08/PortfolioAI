"""
Risk models for portfolio optimization
"""

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
from typing import Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


class RiskModel:
    """Base class for risk models"""
    
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns
        self.n_assets = len(returns.columns)
        self.asset_names = list(returns.columns)
        
    def calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """Calculate portfolio risk given weights"""
        raise NotImplementedError
    
    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions of each asset"""
        raise NotImplementedError


class EqualWeightModel(RiskModel):
    """Equal weight (1/N) portfolio model"""
    
    def optimize_weights(self, constraints=None) -> np.ndarray:
        """Return equal weights for all assets"""
        return np.ones(self.n_assets) / self.n_assets
    
    def calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """Portfolio volatility under equal weighting"""
        cov_matrix = self.returns.cov().values
        return np.sqrt(weights.T @ cov_matrix @ weights)
    
    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Risk contributions under equal weighting"""
        cov_matrix = self.returns.cov().values
        portfolio_vol = self.calculate_portfolio_risk(weights)
        marginal_contrib = cov_matrix @ weights
        return weights * marginal_contrib / portfolio_vol


class MinimumVarianceModel(RiskModel):
    """Minimum variance portfolio optimization"""
    
    def __init__(self, returns: pd.DataFrame, shrinkage: bool = True):
        super().__init__(returns)
        self.shrinkage = shrinkage
        self.cov_matrix = self._estimate_covariance()
    
    def _estimate_covariance(self) -> np.ndarray:
        """Estimate covariance matrix with optional shrinkage"""
        if self.shrinkage:
            lw = LedoitWolf()
            return lw.fit(self.returns.fillna(0)).covariance_
        else:
            return self.returns.cov().values
    
    def optimize_weights(self, constraints=None) -> np.ndarray:
        """Optimize for minimum variance portfolio"""
        # Objective function: minimize portfolio variance
        def objective(weights):
            return weights.T @ self.cov_matrix @ weights
        
        # Default constraints
        if constraints is None:
            from .constraints import create_default_constraints
            constraint_obj = create_default_constraints(self.n_assets)
            constraints_list = constraint_obj.get_constraints()
            bounds = constraint_obj.get_bounds()
        else:
            constraints_list = constraints.get_constraints()
            bounds = constraints.get_bounds()
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints_list
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
            
        return result.x
    
    def calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(weights.T @ self.cov_matrix @ weights)
    
    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions"""
        portfolio_vol = self.calculate_portfolio_risk(weights)
        marginal_contrib = self.cov_matrix @ weights
        return weights * marginal_contrib / portfolio_vol


class MaximumSharpeModel(RiskModel):
    """Maximum Sharpe ratio portfolio optimization"""
    
    def __init__(self, returns: pd.DataFrame, risk_free_rate: float = 0.0, 
                 shrinkage: bool = True):
        super().__init__(returns)
        self.risk_free_rate = risk_free_rate
        self.shrinkage = shrinkage
        self.expected_returns = self._estimate_expected_returns()
        self.cov_matrix = self._estimate_covariance()
    
    def _estimate_expected_returns(self) -> np.ndarray:
        """Estimate expected returns (simple historical mean)"""
        return self.returns.mean().values
    
    def _estimate_covariance(self) -> np.ndarray:
        """Estimate covariance matrix with optional shrinkage"""
        if self.shrinkage:
            lw = LedoitWolf()
            return lw.fit(self.returns.fillna(0)).covariance_
        else:
            return self.returns.cov().values
    
    def optimize_weights(self, constraints=None) -> np.ndarray:
        """Optimize for maximum Sharpe ratio portfolio"""
        # Convert to minimization problem by maximizing negative Sharpe ratio
        def objective(weights):
            portfolio_return = weights.T @ self.expected_returns
            portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
            
            if portfolio_vol < 1e-10:  # Avoid division by zero
                return 1e10
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe_ratio  # Minimize negative Sharpe
        
        # Default constraints
        if constraints is None:
            from .constraints import create_default_constraints
            constraint_obj = create_default_constraints(self.n_assets)
            constraints_list = constraint_obj.get_constraints()
            bounds = constraint_obj.get_bounds()
        else:
            constraints_list = constraints.get_constraints()
            bounds = constraints.get_bounds()
        
        # Initial guess
        x0 = np.ones(self.n_assets) / self.n_assets
        
        # Optimize
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints_list
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
            
        return result.x
    
    def calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(weights.T @ self.cov_matrix @ weights)
    
    def calculate_portfolio_return(self, weights: np.ndarray) -> float:
        """Calculate expected portfolio return"""
        return weights.T @ self.expected_returns
    
    def calculate_sharpe_ratio(self, weights: np.ndarray) -> float:
        """Calculate portfolio Sharpe ratio"""
        portfolio_return = self.calculate_portfolio_return(weights)
        portfolio_vol = self.calculate_portfolio_risk(weights)
        
        if portfolio_vol < 1e-10:
            return 0.0
        
        return (portfolio_return - self.risk_free_rate) / portfolio_vol
    
    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions"""
        portfolio_vol = self.calculate_portfolio_risk(weights)
        marginal_contrib = self.cov_matrix @ weights
        return weights * marginal_contrib / portfolio_vol


class RiskBudgetingModel(RiskModel):
    """Risk budgeting (Risk Parity) portfolio optimization"""
    
    def __init__(self, returns: pd.DataFrame, target_risk_budgets: Optional[np.ndarray] = None,
                 shrinkage: bool = True):
        super().__init__(returns)
        self.shrinkage = shrinkage
        self.cov_matrix = self._estimate_covariance()
        
        # Default to equal risk budgets if not specified
        if target_risk_budgets is None:
            self.target_risk_budgets = np.ones(self.n_assets) / self.n_assets
        else:
            self.target_risk_budgets = target_risk_budgets
    
    def _estimate_covariance(self) -> np.ndarray:
        """Estimate covariance matrix with optional shrinkage"""
        if self.shrinkage:
            lw = LedoitWolf()
            return lw.fit(self.returns.fillna(0)).covariance_
        else:
            return self.returns.cov().values
    
    def calculate_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calculate risk contributions"""
        portfolio_vol = np.sqrt(weights.T @ self.cov_matrix @ weights)
        marginal_contrib = self.cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / portfolio_vol
        return risk_contrib / np.sum(risk_contrib)  # Normalize to sum to 1
    
    def optimize_weights(self, constraints=None) -> np.ndarray:
        """Optimize for risk budgeting portfolio"""
        # Objective function: minimize sum of squared deviations from target risk budgets
        def objective(weights):
            risk_contrib = self.calculate_risk_contributions(weights)
            return np.sum((risk_contrib - self.target_risk_budgets) ** 2)
        
        # Default constraints
        if constraints is None:
            from .constraints import create_default_constraints
            constraint_obj = create_default_constraints(self.n_assets)
            constraints_list = constraint_obj.get_constraints()
            bounds = constraint_obj.get_bounds()
        else:
            constraints_list = constraints.get_constraints()
            bounds = constraints.get_bounds()
        
        # Use inverse volatility as initial guess for risk parity
        vol_diag = np.sqrt(np.diag(self.cov_matrix))
        x0 = (1 / vol_diag) / np.sum(1 / vol_diag)
        
        # Optimize
        result = optimize.minimize(
            objective, x0, method='SLSQP',
            bounds=bounds, constraints=constraints_list
        )
        
        if not result.success:
            print(f"Optimization warning: {result.message}")
            
        return result.x
    
    def calculate_portfolio_risk(self, weights: np.ndarray) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(weights.T @ self.cov_matrix @ weights)


class RiskMetrics:
    """Calculate various risk metrics for portfolios"""
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk (VaR)"""
        return np.percentile(returns, confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR/Expected Shortfall)"""
        var = RiskMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_max_drawdown(cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        return drawdown.min()
    
    @staticmethod
    def calculate_downside_deviation(returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target_return] - target_return
        return np.sqrt(np.mean(downside_returns ** 2))
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, target_return: float = 0.0,
                              risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio"""
        excess_return = returns.mean() - risk_free_rate
        downside_dev = RiskMetrics.calculate_downside_deviation(returns, target_return)
        
        if downside_dev < 1e-10:
            return 0.0
            
        return excess_return / downside_dev
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = returns.mean() * 252  # Assuming daily returns
        cumulative_returns = (1 + returns).cumprod()
        max_dd = RiskMetrics.calculate_max_drawdown(cumulative_returns)
        
        if abs(max_dd) < 1e-10:
            return 0.0
            
        return annual_return / abs(max_dd)
    
    @staticmethod
    def calculate_portfolio_metrics(weights: np.ndarray, returns: pd.DataFrame, 
                                  risk_free_rate: float = 0.0) -> dict:
        """Calculate comprehensive portfolio risk metrics"""
        portfolio_returns = (returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        metrics = {
            'annualized_return': portfolio_returns.mean() * 252,
            'annualized_volatility': portfolio_returns.std() * np.sqrt(252),
            'sharpe_ratio': (portfolio_returns.mean() - risk_free_rate) / portfolio_returns.std(),
            'var_5': RiskMetrics.calculate_var(portfolio_returns, 0.05),
            'cvar_5': RiskMetrics.calculate_cvar(portfolio_returns, 0.05),
            'max_drawdown': RiskMetrics.calculate_max_drawdown(cumulative_returns),
            'sortino_ratio': RiskMetrics.calculate_sortino_ratio(portfolio_returns, 0.0, risk_free_rate),
            'calmar_ratio': RiskMetrics.calculate_calmar_ratio(portfolio_returns),
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis(),
            'positive_periods': (portfolio_returns > 0).sum() / len(portfolio_returns)
        }
        
        return metrics