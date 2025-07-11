"""
Portfolio constraints utilities
"""

import numpy as np
from scipy import optimize
from typing import Dict, List, Optional, Tuple, Union


class PortfolioConstraints:
    """Portfolio constraints for optimization"""
    
    def __init__(self, n_assets: int):
        self.n_assets = n_assets
        self.constraints = []
        self.bounds = [(0, 1) for _ in range(n_assets)]  # Default: long-only
        
    def add_sum_constraint(self, target_sum: float = 1.0):
        """Add constraint that weights sum to target value"""
        constraint = {
            'type': 'eq',
            'fun': lambda w: np.sum(w) - target_sum
        }
        self.constraints.append(constraint)
        return self
    
    def add_long_only_constraint(self):
        """Add long-only constraint (no short selling)"""
        self.bounds = [(0, 1) for _ in range(self.n_assets)]
        return self
    
    def add_weight_bounds(self, min_weight: float = 0.0, max_weight: float = 1.0):
        """Add individual weight bounds"""
        self.bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        return self
    
    def add_sector_constraints(self, sector_mapping: Dict[str, List[int]], 
                             sector_limits: Dict[str, Tuple[float, float]]):
        """Add sector exposure constraints
        
        Args:
            sector_mapping: Dict mapping sector names to list of asset indices
            sector_limits: Dict mapping sector names to (min_exposure, max_exposure)
        """
        for sector, asset_indices in sector_mapping.items():
            if sector in sector_limits:
                min_exp, max_exp = sector_limits[sector]
                
                # Minimum exposure constraint
                if min_exp > 0:
                    constraint = {
                        'type': 'ineq',
                        'fun': lambda w, indices=asset_indices: np.sum(w[indices]) - min_exp
                    }
                    self.constraints.append(constraint)
                
                # Maximum exposure constraint
                if max_exp < 1:
                    constraint = {
                        'type': 'ineq',
                        'fun': lambda w, indices=asset_indices: max_exp - np.sum(w[indices])
                    }
                    self.constraints.append(constraint)
        return self
    
    def add_diversification_constraint(self, max_single_weight: float = 0.1):
        """Add constraint limiting maximum individual weight"""
        self.bounds = [(0, max_single_weight) for _ in range(self.n_assets)]
        return self
    
    def add_turnover_constraint(self, current_weights: np.ndarray, max_turnover: float):
        """Add constraint limiting portfolio turnover"""
        constraint = {
            'type': 'ineq',
            'fun': lambda w: max_turnover - np.sum(np.abs(w - current_weights))
        }
        self.constraints.append(constraint)
        return self
    
    def add_tracking_error_constraint(self, benchmark_weights: np.ndarray, 
                                    cov_matrix: np.ndarray, max_te: float):
        """Add tracking error constraint relative to benchmark"""
        def tracking_error(w):
            active_weights = w - benchmark_weights
            te = np.sqrt(active_weights.T @ cov_matrix @ active_weights)
            return max_te - te
        
        constraint = {
            'type': 'ineq',
            'fun': tracking_error
        }
        self.constraints.append(constraint)
        return self
    
    def add_risk_budget_constraint(self, risk_budgets: np.ndarray, cov_matrix: np.ndarray, 
                                 tolerance: float = 0.01):
        """Add risk budgeting constraints
        
        Args:
            risk_budgets: Target risk contribution for each asset
            cov_matrix: Covariance matrix
            tolerance: Tolerance for risk budget deviations
        """
        def risk_contributions(w):
            portfolio_vol = np.sqrt(w.T @ cov_matrix @ w)
            marginal_contrib = cov_matrix @ w
            risk_contrib = w * marginal_contrib / portfolio_vol
            return risk_contrib / np.sum(risk_contrib)
        
        # Add constraints for each asset's risk contribution
        for i, target_contrib in enumerate(risk_budgets):
            constraint = {
                'type': 'ineq',
                'fun': lambda w, idx=i, target=target_contrib: 
                       tolerance - abs(risk_contributions(w)[idx] - target)
            }
            self.constraints.append(constraint)
        return self
    
    def add_minimum_positions(self, min_positions: int):
        """Add constraint for minimum number of positions"""
        constraint = {
            'type': 'ineq',
            'fun': lambda w: np.sum(w > 1e-6) - min_positions
        }
        self.constraints.append(constraint)
        return self
    
    def add_maximum_positions(self, max_positions: int):
        """Add constraint for maximum number of positions"""
        constraint = {
            'type': 'ineq',
            'fun': lambda w: max_positions - np.sum(w > 1e-6)
        }
        self.constraints.append(constraint)
        return self
    
    def add_liquidity_constraint(self, liquidity_scores: np.ndarray, min_avg_liquidity: float):
        """Add constraint for minimum average portfolio liquidity"""
        constraint = {
            'type': 'ineq',
            'fun': lambda w: np.sum(w * liquidity_scores) - min_avg_liquidity
        }
        self.constraints.append(constraint)
        return self
    
    def get_constraints(self):
        """Get all constraints for scipy optimization"""
        return self.constraints
    
    def get_bounds(self):
        """Get weight bounds for scipy optimization"""
        return self.bounds


class TransactionCosts:
    """Transaction cost modeling"""
    
    def __init__(self, fixed_cost: float = 0.0, proportional_cost: float = 0.001):
        self.fixed_cost = fixed_cost
        self.proportional_cost = proportional_cost
    
    def calculate_costs(self, current_weights: np.ndarray, target_weights: np.ndarray,
                       portfolio_value: float = 1.0) -> float:
        """Calculate transaction costs for rebalancing"""
        trades = np.abs(target_weights - current_weights)
        
        # Fixed costs for each non-zero trade
        fixed_costs = np.sum(trades > 1e-6) * self.fixed_cost
        
        # Proportional costs
        proportional_costs = np.sum(trades) * self.proportional_cost * portfolio_value
        
        return fixed_costs + proportional_costs
    
    def add_cost_penalty(self, objective_func, current_weights: np.ndarray, 
                        cost_penalty: float = 1.0):
        """Add transaction cost penalty to objective function"""
        def penalized_objective(weights):
            original_obj = objective_func(weights)
            cost_penalty_term = cost_penalty * self.calculate_costs(current_weights, weights)
            return original_obj + cost_penalty_term
        
        return penalized_objective


def create_default_constraints(n_assets: int, **kwargs) -> PortfolioConstraints:
    """Create default portfolio constraints"""
    constraints = PortfolioConstraints(n_assets)
    
    # Add sum constraint (weights sum to 1)
    constraints.add_sum_constraint(1.0)
    
    # Add long-only constraint
    constraints.add_long_only_constraint()
    
    # Add diversification constraint if specified
    max_weight = kwargs.get('max_single_weight', 0.2)
    if max_weight < 1.0:
        constraints.add_diversification_constraint(max_weight)
    
    return constraints


def create_sector_mapping(stock_symbols: List[str]) -> Dict[str, List[int]]:
    """Create sector mapping for Indian NSE stocks (simplified)"""
    # Simplified sector classification for NSE stocks
    sectors = {
        'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 
                   'AXISBANK.NS', 'INDUSINDBK.NS'],
        'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'WIPRO.NS', 'TECHM.NS'],
        'Energy': ['RELIANCE.NS', 'ONGC.NS', 'BPCL.NS', 'IOC.NS', 'COALINDIA.NS'],
        'FMCG': ['ITC.NS', 'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'TATACONSUM.NS'],
        'Auto': ['MARUTI.NS', 'TATAMOTORS.NS', 'HEROMOTOCO.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'M&M.NS'],
        'Pharma': ['SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS'],
        'Metals': ['TATASTEEL.NS', 'JSWSTEEL.NS', 'HINDALCO.NS'],
        'Cement': ['ULTRACEMCO.NS', 'GRASIM.NS'],
        'Telecom': ['BHARTIARTL.NS'],
        'Others': []  # Will include remaining stocks
    }
    
    # Create index mapping
    sector_mapping = {}
    classified_stocks = set()
    
    for sector, sector_stocks in sectors.items():
        indices = []
        for stock in sector_stocks:
            if stock in stock_symbols:
                indices.append(stock_symbols.index(stock))
                classified_stocks.add(stock)
        if indices:
            sector_mapping[sector] = indices
    
    # Add unclassified stocks to 'Others'
    other_indices = []
    for i, stock in enumerate(stock_symbols):
        if stock not in classified_stocks:
            other_indices.append(i)
    
    if other_indices:
        sector_mapping['Others'] = other_indices
    
    return sector_mapping