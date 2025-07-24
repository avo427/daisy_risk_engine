import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

class PortfolioRiskScorer:
    """
    Comprehensive portfolio risk scoring system for concentrated tech portfolios.
    
    Evaluates risk across multiple dimensions:
    1. Concentration Risk
    2. Volatility Risk
    3. Market Risk (Beta)
    4. Factor Risk
    5. Correlation Risk
    6. Stress Test Risk
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.risk_weights = {
            'concentration': 0.30,
            'volatility': 0.25,
            'market_risk': 0.15,
            'factor_risk': 0.15,
            'correlation': 0.10,
            'stress_test': 0.05
        }
        
        # Risk thresholds for scoring
        self.thresholds = {
            'concentration': {
                'low': 0.10,      # Largest position < 10%
                'medium': 0.15,   # Largest position < 15%
                'high': 0.20      # Largest position < 20%
            },
            'volatility': {
                'low': 0.20,      # Annual vol < 20%
                'medium': 0.30,   # Annual vol < 30%
                'high': 0.40      # Annual vol < 40%
            },
            'market_risk': {
                'low': 0.8,       # Beta < 0.8
                'medium': 1.2,    # Beta < 1.2
                'high': 1.5       # Beta < 1.5
            },
            'factor_risk': {
                'low': 0.3,       # Max factor exposure < 0.3
                'medium': 0.5,    # Max factor exposure < 0.5
                'high': 0.7       # Max factor exposure < 0.7
            },
            'correlation': {
                'low': 0.5,       # Max correlation < 0.5
                'medium': 0.7,    # Max correlation < 0.7
                'high': 0.8       # Max correlation < 0.8
            }
        }
    
    def calculate_concentration_risk(self, weights: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate concentration risk score and details"""
        if weights.empty:
            return 0.0, {}
        
        active_weights = weights[weights['Weight'] > 0]
        if len(active_weights) == 0:
            return 0.0, {}
        
        largest_position = active_weights['Weight'].max()
        top_3_concentration = active_weights['Weight'].nlargest(3).sum()
        top_5_concentration = active_weights['Weight'].nlargest(5).sum()
        herfindahl_index = (active_weights['Weight'] ** 2).sum()
        
        # Calculate risk score based on largest position
        if largest_position <= self.thresholds['concentration']['low']:
            concentration_score = 0.0
        elif largest_position <= self.thresholds['concentration']['medium']:
            concentration_score = 0.3
        elif largest_position <= self.thresholds['concentration']['high']:
            concentration_score = 0.7
        else:
            concentration_score = 1.0
        
        # Adjust for overall concentration
        if top_3_concentration > 0.60:
            concentration_score = min(1.0, concentration_score + 0.2)
        if top_5_concentration > 0.80:
            concentration_score = min(1.0, concentration_score + 0.1)
        
        details = {
            'largest_position': largest_position,
            'top_3_concentration': top_3_concentration,
            'top_5_concentration': top_5_concentration,
            'herfindahl_index': herfindahl_index,
            'num_positions': len(active_weights)
        }
        
        return concentration_score, details
    
    def calculate_volatility_risk(self, realized_metrics: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate volatility risk score and details"""
        if realized_metrics.empty:
            return 0.0, {}
        
        portfolio_metrics = realized_metrics[realized_metrics['Ticker'] == 'PORTFOLIO']
        if portfolio_metrics.empty:
            return 0.0, {}
        
        portfolio = portfolio_metrics.iloc[0]
        
        # Parse volatility
        try:
            vol_str = str(portfolio.get('Ann. Volatility', ''))
            if '%' in vol_str:
                volatility = float(vol_str.replace('%', '')) / 100
            else:
                volatility = float(vol_str) if vol_str else 0.0
        except:
            volatility = 0.0
        
        # Calculate risk score
        if volatility <= self.thresholds['volatility']['low']:
            vol_score = 0.0
        elif volatility <= self.thresholds['volatility']['medium']:
            vol_score = 0.3
        elif volatility <= self.thresholds['volatility']['high']:
            vol_score = 0.7
        else:
            vol_score = 1.0
        
        # Parse other metrics
        try:
            max_dd_str = str(portfolio.get('Max Drawdown', ''))
            max_drawdown = float(max_dd_str.replace('%', '')) / 100 if '%' in max_dd_str else float(max_dd_str) if max_dd_str else 0.0
        except:
            max_drawdown = 0.0
        
        try:
            sharpe = float(portfolio.get('Sharpe', 0))
        except:
            sharpe = 0.0
        
        details = {
            'annual_volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe
        }
        
        return vol_score, details
    
    def calculate_market_risk(self, realized_metrics: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate market risk (beta) score and details"""
        if realized_metrics.empty:
            return 0.0, {}
        
        portfolio_metrics = realized_metrics[realized_metrics['Ticker'] == 'PORTFOLIO']
        if portfolio_metrics.empty:
            return 0.0, {}
        
        portfolio = portfolio_metrics.iloc[0]
        
        # Parse beta values
        try:
            beta_ndx = float(portfolio.get('Beta (NDX)', 0))
        except:
            beta_ndx = 0.0
        
        try:
            beta_spx = float(portfolio.get('Beta (SPX)', 0))
        except:
            beta_spx = 0.0
        
        # Use NDX beta as primary, fallback to SPX
        beta = beta_ndx if beta_ndx != 0 else beta_spx
        
        # Calculate risk score
        if abs(beta) <= self.thresholds['market_risk']['low']:
            market_score = 0.0
        elif abs(beta) <= self.thresholds['market_risk']['medium']:
            market_score = 0.3
        elif abs(beta) <= self.thresholds['market_risk']['high']:
            market_score = 0.7
        else:
            market_score = 1.0
        
        details = {
            'beta_ndx': beta_ndx,
            'beta_spx': beta_spx,
            'primary_beta': beta
        }
        
        return market_score, details
    
    def calculate_factor_risk(self, factor_exposures: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate factor risk score and details"""
        if factor_exposures.empty:
            return 0.0, {}
        
        # Get latest factor exposures for portfolio
        if 'Date' in factor_exposures.columns and 'Ticker' in factor_exposures.columns:
            latest_date = factor_exposures['Date'].max()
            portfolio_factors = factor_exposures[
                (factor_exposures['Date'] == latest_date) & 
                (factor_exposures['Ticker'] == 'PORTFOLIO')
            ]
            
            if portfolio_factors.empty:
                return 0.0, {}
            
            # Calculate max factor exposure
            max_exposure = portfolio_factors['Beta'].abs().max()
            
            # Calculate risk score
            if max_exposure <= self.thresholds['factor_risk']['low']:
                factor_score = 0.0
            elif max_exposure <= self.thresholds['factor_risk']['medium']:
                factor_score = 0.3
            elif max_exposure <= self.thresholds['factor_risk']['high']:
                factor_score = 0.7
            else:
                factor_score = 1.0
            
            # Get top factor exposures
            top_factors = portfolio_factors.nlargest(3, 'Beta')['Factor'].tolist()
            
            details = {
                'max_factor_exposure': max_exposure,
                'top_factors': top_factors,
                'factor_count': len(portfolio_factors)
            }
            
            return factor_score, details
        
        return 0.0, {}
    
    def calculate_correlation_risk(self, correlation_matrix: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate correlation risk score and details"""
        if correlation_matrix.empty:
            return 0.0, {}
        
        # Find high correlation pairs
        high_corr_pairs = []
        max_correlation = 0.0
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                max_correlation = max(max_correlation, abs(corr_val))
                
                if abs(corr_val) > 0.7:
                    high_corr_pairs.append({
                        'ticker1': correlation_matrix.columns[i],
                        'ticker2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Calculate risk score
        if max_correlation <= self.thresholds['correlation']['low']:
            corr_score = 0.0
        elif max_correlation <= self.thresholds['correlation']['medium']:
            corr_score = 0.3
        elif max_correlation <= self.thresholds['correlation']['high']:
            corr_score = 0.7
        else:
            corr_score = 1.0
        
        details = {
            'max_correlation': max_correlation,
            'high_corr_pairs': len(high_corr_pairs),
            'correlation_pairs': high_corr_pairs[:5]  # Top 5
        }
        
        return corr_score, details
    
    def calculate_liquidity_risk(self, weights: pd.DataFrame) -> Tuple[float, Dict]:
        """Calculate liquidity risk score - not applicable for this portfolio"""
        # Liquidity is not a concern for this portfolio
        return 0.0, {'note': 'Liquidity risk not applicable - portfolio uses liquid instruments'}
    
    def calculate_stress_test_risk(self, stress_test_results: Dict) -> Tuple[float, Dict]:
        """Calculate stress test risk score and details"""
        if not stress_test_results:
            return 0.0, {}
        
        worst_return = 0.0
        stress_scenarios = []
        
        # Check worst-case scenarios
        if 'worst_case_scenarios' in stress_test_results:
            for scenario in stress_test_results['worst_case_scenarios']:
                scenario_return = scenario.get('return', 0)
                worst_return = min(worst_return, scenario_return)
                stress_scenarios.append(scenario)
        
        # Calculate risk score based on worst-case return
        if worst_return >= -0.10:
            stress_score = 0.0
        elif worst_return >= -0.20:
            stress_score = 0.3
        elif worst_return >= -0.30:
            stress_score = 0.7
        else:
            stress_score = 1.0
        
        details = {
            'worst_case_return': worst_return,
            'stress_scenarios': stress_scenarios
        }
        
        return stress_score, details
    
    def calculate_overall_risk_score(self, data: Dict) -> Dict:
        """Calculate overall portfolio risk score and breakdown"""
        
        # Calculate individual risk scores
        concentration_score, concentration_details = self.calculate_concentration_risk(data.get('weights', pd.DataFrame()))
        volatility_score, volatility_details = self.calculate_volatility_risk(data.get('realized', pd.DataFrame()))
        market_score, market_details = self.calculate_market_risk(data.get('realized', pd.DataFrame()))
        factor_score, factor_details = self.calculate_factor_risk(data.get('factor_exposures', pd.DataFrame()))
        correlation_score, correlation_details = self.calculate_correlation_risk(data.get('correlation', pd.DataFrame()))
        stress_score, stress_details = self.calculate_stress_test_risk(data.get('stress_test', {}))
        
        # Calculate weighted overall score
        risk_scores = {
            'concentration': concentration_score,
            'volatility': volatility_score,
            'market_risk': market_score,
            'factor_risk': factor_score,
            'correlation': correlation_score,
            'stress_test': stress_score
        }
        
        overall_score = sum(score * self.risk_weights[risk_type] for risk_type, score in risk_scores.items())
        
        # Determine risk level
        if overall_score <= 0.3:
            risk_level = "LOW"
        elif overall_score <= 0.6:
            risk_level = "MEDIUM"
        elif overall_score <= 0.8:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"
        
        # Compile detailed breakdown
        breakdown = {
            'overall_score': overall_score,
            'risk_level': risk_level,
            'risk_scores': risk_scores,
            'risk_weights': self.risk_weights,
            'details': {
                'concentration': concentration_details,
                'volatility': volatility_details,
                'market_risk': market_details,
                'factor_risk': factor_details,
                'correlation': correlation_details,
                'stress_test': stress_details
            }
        }
        
        return breakdown

def get_risk_score_color(score: float) -> str:
    """Get color for risk score display"""
    if score <= 0.3:
        return "green"
    elif score <= 0.6:
        return "orange"
    elif score <= 0.8:
        return "red"
    else:
        return "darkred"

def get_risk_level_description(level: str) -> str:
    """Get description for risk level"""
    descriptions = {
        "LOW": "Portfolio risk is well-managed and within acceptable limits",
        "MEDIUM": "Portfolio has moderate risk levels that should be monitored",
        "HIGH": "Portfolio has elevated risk levels requiring attention",
        "CRITICAL": "Portfolio has critical risk levels requiring immediate action"
    }
    return descriptions.get(level, "Risk level not determined") 