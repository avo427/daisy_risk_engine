import pandas as pd
import numpy as np
import yaml
import logging
from pathlib import Path
from scipy.stats import norm, t, multivariate_normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json

warnings.filterwarnings("ignore")

class StressTestEngine:
    """
    Institutional-grade stress testing engine incorporating methodologies from
    Bridgewater Associates and Citadel Securities.
    
    Features:
    - Historical scenario analysis (2008 Crisis, 2020 COVID, 2022 Inflation, etc.)
    - Monte Carlo simulations with fat-tailed distributions
    - Factor-based stress tests with correlation breakdown
    - Regime-dependent stress scenarios
    - Portfolio-level and position-level analysis
    - VaR and CVaR stress testing
    """
    
    def __init__(self, config_path="config.yaml"):
        self.config = self._load_config(config_path)
        self.rf_rate = self.config["user_settings"].get("risk_free_rate", 0.05)
        self.annual_factor = self.config["user_settings"].get("trading_days_per_year", 252)
        self.random_state = self.config["user_settings"].get("random_state", 44)
        np.random.seed(self.random_state)
        
        # Load data
        self._load_data()
        
        # Initialize stress scenarios
        self._initialize_scenarios()
        
    def _load_config(self, config_path):
        """Load configuration file."""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def _load_data(self):
        """Load portfolio and market data."""
        try:
            # Load portfolio weights
            weights_df = pd.read_csv(self.config["paths"]["portfolio_weights"])
            weights_df["Ticker"] = weights_df["Ticker"].str.upper()
            self.weights = weights_df[weights_df["MarketValue"] > 0].copy()
            
            # Load price data
            prices = pd.read_csv(self.config["paths"]["recon_prices_output"], 
                               index_col=0, parse_dates=True).dropna(how="all")
            
            # Ensure index is properly formatted as datetime
            if not isinstance(prices.index, pd.DatetimeIndex):
                prices.index = pd.to_datetime(prices.index)
            
            # Forward fill NaN values to preserve historical data
            prices = prices.fillna(method='ffill')
            
            # Calculate returns and drop only the first row (which will be NaN from pct_change)
            self.returns = prices.pct_change().dropna(how='all')
            
            # Ensure returns index is also properly formatted
            if not isinstance(self.returns.index, pd.DatetimeIndex):
                self.returns.index = pd.to_datetime(self.returns.index)
                
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise ValueError(f"Failed to load data: {e}")
        
        # Load factor data if available
        try:
            self.factors = pd.read_csv(self.config["paths"]["factor_returns"], 
                                     index_col=0, parse_dates=True).dropna(how="all")
        except FileNotFoundError:
            self.factors = None
            logging.warning("Factor returns not found - factor-based stress tests disabled")
        
        # Calculate portfolio returns - CRITICAL FIX: Maintain index alignment
        portfolio_tickers = list(self.weights["Ticker"])
        portfolio_returns = self.returns[portfolio_tickers]
        
        # IMPORTANT: Don't dropna() here - it breaks index alignment
        # Instead, fill NaN with 0 for portfolio calculation
        portfolio_returns = portfolio_returns.fillna(0)
        
        weights_series = self.weights.set_index("Ticker")["Weight"]
        self.portfolio_returns = (portfolio_returns * weights_series).sum(axis=1)
        
        # Now we can drop NaN from portfolio returns if needed, but maintain alignment
        # Only drop rows where ALL portfolio tickers are NaN
        self.portfolio_returns = self.portfolio_returns.dropna()
        
        # Re-align returns to match portfolio returns index
        self.returns = self.returns.loc[self.portfolio_returns.index]
        
        # Validate data integrity
        if len(self.returns) != len(self.portfolio_returns):
            raise ValueError(f"Data alignment error: returns length {len(self.returns)} != portfolio length {len(self.portfolio_returns)}")
        
        if not self.returns.index.equals(self.portfolio_returns.index):
            raise ValueError("Data alignment error: returns and portfolio returns have different indices")
        
        logging.info(f"Data loaded successfully: {len(self.returns)} observations from {self.returns.index.min()} to {self.returns.index.max()}")
        
    def _initialize_scenarios(self):
        """Initialize predefined stress scenarios."""
        self.historical_scenarios = {
            "2008_Financial_Crisis": {
                "period": ("2008-09-01", "2009-03-31"),
                "description": "Global Financial Crisis - Lehman Brothers collapse",
                "market_shock": -0.40,
                "volatility_multiplier": 3.0,
                "correlation_breakdown": True
            },
            "2020_COVID_Crash": {
                "period": ("2020-02-20", "2020-03-23"),
                "description": "COVID-19 pandemic market crash",
                "market_shock": -0.35,
                "volatility_multiplier": 4.0,
                "correlation_breakdown": True
            },
            "2022_Inflation_Shock": {
                "period": ("2022-01-01", "2022-12-31"),
                "description": "Inflation-driven market correction",
                "market_shock": -0.20,
                "volatility_multiplier": 2.0,
                "correlation_breakdown": False
            },
            "2018_Q4_Volatility": {
                "period": ("2018-10-01", "2018-12-31"),
                "description": "Q4 2018 volatility spike",
                "market_shock": -0.15,
                "volatility_multiplier": 2.5,
                "correlation_breakdown": True
            },
            "2020_Recovery": {
                "period": ("2020-03-24", "2020-08-31"),
                "description": "Post-COVID recovery rally",
                "market_shock": 0.30,
                "volatility_multiplier": 1.5,
                "correlation_breakdown": False
            },
            "2025_Trump_Tariffs": {
                "period": ("2025-03-31", "2025-06-30"),
                "description": "Trump administration trade war escalation and tariff increases",
                "market_shock": -0.18,
                "volatility_multiplier": 2.2,
                "correlation_breakdown": True
            }
        }
        
        # Factor stress scenarios
        self.factor_stress_scenarios = {
            "Tech_Sector_Crash": {
                "factors": ["AI", "MARKET"],
                "shock_magnitude": -0.25,
                "description": "Technology sector specific crash"
            },
            "Interest_Rate_Spike": {
                "factors": ["RATES"],
                "shock_magnitude": 0.50,
                "description": "Sharp increase in interest rates"
            },
            "Volatility_Explosion": {
                "factors": ["VOLATILITY"],
                "shock_magnitude": 2.0,
                "description": "VIX spike and volatility regime change"
            },
            "Momentum_Reversal": {
                "factors": ["MOMENTUM"],
                "shock_magnitude": -0.30,
                "description": "Momentum factor reversal"
            },
            "Liquidity_Crisis": {
                "factors": ["SIZE", "MIN_VOL"],
                "shock_magnitude": -0.20,
                "description": "Liquidity crisis affecting small caps and low vol"
            }
        }
    
    def historical_scenario_analysis(self, scenario_name=None, custom_period=None):
        """
        Perform historical scenario analysis.
        
        Args:
            scenario_name: Name of predefined scenario
            custom_period: Tuple of (start_date, end_date) for custom period
            
        Returns:
            Dictionary with scenario results
        """
        if scenario_name and scenario_name in self.historical_scenarios:
            scenario = self.historical_scenarios[scenario_name]
            start_date, end_date = scenario["period"]
        elif custom_period:
            start_date, end_date = custom_period
            scenario = {"description": "Custom historical period"}
        else:
            raise ValueError("Must specify either scenario_name or custom_period")
        
        # Convert string dates to pandas Timestamp objects for filtering
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        
        # Validate data alignment before filtering
        if not self.returns.index.equals(self.portfolio_returns.index):
            raise ValueError("Data alignment error: returns and portfolio returns indices don't match")
        
        # Filter data for scenario period
        mask = (self.returns.index >= start_date_ts) & (self.returns.index <= end_date_ts)
        scenario_returns = self.returns[mask]
        scenario_portfolio = self.portfolio_returns[mask]
        
        # Validate filtered data
        if scenario_returns.empty:
            return {"error": "No data available for specified period"}
        
        if len(scenario_returns) != len(scenario_portfolio):
            raise ValueError(f"Filtered data alignment error: returns {len(scenario_returns)} != portfolio {len(scenario_portfolio)}")
        
        # Additional validation for extreme values
        if scenario_portfolio.abs().max() > 1.0:  # More than 100% daily return
            logging.warning(f"Extreme portfolio return detected: {scenario_portfolio.abs().max():.4f}")
        
        # Calculate scenario metrics
        results = {
            "scenario_name": scenario_name or "custom",
            "period": (start_date, end_date),
            "description": scenario.get("description", "Custom scenario"),
            "data_points": len(scenario_returns),
            "portfolio_metrics": self._calculate_portfolio_metrics(scenario_portfolio),
            "position_metrics": self._calculate_position_metrics(scenario_returns),
            "correlation_analysis": self._analyze_correlations(scenario_returns),
            "tail_risk_metrics": self._calculate_tail_risk(scenario_portfolio)
        }
        
        return results
    
    def monte_carlo_stress_test(self, n_simulations=None, time_horizon=252, 
                               confidence_levels=[0.95, 0.99, 0.995]):
        """
        Monte Carlo stress testing with fat-tailed distributions.
        
        Args:
            n_simulations: Number of Monte Carlo simulations (defaults to config value)
            time_horizon: Time horizon in days
            confidence_levels: List of confidence levels for VaR/CVaR
            
        Returns:
            Dictionary with Monte Carlo results
        """
        # Use config value if not specified
        if n_simulations is None:
            n_simulations = self.config["user_settings"].get("monte_carlo_simulations", 10000)
        
        # Validate input data
        if self.portfolio_returns is None or self.portfolio_returns.empty:
            raise ValueError("No portfolio returns data available for Monte Carlo simulation")
        
        # Remove extreme outliers that could skew the simulation
        returns_clean = self.portfolio_returns.dropna()
        
        # Cap extreme returns to prevent unrealistic simulations
        # More than 50% daily return is extremely rare and likely data error
        extreme_threshold = 0.50
        returns_clean = returns_clean.clip(-extreme_threshold, extreme_threshold)
        
        # Log data quality metrics
        logging.info(f"Monte Carlo input: {len(returns_clean)} observations")
        logging.info(f"Return range: {returns_clean.min():.4f} to {returns_clean.max():.4f}")
        logging.info(f"Return std: {returns_clean.std():.4f}")
        
        # Estimate parameters from cleaned historical data
        mu = returns_clean.mean()
        sigma = returns_clean.std()
        
        # Validate parameters are reasonable
        if abs(mu) > 0.1:  # More than 10% daily mean return
            logging.warning(f"Unusually high mean return: {mu:.4f}")
        
        if sigma > 0.1:  # More than 10% daily volatility
            logging.warning(f"Unusually high volatility: {sigma:.4f}")
        
        # Fit t-distribution for fat tails
        try:
            df, loc, scale = t.fit(returns_clean)
            
            # Validate fitted parameters
            if df < 1 or df > 100:
                logging.warning(f"Unusual degrees of freedom: {df:.2f}, using normal distribution")
                df, loc, scale = 30, mu, sigma  # Fallback to reasonable values
                
        except Exception as e:
            logging.warning(f"T-distribution fitting failed: {e}, using normal distribution")
            df, loc, scale = 30, mu, sigma  # Fallback to reasonable values
        
        # Generate simulations with validation
        simulations = []
        for i in range(n_simulations):
            # Use t-distribution for fat-tailed returns
            daily_returns = t.rvs(df, loc=loc, scale=scale, size=time_horizon)
            
            # Validate daily returns are reasonable
            if np.any(np.abs(daily_returns) > 1.0):  # More than 100% daily return
                daily_returns = np.clip(daily_returns, -0.5, 0.5)  # Cap at 50%
            
            cumulative_return = np.prod(1 + daily_returns) - 1
            
            # Validate cumulative return is reasonable
            if abs(cumulative_return) > 10.0:  # More than 1000% cumulative return
                logging.warning(f"Extreme cumulative return in simulation {i}: {cumulative_return:.4f}")
                cumulative_return = np.clip(cumulative_return, -5.0, 5.0)  # Cap at 500%
            
            simulations.append(cumulative_return)
        
        simulations = np.array(simulations)
        
        # Validate simulation results
        if np.any(np.isnan(simulations)) or np.any(np.isinf(simulations)):
            raise ValueError("Monte Carlo simulation produced NaN or infinite values")
        
        # Calculate risk metrics
        results = {
            "n_simulations": n_simulations,
            "time_horizon": time_horizon,
            "input_data_quality": {
                "observations": len(returns_clean),
                "mean_return": float(mu),
                "volatility": float(sigma),
                "min_return": float(returns_clean.min()),
                "max_return": float(returns_clean.max())
            },
            "simulation_statistics": {
                "mean": float(np.mean(simulations)),
                "std": float(np.std(simulations)),
                "skewness": float(self._calculate_skewness(simulations)),
                "kurtosis": float(self._calculate_kurtosis(simulations)),
                "min": float(np.min(simulations)),
                "max": float(np.max(simulations))
            },
            "var_cvar": {}
        }
        
        # Calculate VaR and CVaR for different confidence levels
        for cl in confidence_levels:
            var = np.percentile(simulations, (1 - cl) * 100)
            cvar = np.mean(simulations[simulations <= var])
            results["var_cvar"][f"VaR_{int(cl*100)}"] = float(var)
            results["var_cvar"][f"CVaR_{int(cl*100)}"] = float(cvar)
        
        return results
    
    def factor_stress_test(self, scenario_name=None, custom_shocks=None):
        """
        Factor-based stress testing.
        
        Args:
            scenario_name: Name of predefined factor scenario
            custom_shocks: Dictionary of factor shocks {factor: shock_magnitude}
            
        Returns:
            Dictionary with factor stress test results
        """
        if self.factors is None:
            return {"error": "Factor data not available"}
        
        if scenario_name and scenario_name in self.factor_stress_scenarios:
            scenario = self.factor_stress_scenarios[scenario_name]
            shocks = {factor: scenario["shock_magnitude"] for factor in scenario["factors"]}
        elif custom_shocks:
            shocks = custom_shocks
        else:
            raise ValueError("Must specify either scenario_name or custom_shocks")
        
        # Load factor exposures if available
        try:
            exposures = pd.read_csv(self.config["paths"]["factor_exposures"])
            # Pivot to get portfolio exposures in wide format
            exposures_pivot = exposures.pivot(index="Ticker", columns="Factor", values="Beta")
            portfolio_exposure = exposures_pivot.loc["PORTFOLIO"] if "PORTFOLIO" in exposures_pivot.index else None
            

            
        except FileNotFoundError:
            portfolio_exposure = None
            logging.warning("Factor exposures not found - using simplified approach")
        
        # Calculate factor impact
        results = {
            "scenario": scenario_name or "custom",
            "shocks": shocks,
            "factor_impact": {},
            "portfolio_impact": 0
        }
        
        if portfolio_exposure is not None:
            for factor, shock in shocks.items():
                if factor in portfolio_exposure.index:
                    impact = portfolio_exposure[factor] * shock
                    results["factor_impact"][factor] = impact
                    results["portfolio_impact"] += impact

        
        return results
    
    def regime_dependent_stress_test(self, regime_indicators=None):
        """
        Regime-dependent stress testing based on market conditions.
        
        Args:
            regime_indicators: Dictionary of regime indicators and thresholds
            
        Returns:
            Dictionary with regime-dependent results
        """
        if regime_indicators is None:
            # Default regime indicators
            regime_indicators = {
                "volatility_regime": {"high": 0.25, "extreme": 0.40},
                "correlation_regime": {"high": 0.7, "extreme": 0.9},
                "momentum_regime": {"negative": -0.1, "positive": 0.1}
            }
        
        # Calculate current regime indicators
        current_vol = self.portfolio_returns.rolling(20).std().iloc[-1] * np.sqrt(self.annual_factor)
        current_corr = self.returns.corr().mean().mean()
        current_momentum = self.portfolio_returns.rolling(60).mean().iloc[-1]
        
        # Determine current regime
        regime = "normal"
        if current_vol > regime_indicators["volatility_regime"]["extreme"]:
            regime = "extreme_volatility"
        elif current_vol > regime_indicators["volatility_regime"]["high"]:
            regime = "high_volatility"
        
        if current_corr > regime_indicators["correlation_regime"]["extreme"]:
            regime += "_extreme_correlation"
        elif current_corr > regime_indicators["correlation_regime"]["high"]:
            regime += "_high_correlation"
        
        # Apply regime-specific stress scenarios
        results = {
            "current_regime": regime,
            "regime_indicators": {
                "volatility": current_vol,
                "correlation": current_corr,
                "momentum": current_momentum
            },
            "stress_scenarios": self._get_regime_stress_scenarios(regime)
        }
        
        return results
    
    def _calculate_portfolio_metrics(self, returns):
        """Calculate portfolio-level metrics for stress scenario."""
        if returns.empty:
            return {}
        
        ann_return = (1 + returns.mean()) ** self.annual_factor - 1
        ann_vol = returns.std() * np.sqrt(self.annual_factor)
        sharpe = ((returns.mean() - self.rf_rate / self.annual_factor) / returns.std()) * np.sqrt(self.annual_factor)
        max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        return {
            "annual_return": ann_return,
            "annual_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "total_return": (1 + returns).prod() - 1
        }
    
    def _calculate_position_metrics(self, returns):
        """Calculate position-level metrics for stress scenario."""
        portfolio_tickers = list(self.weights["Ticker"])
        position_metrics = {}
        
        for ticker in portfolio_tickers:
            if ticker in returns.columns:
                ticker_returns = returns[ticker].dropna()
                if not ticker_returns.empty:
                    weight = self.weights[self.weights["Ticker"] == ticker]["Weight"].iloc[0]
                    market_value = self.weights[self.weights["Ticker"] == ticker]["MarketValue"].iloc[0]
                    
                    position_metrics[ticker] = {
                        "weight": weight,
                        "market_value": market_value,
                        "total_return": (1 + ticker_returns).prod() - 1,
                        "volatility": ticker_returns.std() * np.sqrt(self.annual_factor),
                        "var_95": np.percentile(ticker_returns, 5),
                        "dollar_var_95": np.percentile(ticker_returns, 5) * market_value
                    }
        
        return position_metrics
    
    def _analyze_correlations(self, returns):
        """Analyze correlation changes during stress scenario."""
        portfolio_tickers = list(self.weights["Ticker"])
        relevant_tickers = [t for t in portfolio_tickers if t in returns.columns]
        
        if len(relevant_tickers) < 2:
            return {}
        
        # Calculate correlation matrix
        corr_matrix = returns[relevant_tickers].corr()
        
        # Calculate average correlation
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        avg_correlation = upper_triangle.stack().mean()
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "average_correlation": avg_correlation,
            "max_correlation": upper_triangle.stack().max(),
            "min_correlation": upper_triangle.stack().min()
        }
    
    def _calculate_tail_risk(self, returns):
        """Calculate tail risk metrics."""
        if returns.empty:
            return {}
        
        # Expected Shortfall (CVaR) at different confidence levels
        confidence_levels = [0.95, 0.99, 0.995]
        tail_risk = {}
        
        for cl in confidence_levels:
            var = np.percentile(returns, (1 - cl) * 100)
            cvar = returns[returns <= var].mean()
            tail_risk[f"var_{int(cl*100)}"] = var
            tail_risk[f"cvar_{int(cl*100)}"] = cvar
        
        # Tail dependence measures
        tail_risk["tail_dependence"] = self._calculate_tail_dependence(returns)
        
        return tail_risk
    
    def _calculate_tail_dependence(self, returns):
        """Calculate tail dependence coefficient."""
        # Simplified tail dependence calculation
        threshold = np.percentile(returns, 5)
        tail_events = (returns <= threshold).sum()
        total_events = len(returns)
        
        return tail_events / total_events if total_events > 0 else 0
    
    def _calculate_skewness(self, data):
        """Calculate skewness."""
        return ((data - np.mean(data)) ** 3).mean() / (np.std(data) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis."""
        return ((data - np.mean(data)) ** 4).mean() / (np.std(data) ** 4) - 3
    
    def _get_regime_stress_scenarios(self, regime):
        """Get stress scenarios based on current regime."""
        scenarios = {
            "normal": {"vol_multiplier": 1.0, "correlation_impact": 0.0},
            "high_volatility": {"vol_multiplier": 1.5, "correlation_impact": 0.2},
            "extreme_volatility": {"vol_multiplier": 2.0, "correlation_impact": 0.4},
            "high_correlation": {"vol_multiplier": 1.2, "correlation_impact": 0.3},
            "extreme_correlation": {"vol_multiplier": 1.5, "correlation_impact": 0.5}
        }
        
        return scenarios.get(regime, scenarios["normal"])
    
    def run_comprehensive_stress_test(self):
        """
        Run comprehensive stress testing suite.
        
        Returns:
            Dictionary with all stress test results
        """
        try:
            # Verify data is loaded properly
            if self.returns is None or self.returns.empty:
                raise ValueError("No returns data available")
            
            if not isinstance(self.returns.index, pd.DatetimeIndex):
                raise ValueError("Returns index is not a DatetimeIndex")
            
            data_start = pd.Timestamp(self.returns.index.min())
            data_end = pd.Timestamp(self.returns.index.max())
            
            logging.info(f"Data range: {data_start} to {data_end}")
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "historical_scenarios": {},
                "monte_carlo": {},
                "factor_stress": {},
                "regime_analysis": {},
                "summary": {},
                "data_availability": {
                    "available_period": (str(data_start), str(data_end)),
                    "scenarios_included": [],
                    "scenarios_excluded": []
                }
            }
        except Exception as e:
            logging.error(f"Error initializing stress test: {e}")
            raise
        
        # Historical scenario analysis - only include scenarios with available data
        for scenario_name, scenario_config in self.historical_scenarios.items():
            try:
                start_date, end_date = scenario_config["period"]
                
                # Convert string dates to pandas Timestamp objects for comparison
                start_date_ts = pd.Timestamp(start_date)
                end_date_ts = pd.Timestamp(end_date)
                
                logging.info(f"Processing scenario {scenario_name}: {start_date_ts} to {end_date_ts}")
                
                # Check if scenario period overlaps with available data
                if start_date_ts <= data_end and end_date_ts >= data_start:
                    # Scenario overlaps with available data
                    try:
                        scenario_result = self.historical_scenario_analysis(scenario_name)
                        if "error" not in scenario_result:
                            results["historical_scenarios"][scenario_name] = scenario_result
                            results["data_availability"]["scenarios_included"].append(scenario_name)
                        else:
                            results["data_availability"]["scenarios_excluded"].append(
                                f"{scenario_name} (insufficient data)"
                            )
                    except Exception as e:
                        logging.error(f"Error in historical scenario {scenario_name}: {e}")
                        results["data_availability"]["scenarios_excluded"].append(
                            f"{scenario_name} (error: {str(e)})"
                        )
                else:
                    # Scenario completely outside available data range
                    results["data_availability"]["scenarios_excluded"].append(
                        f"{scenario_name} (period {start_date} to {end_date} outside data range {data_start} to {data_end})"
                    )
            except Exception as e:
                logging.error(f"Error processing scenario {scenario_name}: {e}")
                results["data_availability"]["scenarios_excluded"].append(
                    f"{scenario_name} (processing error: {str(e)})"
                )
        
        # Monte Carlo stress test
        try:
            results["monte_carlo"] = self.monte_carlo_stress_test()
        except Exception as e:
            logging.error(f"Error in Monte Carlo stress test: {e}")
            results["monte_carlo"] = {"error": str(e)}
        
        # Factor stress tests
        if self.factors is not None:
            for scenario_name in self.factor_stress_scenarios.keys():
                try:
                    results["factor_stress"][scenario_name] = self.factor_stress_test(scenario_name)
                except Exception as e:
                    logging.error(f"Error in factor stress test {scenario_name}: {e}")
                    results["factor_stress"][scenario_name] = {"error": str(e)}
        
        # Regime analysis
        try:
            results["regime_analysis"] = self.regime_dependent_stress_test()
        except Exception as e:
            logging.error(f"Error in regime analysis: {e}")
            results["regime_analysis"] = {"error": str(e)}
        
        # Generate summary
        results["summary"] = self._generate_stress_test_summary(results)
        
        return results
    
    def _generate_stress_test_summary(self, results):
        """Generate executive summary of stress test results."""
        summary = {
            "worst_case_scenarios": [],
            "key_risk_metrics": {},
            "recommendations": [],
            "data_coverage": {
                "scenarios_analyzed": len(results["historical_scenarios"]),
                "scenarios_excluded": len(results.get("data_availability", {}).get("scenarios_excluded", [])),
                "available_period": results.get("data_availability", {}).get("available_period", "Unknown")
            }
        }
        
        # Find worst-case historical scenario
        worst_historical = None
        worst_return = 0
        
        for scenario_name, scenario_results in results["historical_scenarios"].items():
            if "portfolio_metrics" in scenario_results:
                total_return = scenario_results["portfolio_metrics"].get("total_return", 0)
                if total_return < worst_return:
                    worst_return = total_return
                    worst_historical = scenario_name
        
        if worst_historical:
            summary["worst_case_scenarios"].append({
                "type": "historical",
                "scenario": worst_historical,
                "return": worst_return
            })
        elif results.get("data_availability", {}).get("scenarios_excluded"):
            # Convert string date back to Timestamp to access year
            available_start = pd.Timestamp(results['data_availability']['available_period'][0])
            summary["recommendations"].append(
                f"Consider increasing data history to {max([20, available_start.year + 15])} years to include more historical stress scenarios"
            )
        
        # Monte Carlo worst case
        if "simulation_statistics" in results["monte_carlo"]:
            mc_min = results["monte_carlo"]["simulation_statistics"].get("min", 0)
            summary["worst_case_scenarios"].append({
                "type": "monte_carlo",
                "return": mc_min,
                "confidence": "Based on 10,000 simulations"
            })
        
        # Key risk metrics
        if "var_cvar" in results["monte_carlo"]:
            summary["key_risk_metrics"] = results["monte_carlo"]["var_cvar"]
        
        # Generate recommendations
        if worst_return < -0.20:
            summary["recommendations"].append("Consider reducing portfolio leverage")
        if worst_return < -0.30:
            summary["recommendations"].append("Review concentration in high-risk positions")
        
        return summary
    
    def save_results(self, results, output_path=None):
        """Save stress test results to file."""
        if output_path is None:
            output_path = Path(self.config["paths"].get("stress_test_output", "data/stress_test_results.json"))
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        results_converted = convert_numpy(results)
        
        with open(output_path, "w") as f:
            json.dump(results_converted, f, indent=2, default=str)
        
        logging.info(f"Stress test results saved to {output_path}")
        return output_path


def run_stress_test(config_path="config.yaml", output_path=None):
    """
    Main function to run comprehensive stress testing.
    
    Args:
        config_path: Path to configuration file
        output_path: Path to save results (optional)
    
    Returns:
        Path to saved results file
    """
    try:
        # Initialize stress test engine
        engine = StressTestEngine(config_path)
        
        # Run comprehensive stress test
        results = engine.run_comprehensive_stress_test()
        
        # Save results
        saved_path = engine.save_results(results, output_path)
        
        logging.info("Stress testing completed successfully")
        return saved_path
        
    except Exception as e:
        logging.error(f"Stress testing failed: {e}")
        raise


if __name__ == "__main__":
    # Run stress test from command line
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive stress testing")
    parser.add_argument("--config", default="config.yaml", help="Path to config file")
    parser.add_argument("--output", help="Path to save results")
    
    args = parser.parse_args()
    
    try:
        output_path = run_stress_test(args.config, args.output)
        print(f"Stress test completed. Results saved to: {output_path}")
    except Exception as e:
        print(f"Stress test failed: {e}")
        exit(1) 