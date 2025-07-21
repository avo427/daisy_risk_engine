
# Daisy Risk Engine

## Overview

The **Daisy Risk Engine** is a comprehensive **quantitative risk management platform** designed to evaluate and optimize portfolio performance. It computes both **realized** and **forecasted metrics**, performs **factor exposure analysis**, and enables **volatility-based sizing** for the portfolio. 

The engine is modular, and you can run the following components independently:

1. **Full Pipeline**: Runs the entire pipeline, including risk analysis and factor exposure.
2. **Risk Analysis**: Computes the realized and forecasted risk metrics.
3. **Factor Exposure**: Analyzes the exposure to different factors and computes the factor returns.

The system is designed for portfolio managers and quantitative risk analysts to **interact with data**, **adjust risk parameters**, and **visualize** various financial metrics through an intuitive **Streamlit** dashboard.

---

## Key Features

- **Realized Risk Metrics**: 
  - Compute **annual returns**, **volatility**, **max drawdown**, and other risk metrics like **VaR** (Value-at-Risk) and **CVaR** (Conditional Value-at-Risk).
  
- **Forecast Risk**: 
  - Utilizes **EWMA**, **GARCH**, and **EGARCH** models to predict risk metrics like **VaR**, **CVaR**, and forecasted volatility.

- **Factor Exposure Analysis**: 
  - Assess the portfolio’s exposure to various factors like **Market**, **Volatility**, **Momentum**, **Size**, etc.

- **Volatility-Based Sizing**: 
  - Adjust portfolio positions based on **volatility forecasts**, ensuring that more volatile assets have lower weight.

- **Thematic and Proxy Management**: 
  - Define and adjust **themes** and **proxies** to guide stock selection and factor analysis.

---

## Architecture

### Modular Design

The **Daisy Risk Engine** is composed of several independent, modular components. Each module performs a specific task in the risk analysis pipeline. These modules include:

1. **Portfolio Loading**: 
   - Loads portfolio positions (tickers, weights, market values) from an **Excel file**.
   
2. **Price Data**: 
   - Downloads historical price data for each asset (and associated proxies) from **Yahoo Finance**.
   
3. **Realized Metrics Calculation**: 
   - Computes realized metrics, including annualized return, volatility, and other performance/risk metrics.

4. **Forecast Risk Calculation**: 
   - Computes forecasted risk metrics such as **VaR**, **CVaR**, and **volatility** using advanced models like **EWMA**, **GARCH**, and **EGARCH**.

5. **Factor Exposure and Returns**: 
   - Measures the portfolio's sensitivity to various factors (e.g., market, volatility, size, etc.) and computes factor exposures.

6. **Volatility-Based Sizing**: 
   - Adjusts portfolio positions based on forecasted volatility to better balance risk exposure.

### Streamlit Dashboard

The **Streamlit** dashboard is designed to provide an easy-to-use interface for controlling the risk engine pipeline and visualizing results. Users can:

- **Run Pipeline Components**: Execute the full pipeline or specific components like risk analysis or factor exposure.
- **View Metrics**: View both realized and forecasted metrics, factor exposures, and portfolio sizing.
- **Visualize Results**: Explore various interactive **charts**, **graphs**, and **tables** to make data-driven decisions.

---

## Installation & Setup

Follow these steps to install and run the **Daisy Risk Engine**:

1. **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd daisy_risk_engine
    ```

2. **Install dependencies**:
    Make sure you have all the necessary Python libraries installed using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Configure `config.yaml`**:
    Modify the `config.yaml` file to include your portfolio tickers, weights, and other risk parameters. Ensure the portfolio **Excel file** (`Portfolio.xlsm`) is available in the correct directory.

    ### Portfolio Excel File Configuration:
    - **Path**: Specify the path to the Excel file in `config.yaml`.
    - **Named Ranges**: Define the named ranges for **tickers**, **weights**, and **market values**. These ranges should each be a **1xN or Nx1 vector** (row or column vector with matching dimensions).

    Example configuration:
    ```yaml
    excel_path: ../Portfolio.xlsm  # Path to the portfolio Excel file
    sheet_name: Dashboard  # Name of the sheet containing portfolio data
    named_ranges:
      tickers: Portfolio_Tickers  # Named range for tickers in the portfolio
      weights: Portfolio_Weights  # Named range for portfolio weights
      market_values: Portfolio_MarketValues  # Named range for portfolio market values
    ```

4. **Run the Streamlit app**:
    Start the app using Streamlit:
    ```bash
    streamlit run app.py
    ```

5. **Running the pipeline**:
    - From the **Streamlit interface**, you can choose to run the full pipeline or specific components (`full`, `risk`, or `factor`).

---

## Pipeline Functions Input/Output

### 1. **`load_portfolio.py` (Load Portfolio)**

- **Input**:
    - Excel file (`Portfolio.xlsm`) with tickers, weights, and market values.
    - Config file (`config.yaml`): Portfolio weights and tickers.

- **Output**:
    - DataFrame with columns: `Ticker`, `Weight`, `Market Value`.

---

### 2. **`download_prices.py` (Download Prices)**

- **Input**:
    - Tickers list (portfolio tickers + proxies).
    - Config file (`config.yaml`): Start and end date for historical data.

- **Output**:
    - DataFrame: Historical price data for tickers, columns: `Date`, `Ticker1`, `Ticker2`, ..., `TickerN`.

---

### 3. **`reconstruct_prices.py` (Reconstruct Missing Prices)**

- **Input**:
    - Raw price data (from `download_prices.py`).
    - Portfolio tickers list.
    - Config file (`config.yaml`): Proxies and fallback proxies.

- **Output**:
    - Reconstructed price history for tickers, using proxies where necessary.

---

### 4. **`realized.py` (Realized Metrics Calculation)**

- **Input**:
    - Portfolio weights (from `load_portfolio.py`).
    - Reconstructed price data (from `reconstruct_prices.py`).
    - Config file (`config.yaml`): Risk-free rate, tickers, and other settings.

- **Output**:
    - DataFrame with realized metrics: `Annualized Return`, `Annualized Volatility`, `Max Drawdown`, `VaR (95%)`, `CVaR (95%)`, `Sharpe`, `Sortino`, etc.

---

### 5. **`forecast.py` (Forecast Risk Calculation)**

- **Input**:
    - Portfolio weights (from `load_portfolio.py`).
    - Reconstructed price data (from `reconstruct_prices.py`).
    - Config file (`config.yaml`): Forecast models (EWMA, GARCH, EGARCH), risk-free rate.

- **Output**:
    - DataFrame with forecasted metrics: `VaR`, `CVaR`, `Volatility` (for various models and timeframes).

---

### 6. **`factor_loader.py` (Factor Loading)**

- **Input**:
    - Portfolio tickers.
    - Config file (`config.yaml`): Factor definitions, proxy tickers.

- **Output**:
    - DataFrame with factor returns for defined factors (e.g., market, volatility, size, momentum).

---

### 7. **`factor_engine.py` (Factor Engine)**

- **Input**:
    - Portfolio weights (from `load_portfolio.py`).
    - Factor returns (from `factor_loader.py`).
    - Config file (`config.yaml`): Risk-free rate, factor exposure models.

- **Output**:
    - DataFrame with factor exposures: `Factor`, `Beta (Exposure)`, `Risk Contribution`.

---

### 8. **`app.py` (Streamlit Dashboard)**

- **Input**:
    - User inputs (via Streamlit controls).
    - Data files: CSV files generated by the pipeline (e.g., `realized_metrics.csv`, `forecast_metrics.csv`).

- **Output**:
    - Interactive visualizations: Tables, charts, and graphs showing risk metrics, factor exposures, portfolio sizing, etc.

---

## Conclusion

The **Daisy Risk Engine** provides a **robust framework** for evaluating portfolio risk and performance. It integrates portfolio management, risk analysis, forecasting, and factor analysis into a single, **unified platform**. With a **modular design** and clear input/output separation, this system enables users to **compute** and **visualize** various financial metrics essential for making **data-driven investment decisions**.

---

This **README** serves as a comprehensive guide to setting up, running, and understanding the **Daisy Risk Engine**.
