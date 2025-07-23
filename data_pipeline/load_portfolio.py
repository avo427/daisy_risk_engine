import os
import yaml
import pandas as pd
from openpyxl import load_workbook
from datetime import datetime

def load_portfolio_from_excel(config_path="config.yaml") -> pd.DataFrame:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)["loader"]

    excel_path = config["excel_path"]
    sheet_name = config["sheet_name"]
    named_ranges = config["named_ranges"]
    output_file = config["output_file"]
    versioned_output_dir = config["versioned_output_dir"]

    # Ensure output directories exist
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(versioned_output_dir, exist_ok=True)

    # Load Excel workbook
    wb = load_workbook(excel_path, data_only=True)
    sheet = wb[sheet_name]

    def extract_named_range_values(wb, sheet, range_name):
        try:
            dn = wb.defined_names[range_name]
        except KeyError:
            raise KeyError(f"ERROR: Named range '{range_name}' not found in workbook.")

        destinations = list(dn.destinations)
        if not destinations:
            raise ValueError(f"ERROR: No destinations found for named range '{range_name}'.")

        for title, ref in destinations:
            if title != sheet.title:
                continue
            cells = sheet[ref]
            return [cell.value for row in cells for cell in row]

        raise ValueError(f"ERROR: Named range '{range_name}' not defined on sheet '{sheet.title}'.")

    tickers = extract_named_range_values(wb, sheet, named_ranges["tickers"])
    weights = extract_named_range_values(wb, sheet, named_ranges["weights"])
    market_values = extract_named_range_values(wb, sheet, named_ranges["market_values"])

    if not (len(tickers) == len(weights) == len(market_values)):
        raise ValueError(
            f"ERROR: Named range length mismatch:\n"
            f"Tickers: {len(tickers)}\n"
            f"Weights: {len(weights)}\n"
            f"MarketValues: {len(market_values)}"
        )

    df = pd.DataFrame({
        "Ticker": tickers,
        "Weight": weights,
        "MarketValue": market_values
    }).dropna()

    if df.empty:
        raise ValueError("ERROR: DataFrame is empty after parsing and cleaning. Check Excel content.")

    # Clean tickers and numbers
    df["Ticker"] = df["Ticker"].astype(str).str.strip().str.replace(r"\$", "SGOV", regex=True)
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")
    df["MarketValue"] = pd.to_numeric(df["MarketValue"], errors="coerce")

    # Group duplicates
    df = df.groupby("Ticker", as_index=False).agg({
        "Weight": "sum",
        "MarketValue": "sum"
    })

    # Validate weights sum to 1.0
    total_weight = df["Weight"].sum()
    if not abs(total_weight - 1.0) < 1e-6:
        raise ValueError(f"ERROR: Portfolio weights sum to {total_weight:.4f}, expected 1.0.")

    # Normalize ticker strings
    df["Ticker"] = df["Ticker"].str.replace("/", "-")

    # Save main output
    df.to_csv(output_file, index=False)

    # Save versioned copy
    stamp = datetime.today().strftime("%Y%m%d")
    backup_path = os.path.join(versioned_output_dir, f"portfolio_weights_{stamp}.csv")
    df.to_csv(backup_path, index=False)

    return df
