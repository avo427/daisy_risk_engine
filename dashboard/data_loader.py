import pandas as pd
from pathlib import Path
from streamlit import cache_data

def load_csv(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col=None)

@cache_data
def load_all_data(project_root, paths):
    return {
        "realized": load_csv(project_root / paths["realized_output"]),
        "roll": load_csv(project_root / paths["realized_rolling_output"]),
        "corr": load_csv(project_root / paths["correlation_matrix"]),
        "vol": load_csv(project_root / paths["vol_contribution"]),
        "forecast": load_csv(project_root / paths["forecast_output"]),
        "forecast_roll": load_csv(project_root / paths["forecast_rolling_output"]),
    } 