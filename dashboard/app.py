import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to sys.path for main.py imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Import modularized components
from config import load_config, save_config
from data_loader import load_all_data
from ui.sidebar import render_sidebar
from ui.tabs.realized import realized_tab
from ui.tabs.forecast import forecast_tab
from ui.tabs.sizing import sizing_tab
from ui.tabs.factors import factors_tab
from ui.tabs.themes import themes_tab
from ui.tabs.prices import prices_tab

# === Load Config ===
config = load_config(project_root)
user_settings = config.get("user_settings", {})
paths = config.get("paths", {})

def reload_data():
    return load_all_data(project_root, paths)

data = reload_data()

# === UI Setup ===
st.set_page_config(page_title="Daisy Risk Engine", layout="wide")
st.title("Daisy Risk Engine Dashboard")

# === Sidebar Controls ===
with st.sidebar:
    render_sidebar(config, lambda c: save_config(project_root, c), lambda: reload_data(), project_root, paths, user_settings)

# === Initialize session data ===
if "realized" not in st.session_state:
    st.session_state.update(data)

df_realized = st.session_state.get("realized", pd.DataFrame())
df_roll = st.session_state.get("roll", pd.DataFrame())
df_corr = st.session_state.get("corr", pd.DataFrame())
df_vol = st.session_state.get("vol", pd.DataFrame())
df_forecast = st.session_state.get("forecast", pd.DataFrame())
df_fore_roll = st.session_state.get("forecast_roll", pd.DataFrame())

# === Tabs ===
tabs = st.tabs([
    "Realized Risk",
    "Forecast Risk",
    "Volatility-Based Sizing",
    "Factor Exposure",
    "Themes & Proxies",
    "Reconstructed Prices"
])

with tabs[0]:
    realized_tab(df_realized, df_roll, df_corr, df_vol)
with tabs[1]:
    forecast_tab(df_forecast, df_fore_roll, project_root, paths)
with tabs[2]:
    sizing_tab(project_root, paths)
with tabs[3]:
    factors_tab(project_root, paths)
with tabs[4]:
    themes_tab(config, lambda c: save_config(project_root, c))
with tabs[5]:
    prices_tab(project_root, paths)