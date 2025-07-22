#!/bin/bash

# === Navigate to parent project folder from windows/ subfolder ===
cd "$(dirname "$0")/.."

# === Activate the virtual environment ===
source .venv/bin/activate

# === Run the Streamlit dashboard ===
streamlit run dashboard/app.py
