#!/bin/bash

# Go to project folder
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

# Run Streamlit dashboard
streamlit run dashboard/app.py

