#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate
streamlit run dashboard/app.py

