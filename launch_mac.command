#!/bin/bash
cd "$(dirname "$0")"
nohup streamlit run dashboard/app.py > /dev/null 2>&1 & 