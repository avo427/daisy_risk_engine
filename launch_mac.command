#!/bin/bash
cd "$(dirname "$0")"
nohup streamlit run dashboard/app.py --server.port 8501 > /tmp/streamlit.log 2>&1 &
sleep 3
open http://localhost:8501 