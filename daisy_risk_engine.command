#!/bin/bash

# === Daisy Risk Engine Launcher for macOS ===

# Get the directory of this script
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Optional: Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo "✅ Activating virtual environment..."
    source venv/bin/activate
else
    echo "⚠️  No virtual environment found. Using system Python..."
fi

# Launch Streamlit dashboard in background
echo "🚀 Starting Daisy Risk Engine..."
streamlit run dashboard/app.py --server.headless true &

# Wait a few seconds for server to start
sleep 3

# Open in Google Chrome (new window)
echo "🌐 Opening in Chrome..."
open -na "Google Chrome" --args --new-window "http://localhost:8501"
