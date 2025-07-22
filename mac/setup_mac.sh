#!/bin/bash

# Simple setup for Daisy Risk Engine (Mac)

echo "🔧 Creating virtual environment in ../.venv..."
python3 -m venv ../.venv

echo "📦 Activating virtual environment..."
source ../.venv/bin/activate

echo "📚 Installing dependencies from ../requirements.txt..."
pip install --upgrade pip
pip install -r ../requirements.txt

echo "✅ Setup complete. To activate later:"
echo "   source ../.venv/bin/activate"