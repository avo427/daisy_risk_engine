#!/bin/bash

# Simple setup for Daisy Risk Engine (inside 'mac' folder)

echo "🔧 Creating virtual environment in ../.venv..."
python3 -m venv ../.venv

echo "📦 Activating virtual environment..."
source ../.venv/bin/activate

echo "📚 Installing dependencies from ../requirements.txt..."
pip install --upgrade pip
pip install -r ../requirements.txt

echo "✅ Setup complete. To activate later:"
echo "   source ../.venv/bin/activate"
#!/bin/bash

# Simple setup for Daisy Risk Engine

echo "🔧 Creating virtual environment..."
python3 -m venv .venv

echo "📦 Activating virtual environment..."
source .venv/bin/activate

echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Setup complete. To activate:"
echo "   source .venv/bin/activate"

