#!/bin/bash

# Simple setup for Daisy Risk Engine (Windows)

@echo off
echo 🔧 Creating virtual environment in ..\.venv...
python -m venv ..\.venv

echo 📦 Activating virtual environment...
call ..\.venv\Scripts\activate.bat

echo 📚 Installing dependencies from ..\requirements.txt...
python -m pip install --upgrade pip
pip install -r ..\requirements.txt

echo ✅ Setup complete. To activate later:
echo     ..\.venv\Scripts\activate.bat
pause


