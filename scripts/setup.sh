#!/usr/bin/env bash
# Set up the development environment.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Creating virtual environment..."
uv venv

echo "Installing dependencies..."
uv pip install -e ".[dev]"

echo "Downloading training data..."
chmod +x data/download_ud.sh data/download_conll.sh
./data/download_ud.sh

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
