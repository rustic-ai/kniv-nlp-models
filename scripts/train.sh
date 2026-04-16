#!/usr/bin/env bash
# Train a model.  Usage: ./scripts/train.sh distilroberta-nlp-en
set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model-name>}"
MODEL_DIR="models/${MODEL_NAME}"

if [ ! -d "${MODEL_DIR}" ]; then
    echo "Error: Model directory ${MODEL_DIR} does not exist"
    exit 1
fi

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "Preparing data for ${MODEL_NAME}..."
python "${MODEL_DIR}/prepare_data.py"

echo "Training ${MODEL_NAME}..."
python "${MODEL_DIR}/train.py"

echo "Training complete. Model saved to outputs/${MODEL_NAME}/"
