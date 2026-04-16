#!/usr/bin/env bash
# Export a trained model to ONNX + quantize.  Usage: ./scripts/export.sh distilroberta-nlp-en
set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model-name>}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "Exporting ${MODEL_NAME} to ONNX..."
python -m shared.export_onnx \
    --model-dir "outputs/${MODEL_NAME}" \
    --output-dir "onnx-output/${MODEL_NAME}"

echo "Export complete. ONNX files in onnx-output/${MODEL_NAME}/"
