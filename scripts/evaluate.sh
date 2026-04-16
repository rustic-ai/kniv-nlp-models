#!/usr/bin/env bash
# Evaluate a trained model.  Usage: ./scripts/evaluate.sh distilroberta-nlp-en
set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model-name>}"
MODEL_DIR="models/${MODEL_NAME}"
OUTPUT_DIR="outputs/${MODEL_NAME}/final"
ONNX_DIR="onnx-output/${MODEL_NAME}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=========================================="
echo "Evaluation: ${MODEL_NAME}"
echo "=========================================="

# 1. PyTorch model evaluation
if [ -d "${OUTPUT_DIR}" ]; then
    echo ""
    echo "--- PyTorch model ---"
    python "${MODEL_DIR}/evaluate_model.py" --model-dir "${OUTPUT_DIR}"
else
    echo "⚠ No trained model at ${OUTPUT_DIR}. Skipping PyTorch evaluation."
fi

# 2. ONNX model evaluation
ONNX_FILE="${ONNX_DIR}/model-int8.onnx"
if [ -f "${ONNX_FILE}" ]; then
    echo ""
    echo "--- ONNX model (INT8) ---"
    python "${MODEL_DIR}/evaluate_onnx.py" \
        --onnx-path "${ONNX_FILE}" \
        --tokenizer-dir "${OUTPUT_DIR}" \
        --pytorch-results "${OUTPUT_DIR}/eval_results.json"
else
    echo "⚠ No ONNX model at ${ONNX_FILE}. Skipping ONNX evaluation."
fi

# 3. Baseline comparison
echo ""
echo "--- Baselines ---"
python "${MODEL_DIR}/evaluate_baseline.py" || echo "⚠ Baseline evaluation failed (spaCy may not be installed)"

echo ""
echo "Evaluation complete."
