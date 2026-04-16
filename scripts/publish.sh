#!/usr/bin/env bash
# Publish a model to HuggingFace Hub.  Usage: ./scripts/publish.sh distilroberta-nlp-en
set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model-name>}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "Publishing ${MODEL_NAME} to HuggingFace..."
python -m shared.hf_publish \
    --model-name "${MODEL_NAME}" \
    --onnx-dir "onnx-output/${MODEL_NAME}" \
    --org dragonscale-ai

echo "Published to https://huggingface.co/dragonscale-ai/uniko-${MODEL_NAME}"
