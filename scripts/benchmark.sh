#!/usr/bin/env bash
# Benchmark inference latency.  Usage: ./scripts/benchmark.sh distilroberta-nlp-en
set -euo pipefail

MODEL_NAME="${1:?Usage: $0 <model-name>}"
ONNX_DIR="onnx-output/${MODEL_NAME}"

cd "$(dirname "$0")/.."
source .venv/bin/activate

echo "=========================================="
echo "Latency Benchmark: ${MODEL_NAME}"
echo "=========================================="

for variant in "model.onnx" "model-int8.onnx"; do
    ONNX_FILE="${ONNX_DIR}/${variant}"
    if [ -f "${ONNX_FILE}" ]; then
        echo ""
        echo "--- ${variant} ---"
        python -c "
import onnxruntime as ort
import numpy as np
import time
from transformers import AutoTokenizer
from pathlib import Path

tokenizer = AutoTokenizer.from_pretrained('outputs/${MODEL_NAME}/final')
session = ort.InferenceSession('${ONNX_FILE}')

texts = [
    'Caroline went to the hospital yesterday.',
    'The server runs on port 9090 and handles HTTP requests from the load balancer.',
    'Did you know that the adoption process requires a home study before they can proceed with the application?',
    'Hey!',
    'I prefer Python over Java for data science work.',
]

# Warmup
for t in texts:
    enc = tokenizer(t, max_length=128, padding='max_length', truncation=True, return_tensors='np')
    session.run(None, {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']})

# Benchmark
latencies = []
for _ in range(20):
    for t in texts:
        enc = tokenizer(t, max_length=128, padding='max_length', truncation=True, return_tensors='np')
        start = time.perf_counter()
        session.run(None, {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask']})
        latencies.append((time.perf_counter() - start) * 1000)

latencies.sort()
print(f'  Sentences: {len(latencies)}')
print(f'  Mean:  {np.mean(latencies):.1f}ms')
print(f'  P50:   {np.percentile(latencies, 50):.1f}ms')
print(f'  P95:   {np.percentile(latencies, 95):.1f}ms')
print(f'  P99:   {np.percentile(latencies, 99):.1f}ms')
print(f'  Min:   {min(latencies):.1f}ms')
print(f'  Max:   {max(latencies):.1f}ms')
"
    else
        echo "⚠ ${ONNX_FILE} not found"
    fi
done
