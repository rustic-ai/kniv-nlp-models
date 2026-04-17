#!/usr/bin/env bash
# Collect raw text for a domain.  Usage: ./corpus/scripts/collect.sh conversation
set -euo pipefail

DOMAIN="${1:?Usage: $0 <domain-name>}"
DOMAIN_DIR="corpus/domains/${DOMAIN}"

if [ ! -d "${DOMAIN_DIR}" ]; then
    echo "Error: Domain directory ${DOMAIN_DIR} does not exist"
    exit 1
fi

cd "$(dirname "$0")/../.."

echo "Collecting raw text for domain: ${DOMAIN}"
uv run python "${DOMAIN_DIR}/collect.py"

echo "Preprocessing..."
uv run python "${DOMAIN_DIR}/preprocess.py"

echo "Collection complete. Output: corpus/output/raw/${DOMAIN}/sentences.jsonl"
