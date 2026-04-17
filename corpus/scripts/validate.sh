#!/usr/bin/env bash
# Validate annotations with GPT-5.4 mini.  Usage: ./corpus/scripts/validate.sh conversation [--check|--retrieve]
set -euo pipefail

DOMAIN="${1:?Usage: $0 <domain-name> [--check|--retrieve]}"
shift
EXTRA_ARGS="${@:-}"

cd "$(dirname "$0")/../.."

echo "Validation for domain: ${DOMAIN}"
uv run python -m corpus.pipeline.validate --domain "${DOMAIN}" ${EXTRA_ARGS}
