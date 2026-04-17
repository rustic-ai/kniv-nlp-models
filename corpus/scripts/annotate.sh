#!/usr/bin/env bash
# Annotate a domain with spaCy.  Usage: ./corpus/scripts/annotate.sh conversation
set -euo pipefail

DOMAIN="${1:?Usage: $0 <domain-name>}"

cd "$(dirname "$0")/../.."

echo "Annotating domain: ${DOMAIN}"
uv run python -m corpus.pipeline.annotate --domain "${DOMAIN}"

echo "Computing statistics..."
uv run python -m corpus.pipeline.stats --domain "${DOMAIN}"
