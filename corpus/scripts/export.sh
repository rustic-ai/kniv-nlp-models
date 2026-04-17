#!/usr/bin/env bash
# Export merged corpus.  Usage: ./corpus/scripts/export.sh conversation narrative technical
set -euo pipefail

DOMAINS="${@:?Usage: $0 <domain1> <domain2> ...}"

cd "$(dirname "$0")/../.."

echo "Exporting corpus from domains: ${DOMAINS}"
uv run python -m corpus.pipeline.export --domains ${DOMAINS}

echo "Final corpus statistics:"
uv run python -m corpus.pipeline.stats --final
