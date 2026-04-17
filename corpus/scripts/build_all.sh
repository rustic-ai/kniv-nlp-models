#!/usr/bin/env bash
# Build the full corpus: collect + annotate + validate + export for all domains.
set -euo pipefail

cd "$(dirname "$0")/../.."

DOMAINS=(conversation narrative technical news encyclopedic)
COMPLETED_DOMAINS=()

for domain in "${DOMAINS[@]}"; do
    echo ""
    echo "============================================"
    echo "  Domain: ${domain}"
    echo "============================================"

    if [ ! -f "corpus/domains/${domain}/collect.py" ]; then
        echo "  ⚠ No collect.py — skipping"
        continue
    fi

    echo "  Step 1: Collect..."
    ./corpus/scripts/collect.sh "${domain}" || { echo "  ✗ Collection failed"; continue; }

    echo "  Step 2: Annotate..."
    ./corpus/scripts/annotate.sh "${domain}" || { echo "  ✗ Annotation failed"; continue; }

    echo "  Step 3: Validate (submitting batch)..."
    ./corpus/scripts/validate.sh "${domain}" || { echo "  ✗ Validation submission failed"; continue; }

    COMPLETED_DOMAINS+=("${domain}")
    echo "  ✓ Domain ${domain} processed"
done

echo ""
echo "============================================"
echo "  Processed domains: ${COMPLETED_DOMAINS[*]}"
echo "============================================"
echo ""
echo "Validation batches are running asynchronously."
echo "Check status: ./corpus/scripts/validate.sh <domain> --check"
echo "When all batches complete, run:"
echo "  ./corpus/scripts/export.sh ${COMPLETED_DOMAINS[*]}"
