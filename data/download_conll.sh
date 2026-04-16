#!/usr/bin/env bash
# CoNLL-2003 NER dataset
#
# NOTE: The original CoNLL-2003 shared task data requires a Reuters corpus
# license. However, the dataset is available via HuggingFace Datasets
# which handles the download automatically in the training script.
#
# This script documents the data source and provides manual download
# instructions if needed.

set -euo pipefail

echo "CoNLL-2003 NER Dataset"
echo "======================"
echo ""
echo "The CoNLL-2003 dataset is loaded automatically via HuggingFace Datasets"
echo "in the training script (prepare_data.py):"
echo ""
echo '  from datasets import load_dataset'
echo '  dataset = load_dataset("conll2003")'
echo ""
echo "Entity types: PER, ORG, LOC, MISC"
echo "~20K sentences (train: 14K, dev: 3.3K, test: 3.5K)"
echo ""
echo "No manual download needed."
