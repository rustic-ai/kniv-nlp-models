#!/usr/bin/env bash
# Download Universal Dependencies English EWT treebank v2.14
# License: CC BY-SA 4.0
# Contains: POS tags, morphological features, dependency annotations
# ~16K sentences, ~254K tokens

set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")" && pwd)"
UD_DIR="${DATA_DIR}/ud-english-ewt"

if [ -d "${UD_DIR}" ]; then
    echo "UD English EWT already downloaded at ${UD_DIR}"
    exit 0
fi

echo "Downloading UD English EWT v2.14..."
mkdir -p "${UD_DIR}"
cd "${UD_DIR}"

curl -L -o ud-english-ewt.zip \
    "https://github.com/UniversalDependencies/UD_English-EWT/archive/refs/tags/r2.14.zip"

unzip -q ud-english-ewt.zip
mv UD_English-EWT-r2.14/* .
rmdir UD_English-EWT-r2.14
rm ud-english-ewt.zip

echo "Downloaded to ${UD_DIR}"
echo "Files:"
ls -la *.conllu
echo ""
echo "Train: $(grep -c '^# sent_id' en_ewt-ud-train.conllu) sentences"
echo "Dev:   $(grep -c '^# sent_id' en_ewt-ud-dev.conllu) sentences"
echo "Test:  $(grep -c '^# sent_id' en_ewt-ud-test.conllu) sentences"
