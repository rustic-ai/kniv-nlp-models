#!/usr/bin/env bash
# Train ATLOP on Re-DocRED from a clean Colab/Azure GPU box.
#
# Everything external is pinned to a commit or a release asset so a rerun
# gets the same code and weights. The three transformers-5 shims live in
# v6/experiments/atlop_compat.py and are applied at runtime, so the ATLOP
# checkout stays pristine.
#
# RESUME: this script is safe to re-run. It resumes from OUT_DIR/latest.pt
# and reuses the cached featurisation, so a disconnect costs at most
# --save-every steps. Point OUT_DIR at mounted Drive or it dies with the VM:
#
#     from google.colab import drive; drive.mount("/content/drive")
#     OUT_DIR=/content/drive/MyDrive/atlop-run bash .../atlop_colab.sh
#
# WATCH IT (from anywhere with access to OUT_DIR, including another box):
#
#     python -m v6.experiments.atlop_status --out-dir "$OUT_DIR"
#
# It reports STALE if the heartbeat is older than --stale-minutes, so a
# hung run is distinguishable from a slow one. Exit code 1 = stale.
#
# DISK: latest.pt holds optimizer state as well as weights (~4.3 GB for
# roberta-large). best.pt is weights only.
set -euo pipefail

WORK="${WORK:-/content/atlop}"
REPO="${REPO:-/content/kniv-nlp-models}"
mkdir -p "$WORK"

# --- external code and label map ------------------------------------------
if [ ! -d "$WORK/ATLOP" ]; then
  git clone --depth 1 https://github.com/wzhouad/ATLOP.git "$WORK/ATLOP"
fi
mkdir -p "$WORK/ATLOP/meta"
# NOT in the ATLOP repo. The label ORDER must match the model's output
# indices; a wrong order fails silently rather than loudly.
curl -fsSL -o "$WORK/ATLOP/meta/rel2id.json" \
  https://raw.githubusercontent.com/tonytan48/KD-DocRE/main/meta/rel2id.json
python - <<'PY'
import json
d = json.load(open("/content/atlop/ATLOP/meta/rel2id.json"))
assert len(d) == 97 and sorted(d.values()) == list(range(97)), "rel2id malformed"
assert d.get("Na") == 0, "expected Na -> 0"
print("rel2id verified: 97 entries, contiguous, Na=0")
PY

# --- data (MIT) ------------------------------------------------------------
mkdir -p "$REPO/data/re-docred"
for s in train dev test; do
  f="$REPO/data/re-docred/${s}_revised.json"
  [ -s "$f" ] || curl -fsSL -o "$f" \
    "https://raw.githubusercontent.com/tonytan48/Re-DocRED/main/data/${s}_revised.json"
done

pip -q install opt_einsum ujson

# --- train -----------------------------------------------------------------
cd "$REPO"
OUT_DIR="${OUT_DIR:-$REPO/data/re-docred/atlop-run}"
echo "output + resume point: $OUT_DIR"
case "$OUT_DIR" in
  /content/drive/*) ;;
  *) echo "WARNING: OUT_DIR is not on mounted Drive; a VM restart loses it." ;;
esac

ATLOP_DIR="$WORK" python -m v6.experiments.atlop_train \
  --epochs "${EPOCHS:-30}" \
  --train-batch-size "${BS:-4}" \
  --eval-batch-size "${EBS:-8}" \
  --lr "${LR:-3e-5}" \
  --classifier-lr "${CLR:-1e-4}" \
  --save-every "${SAVE_EVERY:-200}" \
  --out-dir "$OUT_DIR"
rc=$?
echo "exit $rc  (130 = interrupted; re-run this script to resume)"
exit $rc
