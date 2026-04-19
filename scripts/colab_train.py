"""Self-contained Colab training script for kniv DeBERTa-v3-large teacher.

Run this in a Google Colab notebook with GPU/TPU runtime.
Paste into a cell or upload and run:

    !python colab_train.py

It will:
1. Install dependencies
2. Download training data from HuggingFace
3. Clone the training code from GitHub
4. Run the teacher training with HF Trainer
"""

import subprocess
import sys
import os

# ── Step 1: Install dependencies ──────────────────────────────────
print("Installing dependencies...", flush=True)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "torch", "transformers", "datasets", "sentencepiece",
    "pyyaml", "seqeval", "conllu", "accelerate",
    "huggingface_hub",
])

# ── Step 2: Download training data from HuggingFace ───────────────
print("\nDownloading training data...", flush=True)
from huggingface_hub import snapshot_download

snapshot_download(
    "dragonscale-ai/kniv-corpus-en",
    repo_type="dataset",
    allow_patterns="prepared/*",
    local_dir="kniv-data",
)
print("Training data downloaded to kniv-data/", flush=True)

# ── Step 3: Clone the repo ────────────────────────────────────────
print("\nCloning training code...", flush=True)
if not os.path.exists("kniv-nlp-models"):
    subprocess.check_call([
        "git", "clone", "--depth", "1",
        "https://github.com/rustic-ai/kniv-nlp-models.git",
    ])

# ── Step 4: Setup paths ──────────────────────────────────────────
os.chdir("kniv-nlp-models")
# Symlink the downloaded data into the expected location
os.makedirs("data/prepared", exist_ok=True)
if not os.path.exists("data/prepared/deberta-v3-large-nlp-en"):
    os.symlink(
        os.path.abspath("../kniv-data/prepared/deberta-v3-large-nlp-en"),
        "data/prepared/deberta-v3-large-nlp-en",
    )
print(f"Working directory: {os.getcwd()}", flush=True)
print(f"Data linked: {os.path.exists('data/prepared/deberta-v3-large-nlp-en')}", flush=True)

# ── Step 5: Run training ──────────────────────────────────────────
print("\n" + "=" * 60, flush=True)
print("Starting teacher training: DeBERTa-v3-large", flush=True)
print("=" * 60, flush=True)

# Add model dir to path for imports
sys.path.insert(0, "models/deberta-v3-large-nlp-en")
sys.path.insert(0, ".")

# Import and run
from models_deberta_v3_large = __import__("importlib").import_module(
    "models.deberta-v3-large-nlp-en.train"
) if False else None

# Just run the script directly
subprocess.check_call([
    sys.executable, "models/deberta-v3-large-nlp-en/train.py",
])

print("\nTraining complete!", flush=True)
