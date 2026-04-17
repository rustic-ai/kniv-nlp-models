"""Shared configuration for the corpus pipeline."""

from pathlib import Path

# Paths
CORPUS_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = CORPUS_ROOT / "output"
RAW_DIR = OUTPUT_DIR / "raw"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"
VALIDATED_DIR = OUTPUT_DIR / "validated"
FINAL_DIR = OUTPUT_DIR / "final"

# spaCy teacher model
SPACY_MODEL = "en_core_web_trf"

# OpenAI validation model
OPENAI_MODEL = "gpt-5.4-mini"
VALIDATION_BATCH_SIZE = 50  # sentences per API call
VALIDATION_PERCENT = 100    # validate 100% of corpus

# Corpus parameters
MIN_SENTENCE_LENGTH = 5     # words
MAX_SENTENCE_LENGTH = 150   # words
MIN_SENTENCE_CHARS = 15     # characters
