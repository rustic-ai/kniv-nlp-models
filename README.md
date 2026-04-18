# uniko-nlp-models

Multi-task NLP model training and LLM domain adaptation for the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system.

**Published models:** [dragonscale-ai on HuggingFace](https://huggingface.co/dragonscale-ai)

## Models

### NLP Models (Token Classification)

Single forward pass → NER + POS + dependency parsing + sentence classification.

| Model | Base | Params | Method | Use case |
|-------|------|--------|--------|----------|
| `deberta-v3-large-nlp-en` | DeBERTa-v3-large | 304M | Direct training | Server-side, highest accuracy |
| `deberta-v3-nlp-en` | DeBERTa-v3-small | 44M | Distilled from large | Edge/embedded, uniko runtime |

### LLM Models (Domain-Adapted Reasoning)

| Model | Base | Params | Method | Use case |
|-------|------|--------|--------|----------|
| `phi4-mini-llm-en` | Phi-4-mini-reasoning | 3.8B | LoRA fine-tuning | Business domain reasoning |

## Architecture

### NLP: Shared encoder + 4 task heads

```
              ┌── NER head  (BIO tags: PER, ORG, LOC, MISC)
              │
Encoder  ─────├── POS head  (UPOS: NOUN, VERB, ADJ, ...)
(shared)      │
              ├── Dep head  (dep2label: +1@nsubj@VERB, ...)
              │
              └── CLS head  (intent: statement, question, command, ...)
```

Dependency parsing uses [dep2label](https://aclanthology.org/N19-1077/) (Strzyz et al., 2019) — parsing reframed as token classification.

### Knowledge Distillation

```
DeBERTa-v3-large (teacher, 304M)
    │
    ├── Publish as server-side model
    │
    └── Generate soft labels
            │
            ▼
DeBERTa-v3-small (student, 44M)
    │   trained on hard labels + teacher soft labels
    │   (KL divergence + cross-entropy)
    │
    └── Publish as edge model
```

### LLM: LoRA domain adaptation

```
Phi-4-mini-reasoning (3.8B, frozen)
    │
    └── LoRA adapters (rank 16, ~50MB)
            trained on business corpus (full documents)
```

## Corpus

Multi-domain corpus with GPT-5.4-mini validation of spaCy annotations.

| Domain | Sentences | Sources | Status |
|--------|-----------|---------|--------|
| Conversation | 100K | DailyDialog | Validated |
| Narrative | 52K | Project Gutenberg | Validated |
| Business | 198K | SEC EDGAR, Enron, OpenStax, Odoo, Wikipedia, CUAD, OpenAlex | Validated |
| Technical | 193K | Wikipedia (CS), Python docs | Collected |
| News | 58K | Wikinews, Wikipedia | Collected |
| Encyclopedic | 66K | Wikipedia (general knowledge) | Collected |

Pipeline: `collect → preprocess → annotate (spaCy) → validate (GPT-5.4-mini) → export`

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# Download training data
./data/download_ud.sh

# ── NLP Model Training ──

# Train teacher
./scripts/train.sh deberta-v3-large-nlp-en

# Generate soft labels
uv run python models/deberta-v3-large-nlp-en/generate_soft_labels.py \
    --model-dir outputs/deberta-v3-large-nlp-en/best

# Train student via distillation
uv run python models/deberta-v3-nlp-en/distill.py \
    --soft-labels outputs/deberta-v3-large-nlp-en/soft_labels

# Export to ONNX + quantize
./scripts/export.sh deberta-v3-nlp-en

# Publish to HuggingFace
./scripts/publish.sh deberta-v3-nlp-en

# ── LLM Training ──

uv pip install -e ".[llm]"
uv run python models/phi4-mini-llm-en/prepare_data.py
uv run python models/phi4-mini-llm-en/train.py

# ── Corpus Building ──

uv pip install -e ".[corpus]"
uv run python -m corpus.domains.business.collect
uv run python -m corpus.domains.business.preprocess
uv run python -m corpus.pipeline.annotate --domain business
uv run python -m corpus.pipeline.validate --domain business
```

## What Gets Published

### To GitHub (this repo)
- Training code, model configs, corpus pipelines
- No weights, no data, no outputs

### To HuggingFace ([dragonscale-ai](https://huggingface.co/dragonscale-ai))
- ONNX models (INT8 quantized) + tokenizers + model cards
- `uniko-deberta-v3-large-nlp-en` — server-side NLP
- `uniko-deberta-v3-nlp-en` — edge NLP (distilled)
- `uniko-phi4-mini-llm-en` — business domain LLM (LoRA adapters)

## Training Data

| Dataset | Tasks | License |
|---------|-------|---------|
| [UD English EWT v2.14](https://universaldependencies.org/) | POS, Dep | CC BY-SA 4.0 |
| [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) | NER | Research use |
| [Custom corpus](corpus/) | All tasks | Mixed (see domain READMEs) |

## Project Structure

```
models/
  deberta-v3-large-nlp-en/   # NLP teacher (304M)
  deberta-v3-nlp-en/          # NLP student (44M, distilled)
  phi4-mini-llm-en/           # LLM domain adaptation
  distilroberta-nlp-en/       # Legacy reference model

corpus/
  domains/                     # Per-domain collect + preprocess
    business/                  # 7 sources, 1.2M raw sentences
    conversation/              # DailyDialog
    narrative/                 # Project Gutenberg
    technical/                 # Wikipedia CS + Python docs
    news/                      # Wikinews + Wikipedia
    encyclopedic/              # Wikipedia general knowledge
  pipeline/                    # annotate → validate → export

shared/                        # ONNX export, HF publish, evaluation
scripts/                       # Shell wrappers for train/export/publish
```

## License

Apache-2.0
