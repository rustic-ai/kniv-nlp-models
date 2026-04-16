# uniko-nlp-models

Multi-task NLP model training for the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system.

## What This Is

Training code and tooling to produce ONNX models that run NER, POS tagging, and dependency parsing in a single forward pass. These models power uniko's entity extraction, observation extraction, and fact derivation pipelines.

**Trained models are published to HuggingFace:** [dragonscale-ai](https://huggingface.co/dragonscale-ai)

## Models

| Model | Base | Tasks | Size (INT8) | Languages |
|-------|------|-------|-------------|-----------|
| `distilroberta-nlp-en` | DistilRoBERTa | NER + POS + Dep | ~80MB | English |

## Architecture

One shared transformer encoder with three lightweight classification heads:

```
              ┌── NER head (BIO tags: PER, ORG, LOC, MISC)
Input →       │
Encoder  ─────├── POS head (UPOS: NOUN, VERB, ADJ, ...)
(shared)      │
              └── Dep head (dep2label: +1@nsubj@VERB, ...)
```

Dependency parsing uses the [dep2label](https://aclanthology.org/N19-1077/) approach (Strzyz et al., 2019) which reframes parsing as token classification — no transition-based decoder needed.

## Quick Start

```bash
# Setup
uv venv && source .venv/bin/activate
uv pip install -e .

# Download training data
./data/download_ud.sh

# Train
./scripts/train.sh distilroberta-nlp-en

# Export to ONNX + quantize
./scripts/export.sh distilroberta-nlp-en

# Publish to HuggingFace
./scripts/publish.sh distilroberta-nlp-en
```

## Training Data

| Dataset | Tasks | Source | License |
|---------|-------|--------|---------|
| [UD English EWT v2.14](https://universaldependencies.org/) | POS, Dep | Universal Dependencies | CC BY-SA 4.0 |
| [CoNLL-2003](https://www.clips.uantwerpen.be/conll2003/ner/) | NER | Reuters corpus | Research use |

## Adding a New Model

1. Create `models/{model-name}/` with the standard file structure
2. Write `config.yaml` with hyperparameters and data paths
3. Implement `prepare_data.py` for any data format conversion
4. Run `./scripts/train.sh {model-name}`
5. Run `./scripts/export.sh {model-name}`
6. Run `./scripts/publish.sh {model-name}`

## License

Apache-2.0
