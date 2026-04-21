---
language:
  - en
license: apache-2.0
tags:
  - ner
  - pos
  - dependency-parsing
  - token-classification
  - sequence-classification
  - multi-task
  - onnx
  - deberta-v3
  - knowledge-distillation
  - teacher-model
datasets:
  - dragonscale-ai/kniv-corpus-en
  - universal_dependencies
pipeline_tag: token-classification
model-index:
  - name: kniv-deberta-v3-large-nlp-en
    results:
      - task:
          type: token-classification
          name: NER
        metrics:
          - type: f1
            value: 0.725
            name: F1
      - task:
          type: token-classification
          name: POS
        metrics:
          - type: accuracy
            value: 0.984
            name: Accuracy
      - task:
          type: token-classification
          name: Dependency Parsing
        metrics:
          - type: accuracy
            value: 0.871
            name: UAS
      - task:
          type: text-classification
          name: Dialog Act
        metrics:
          - type: f1
            value: 0.493
            name: Macro F1
---

# kniv-deberta-v3-large-nlp-en

Multi-task NLP teacher model for English: NER + POS tagging + dependency parsing + sentence classification in a single forward pass.

Part of the [kniv-nlp-models](https://github.com/rustic-ai/kniv-nlp-models) project, powering the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system.

## Model Details

| | |
|---|---|
| **Base model** | [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) |
| **Parameters** | 435M (24 layers, 1024 hidden) |
| **Max sequence length** | 128 tokens |
| **Format** | PyTorch + ONNX (FP32 + INT8) |
| **Training data** | [kniv-corpus-en](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en) (gold-filtered) + UD English EWT |
| **License** | Apache-2.0 |
| **Use** | Server-side NLP; teacher for knowledge distillation |

## Results

| Head | Task | Metric | Score |
|------|------|--------|-------|
| NER | Named entity recognition (18 types) | F1 | **0.725** |
| POS | Part-of-speech tagging (17 UPOS) | Accuracy | **0.984** |
| DEP | Dependency parsing (dep2label) | UAS | **0.871** |
| CLS | Dialog act classification (9 labels) | Macro F1 | **0.493** |
| | | **Composite** | **0.823** |

## Architecture

Shared DeBERTa-v3-large encoder with four linear heads. One forward pass, four outputs.

```
DeBERTa-v3-large encoder (435M params, 24 layers, 1024 hidden)
  +-- NER head: Linear(1024, 37)     -- per-token BIO entity tags
  +-- POS head: Linear(1024, 17)     -- per-token UPOS tags
  +-- Dep head: Linear(1024, 1411)   -- per-token dep2label tags
  +-- CLS head: Linear(1024, 9)      -- per-sequence dialog act
```

### NER Entity Types (18)

PERSON, ORG, GPE, LOC, DATE, TIME, MONEY, PERCENT, QUANTITY, ORDINAL, CARDINAL, NORP, FAC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE

### CLS Dialog Act Labels (9)

inform, correction, agreement, question, plan_commit, request, feedback, social, filler

### dep2label Encoding

Dependencies encoded as token labels using [rel-pos](https://aclanthology.org/N19-1077/) (Strzyz et al., 2019):

```
+1@nsubj@VERB   ->  "1st VERB to the right, relation=nsubj"
-2@det@NOUN     ->  "2nd NOUN to the left, relation=det"
 0@root@ROOT    ->  "root of the sentence"
```

## Training

Trained on gold-filtered [kniv-corpus-en](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en):
- **NER**: 45,000 examples (gold-filtered, domain-balanced from 237K)
- **POS + DEP**: 12,544 examples (UD English EWT v2.14, expert-annotated)
- **CLS**: 57,544 examples (NER + UD combined, GPT-5.4-nano classified)

| Parameter | Value |
|-----------|-------|
| Batch size | 64 |
| Learning rate | 1e-5 |
| Epochs | 5 |
| Precision | fp32 (gradient checkpointing) |
| Warmup | 10% |
| Loss weights | NER: 1.0, POS: 1.0, Dep: 1.0, CLS: 0.5 |
| Hardware | NVIDIA A100 40GB |

## Usage

### Python (ONNX Runtime)

```python
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer
import json

# Load
session = ort.InferenceSession("model-int8.onnx")
tokenizer = AutoTokenizer.from_pretrained(".")
with open("label_maps.json") as f:
    labels = json.load(f)

# Tokenize
text = "Caroline went to the hospital in New York."
enc = tokenizer(text, return_tensors="np", padding="max_length", max_length=128)

# Inference (single forward pass -> 4 outputs)
outputs = session.run(None, {
    "input_ids": enc["input_ids"],
    "attention_mask": enc["attention_mask"],
})
ner_logits, pos_logits, dep_logits, cls_logits = outputs

# Decode NER
tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
ner_preds = [labels["ner_labels"][i] for i in ner_logits[0].argmax(axis=-1)]
for tok, ner in zip(tokens, ner_preds):
    if ner != "O":
        print(f"  {tok}: {ner}")
```

### Rust (ONNX Runtime)

```rust
use ort::{Session, Value};
use ndarray::Array2;
use tokenizers::Tokenizer;

let session = Session::builder()?
    .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
    .commit_from_file("model-int8.onnx")?;

let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let encoding = tokenizer.encode("Caroline went to the hospital.", true)?;

let outputs = session.run(ort::inputs![
    Array2::from_shape_vec((1, 128), encoding.get_ids().to_vec())?,
    Array2::from_shape_vec((1, 128), encoding.get_attention_mask().to_vec())?,
]?)?;

// outputs: ner_logits, pos_logits, dep_logits, cls_logits
```

## Files

| File | Size | Description |
|------|------|-------------|
| `model.onnx` | 1,663 MB | FP32 ONNX model |
| `model-int8.onnx` | 612 MB | INT8 quantized (dynamic) |
| `model.pt` | 1,670 MB | PyTorch weights |
| `label_maps.json` | <1 MB | NER/POS/DEP/CLS label vocabularies |
| `tokenizer.json` | 8 MB | DeBERTa-v3 tokenizer |

## Important: Use This Model's Tokenizer

Always load the tokenizer from this repo, **not** from `microsoft/deberta-v3-large`. The upstream HuggingFace tokenizer may omit BOS/EOS special tokens, shifting all positions and producing incorrect results.

```python
# Correct
tokenizer = AutoTokenizer.from_pretrained("dragonscale-ai/kniv-deberta-v3-large-nlp-en")

# WRONG — may omit special tokens
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
```

## Limitations

- **English only**
- **Max 128 tokens** — longer inputs truncated
- **CLS labels are GPT-classified** — not human-annotated, macro F1 reflects imbalanced rare labels
- **Server-side model** — 435M params, not for edge/mobile. Use the distilled student for that.

## Source

- **Code**: [rustic-ai/kniv-nlp-models](https://github.com/rustic-ai/kniv-nlp-models)
- **Dataset**: [dragonscale-ai/kniv-corpus-en](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en)

## Citation

```bibtex
@misc{kniv-deberta-v3-large-2026,
  title={kniv-deberta-v3-large-nlp-en: Multi-task NLP Teacher Model},
  author={Dragonscale Industries Inc.},
  year={2026},
  url={https://huggingface.co/dragonscale-ai/kniv-deberta-v3-large-nlp-en}
}
```
