---
language: en
license: apache-2.0
library_name: onnxruntime
tags:
  - ner
  - pos
  - dependency-parsing
  - token-classification
  - sequence-classification
  - multi-task
  - onnx
  - int8
  - quantized
  - deberta
  - deberta-v3
  - knowledge-distillation
  - edge
  - embedded
datasets:
  - dragonscale-ai/kniv-corpus-en
  - universal_dependencies
  - rjac/kaggle-entity-annotated-corpus-ner-dataset
pipeline_tag: token-classification
model-index:
  - name: kniv-deberta-v3-nlp-en
    results:
      - task:
          type: token-classification
          name: NER
        metrics:
          - type: f1
            value: 0.730
            name: F1
      - task:
          type: token-classification
          name: POS
        metrics:
          - type: accuracy
            value: 0.978
            name: Accuracy
      - task:
          type: token-classification
          name: Dependency Parsing
        metrics:
          - type: accuracy
            value: 0.864
            name: UAS
      - task:
          type: text-classification
          name: Dialog Act
        metrics:
          - type: f1
            value: 0.414
            name: Macro F1
---

# kniv-deberta-v3-nlp-en

Multi-task NLP model for English that performs **NER**, **POS tagging**, **dependency parsing**, and **sentence classification** in a single forward pass. Designed for edge and embedded deployment as the core linguistic analysis layer of the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system.

This is the **student** model, trained via knowledge distillation from [kniv-deberta-v3-large-nlp-en](https://huggingface.co/dragonscale-ai/kniv-deberta-v3-large-nlp-en) (435M parameter teacher). Shipped as an INT8-quantized ONNX model for efficient CPU inference.

## Model Details

| Property | Value |
|---|---|
| **Published by** | [Dragonscale Industries Inc.](https://huggingface.co/dragonscale-ai) |
| **GitHub** | [rustic-ai/kniv-nlp-models](https://github.com/rustic-ai/kniv-nlp-models) |
| **Base model** | [microsoft/deberta-v3-small](https://huggingface.co/microsoft/deberta-v3-small) |
| **Parameters** | 142M (6 layers, 768 hidden) |
| **Teacher** | kniv-deberta-v3-large-nlp-en (435M, DeBERTa-v3-large) |
| **Format** | ONNX (INT8 quantized) |
| **Max sequence length** | 128 tokens |
| **License** | Apache-2.0 |

### Why DeBERTa-v3-small?

Compared to DistilRoBERTa (82M parameters), DeBERTa-v3-small offers:

- **Smaller transformer body** (44M non-embedding params vs 82M) at the same depth (6 layers, 768 hidden); 142M total with 128K-token SentencePiece vocabulary
- **Disentangled attention** -- separate content and position representations improve token-level tasks like NER and dependency parsing
- **RTD pretraining** -- ELECTRA-style replaced token detection yields stronger contextual representations from fewer parameters

## Results

### Dev set (best epoch 10, 500-sample eval)

| Head | Task | Metric | Score |
|------|------|--------|-------|
| NER | Named entity recognition (18 types) | F1 | **0.730** |
| POS | Part-of-speech tagging (17 UPOS) | Accuracy | **0.978** |
| DEP | Dependency parsing (dep2label) | UAS | **0.864** |
| CLS | Dialog act classification (9 labels) | Macro F1 | **0.414** |
| | | **Composite** | **0.813** |

### External benchmarks

| Benchmark | Metric | Student | Teacher |
|-----------|--------|---------|---------|
| OntoNotes 5.0 | NER F1 | **0.752** | 0.731 |
| UD EWT | POS Accuracy | 0.979 | **0.980** |
| UD EWT | DEP UAS | 0.866 | **0.871** |
| UD EWT | DEP LAS | 0.844 | **0.848** |
| DailyDialog | CLS Accuracy | **0.701** | 0.669 |

The student matches or exceeds the teacher on NER and CLS external benchmarks despite being 3x smaller, likely due to regularization benefits of the smaller architecture and longer distillation training (10 epochs vs 5).

## Architecture

Shared DeBERTa-v3-small encoder with four linear classification heads. One encoder forward pass produces all four outputs.

```
DeBERTa-v3-small encoder (shared, 142M params)
  +-- NER head: Linear(768, 37)     -- per-token BIO entity tags
  +-- POS head: Linear(768, 17)     -- per-token UPOS tags
  +-- Dep head: Linear(768, 1411)   -- per-token dep2label composite tags
  +-- CLS head: Linear(768, 9)      -- per-sequence dialog act (from [CLS] token)
```

### Task heads

| Head | Output dim | Label scheme | Training data |
|---|---|---|---|
| **NER** | 37 (BIO over 18 entity types) | BIO tagging | kniv corpus + GMB |
| **POS** | 17 (UPOS tagset) | Flat labels | UD English EWT |
| **Dep** | 1411 (dep2label composite tags) | Composite labels | UD English EWT |
| **CLS** | 9 (dialog act) | Flat labels | GPT-5.4-nano classified |

### NER entity types (18)

The NER head uses the 18-type entity scheme from spaCy `en_core_web_trf`, encoded in BIO format (37 labels = 1 `O` + 18 x 2 `B-`/`I-`):

| Entity type | Description |
|---|---|
| `PERSON` | People, including fictional |
| `NORP` | Nationalities, religious, or political groups |
| `FAC` | Facilities -- buildings, airports, highways, bridges |
| `ORG` | Companies, agencies, institutions |
| `GPE` | Geopolitical entities -- countries, cities, states |
| `LOC` | Non-GPE locations -- mountain ranges, bodies of water |
| `PRODUCT` | Objects, vehicles, foods, etc. (not services) |
| `EVENT` | Named hurricanes, battles, wars, sports events |
| `WORK_OF_ART` | Titles of books, songs, etc. |
| `LAW` | Named documents made into laws |
| `LANGUAGE` | Any named language |
| `DATE` | Absolute or relative dates or periods |
| `TIME` | Times smaller than a day |
| `PERCENT` | Percentage, including `%` |
| `MONEY` | Monetary values, including unit |
| `QUANTITY` | Measurements (weight, distance, etc.) |
| `ORDINAL` | "first", "second", etc. |
| `CARDINAL` | Numerals that are not another type |

### CLS dialog act labels (9)

| Label | Description |
|---|---|
| `inform` | Declarative factual content |
| `correction` | Correcting prior information |
| `agreement` | Agreement or confirmation |
| `question` | Genuine question |
| `plan_commit` | Stating intent or commitment |
| `request` | Imperative or request |
| `feedback` | Reaction or backchannel |
| `social` | Greetings, closings, politeness |
| `filler` | Discourse markers, hesitations |

## dep2label encoding

Dependency parsing is reformulated as token classification using the **rel-pos** encoding from [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/). Each token receives a composite label encoding its head attachment, making dependency parsing compatible with the same multi-task token classification architecture used for NER and POS.

Each label has the format `{signed_offset}@{relation}@{head_UPOS}`:

```
+1@nsubj@VERB
```

This means:
- `+1` -- the head is the **1st token of matching POS to the right**
- `nsubj` -- the **dependency relation** to the head
- `VERB` -- the **UPOS tag of the head** token

Decoding back to a dependency tree is **O(n)** per sentence. The label vocabulary is constructed from the training data and typically contains 800--1200 unique composite tags for UD English EWT.

## Knowledge distillation

The student model is trained using a combined loss:

```
L = alpha * L_hard + (1 - alpha) * T^2 * KL(softmax(z_s/T) || softmax(z_t/T))
```

| Parameter | Value |
|---|---|
| Teacher | kniv-deberta-v3-large-nlp-en (DeBERTa-v3-large, 435M params) |
| Temperature (T) | 3.0 |
| Alpha | 0.5 (equal weight to hard labels and soft labels) |
| Hard loss | Cross-entropy on ground truth annotations |
| Soft loss | KL divergence on temperature-scaled teacher logits |

Soft labels are generated per-task by running the trained teacher over the full training set. The student learns from both signals for all four task heads simultaneously.

## Training data

| Dataset | Tasks | License | Notes |
|---|---|---|---|
| [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/) | POS, Dep, CLS | CC BY-SA 4.0 | Universal Dependencies English Web Treebank |
| kniv corpus | NER, CLS | Mixed open | spaCy + GPT validated annotations, 18 entity types |
| [GMB (Groningen Meaning Bank)](https://huggingface.co/datasets/rjac/kaggle-entity-annotated-corpus-ner-dataset) | NER | CC BY 4.0 | Human-corrected annotations, mapped to 18-type scheme |

GMB's 8 entity types are mapped to the 18-type scheme: `per`->`PERSON`, `org`->`ORG`, `gpe`->`GPE`, `geo`->`LOC`, `tim`->`DATE`, `art`->`PRODUCT`, `eve`->`EVENT`, `nat`->`EVENT`.

This model does **not** use CoNLL-2003 data.

## Training configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max grad norm | 1.0 |
| Batch size | 16 |
| Epochs | 10 (with early stopping, patience=3) |
| Dropout | 0.1 |
| NER/POS/Dep loss weight | 1.0 |
| CLS loss weight | 0.5 |

## Intended use

This model is designed for **edge and embedded NLP** as part of the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system. It provides the core linguistic analysis pipeline -- entity recognition, part-of-speech tagging, syntactic parsing, and sentence classification -- in a single efficient forward pass.

**Primary use cases:**
- On-device NLP for privacy-sensitive applications
- Real-time text analysis in resource-constrained environments
- Linguistic feature extraction for downstream cognitive memory operations

**Out-of-scope uses:**
- High-stakes decision-making without human review
- Languages other than English
- Documents exceeding 128 tokens without chunking

## Usage with ONNX Runtime

### Rust (ort crate)

```rust
use ort::{Session, Value};
use tokenizers::Tokenizer;

// Load model and tokenizer
let session = Session::builder()?
    .with_model_from_file("model-int8.onnx")?;
let tokenizer = Tokenizer::from_file("tokenizer.json")?;

// Tokenize
let encoding = tokenizer.encode("Caroline went to the hospital.", true)?;
let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
let attention_mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();

// Run inference -- single forward pass, four outputs
let outputs = session.run(ort::inputs![input_ids, attention_mask]?)?;
let ner_logits = outputs["ner_logits"].extract_tensor::<f32>()?;
let pos_logits = outputs["pos_logits"].extract_tensor::<f32>()?;
let dep_logits = outputs["dep_logits"].extract_tensor::<f32>()?;
let cls_logits = outputs["cls_logits"].extract_tensor::<f32>()?;

// Argmax over last dimension to get predicted label indices
```

### Python (onnxruntime)

```python
import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np

session = ort.InferenceSession("model-int8.onnx")
tokenizer = AutoTokenizer.from_pretrained("dragonscale-ai/kniv-deberta-v3-nlp-en")

encoding = tokenizer(
    "Caroline went to the hospital.",
    max_length=128, padding="max_length", truncation=True,
    return_tensors="np",
)

outputs = session.run(None, {
    "input_ids": encoding["input_ids"],
    "attention_mask": encoding["attention_mask"],
})

ner_logits, pos_logits, dep_logits, cls_logits = outputs
ner_preds = np.argmax(ner_logits, axis=-1)
pos_preds = np.argmax(pos_logits, axis=-1)
dep_preds = np.argmax(dep_logits, axis=-1)
cls_pred = np.argmax(cls_logits, axis=-1)
```

## Limitations

- **English only.** No multilingual support.
- **128-token context window.** Longer documents require sentence-level chunking before inference.
- **NER coverage.** Entity types are limited to the 18 spaCy categories. Domain-specific entities (e.g., biomedical, legal terminology) are not covered.
- **CLS labels are GPT-classified.** Dialog act labels are generated by GPT-5.4-nano, not human-annotated. Macro F1 reflects imbalanced rare labels (correction, filler).
- **dep2label decoding errors.** The rel-pos encoding can fail to reconstruct a valid tree if the predicted head POS does not exist in the expected direction. Such tokens receive a fallback head of -1.
- **Quantization trade-off.** INT8 quantization reduces model size and latency but may degrade accuracy by up to 0.5% on individual tasks.

## Training and export

```bash
git clone https://github.com/rustic-ai/kniv-nlp-models
cd kniv-nlp-models

# 1. Train teacher
python models/deberta-v3-large-nlp-en/train.py

# 2. Generate soft labels from teacher
python models/deberta-v3-large-nlp-en/generate_soft_labels.py \
    --model-dir outputs/deberta-v3-large-nlp-en/best

# 3. Prepare student data
python models/deberta-v3-nlp-en/prepare_data.py

# 4. Distill student from teacher
python models/deberta-v3-nlp-en/distill.py \
    --soft-labels outputs/deberta-v3-large-nlp-en/soft_labels

# 5. Export to ONNX (INT8 quantized)
./scripts/export.sh deberta-v3-nlp-en
```

## Citation

If you use this model, please cite the dep2label encoding:

```bibtex
@inproceedings{strzyz-etal-2019-viable,
    title = "Viable Dependency Parsing as Sequence Labeling",
    author = "Strzyz, Michalina and Vilares, David and G{\'o}mez-Rodr{\'\i}guez, Carlos",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N19-1077",
}
```

The DeBERTa-v3 base model:

```bibtex
@inproceedings{he2021debertav3,
    title = "{D}e{BERT}a{V}3: Improving {D}e{BERT}a using {ELECTRA}-Style Pre-Training with Gradient-Disentangled Embedding Sharing",
    author = "He, Pengcheng and Gao, Jianfeng and Chen, Weizhu",
    booktitle = "International Conference on Learning Representations",
    year = "2023",
    url = "https://openreview.net/forum?id=sE7-XhLxHA",
}
```

Knowledge distillation approach:

```bibtex
@article{hinton2015distilling,
    title = "Distilling the Knowledge in a Neural Network",
    author = "Hinton, Geoffrey and Vinyals, Oriol and Dean, Jeff",
    journal = "arXiv preprint arXiv:1503.02531",
    year = "2015",
}
```
