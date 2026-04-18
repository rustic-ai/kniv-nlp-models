---
language: en
license: apache-2.0
tags:
  - ner
  - pos
  - dependency-parsing
  - token-classification
  - multi-task
  - onnx
  - deberta
datasets:
  - conll2003
  - universal_dependencies
pipeline_tag: token-classification
---

# uniko-deberta-v3-nlp-en

Multi-task NLP model for English: NER + POS tagging + dependency parsing + sentence classification in a single forward pass.

## Model Details

- **Base:** `microsoft/deberta-v3-small` (44M parameters, 6 layers, 768 hidden)
- **Tasks:** NER (CoNLL-03), POS (UD English EWT), Dependency Parsing (UD English EWT, dep2label encoding), Sentence Classification (CLS)
- **Format:** ONNX (INT8 quantized)
- **Intended use:** Embedded NLP pipeline for the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system

## Why DeBERTa-v3-small?

Compared to DistilRoBERTa (82M params), DeBERTa-v3-small offers:
- **Half the parameters** (44M vs 82M) with the same depth (6 layers, 768 hidden)
- **Disentangled attention** — separate content and position representations improve token-level tasks
- **RTD pretraining** — ELECTRA-style replaced token detection yields stronger representations

## Architecture

Shared DeBERTa-v3-small encoder with four linear classification heads:

| Head | Labels | Training Data |
|------|--------|---------------|
| NER | 9 (BIO: PER, ORG, LOC, MISC) | CoNLL-2003 |
| POS | 17 (UPOS tagset) | UD English EWT |
| Dep | ~800-1200 (dep2label composite tags) | UD English EWT |
| CLS | 7 (statement, question, command, ...) | Bootstrap labels |

## dep2label Encoding

Dependency parsing is reformulated as token classification using the rel-pos encoding from [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/):

Each token receives a label like `+1@nsubj@VERB` meaning:
- `+1` -> head is the 1st token of matching POS to the right
- `nsubj` -> dependency relation
- `VERB` -> POS tag of the head token

Decoding back to a dependency tree is O(n) per sentence.

## Usage with ONNX Runtime (Rust)

```rust
// Load ONNX model via ort crate
let session = ort::Session::builder()?.with_model("model-int8.onnx")?;

// Tokenize input
let encoding = tokenizer.encode("Caroline went to the hospital.", true);

// Run inference — single forward pass, four outputs
let outputs = session.run(inputs)?;
let ner_logits = &outputs["ner_logits"];
let pos_logits = &outputs["pos_logits"];
let dep_logits = &outputs["dep_logits"];
let cls_logits = &outputs["cls_logits"];
```

## Training

```bash
git clone https://github.com/rustic-ai/kniv-nlp-models
cd kniv-nlp-models
./scripts/setup.sh
./scripts/train.sh deberta-v3-nlp-en
./scripts/export.sh deberta-v3-nlp-en
```

## Citation

```bibtex
@inproceedings{strzyz-etal-2019-viable,
    title = "Viable Dependency Parsing as Sequence Labeling",
    author = "Strzyz, Michalina and Vilares, David and G{\'o}mez-Rodr{\'\i}guez, Carlos",
    booktitle = "Proceedings of NAACL",
    year = "2019"
}
```
