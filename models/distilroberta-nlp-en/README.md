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
datasets:
  - conll2003
  - universal_dependencies
pipeline_tag: token-classification
---

# kniv-distilroberta-nlp-en

Multi-task NLP model for English: NER + POS tagging + dependency parsing in a single forward pass.

## Model Details

- **Base:** `distilroberta-base` (82M parameters)
- **Tasks:** NER (CoNLL-03), POS (UD English EWT), Dependency Parsing (UD English EWT, dep2label encoding)
- **Format:** ONNX (INT8 quantized, ~80MB)
- **Intended use:** Embedded NLP pipeline for the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system

## Architecture

Shared DistilRoBERTa encoder with three linear classification heads:

| Head | Labels | Training Data |
|------|--------|---------------|
| NER | 9 (BIO: PER, ORG, LOC, MISC) | CoNLL-2003 |
| POS | 17 (UPOS tagset) | UD English EWT |
| Dep | ~800-1200 (dep2label composite tags) | UD English EWT |

## dep2label Encoding

Dependency parsing is reformulated as token classification using the rel-pos encoding from [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/):

Each token receives a label like `+1@nsubj@VERB` meaning:
- `+1` → head is the 1st token of matching POS to the right
- `nsubj` → dependency relation
- `VERB` → POS tag of the head token

Decoding back to a dependency tree is O(n) per sentence.

## Usage with ONNX Runtime (Rust)

```rust
// Load ONNX model via ort crate
let session = ort::Session::builder()?.with_model("model-int8.onnx")?;

// Tokenize input
let encoding = tokenizer.encode("Caroline went to the hospital.", true);

// Run inference — single forward pass, three outputs
let outputs = session.run(inputs)?;
let ner_logits = &outputs["ner_logits"];
let pos_logits = &outputs["pos_logits"];
let dep_logits = &outputs["dep_logits"];
```

## Training

```bash
git clone https://github.com/rustic-ai/kniv-nlp-models
cd kniv-nlp-models
./scripts/setup.sh
./scripts/train.sh distilroberta-nlp-en
./scripts/export.sh distilroberta-nlp-en
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
