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
  - universal_dependencies
  - rjac/kaggle-entity-annotated-corpus-ner-dataset
pipeline_tag: token-classification
model-index:
  - name: kniv-deberta-v3-large-nlp-en
    results: []
---

# kniv-deberta-v3-large-nlp-en

Multi-task NLP teacher model for English: NER + POS tagging + dependency parsing + sentence classification in a single forward pass. Built for server-side inference and as the teacher in a knowledge distillation pipeline for the smaller `kniv-deberta-v3-nlp-en` student model.

Part of the [kniv-nlp-models](https://github.com/dragonscale-ai/kniv-nlp-models) project, powering the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system.

## Model Details

| | |
|---|---|
| **Published by** | [dragonscale-ai](https://huggingface.co/dragonscale-ai) |
| **Base model** | [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) |
| **Parameters** | 304M (24 layers, 1024 hidden dimension) |
| **Max sequence length** | 128 tokens |
| **Format** | ONNX (INT8 quantized) |
| **License** | Apache-2.0 |
| **Intended use** | Server-side NLP pipeline; teacher model for knowledge distillation |

## Architecture

Shared DeBERTa-v3-large encoder with four linear classification heads. One encoder forward pass produces all four outputs simultaneously.

```
DeBERTa-v3-large encoder (304M params, 24 layers, 1024 hidden)
  +-- NER head: Linear(1024, 37)   -- per-token BIO entity tags
  +-- POS head: Linear(1024, 17)   -- per-token UPOS tags
  +-- Dep head: Linear(1024, ~800-1200)  -- per-token dep2label tags
  +-- CLS head: Linear(1024, 7)    -- per-sequence sentence type
```

| Head | Task | Labels | Label scheme | Training data | License |
|------|------|--------|--------------|---------------|---------|
| NER | Named entity recognition | 37 (BIO tags for 18 entity types) | BIO: PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL | kniv corpus + GMB | Mixed open / CC BY 4.0 |
| POS | Part-of-speech tagging | 17 (Universal POS tagset) | ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X | UD English EWT | CC BY-SA 4.0 |
| Dep | Dependency parsing | ~800-1200 (dep2label composite tags) | rel-pos encoding (see below) | UD English EWT | CC BY-SA 4.0 |
| CLS | Sentence classification | 7 | statement, question, question_fact, command, greeting, filler, acknowledgment | Bootstrap labels from UD EWT + kniv corpus | -- |

Token-level heads (NER, POS, Dep) operate on every hidden state in the sequence. The CLS head operates on the first token (`[CLS]`) hidden state only.

## dep2label Encoding

Dependency parsing is reformulated as token classification using the **rel-pos** encoding from [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/). This eliminates the need for a separate graph-based parser -- the dependency tree is predicted as a label per token, just like NER or POS.

Each token receives a composite label with three parts:

```
{signed_offset}@{relation}@{head_UPOS}
```

| Component | Meaning | Example |
|-----------|---------|---------|
| `signed_offset` | How many tokens of the matching POS to count left (-) or right (+) to find the head | `+1`, `-2` |
| `relation` | Dependency relation (UD deprel) | `nsubj`, `det`, `root` |
| `head_UPOS` | POS tag of the head token | `VERB`, `NOUN`, `ROOT` |

**Examples:**

- `+1@nsubj@VERB` -- head is the 1st VERB to the right, relation is nsubj
- `-2@det@NOUN` -- head is the 2nd NOUN to the left, relation is det
- `0@root@ROOT` -- this token is the root of the sentence

Decoding back to a full dependency tree is O(n) per sentence: scan in the indicated direction, count tokens matching the head POS, and stop at the offset-th match.

The label vocabulary size (typically 800-1200 unique tags) is determined by the training data and depends on the combinations of offsets, relations, and head POS tags actually observed in UD English EWT.

## Training Data

| Dataset | Description | License |
|---------|-------------|---------|
| [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/) | Universal Dependencies English Web Treebank -- POS, dependency, and sentence text for CLS bootstrap | CC BY-SA 4.0 |
| kniv corpus | Validated NER annotations (spaCy + GPT validated, 18 entity types) from the kniv corpus pipeline | Mixed open licenses |
| [GMB](https://huggingface.co/datasets/rjac/kaggle-entity-annotated-corpus-ner-dataset) | Groningen Meaning Bank -- human-corrected NER annotations mapped to the 18-type spaCy scheme | CC BY 4.0 |

This model does **not** use CoNLL-2003 data.

## Training

### Prerequisites

```bash
git clone https://github.com/dragonscale-ai/kniv-nlp-models
cd kniv-nlp-models
pip install -e .
```

### Data preparation

```bash
python models/deberta-v3-large-nlp-en/prepare_data.py
```

### Train the teacher

```bash
python models/deberta-v3-large-nlp-en/train.py
```

Training configuration (from `config.yaml`):

| Parameter | Value |
|-----------|-------|
| Encoder | `microsoft/deberta-v3-large` |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.1 |
| Weight decay | 0.01 |
| Max gradient norm | 1.0 |
| Epochs | 10 (early stopping, patience 3) |
| Loss weights | NER: 1.0, POS: 1.0, Dep: 1.0, CLS: 0.5 |

The composite metric for best-checkpoint selection is: `0.3 * NER_F1 + 0.3 * POS_Acc + 0.3 * Dep_UAS + 0.1 * CLS_F1`.

### Generate soft labels for distillation

After training, generate soft targets for the student model:

```bash
python models/deberta-v3-large-nlp-en/generate_soft_labels.py \
    --model-dir outputs/deberta-v3-large-nlp-en/best
```

### Export to ONNX

```bash
python models/deberta-v3-large-nlp-en/evaluate_onnx.py
```

## Usage with ONNX Runtime (Rust)

```rust
use ort::{Session, Value};
use ndarray::Array2;
use tokenizers::Tokenizer;

// Load the INT8 quantized ONNX model
let session = Session::builder()?
    .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
    .commit_from_file("model-int8.onnx")?;

// Load label maps from label_maps.json
let label_maps: LabelMaps = serde_json::from_reader(
    std::fs::File::open("label_maps.json")?
)?;

// Tokenize
let tokenizer = Tokenizer::from_file("tokenizer.json")?;
let encoding = tokenizer.encode("Caroline went to the hospital.", true)?;
let input_ids = encoding.get_ids();
let attention_mask = encoding.get_attention_mask();

// Build input tensors
let ids = Array2::from_shape_vec(
    (1, input_ids.len()),
    input_ids.iter().map(|&x| x as i64).collect(),
)?;
let mask = Array2::from_shape_vec(
    (1, attention_mask.len()),
    attention_mask.iter().map(|&x| x as i64).collect(),
)?;

// Single forward pass -> four outputs
let outputs = session.run(ort::inputs![ids, mask]?)?;

let ner_logits  = outputs["ner_logits"].try_extract_tensor::<f32>()?;
let pos_logits  = outputs["pos_logits"].try_extract_tensor::<f32>()?;
let dep_logits  = outputs["dep_logits"].try_extract_tensor::<f32>()?;
let cls_logits  = outputs["cls_logits"].try_extract_tensor::<f32>()?;

// Argmax per token for NER predictions
let ner_preds: Vec<&str> = ner_logits
    .slice(ndarray::s![0, .., ..])
    .outer_iter()
    .map(|row| {
        let idx = row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        label_maps.ner_labels[idx].as_str()
    })
    .collect();

// Decode dep2label tags back to a dependency tree
let dep_preds: Vec<&str> = dep_logits
    .slice(ndarray::s![0, .., ..])
    .outer_iter()
    .map(|row| {
        let idx = row.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap().0;
        label_maps.dep_labels[idx].as_str()
    })
    .collect();
// dep2label::decode_sentence(&dep_preds, &pos_preds) -> (heads, deprels)
```

## Limitations

- **English only.** This model was trained exclusively on English data and will not generalize to other languages.
- **Max 128 tokens.** Inputs longer than 128 subword tokens are truncated. This is sufficient for most single sentences but not for paragraph-level text.
- **CLS labels are bootstrapped.** Sentence classification labels were generated by rule-based heuristics, not human annotation. CLS accuracy may be lower than the other heads and the label distribution is skewed toward "statement."
- **dep2label requires POS.** Decoding dependency trees from dep2label tags requires predicted POS tags as input. POS errors propagate into dependency errors.
- **NER entity types limited to 18.** The model uses spaCy's 18-type entity scheme. Entity types outside this set (e.g., medical, scientific) are not covered.
- **Server-side model.** At 304M parameters, this model is not suitable for edge or mobile deployment. Use the distilled `kniv-deberta-v3-nlp-en` (44M params) student model for resource-constrained environments.
- **No coreference or relation extraction.** The model handles token-level and sentence-level classification only.

## Citation

```bibtex
@inproceedings{strzyz-etal-2019-viable,
    title = "Viable Dependency Parsing as Sequence Labeling",
    author = "Strzyz, Michalina and Vilares, David and G{\'o}mez-Rodr{\'\i}guez, Carlos",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2019",
    url = "https://aclanthology.org/N19-1077/",
}
```

```bibtex
@inproceedings{he2021debertav3,
    title = "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing",
    author = "He, Pengcheng and Gao, Jianfeng and Chen, Weizhu",
    booktitle = "International Conference on Learning Representations",
    year = "2023",
    url = "https://arxiv.org/abs/2111.09543",
}
```
