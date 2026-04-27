---
language: en
license: cc-by-sa-4.0
library_name: transformers
base_model: microsoft/deberta-v3-large
tags:
  - multi-task
  - pos
  - ner
  - dependency-parsing
  - semantic-role-labeling
  - dialog-act-classification
  - deberta-v3
  - cascade
  - nlp
  - rustic
datasets:
  - universal-dependencies/universal_dependencies
  - dragonscale-ai/kniv-corpus-en
  - google-research-datasets/dstc8-schema-guided-dialogue
pipeline_tag: token-classification
model-index:
  - name: kniv-deberta-nlp-base-en-large
    results:
      - task:
          type: token-classification
          name: POS Tagging
        dataset:
          type: universal-dependencies/universal_dependencies
          name: UD English EWT
          split: test
        metrics:
          - type: accuracy
            value: 0.977
      - task:
          type: token-classification
          name: Named Entity Recognition
        dataset:
          type: tner/ontonotes5
          name: OntoNotes 5.0
          split: test
        metrics:
          - type: f1
            value: 0.889
            name: F1 (micro)
      - task:
          type: token-classification
          name: Named Entity Recognition
        dataset:
          type: eriktks/conll2003
          name: CoNLL-2003
          split: test
        metrics:
          - type: f1
            value: 0.794
            name: F1 (mapped 18 to 4 types)
      - task:
          type: token-classification
          name: Dependency Parsing
        dataset:
          type: universal-dependencies/universal_dependencies
          name: UD English EWT
          split: test
        metrics:
          - type: accuracy
            value: 0.944
            name: UAS
          - type: accuracy
            value: 0.923
            name: LAS
      - task:
          type: token-classification
          name: Semantic Role Labeling
        dataset:
          type: conll2012_ontonotesv5
          name: PropBank EWT
          split: test
        metrics:
          - type: f1
            value: 0.843
      - task:
          type: text-classification
          name: Dialog Act Classification
        dataset:
          type: schema_guided_dstc8
          name: SGD + GPT
          split: dev
        metrics:
          - type: f1
            value: 0.951
            name: Macro F1
---

# kniv-deberta-nlp-base-en-large v5

A multi-task NLP model that performs 5 language analysis tasks from a
single [DeBERTa-v3-large](https://huggingface.co/microsoft/deberta-v3-large)
encoder pass: POS tagging, Named Entity Recognition, Dependency Parsing,
Semantic Role Labeling, and Dialog Act Classification.

Part of the [Rustic](https://rustic.ai) initiative by
[Dragonscale Industries Inc.](https://dragonscale.ai)

| | |
|---|---|
| Source code | [GitHub](https://github.com/rustic-ai/kniv-nlp-models) |
| Training data | [dragonscale-ai/kniv-corpus-en](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en) |
| Demo | [`examples/cascade_demo.py`](https://github.com/rustic-ai/kniv-nlp-models/blob/main/examples/cascade_demo.py) |
| Parameters | 443M (434M encoder + 9.5M heads) |
| Download | 1.74 GB (PyTorch) / 1.78 GB (ONNX FP32) / 654 MB (ONNX INT8) |
| License | CC-BY-SA-4.0 |

## Quick Start

### ONNX

```bash
pip install torch transformers==5.6.2 onnxruntime
python models/kniv-deberta-nlp-base-en-large/export_onnx.py
```

```python
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dragonscale-ai/kniv-deberta-nlp-base-en-large")
session = ort.InferenceSession("onnx/cascade.onnx")
pos, ner, arc, label, srl, cls = session.run(None, {
    "input_ids": input_ids,           # int64 [batch, seq]
    "attention_mask": attention_mask,  # int64 [batch, seq]
    "predicate_idx": predicate_idx,   # int64 [batch] — verb token index (0 if unused)
})
```

### PyTorch

```bash
pip install torch transformers==5.6.2 seqeval
python examples/cascade_demo.py --model models/kniv-deberta-nlp-base-en-large
```

The [demo script](https://github.com/rustic-ai/kniv-nlp-models/blob/main/examples/cascade_demo.py)
is self-contained — it loads the model, runs all 5 heads, and prints
POS tags, NER entities, dependency tree, SRL frames, and dialog acts.

## Benchmark Results

All benchmarks use standard public test sets. No benchmark data was used
during training. Results are reproducible via the included
[benchmark scripts](https://github.com/rustic-ai/kniv-nlp-models/tree/main/models/kniv-deberta-nlp-base-en-large).

| Head | Score | Metric | Benchmark | Split |
|------|-------|--------|-----------|-------|
| POS | 0.977 | Accuracy | [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/) | test |
| NER | 0.889 | F1 (micro) | [OntoNotes 5.0](https://huggingface.co/datasets/tner/ontonotes5) | test |
| DEP | 0.944 / 0.923 | UAS / LAS | [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/) | test |
| SRL | 0.843 | F1 | PropBank EWT | test |
| CLS | 0.951 | Macro F1 | SGD + GPT (8 labels) | dev |

NER was also evaluated on [CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003)
(F1 = 0.794) with entity type mapping (18 OntoNotes types mapped to 4
CoNLL types; numeric entities like DATE and CARDINAL have no CoNLL
equivalent and are mapped to O).

CLS was cross-evaluated on [DailyDialog](https://huggingface.co/datasets/daily_dialog)
(accuracy = 0.613) with lossy 8-to-4 label mapping.

```bash
# Reproduce benchmarks
python models/kniv-deberta-nlp-base-en-large/download_benchmarks.py
python models/kniv-deberta-nlp-base-en-large/benchmark_standard.py
```

## Architecture

One encoder, five task heads. The encoder runs once and produces 25
hidden state tensors (embedding layer + 24 transformer layers). Each
head selects its optimal layers via a learned ScalarMix and applies a
task-specific classifier. A predicate embedding is added at the
embedding level for SRL, so all heads — including SRL — share the
same encoder pass.

```
DeBERTa-v3-large + pred_embedding
│
├─ ScalarMix(0-8)  → Linear(17)                              → POS
├─ ScalarMix(5-10) → BiLSTM(256) → +POS probs → MLP(37)     → NER  [Viterbi]
├─ ScalarMix(12-18)→ +POS/NER probs → Biaffine(arc+label)    → DEP
├─ last_hidden     → MLP(42)                                  → SRL  [Viterbi]
└─ ScalarMix(all)  → AttentionPool → MLP(8)                   → CLS
```

**ScalarMix**: Learned softmax-weighted average of all 25 encoder layers.
Each head discovers which layers are most useful — POS reads lower layers
(morphosyntax), DEP reads mid-high layers (structure), SRL reads the top
(semantics).

**Cascade**: POS probabilities feed into NER; POS + NER probabilities
feed into DEP. Upstream outputs are detached (no gradient flow), so heads
train independently but benefit from upstream predictions at inference.

**Predicate embedding**: `Embedding(2, 1024)` added at the encoder
embedding level. Marks one token as the SRL predicate before all 24
attention layers, so the encoder produces predicate-aware representations
for SRL without a separate forward pass.

**Decoding**: POS, DEP, and CLS use argmax. NER and SRL use Viterbi
decoding with constrained BIO transitions (I-X can only follow B-X or
I-X of the same type).

## Training

### Bottom-Up Layer-Selective Training

The encoder is shaped progressively from lower to upper layers:

| Phase | Task | Layers Unfrozen | Encoder LR | Epochs | Data |
|-------|------|----------------|------------|--------|------|
| 1 | POS | All (warm-up) | 2e-5 | 10 | 12.5K [UD EWT](https://universaldependencies.org/treebanks/en_ewt/) gold |
| 2 | NER | 5-12 | 2e-5 / 5e-5 | 15 | 195K SpanMarker silver |
| 3 | DEP | 12-18 | 2e-5 | 20 | 12.5K UD EWT gold |
| 4 | SRL | All | 3e-6 | 3 | 200K AllenNLP silver + 41K [PropBank](https://github.com/propbank/propbank-release) gold |
| 5 | CLS | Frozen | — | 5 | 60K [SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) + GPT |

After all phases, POS/NER/DEP heads are retrained for 3 epochs on the
frozen final encoder to recover from minor forgetting.

## ONNX Export

```bash
python models/kniv-deberta-nlp-base-en-large/export_onnx.py
```

Exports `cascade.onnx` (FP32) and `cascade-int8.onnx` (INT8 quantized)
with all 5 heads. One call returns all 6 output tensors.

| Variant | Size | Speed | Quality |
|---------|------|-------|---------|
| `cascade.onnx` (FP32) | 1,776 MB | baseline | reference |
| `cascade-int8.onnx` (INT8) | 654 MB | ~2x faster | -0.2% POS accuracy |

| Input | Shape | Description |
|-------|-------|-------------|
| `input_ids` | [B, S] | Tokenized input (int64) |
| `attention_mask` | [B, S] | Padding mask (int64) |
| `predicate_idx` | [B] | SRL predicate token index (int64, 0 if unused) |

| Output | Shape | Description |
|--------|-------|-------------|
| `pos_logits` | [B, S, 17] | POS tag scores |
| `ner_logits` | [B, S, 37] | NER BIO tag scores |
| `arc_scores` | [B, S, S] | DEP head selection scores |
| `label_scores` | [B, S, S, 53] | DEP relation label scores |
| `srl_logits` | [B, S, 42] | SRL BIO tag scores |
| `cls_logits` | [B, 8] | Dialog act scores |

## Labels

| Head | Count | Tags |
|------|-------|------|
| POS | 17 | ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X |
| NER | 37 | BIO tags for: PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL |
| DEP | 53 | UD relations: root, nsubj, obj, obl, amod, det, case, conj, cc, advmod, aux, mark, xcomp, ccomp, acl, advcl, ... |
| SRL | 42 | BIO tags for: V, ARG0-4, ARGM-TMP, ARGM-LOC, ARGM-MNR, ARGM-CAU, ARGM-PRP, ARGM-NEG, ARGM-ADV, ARGM-DIR, ARGM-DIS, ARGM-EXT, ARGM-MOD, ARGM-PRD, ARGM-GOL, ARGM-COM, ARGM-REC |
| CLS | 8 | inform, request, question, confirm, reject, offer, social, status |

## Limitations

- **English only.** Encoder and training data are English.
- **NER trained on silver labels.** Performance may degrade on domains far from the training corpus.
- **CLS trained on dialog.** Optimized for conversational text; may misclassify news or documents.
- **SRL requires predicate index.** The demo identifies predicates via POS=VERB, which misses nominal predicates.
- **DEP uses greedy decoding.** No MST constraint — output may not form a valid tree in all cases.
- **Requires transformers==5.6.2.** Other versions produce incorrect outputs.

## Model Family

| Model | Heads | Status |
|-------|-------|--------|
| **kniv-deberta-nlp-base-en-large** | POS, NER, DEP, SRL, CLS | Released |
| kniv-deberta-nlp-tier1-en-large | +Lemma, Morph, Keyword | Planned |
| kniv-deberta-nlp-tier2-en-large | +Sentiment, Intent, Punct, Truecase | Planned |
| kniv-deberta-nlp-tier3-en-large | +QA, NLI, RelEx, Events, STS | Planned |
| kniv-deberta-nlp-base-en-base | Distilled student | Planned |

## Citation

```bibtex
@misc{kniv-cascade-2026,
  title={kniv-deberta-nlp-base-en-large: Multi-Task NLP Cascade on DeBERTa-v3},
  author={Dragonscale Industries Inc.},
  year={2026},
  url={https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-large}
}
```

## License

CC-BY-SA-4.0

Built by [Dragonscale Industries Inc.](https://dragonscale.ai) | [Rustic](https://rustic.ai)
