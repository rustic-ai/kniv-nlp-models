---
language: en
license: cc-by-sa-4.0
library_name: transformers
base_model: microsoft/deberta-v3-base
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
  - knowledge-distillation
  - student-model
datasets:
  - universal-dependencies/universal_dependencies
  - dragonscale-ai/kniv-corpus-en
  - google-research-datasets/dstc8-schema-guided-dialogue
pipeline_tag: token-classification
model-index:
  - name: kniv-deberta-nlp-base-en-base
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
            value: 0.970
            name: POS Accuracy
      - task:
          type: token-classification
          name: Named Entity Recognition
        dataset:
          type: eriktks/conll2003
          name: CoNLL-2003
          split: test
        metrics:
          - type: f1
            value: 0.784
            name: F1 (mapped 18→4 types)
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
            value: 0.925
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
            value: 0.848
            name: F1
      - task:
          type: text-classification
          name: Dialog Act Classification
        dataset:
          type: schema_guided_dstc8
          name: SGD + GPT
          split: dev
        metrics:
          - type: f1
            value: 0.947
            name: Macro F1
---

# kniv-deberta-nlp-base-en-base

A multi-task NLP student model that performs the same 5 language analysis
tasks as our [production teacher][teacher] from a single
[DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) encoder
pass: POS tagging, Named Entity Recognition, Dependency Parsing, Semantic
Role Labeling, and Dialog Act Classification.

This is the **highest-quality student** in the kniv cascade family —
2.2× smaller than the teacher while approaching teacher-level accuracy on
every head. Recommended when accuracy matters more than size and the full
teacher is impractical to deploy.

Part of the [Rustic](https://rustic.ai) initiative by
[Dragonscale Industries Inc.](https://dragonscale.ai)

[teacher]: https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-large

| | |
|---|---|
| Source code | [GitHub](https://github.com/rustic-ai/kniv-nlp-models) |
| Teacher | [`kniv-deberta-nlp-base-en-large`][teacher] (443M) |
| Encoder | [DeBERTa-v3-base](https://huggingface.co/microsoft/deberta-v3-base) (768d, 12 layers) |
| Parameters | 200M (184M encoder + 16M heads) |
| Compression | 2.2× smaller than teacher |
| Download | 798 MB (PyTorch) / 799 MB (ONNX FP32) / 262 MB (ONNX INT8) |
| License | CC-BY-SA-4.0 |

## Quick Start

### ONNX

```bash
pip install torch transformers==5.6.2 onnxruntime
```

```python
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dragonscale-ai/kniv-deberta-nlp-base-en-base")
session = ort.InferenceSession("onnx/cascade.onnx")
pos, ner, arc, label, srl, cls = session.run(None, {
    "input_ids": input_ids,           # int64 [batch, seq]
    "attention_mask": attention_mask, # int64 [batch, seq]
    "predicate_idx": predicate_idx,   # int64 [batch] — verb token index (0 if unused)
})
```

### PyTorch

```bash
pip install torch transformers==5.6.2 seqeval
python examples/cascade_demo.py --model models/kniv-deberta-nlp-base-en-base
```

The [demo script](https://github.com/rustic-ai/kniv-nlp-models/blob/main/examples/cascade_demo.py)
loads the model, runs all 5 heads, and prints POS tags, NER entities,
dependency tree, SRL frames, and dialog acts.

## Benchmark Results

All benchmarks use standard public test sets. No benchmark data was used
during training. Results are reproducible via the included
[benchmark scripts](https://github.com/rustic-ai/kniv-nlp-models/tree/main/models).

| Head | Score | Metric | Benchmark | Split |
|------|-------|--------|-----------|-------|
| POS | 0.970 | Accuracy | [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/) | test |
| NER | 0.784 | F1 | [CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) (mapped) | test |
| DEP | 0.944 / 0.925 | UAS / LAS | [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/) | test |
| SRL | 0.848 | F1 | PropBank EWT | test |
| CLS | 0.947 | Macro F1 | SGD + GPT (8 labels, internal) | dev |

NER on CoNLL-2003 was evaluated by mapping our 18 OntoNotes entity types to
the 4 CoNLL types (PER, ORG, LOC, MISC). Numeric entities (DATE, MONEY,
PERCENT, QUANTITY, ORDINAL, CARDINAL, TIME) have no CoNLL equivalent and
are mapped to O — this is a strictly harder protocol than CoNLL-trained
baselines.

CLS DailyDialog cross-evaluation accuracy: 0.598 (with lossy 8→4 label
mapping; informationally lossy comparison).

```bash
# Reproduce benchmarks
python models/download_benchmarks.py
python models/student_benchmark_standard.py \
    --model-dir models/kniv-deberta-nlp-base-en-base --backend pytorch
```

## Runtime Performance

Single-call latency on NVIDIA RTX 4070 Laptop GPU (CUDA 13.0,
ONNX Runtime 1.25.1 with CUDA 13 build):

| Runtime | bs=1 seq=64 | bs=1 seq=128 | bs=32 seq=128 (sent/s) |
|---------|-------------|--------------|------------------------|
| ONNX FP32 CUDA | **7.98 ms** | **11.07 ms** | **209** |
| PyTorch CUDA | 14.47 ms | 17.70 ms | 207 |
| ONNX INT8 CPU | 37.04 ms | 53.20 ms | 28 |
| ONNX FP32 CPU | 43.69 ms | 72.33 ms | 16 |
| ONNX INT8 CUDA | 27.37 ms | 46.29 ms | 52 |
| PyTorch CPU | 106.78 ms | 148.74 ms | 11 |

**Recommended runtimes**:
- **GPU server (production sweet spot)**: ONNX FP32 CUDA at 11 ms latency, 209 sent/s
- **CPU edge / embedded**: ONNX INT8 CPU at 53 ms, 262 MB model size — but read the INT8 caveat below

INT8 quality drops vs FP32 (per-channel dynamic quantization):
POS −1.6, NER −16.5, DEP UAS/LAS −2.6/−3.5, SRL −5.8, CLS internal −14.1.
INT8 drift on the 12-layer × 768d encoder is meaningfully larger than
on smaller siblings — the deeper cascade compounds quantization error,
hitting NER and CLS hardest. For latency-critical CPU deployment where
INT8 quality is unacceptable, prefer the
[small](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-small)
variant (smaller drops at 2.8× compression) or the
[xsmall](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall)
variant (smallest drops, 5.9× compression).

## Architecture

Identical cascade structure to the teacher, with all sizes auto-scaled to
the encoder hidden dimension (768d):

```
DeBERTa-v3-base + pred_embedding
│
├─ ScalarMix(all)  → Linear(17)                              → POS
├─ ScalarMix(all)  → BiLSTM(192) → +POS probs → MLP(37)      → NER  [Viterbi]
├─ ScalarMix(all)  → BiLSTM(192) → +POS/NER probs → Biaffine → DEP
├─ ScalarMix(all)  → +pred interaction features +POS+DEP probs
│                    → BiLSTM(384) → MLP(42)                  → SRL  [Viterbi]
└─ ScalarMix(all)  → AttentionPool → MLP(8)                   → CLS
```

The architecture mirrors the teacher's design (see the
[teacher's model card][teacher] for the full ScalarMix / cascade /
predicate embedding rationale). The student auto-scales each head's internal
dimensions proportionally to the encoder width.

## Training

This model is a **distilled student** of
[`kniv-deberta-nlp-base-en-large`][teacher].

The training pipeline is two-stage:

1. **Stage 1 — Distillation** (~8 epochs): student learns from teacher's soft
   logits (KL on POS/NER/SRL/CLS, hard CE for DEP arc/relation), teacher's
   intermediate hidden states (PKD-style MSE matching), and consistency
   regularization (R-Drop).

2. **Stage 2 — Fine-tune**: SRL gold + teacher silver supervision, then
   CLS fine-tune with frozen encoder/SRL components to preserve the SRL
   peak achieved in Stage 2a.

See [`docs/design-knowledge-distillation.md`](https://github.com/rustic-ai/kniv-nlp-models/blob/main/docs/design-knowledge-distillation.md)
for the distillation methodology in full.

## Limitations

- **English only.** Encoder and training data are English.
- **Same data caveats as teacher.** NER trained on silver labels (SpanMarker);
  CLS trained on dialog data (may misclassify news/documents); SRL requires
  predicate index; DEP uses greedy decoding (no MST).
- **Requires `transformers==5.6.2`.** Other versions produce incorrect
  outputs.

## Model Family

| Model | Encoder | Params | Compression | SRL F1 |
|-------|---------|--------|-------------|--------|
| [kniv-deberta-nlp-base-en-xsmall](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall) | DeBERTa-v3-xsmall (384d, 12L) | 74.7M | 5.9× | 0.829 |
| [kniv-deberta-nlp-base-en-small](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-small) | DeBERTa-v3-small (768d, 6L) | 157.1M | 2.8× | 0.831 |
| **kniv-deberta-nlp-base-en-base** | DeBERTa-v3-base (768d, 12L) | 200M | 2.2× | 0.848 |
| [kniv-deberta-nlp-base-en-large][teacher] | DeBERTa-v3-large (1024d, 24L) | 443M | — | 0.843 |

## Citation

```bibtex
@misc{kniv-cascade-2026,
  title={kniv-deberta-nlp-base-en-base: Distilled Multi-Task NLP Cascade},
  author={Dragonscale Industries Inc.},
  year={2026},
  url={https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-base}
}
```

## License

CC-BY-SA-4.0

Built by [Dragonscale Industries Inc.](https://dragonscale.ai) | [Rustic](https://rustic.ai)
