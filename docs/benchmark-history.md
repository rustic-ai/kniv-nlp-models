# Benchmark History

Internal reference of benchmark scores across kniv model generations. Captures
both deprecated experimental models and the current production cascade family,
with notes on what's directly comparable and what isn't.

## Scope

This document is a scratchpad of evaluation numbers we want to be able to look
up later — for release notes, blog posts, or to answer "did the new model
regress?" questions when they come up.

It is intentionally not a polished public artifact; numbers and caveats live
together in one place so we don't lose context.

## Generations

```
v1: distilroberta-nlp-en           (deprecated, CoNLL-3 specialist)
v2: deberta-v3-nlp-en              (deprecated, experimental small)
v3: deberta-v3-large-nlp-en        (deprecated, experimental large)
v5: kniv-deberta-nlp-base-en-large (current teacher, production)
v5 students:
    kniv-deberta-nlp-base-en-{xsmall,small,base}  (distilled, current)
```

The `kniv-deberta-nlp-base-en-*` family is the current production line. All
prior models on Hugging Face are marked `deprecated, experimental`.

## Deprecated experimental models

These were research artifacts from the multi-task NLP exploration before the
cascade design and bottom-up training methodology. Scores below are taken from
the published model-index YAML on each repo.

### kniv-distilroberta-nlp-en

- Encoder: `distilroberta-base` (82M params)
- Heads: NER, POS, DEP only
- Trained directly on the in-domain task data (CoNLL-2003 for NER, UD EWT for
  POS/DEP)
- Format: ONNX INT8 (~80 MB)

| Head | Score | Metric | Test set |
|------|-------|--------|----------|
| NER  | 0.909 | F1 | CoNLL-2003 (4 types: PER, ORG, LOC, MISC) |
| POS  | 0.976 | Accuracy | UD English EWT |
| DEP  | 0.880 / 0.866 | UAS / LAS | UD English EWT |

### kniv-deberta-v3-nlp-en

- Encoder: DeBERTa-v3 (compact)
- Multi-task experiment with NER, POS, DEP, dialog act CLS
- Did not have SRL

| Head | Score | Metric |
|------|-------|--------|
| NER | 0.730 | F1 |
| POS | 0.978 | Accuracy |
| DEP | 0.864 | UAS |
| CLS | 0.414 | Macro F1 |

### kniv-deberta-v3-large-nlp-en

- Encoder: DeBERTa-v3-large
- Larger version of the deberta-v3-nlp-en experiment
- Did not have SRL

| Head | Score | Metric |
|------|-------|--------|
| NER | 0.725 | F1 |
| POS | 0.984 | Accuracy |
| DEP | 0.871 | UAS |
| CLS | 0.493 | Macro F1 |

## Current production family — v5 cascade

Five-head cascade (POS, NER, DEP, SRL, CLS) with predicate embedding at the
encoder embedding level, ScalarMix-per-head, and POS→NER→DEP→SRL feature
cascade. All evaluated on the same standard public test sets.

NER is trained on **OntoNotes 5.0 (18 types)**. We evaluate against
CoNLL-2003 with an 18→4 type mapping for cross-comparison; numeric entities
(DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL) have no CoNLL
equivalent and are mapped to O. This is a strictly harder protocol than
training-and-testing-on-CoNLL.

### Teacher: kniv-deberta-nlp-base-en-large

- Encoder: DeBERTa-v3-large (1024d, 24 layers, ~443M params)
- Released April 2026 (v5)

| Head | Score | Metric | Test set |
|------|-------|--------|----------|
| POS | 0.977 | Accuracy | UD English EWT test |
| NER | 0.889 | F1 (micro) | OntoNotes 5.0 test |
| NER | 0.794 | F1 | CoNLL-2003 test (mapped) |
| DEP | 0.944 / 0.923 | UAS / LAS | UD English EWT test |
| SRL | 0.843 | F1 | PropBank EWT test |
| CLS | 0.951 | Macro F1 | SGD+GPT dev (8 labels, internal) |

### Student: kniv-deberta-nlp-base-en-xsmall

- Encoder: DeBERTa-v3-xsmall (384d, 12 layers, ~75M params)
- Compression ratio vs teacher: 5.9×
- Distilled with v2 pipeline (logit KL + hard CE + hidden-state MSE + R-Drop,
  per-verb teacher silver SRL, Pattern C sequential Stage 2)
- PyTorch checkpoint: 299 MB; ONNX FP32: 300 MB; ONNX INT8: 92 MB

| Head | Score | Metric | Test set |
|------|-------|--------|----------|
| POS | 0.963 | Accuracy | UD English EWT test |
| NER | 0.774 | F1 | CoNLL-2003 test (mapped) |
| DEP | 0.942 / 0.920 | UAS / LAS | UD English EWT test |
| SRL | 0.829 | F1 | PropBank EWT test |
| CLS | 0.938 | Macro F1 | SGD+GPT dev (internal) |
| CLS | 0.585 | Accuracy | DailyDialog test (mapped 8→4) |

INT8 quality drops vs FP32: POS −0.21, NER −0.99, DEP −0.03, SRL −0.84, CLS internal −0.87.

### Student: kniv-deberta-nlp-base-en-small

- Encoder: DeBERTa-v3-small (768d, 6 layers, ~157M params)
- Compression ratio vs teacher: 2.8×
- Same v2 distillation pipeline
- PyTorch checkpoint: 628 MB; ONNX FP32: 629 MB; ONNX INT8: 190 MB

| Head | Score | Metric | Test set |
|------|-------|--------|----------|
| POS | 0.970 | Accuracy | UD English EWT test |
| NER | 0.779 | F1 | CoNLL-2003 test (mapped) |
| DEP | 0.942 / 0.922 | UAS / LAS | UD English EWT test |
| SRL | 0.831 | F1 | PropBank EWT test |
| CLS | 0.947 | Macro F1 | SGD+GPT dev (internal) |
| CLS | 0.593 | Accuracy | DailyDialog test (mapped 8→4) |

INT8 quality drops vs FP32: POS −0.54, NER −2.68, DEP −0.95, SRL −1.80, CLS internal −2.58.

INT8 hits the small encoder harder than xsmall — its 768d weight rows have
more dynamic range to compress.

### Student: kniv-deberta-nlp-base-en-base

- Encoder: DeBERTa-v3-base (768d, 12 layers, ~200M params)
- Compression ratio vs teacher: 2.2×
- Status: Pattern C training in progress at time of writing (Stage 1 complete,
  Stage 2 pending)

## Runtime performance — xsmall vs small

Measured on a single-GPU workstation: NVIDIA RTX 4070 Laptop GPU (8 GB),
CUDA 13.0, ONNX Runtime 1.25.1 (CUDA-13 build), PyTorch 2.x.

The numbers below are with **CUDA EP working correctly**. Earlier runs had
ORT silently falling back to CPU because the default `onnxruntime-gpu` PyPI
wheel is built against CUDA 12 while the system has CUDA 13. The fix:

```bash
# CUDA-13 build of onnxruntime-gpu
uv pip install --reinstall onnxruntime-gpu --extra-index-url \
    https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-13/pypi/simple/

# cuDNN 9 ships in nvidia-cudnn-cu13; needs LD_LIBRARY_PATH at runtime
uv pip install nvidia-cudnn-cu13
export LD_LIBRARY_PATH="$VENV/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
```

### Latency at batch_size=1 (mean ms per call)

| Runtime | xsmall seq=64 | xsmall seq=128 | small seq=64 | small seq=128 |
|---------|---------------|----------------|--------------|---------------|
| **onnx_fp32_cuda** | **5.79** | **8.09** | **6.62** | **8.96** |
| pytorch_cuda     | 12.15  | 12.92  | 10.04  | 13.53  |
| onnx_int8_cuda   | 13.03  | 19.97  | 13.60  | 23.69  |
| onnx_int8_cpu    | 22.22  | 34.49  | 21.77  | 31.84  |
| onnx_fp32_cpu    | 24.75  | 43.14  | 29.41  | 46.88  |
| pytorch_cpu      | 129.17 | 63.51  | 80.59  | 99.38  |

### Throughput at batch_size=32 (sentences/sec)

| Runtime | xsmall seq=64 | xsmall seq=128 | small seq=64 | small seq=128 |
|---------|---------------|----------------|--------------|---------------|
| **onnx_fp32_cuda** | **1240** | 502 | **908** | 355 |
| pytorch_cuda     | 1192 | **599** | 770 | **363** |
| onnx_int8_cuda   |  181 | 131 | 184 |  91 |
| onnx_int8_cpu    |  124 |  61 |  98 |  51 |
| onnx_fp32_cpu    |   94 |  47 |  58 |  29 |
| pytorch_cpu      |   24 |  30 |  38 |  21 |

### Speedup vs pytorch_cpu (bs=1, seq=128)

| Runtime | xsmall | small |
|---|---|---|
| **onnx_fp32_cuda** | **7.85×** | **11.09×** |
| pytorch_cuda     | 4.92× | 7.35× |
| onnx_int8_cuda   | 3.18× | 4.19× |
| onnx_int8_cpu    | 1.84× | 3.12× |
| onnx_fp32_cpu    | 1.47× | 2.12× |

### Notes on the runtime numbers

- **ONNX FP32 CUDA is the fastest GPU runtime** — beats PyTorch CUDA by
  ~30-40% on latency at bs=1. ORT fuses ops in the cascade graph that
  PyTorch can't.
- **ONNX INT8 CUDA underperforms FP32 CUDA on this hardware** — modern
  Tensor Cores are FP32/FP16-optimized, INT8 paths often have higher kernel
  launch overhead. INT8 stays the right choice for **CPU** deployment, not
  GPU.
- **small at 9 ms / 908 sent/s on GPU** with FP32 is the production sweet
  spot. The 2.1× param count vs xsmall barely shows because these models
  are memory-bandwidth bound rather than compute bound at bs=1.
- **xsmall + ONNX INT8 CPU at ~22 ms** is the practical edge / embedded
  deployment option (no GPU required, 92 MB on disk).
- The earlier 7.6-second pytorch_cpu number for small (in older perf logs)
  was an outlier — likely thermal throttling. Repeat runs gave 80-99 ms,
  consistent with ~2× xsmall's 64 ms.

### Deployment picks

| Use case | Runtime | Latency (seq=128) | Throughput (bs=32 seq=128) |
|---|---|---|---|
| GPU server, low latency (xsmall) | xsmall + onnx_fp32_cuda | **8.1 ms** | 502 sent/s |
| GPU server, max quality | small + onnx_fp32_cuda | **9.0 ms** | 355 sent/s |
| GPU server, max throughput | xsmall or small + pytorch_cuda | 13–14 ms | 599 / 363 sent/s |
| Edge / embedded | xsmall + onnx_int8_cpu | 34 ms | 61 sent/s |
| CPU-only server, balanced | xsmall + onnx_int8_cpu | 34 ms | 61 sent/s |

For uniko's typical workload (single message per turn, latency-sensitive),
**small + onnx_fp32_cuda at 9 ms** is the production sweet spot — within
0.1–1.4 pts of teacher quality on every head, and faster than
pytorch_cuda by 33%.

## Old vs new — what's directly comparable

| Metric | distilroberta v1 | xsmall v5 | small v5 | Teacher v5 |
|---|---|---|---|---|
| Params | 82M | 75M | 157M | 443M |
| POS Acc (UD EWT) | 0.976 | 0.963 | 0.970 | 0.977 |
| NER F1 (CoNLL-3, 4 types) | 0.909 | 0.774 | 0.779 | 0.794 |
| DEP UAS (UD EWT) | 0.880 | **0.942** | 0.942 | 0.944 |
| DEP LAS (UD EWT) | 0.866 | **0.920** | 0.922 | 0.923 |

### What's directly comparable

- **POS** — same UD EWT test set, both unmapped. Old model is essentially tied
  with the new students at 0.97-0.98.
- **DEP** — same UD EWT test set. The new students are **+6 UAS pts and
  +5-6 LAS pts** ahead of the old distilroberta. The dep_proj + biaffine
  cascade head with POS/NER conditioning is fundamentally a better parser.

### What's NOT directly comparable (NER specifically)

The old distilroberta was **trained on CoNLL-2003** with 4 entity types and
evaluated on its in-domain test set, so 0.909 F1 reflects in-domain
specialization.

The new students are **trained on OntoNotes 5.0** with 18 entity types and
evaluated on CoNLL-2003 via type mapping (with numeric entities discarded
because CoNLL has no equivalent). This is strictly harder.

If we compare on each model's natural test distribution:
- Old distilroberta on CoNLL-3 (4 types, in-domain): 0.909 F1
- New teacher on OntoNotes (18 types, in-domain): 0.889 F1
- New teacher on CoNLL-3 (mapped 18→4, out-of-domain): 0.794 F1

The new teacher is doing harder work (4× more entity types, broader domain
coverage) and dropping 11.5 pts on the CoNLL mapping protocol. That's a
fair-deal trade for the additional generality.

## What's new in the v5 cascade family

Capabilities the deprecated models didn't have:

- **SRL** (Semantic Role Labeling, PropBank schema): 0.829-0.843 F1 across the
  family. New head, no historical comparison possible.
- **CLS** (Dialog Act Classification, 8 labels): 0.938-0.951 Macro F1 — a
  large jump from the deprecated deberta-v3 line which scored 0.41-0.49 on a
  4-label task.
- **POS→NER→DEP→SRL cascade** with detached probability features feeding
  downstream heads.
- **Predicate embedding at encoder embedding level** for SRL (every encoder
  layer is predicate-aware).
- **Bottom-up layer-selective training** for the teacher (each phase unfreezes
  a different layer range).
- **Distillation pipeline** with logit KL + hard CE + hidden-state MSE +
  R-Drop, plus per-verb teacher silver SRL data (~404K verb-level examples).

## How to update this file

When new benchmark numbers come in (e.g. when base finishes), append the new
row in the corresponding "Student" section above. Don't overwrite — keep
the trail of previous numbers for ablation reference. If a model is later
retrained or improved, add a new row with date or version suffix rather than
silently mutating the table.

Quality drop figures (INT8 vs FP32, ONNX vs PyTorch) come from
`models/<model-dir>/benchmark_results.json` and are worth recording when a
new export pass is done.
