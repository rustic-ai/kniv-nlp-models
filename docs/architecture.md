# Architecture

This document describes the system architecture of kniv-nlp-models — the NLP and LLM training framework for the [uniko](https://github.com/rustic-ai/uniko) cognitive memory system.

## Overview

The project produces two types of models from a shared multi-domain corpus:

```
                        ┌─────────────────────────────────┐
                        │         Multi-Domain Corpus      │
                        │  conversation | narrative        │
                        │  business | technical            │
                        │  news | encyclopedic             │
                        └────────┬──────────────┬──────────┘
                                 │              │
                    sentence-split           full documents
                                 │              │
                    ┌────────────▼──┐    ┌──────▼────────────┐
                    │  NLP Models   │    │    LLM Models      │
                    │  (encoder +   │    │    (LoRA domain     │
                    │   task heads) │    │     adaptation)     │
                    └──────────────┘    └────────────────────┘
```

## NLP Models: Multi-Task Token Classification

### Architecture

A shared transformer encoder feeds four lightweight classification heads. One forward pass produces all four outputs.

```
Input text: "Apple reported $394B revenue in Q4."
                    │
                    ▼
         ┌──────────────────┐
         │  Shared Encoder   │
         │  (DeBERTa-v3)     │
         │  frozen or        │
         │  fine-tuned        │
         └──┬──┬──┬──┬──────┘
            │  │  │  │
    ┌───────┘  │  │  └────────┐
    ▼          ▼  ▼           ▼
┌───────┐ ┌──────┐ ┌──────┐ ┌──────┐
│  NER  │ │ POS  │ │ Dep  │ │ CLS  │
│ head  │ │ head │ │ head │ │ head │
└───┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
    │        │        │        │
    ▼        ▼        ▼        ▼
 B-ORG    PROPN   +1@nsubj  statement
   O      VERB     0@root
 B-MONEY  NUM    -1@obj
   ...     ...     ...
```

### Task Heads

| Head | Output | Labels | Training Data |
|------|--------|--------|---------------|
| **NER** | Per-token BIO tags | 37 (BIO for 18 entity types: PERSON, ORG, GPE, ...) | kniv corpus |
| **POS** | Per-token UPOS tags | 17 (NOUN, VERB, ADJ, ...) | UD English EWT |
| **Dep** | Per-token dep2label | ~800-1200 composite tags | UD English EWT |
| **CLS** | Per-sentence class | 7 (statement, question, command, ...) | Bootstrap labels |

Each head is a single `Linear(hidden_size, num_labels)` layer — no extra complexity.

### dep2label Encoding

Dependency parsing is reframed as token classification using the rel-pos scheme from [Strzyz et al. (2019)](https://aclanthology.org/N19-1077/).

Each token receives a label encoding three pieces of information:

```
+1@nsubj@VERB
 │   │     │
 │   │     └── POS tag of the head token
 │   └──────── dependency relation
 └──────────── signed offset (1st VERB to the right)
```

**Encoding** (tree → labels): For each token, find its head, compute the relative position among tokens of the same POS, and concatenate with the relation and head POS.

**Decoding** (labels → tree): For each token, parse the label, scan in the indicated direction counting tokens of the target POS, and resolve to an absolute head index. O(n) per sentence.

### Model Variants

| Model | Params | Encoder | Purpose |
|-------|--------|---------|---------|
| `deberta-v3-large-nlp-en` | 304M | `microsoft/deberta-v3-large` | Teacher / server-side |
| `deberta-v3-nlp-en` | 44M | `microsoft/deberta-v3-small` | Student / edge (distilled) |
| `distilroberta-nlp-en` | 82M | `distilroberta-base` | Legacy reference |

## Knowledge Distillation

The small student model (44M) is trained using knowledge distillation from the large teacher (304M), producing near-teacher accuracy at a fraction of the size.

### Process

```
Step 1: Train Teacher
    DeBERTa-v3-large (304M) trained on hard labels
    → Best checkpoint saved

Step 2: Generate Soft Labels
    Teacher runs inference on all training data
    → Saves raw logits (pre-softmax) for all 4 heads
    → NER logits, POS logits, Dep logits, CLS logits

Step 3: Train Student with Distillation
    DeBERTa-v3-small (44M) trained on combined loss:
    
    loss = α × CE(student, hard_labels)
         + (1-α) × KL(student/T, teacher/T) × T²
    
    where T = temperature, α = hard loss weight
```

### Why Temperature Matters

At temperature T=1 (standard softmax), the teacher's output for a token might be:
```
NOUN: 0.92, PROPN: 0.05, VERB: 0.03
```
The dominant class overwhelms — the student barely learns from the minority classes.

At temperature T=3, the same logits become:
```
NOUN: 0.45, PROPN: 0.30, VERB: 0.25
```
Now the student can learn the inter-class relationships — "this is probably NOUN but has PROPN characteristics." This is the "dark knowledge" that makes distillation work.

### Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `temperature` | 3.0 | 2-6 | Higher = softer distributions, more knowledge transfer |
| `alpha` | 0.5 | 0.3-0.7 | Higher = more weight on hard labels |

## LLM Models: Domain-Adapted Reasoning

### Architecture

LoRA (Low-Rank Adaptation) adapters are inserted into the base model's attention and MLP layers. The base model weights are frozen; only the small adapter matrices are trained.

```
Base Model (Phi-4-mini-reasoning, 3.8B, frozen)
    │
    ├── Self-Attention
    │   ├── q_proj ← LoRA(r=16)
    │   ├── k_proj ← LoRA(r=16)
    │   ├── v_proj ← LoRA(r=16)
    │   └── o_proj ← LoRA(r=16)
    │
    └── MLP
        ├── gate_proj ← LoRA(r=16)
        ├── up_proj   ← LoRA(r=16)
        └── down_proj ← LoRA(r=16)

Trainable parameters: ~50MB (vs 7GB for full model)
```

### Training Data

Unlike the NLP models which use sentence-split annotated data, the LLM uses **full documents** — preserving discourse structure, paragraph flow, and cross-sentence reasoning.

Documents are tokenized and chunked into fixed-length sequences (4096 tokens) with a sliding window overlap (128 tokens) for context continuity.

### Supported Base Models

The pipeline is base-model-agnostic. Change `model.base` in config.yaml:

| Model | Params | License | Strength |
|-------|--------|---------|----------|
| `microsoft/Phi-4-mini-reasoning` | 3.8B | MIT | Best reasoning under 4B |
| `Qwen/Qwen3-4B` | 4B | Apache 2.0 | Thinking mode toggle |
| `google/gemma-4-E4B` | 4B eff. | Apache 2.0 | MoE, multimodal |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 7B | MIT | Strongest small reasoner |

## Deployment

### NLP Models

Exported to ONNX and quantized to INT8 for deployment:

```
PyTorch model (304M / 44M)
    → ONNX export (opset 14, dynamic axes)
    → INT8 dynamic quantization
    → ~80MB / ~25MB on disk
    → Deploy via ONNX Runtime (Python, Rust, C++)
```

### LLM Models

LoRA adapters are merged into the base model for deployment:

```python
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, adapter_path)
merged = model.merge_and_unload()  # full model with adapted weights
```
