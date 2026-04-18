# Training Workflows

This document covers all training workflows for the kniv-nlp-models project: multi-task NLP model training, knowledge distillation, LLM domain adaptation, and ONNX export.

## Model Overview

| Model | Encoder | Params | Role |
|-------|---------|--------|------|
| `deberta-v3-large-nlp-en` | DeBERTa-v3-large | 304M | Teacher (NLP) |
| `deberta-v3-nlp-en` | DeBERTa-v3-small | 44M | Student (NLP) |
| `phi4-mini-llm-en` | Phi-4-mini-reasoning | 3.8B | LLM domain adaptation |

The NLP models are multi-task with four heads sharing a single encoder:

- **NER** -- 18 entity types (BIO scheme) from kniv corpus + GMB
- **POS** -- 17 UPOS tags from UD English EWT
- **Dep** -- dependency parsing via dep2label encoding
- **CLS** -- 7-class sentence intent classification

## 1. Multi-Task NLP Training

### Data Preparation

Each model variant has its own `prepare_data.py` that produces JSON files under `data/prepared/<model>/`.

```bash
uv run python models/deberta-v3-large-nlp-en/prepare_data.py
uv run python models/deberta-v3-nlp-en/prepare_data.py
```

Data sources:

- **UD English EWT** (`data/ud-english-ewt/`) -- provides POS tags, dependency heads/relations, and sentence text for CLS bootstrapping.
- **kniv corpus** (`corpus/output/annotated/`) -- NER annotations from spaCy + GPT validation, split by document into train/dev/test (80/10/10).
- **GMB** (Groningen Meaning Bank, loaded from HuggingFace) -- additional human-corrected NER data. GMB's 8 entity types are mapped to the 18-type spaCy scheme.

Preparation steps:

1. Load UD EWT CoNLL-U files and extract words, UPOS, heads, deprels.
2. Convert dependency trees to dep2label format (`{signed_offset}@{relation}@{head_UPOS}`).
3. Load kniv corpus NER annotations and convert spans to BIO tags.
4. Load GMB NER data and map entity types to the unified scheme.
5. Bootstrap CLS labels using rule-based heuristics (greetings, questions, commands, etc.).
6. Merge NER sources and save `label_vocabs.json` with all label vocabularies.

Output files:

```
data/prepared/<model>/
  ner_train.json, ner_dev.json, ner_test.json
  ud_train.json, ud_dev.json, ud_test.json
  label_vocabs.json
```

### Training

```bash
uv run python models/deberta-v3-large-nlp-en/train.py   # teacher
uv run python models/deberta-v3-nlp-en/train.py          # student (standalone)
```

Training uses task alternation: each epoch iterates through all four task dataloaders round-robin, processing one batch per task per cycle until all dataloaders are exhausted.

Key hyperparameters (from `config.yaml`):

| Parameter | Teacher (large) | Student (small) |
|-----------|----------------|-----------------|
| Batch size | 8 | 16 |
| Learning rate | 2e-5 | 3e-5 |
| Epochs | 10 | 10 |
| Warmup ratio | 0.1 | 0.1 |
| Max sequence length | 128 | 128 |

**Loss weights** control the contribution of each task to the total loss:

- NER: 1.0, POS: 1.0, Dep: 1.0, CLS: 0.5

Token-level tasks (NER, POS, Dep) use `CrossEntropyLoss(ignore_index=-100)` with subword alignment -- only the first subword token of each word receives a label. The CLS task uses standard `CrossEntropyLoss` on the `[CLS]` token representation.

### Evaluation and Checkpointing

Dev-set evaluation runs after every epoch, computing:

- **NER**: entity-level precision, recall, F1 (seqeval)
- **POS**: token-level accuracy
- **Dep**: UAS and LAS (decoded from dep2label predictions)
- **CLS**: macro F1

Best checkpoint selection uses a **composite metric**:

```
composite = 0.3 * NER_F1 + 0.3 * POS_Acc + 0.3 * Dep_UAS + 0.1 * CLS_F1
```

**Early stopping** halts training after 3 consecutive epochs without improvement on the composite metric.

Output structure:

```
outputs/<model>/
  best/          -- best checkpoint by composite metric
  final/         -- last epoch checkpoint
  epoch-N/       -- per-epoch checkpoints (if save_every_epoch: true)
  training_history.json
```

## 2. Knowledge Distillation

Knowledge distillation transfers the teacher's learned representations to the smaller student model. The full pipeline has three stages.

### Step 1: Train the Teacher

```bash
uv run python models/deberta-v3-large-nlp-en/prepare_data.py
uv run python models/deberta-v3-large-nlp-en/train.py
```

### Step 2: Generate Soft Labels

The teacher runs inference on all training data and saves raw logits (pre-softmax) for each task head:

```bash
uv run python models/deberta-v3-large-nlp-en/generate_soft_labels.py \
    --model-dir outputs/deberta-v3-large-nlp-en/best
```

Output:

```
outputs/deberta-v3-large-nlp-en/soft_labels/
  ner_logits.pt    -- [N, seq_len, num_ner_labels]
  pos_logits.pt    -- [N, seq_len, num_pos_labels]
  dep_logits.pt    -- [N, seq_len, num_dep_labels]
  cls_logits.pt    -- [N, num_cls_labels]
  metadata.json    -- label maps, counts, model info
```

### Step 3: Train the Student with Distillation

```bash
uv run python models/deberta-v3-nlp-en/prepare_data.py
uv run python models/deberta-v3-nlp-en/distill.py \
    --soft-labels outputs/deberta-v3-large-nlp-en/soft_labels \
    --temperature 3.0 \
    --alpha 0.5
```

The student learns from two signals simultaneously:

1. **Hard labels** -- ground truth annotations via cross-entropy loss
2. **Soft labels** -- teacher logits via KL divergence with temperature scaling

The combined distillation loss for each task is:

```
L = alpha * CE(student, hard_labels) + (1 - alpha) * KL(student/T, teacher/T) * T^2
```

- **Temperature** (`--temperature`, default 3.0) -- softens probability distributions, revealing inter-class relationships the teacher learned. Higher values produce softer distributions.
- **Alpha** (`--alpha`, default 0.5) -- balances hard and soft loss. At 0.5, both signals contribute equally.
- **T^2 scaling** -- compensates for the gradient magnitude reduction from temperature scaling (per Hinton et al.).

For token-level tasks, KL divergence is masked to exclude padding tokens (where `labels == -100`).

The student is evaluated on the same dev sets as the teacher using hard-label metrics, enabling direct comparison. Distilled outputs are saved under `outputs/deberta-v3-nlp-en/distilled/`.

## 3. LLM Domain Adaptation

### Data Preparation

The LLM pipeline uses full documents (not sentence-split) to preserve discourse structure and cross-paragraph context:

```bash
uv run python models/phi4-mini-llm-en/prepare_data.py
```

Source documents are loaded from the business corpus:

- SEC Edgar filings (10-K annual reports)
- Enron email corpus
- OpenStax business textbooks
- Odoo ERP documentation
- Wikipedia business articles
- CUAD contracts
- S2ORC academic abstracts

Documents are tokenized and chunked into 4096-token sequences with 128-token sliding window overlap. The dev set is split at the document level (5%), not the chunk level, so evaluation measures generalization to unseen documents.

Output:

```
data/prepared/phi4-mini-llm-en/
  train.jsonl    -- pre-tokenized chunks (input_ids)
  dev.jsonl
  meta.json      -- token counts, chunk counts, model info
```

### Training with LoRA

```bash
uv run python models/phi4-mini-llm-en/train.py
uv run python models/phi4-mini-llm-en/train.py --resume outputs/phi4-mini-llm-en/checkpoint-1000
```

Training uses LoRA (Low-Rank Adaptation) via the `peft` library. Only adapter weights are trained; the base model is frozen.

LoRA configuration:

| Parameter | Value |
|-----------|-------|
| Rank (r) | 16 |
| Alpha | 32 |
| Dropout | 0.05 |
| Target modules | q/k/v/o_proj, gate/up/down_proj |

Training configuration:

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 8 (effective batch = 32) |
| Learning rate | 2e-4 |
| Scheduler | cosine |
| Precision | bfloat16 |
| Eval/save interval | every 500 steps |

The trainer uses HuggingFace `Trainer` with `DataCollatorForLanguageModeling` (causal LM, not masked). Best checkpoint is selected by eval loss, with the 3 most recent checkpoints retained.

Switching base models -- change `model.base` in `models/phi4-mini-llm-en/config.yaml`:

```yaml
model:
  base: "Qwen/Qwen3-4B"           # or google/gemma-4-E4B, deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
```

Output:

```
outputs/phi4-mini-llm-en/
  checkpoint-500/, checkpoint-1000/, ...  -- intermediate checkpoints
  final/                                  -- LoRA adapters (~50MB) + tokenizer + config
```

To merge adapters into the base model for deployment:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("microsoft/Phi-4-mini-reasoning")
model = PeftModel.from_pretrained(base, "outputs/phi4-mini-llm-en/final")
merged = model.merge_and_unload()
```

## 4. ONNX Export

Export a trained NLP model to ONNX for production inference:

```bash
uv run python -m shared.export_onnx \
    --model-dir outputs/deberta-v3-nlp-en/best \
    --encoder microsoft/deberta-v3-small \
    --output-dir onnx-output/deberta-v3-nlp-en
```

The export pipeline:

1. **FP32 export** -- `torch.onnx.export` with opset 14 and dynamic axes for variable batch size and sequence length.
2. **INT8 quantization** -- `onnxruntime.quantization.quantize_dynamic` with QInt8 weight type.
3. **Parity validation** -- runs inference on a test sentence through both PyTorch and ONNX, verifying max absolute difference < 0.01 across all four task heads.

All four task outputs (ner/pos/dep/cls logits) are exported as named ONNX outputs with dynamic axes. The tokenizer and label maps are copied to the output directory for self-contained deployment.

Output:

```
onnx-output/<model>/
  model.onnx         -- FP32 ONNX model
  model-int8.onnx    -- INT8 quantized model
  tokenizer files    -- tokenizer config, vocab, etc.
  label_maps.json    -- NER/POS/Dep/CLS label vocabularies
```

## Quick Reference

Full pipeline from scratch:

```bash
# 1. Prepare data
uv run python models/deberta-v3-large-nlp-en/prepare_data.py
uv run python models/deberta-v3-nlp-en/prepare_data.py
uv run python models/phi4-mini-llm-en/prepare_data.py

# 2. Train teacher
uv run python models/deberta-v3-large-nlp-en/train.py

# 3. Generate soft labels
uv run python models/deberta-v3-large-nlp-en/generate_soft_labels.py \
    --model-dir outputs/deberta-v3-large-nlp-en/best

# 4. Distill student
uv run python models/deberta-v3-nlp-en/distill.py \
    --soft-labels outputs/deberta-v3-large-nlp-en/soft_labels \
    --temperature 3.0 --alpha 0.5

# 5. Export to ONNX
uv run python -m shared.export_onnx \
    --model-dir outputs/deberta-v3-nlp-en/distilled/best \
    --encoder microsoft/deberta-v3-small \
    --output-dir onnx-output/deberta-v3-nlp-en

# 6. Train LLM adapter
uv run python models/phi4-mini-llm-en/train.py
```
