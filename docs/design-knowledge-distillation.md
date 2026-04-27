# Design: Knowledge Distillation for Student Cascade Models

## 1. Overview

Use the trained kniv-deberta-nlp-base-en-large (teacher) to generate
soft labels for a large corpus, then train smaller student models on
those labels. Students learn all 5 heads simultaneously in a single
training session.

```
Teacher (434M, trained)
    │
    │  Run on corpus → save soft predictions
    ▼
Distillation Dataset
    │  soft logits for POS, NER, DEP, SRL, CLS
    │
    ├──► DeBERTa-v3-base (86M)
    ├──► DeBERTa-v3-xsmall (22M)
    ├──► ModernBERT-base (150M)
    └──► NeoBERT (250M)
```

## 2. Distillation Dataset

### 2.1 Source Corpus

Use the same annotated corpus used for training, plus any additional
unlabeled text. The teacher produces soft labels for everything — no
gold labels needed (though gold can supplement).

| Source | Sentences | Used For |
|--------|-----------|----------|
| Training corpus (annotated) | ~200K | All heads |
| UD EWT train | 12.5K | POS, DEP (gold + teacher soft) |
| SpanMarker NER train | 195K | NER (silver + teacher soft) |
| SRL silver + gold | 240K | SRL (teacher soft) |
| Additional unlabeled corpus | up to 500K | All heads (teacher soft only) |

Target: **~500K sentences** of teacher-labeled data.

### 2.2 Dataset Format

Each example contains the input text plus teacher soft predictions for
all 5 heads:

```json
{
    "words": ["Steve", "Jobs", "founded", "Apple", "in", "1976", "."],
    "input_ids": [1, 1801, 9265, 5765, 3121, 287, 21114, 260, 2],
    "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1],

    "pos_logits": [[...], [...], ...],       // [seq, 17] float16
    "ner_logits": [[...], [...], ...],       // [seq, 37] float16
    "arc_scores": [[...], [...], ...],       // [seq, seq] float16
    "label_scores": [[...], [...], ...],     // [seq, seq, 53] float16 (sparse: top-5)
    "srl_logits": [[...], [...], ...],       // [seq, 42] float16
    "cls_logits": [...],                     // [8] float16

    "predicate_idx": 2,                     // token index of verb for SRL

    // Optional gold labels (where available)
    "pos_tags": ["PROPN", "PROPN", "VERB", "PROPN", "ADP", "NUM", "PUNCT"],
    "ner_tags": ["B-PERSON", "I-PERSON", "O", "B-ORG", "O", "B-DATE", "O"],
}
```

### 2.3 Storage Optimization

Full logits for 500K sentences are large. Optimizations:

| Optimization | Savings |
|-------------|---------|
| float16 for logits | 50% |
| Top-k sparse for DEP label_scores (k=5 per position) | ~90% for label_scores |
| Omit padding positions | variable |
| Store as memory-mapped numpy arrays (not JSON) | faster I/O |

Estimated dataset size: **~50-80 GB** for 500K sentences with all logits.

Alternatively, store only the top-k logits or temperature-scaled
probabilities instead of full logits. Temperature T=2-4 is standard
for distillation.

### 2.4 Generator Script

```
models/kniv-deberta-nlp-base-en-large/generate_distillation_data.py

Input:  corpus sentences (JSONL)
Output: distillation dataset (arrow/parquet files)

Pipeline:
  1. Load teacher model (PyTorch or ONNX)
  2. For each batch of sentences:
     a. Tokenize with teacher's tokenizer
     b. Run teacher forward (single pass, all heads)
     c. Save soft logits (float16) + hard predictions
     d. For SRL: detect verbs from POS, run per-predicate
  3. Shard output into multiple files for parallel training
```

## 3. Student Architecture

### 3.1 Head Design

Students use the same head architecture as the teacher, scaled to match
the student encoder's hidden dimension:

```
Teacher (H=1024):  ScalarMix(25) → BiLSTM(1024, 256) → MLP(1024, 37)
Student (H=768):   ScalarMix(13) → BiLSTM(768, 192)  → MLP(768, 37)
Student (H=384):   ScalarMix(7)  → BiLSTM(384, 96)   → MLP(384, 37)
```

Each student gets:
- Its own ScalarMix (number of layers matches the student encoder)
- Scaled BiLSTM/MLP dimensions (proportional to hidden size)
- Same label sets (POS=17, NER=37, DEP=53, SRL=42, CLS=8)
- Same cascade: POS→NER→DEP
- Same predicate embedding for SRL
- Same AttentionPool for CLS

### 3.2 Student Configurations

| Student | Encoder | Layers | Hidden | ScalarMix | Est. Head Params |
|---------|---------|--------|--------|-----------|-----------------|
| base | deberta-v3-base | 12 | 768 | 13 | ~5M |
| xsmall | deberta-v3-xsmall | 6 | 384 | 7 | ~1.5M |
| modern-base | ModernBERT-base | 22 | 768 | 23 | ~5M |
| neo | NeoBERT | 24 | 1024 | 25 | ~9.5M |

### 3.3 Predicate Embedding Scaling

Teacher uses `Embedding(2, 1024)`. Students scale to their hidden dim:

```python
pred_embedding = nn.Embedding(2, student_hidden_dim)
nn.init.zeros_(pred_embedding.weight)
```

### 3.4 Biaffine DEP Scaling

Arc and label dimensions scale proportionally:

```python
# Teacher: arc_dim=512, label_dim=128 (for H=1024)
# Student scales: arc_dim = H//2, label_dim = H//8
arc_dim = student_hidden_dim // 2    # 384 for base, 192 for xsmall
label_dim = student_hidden_dim // 8  # 96 for base, 48 for xsmall
```

## 4. Training

### 4.1 Loss Function

Combined distillation + hard label loss:

```python
def distillation_loss(student_logits, teacher_logits, hard_labels=None,
                      temperature=3.0, alpha=0.7):
    """
    KL divergence on softened predictions + optional hard label CE.

    Args:
        student_logits: [B, S, C] raw student logits
        teacher_logits: [B, S, C] raw teacher logits (from dataset)
        hard_labels:    [B, S] integer labels (-100 for ignore)
        temperature:    softening temperature (higher = softer)
        alpha:          weight for distillation vs hard label loss
    """
    # Soft target loss (KL divergence)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean")
    kl_loss = kl_loss * (temperature ** 2)  # scale by T^2

    if hard_labels is not None:
        # Hard label loss (standard CE)
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            hard_labels.view(-1),
            ignore_index=-100
        )
        return alpha * kl_loss + (1 - alpha) * ce_loss

    return kl_loss
```

### 4.2 Multi-Head Loss

All 5 heads train simultaneously with weighted losses:

```python
loss = (
    1.0 * distillation_loss(student.pos, teacher.pos, gold_pos)
  + 1.5 * distillation_loss(student.ner, teacher.ner, gold_ner)
  + 1.5 * distillation_loss(student.dep_arc, teacher.dep_arc, gold_heads)
  + 1.0 * distillation_loss(student.dep_label, teacher.dep_label, gold_rels)
  + 2.0 * distillation_loss(student.srl, teacher.srl, gold_srl)
  + 1.0 * distillation_loss(student.cls, teacher.cls, gold_cls)
)
```

Higher weights on harder tasks (SRL, NER, DEP) to prevent them from
being drowned out by the easier POS and CLS losses.

### 4.3 Training Configuration

```python
# All heads, single session, full encoder unfrozen
optimizer = AdamW([
    {"params": encoder.parameters(), "lr": 2e-5},
    {"params": all_head_params, "lr": 1e-3},
], weight_decay=0.01)

epochs = 5
batch_size = 64  # students are smaller, can use larger batches
temperature = 3.0
alpha = 0.7  # 70% distillation, 30% hard labels
warmup = 10%
scheduler = linear_with_warmup
```

### 4.4 Why Single Session Works for Students

The teacher required bottom-up layer-selective training because it was
shaping a vanilla encoder from scratch — each task needed different
layers tuned without destroying others.

Students don't need this because:
1. **Soft labels are smooth.** KL divergence on softened probabilities
   is a much gentler loss than hard CE. No sharp gradients that cause
   catastrophic forgetting.
2. **All information comes from one teacher.** The teacher's predictions
   are internally consistent — POS, NER, DEP, SRL, CLS all came from
   the same encoder. The student learns to replicate this consistency.
3. **Smaller models are more plastic.** With 6-12 layers, the student
   doesn't have the "layer specialization" problem of the 24-layer
   teacher. Every layer contributes to every task.

## 5. Student Encoder Notes

### 5.1 DeBERTa-v3-base (86M)

```python
encoder = AutoModel.from_pretrained("microsoft/deberta-v3-base")
# 12 layers, 768 hidden, 86M params
# Same architecture as teacher — just fewer, narrower layers
# Tokenizer compatible (same SentencePiece vocab)
```

### 5.2 DeBERTa-v3-xsmall (22M)

```python
encoder = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")
# 6 layers, 384 hidden, 22M params
# Smallest DeBERTa variant
# Same tokenizer
```

### 5.3 ModernBERT-base (150M)

```python
encoder = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
# 22 layers, 768 hidden, 150M params
# Faster than DeBERTa due to Flash Attention, no position bias
# DIFFERENT tokenizer — need to re-tokenize with ModernBERT tokenizer
# for distillation data, or re-run teacher per-token alignment
```

**Tokenizer mismatch**: ModernBERT uses a different tokenizer than
DeBERTa. The distillation dataset must be re-tokenized with each
student's tokenizer. Teacher logits need to be aligned to the student's
subword tokens.

Options:
- **Word-level distillation**: Store teacher predictions at word level
  (first subword only), align to any student's subwords at training time.
  This is the cleanest approach.
- **Re-run teacher**: For non-DeBERTa students, re-run the teacher with
  the student's tokenization. Expensive but exact.

**Recommendation**: Store teacher predictions at **word level** in the
distillation dataset. At student training time, each student's
dataloader aligns word-level predictions to its own subword tokens.

### 5.4 NeoBERT (250M)

```python
encoder = AutoModel.from_pretrained("neobert/neobert-base")  # verify name
# 24 layers, 1024 hidden, 250M params
# Latest BERT variant, strong on MTEB
# Different tokenizer — same word-level alignment needed
```

## 6. Distillation Data Generator

### 6.1 Script

```
models/kniv-deberta-nlp-base-en-large/generate_distillation_data.py

Usage:
    python generate_distillation_data.py \
        --model models/kniv-deberta-nlp-base-en-large \
        --corpus corpus/output/annotated \
        --output data/distillation \
        --max-sentences 500000 \
        --batch-size 32 \
        --device cuda
```

### 6.2 Output Format

Word-level predictions stored as Arrow/Parquet for fast I/O:

```
data/distillation/
├── shard_000.parquet    (~50K sentences each)
├── shard_001.parquet
├── ...
├── shard_009.parquet
└── metadata.json        (teacher version, corpus stats)
```

Each row:

| Column | Type | Shape | Description |
|--------|------|-------|-------------|
| words | list[str] | [N] | Word-level tokens |
| pos_probs | list[list[float16]] | [N, 17] | Softmax POS probs |
| ner_probs | list[list[float16]] | [N, 37] | Softmax NER probs |
| dep_arc_probs | list[list[float16]] | [N, N] | Softmax arc probs |
| dep_label_top5 | list[list[tuple]] | [N, 5] | Top-5 (idx, prob) per position |
| srl_probs | list[list[float16]] | [N, 42] | Softmax SRL probs |
| cls_probs | list[float16] | [8] | Softmax CLS probs |
| predicate_word_idx | int | scalar | Verb index for SRL |
| pos_hard | list[str] | [N] | Argmax POS tags |
| ner_hard | list[str] | [N] | Viterbi NER tags |

Word-level storage means any student tokenizer can align at training time.

### 6.3 Alignment at Training Time

```python
class DistillationDataset(Dataset):
    """Aligns word-level teacher predictions to student subwords."""

    def __init__(self, parquet_path, student_tokenizer, max_length=128):
        self.data = pq.read_table(parquet_path).to_pandas()
        self.tokenizer = student_tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        words = row["words"]

        # Tokenize with student's tokenizer
        enc = self.tokenizer(words, is_split_into_words=True,
                             max_length=self.max_length, truncation=True,
                             padding="max_length", return_tensors="pt")

        # Align teacher word-level probs to student subword tokens
        word_ids = enc.word_ids()
        teacher_pos = torch.zeros(self.max_length, 17)
        teacher_ner = torch.zeros(self.max_length, 37)
        # ... fill from word_ids mapping ...

        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev and wid < len(words):
                teacher_pos[k] = torch.tensor(row["pos_probs"][wid])
                teacher_ner[k] = torch.tensor(row["ner_probs"][wid])
            prev = wid

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "teacher_pos": teacher_pos,
            "teacher_ner": teacher_ner,
            # ... other heads ...
        }
```

## 7. Training Script

```
models/kniv-deberta-nlp-base-en-large/train_student.py

Usage:
    python train_student.py \
        --student microsoft/deberta-v3-base \
        --distillation-data data/distillation \
        --output models/kniv-deberta-nlp-base-en-base \
        --epochs 5 --batch-size 64 --temperature 3.0
```

The script:
1. Loads the student encoder from HuggingFace
2. Builds scaled head architectures (ScalarMix, BiLSTM, Biaffine, etc.)
3. Loads distillation dataset with student-specific tokenizer alignment
4. Trains all heads simultaneously with KL + CE loss
5. Evaluates on standard benchmarks after training
6. Saves in the same checkpoint format as the teacher

## 8. Expected Results

Based on typical distillation results in the literature:

| Student | Params | POS | NER | DEP | SRL | CLS |
|---------|--------|-----|-----|-----|-----|-----|
| Teacher (large) | 434M | 0.977 | 0.889 | 0.944 | 0.843 | 0.951 |
| DeBERTa-base | 86M | ~0.970 | ~0.860 | ~0.920 | ~0.790 | ~0.935 |
| DeBERTa-xsmall | 22M | ~0.955 | ~0.820 | ~0.890 | ~0.720 | ~0.910 |
| ModernBERT-base | 150M | ~0.973 | ~0.870 | ~0.930 | ~0.810 | ~0.940 |
| NeoBERT | 250M | ~0.975 | ~0.880 | ~0.935 | ~0.825 | ~0.945 |

Typical distillation gap: 1-5 points per head depending on the
compression ratio. ModernBERT may outperform DeBERTa-base despite
similar size due to architectural improvements (Flash Attention,
longer pre-training).

## 9. Implementation Plan

### Step 1: Generate distillation dataset (~4-6 hours GPU)

```
generate_distillation_data.py
  Input: 500K corpus sentences
  Output: 10 parquet shards (~50-80 GB total)
```

### Step 2: Train DeBERTa-base student (~2-3 hours GPU)

First student — validates the pipeline and establishes baseline.

### Step 3: Train other students in parallel

Once the pipeline works, train DeBERTa-xsmall, ModernBERT, NeoBERT
concurrently (different GPUs or sequential on one GPU).

### Step 4: Benchmark all students

Run standard benchmarks on all students, compare to teacher.

### Step 5: Export to ONNX + quantize

Each student gets: model.pt, cascade.onnx, cascade-int8.onnx.

## 10. File Structure

```
models/
├── kniv-deberta-nlp-base-en-large/     (teacher)
│   ├── model.pt
│   ├── generate_distillation_data.py   (new)
│   └── train_student.py                (new)
├── kniv-deberta-nlp-base-en-base/      (student: deberta-base)
│   ├── model.pt
│   └── README.md
├── kniv-deberta-nlp-base-en-xsmall/    (student: deberta-xsmall)
├── kniv-modernbert-nlp-base-en-base/   (student: modernbert)
└── kniv-neobert-nlp-base-en/           (student: neobert)

data/
└── distillation/
    ├── shard_000.parquet
    ├── ...
    └── metadata.json
```
