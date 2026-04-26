# Design: Extractive QA Head for kniv-cascade

## 1. Problem Statement

Given a question and a context passage, extract the minimal span of text
from the context that answers the question — or predict "unanswerable" if
no answer exists.

```
Question:  "Who founded Apple?"
Context:   "Apple was founded by Steve Jobs and Steve Wozniak in 1976."
Answer:    "Steve Jobs and Steve Wozniak"
           ─────────── span ───────────
           start=5      end=9   (token indices)
```

This is the **SQuAD-style extractive QA** formulation — the most widely
studied QA paradigm and a natural extension of our cascade model's
existing span extraction capabilities (NER, SRL).

---

## 2. Why the Cascade Gives Us an Advantage

Standard extractive QA models (BERT-for-SQuAD) learn everything from
scratch: entity awareness, argument structure, syntactic roles — all
implicitly from (question, answer_span) supervision.

Our cascade model already provides this understanding explicitly:

```
Question: "When was Apple founded?"
Context:  "Apple was founded by Steve Jobs and Steve Wozniak in 1976."

Cascade outputs for context tokens:
  POS:  PROPN  AUX  VERB   ADP  PROPN  PROPN  CCONJ  PROPN  PROPN  ADP  NUM  PUNCT
  NER:  B-ORG  O    O      O    B-PER  I-PER  O      B-PER  I-PER  O    B-DATE O
  SRL:  B-ARG1 O    V      O    B-ARG0 I-ARG0 I-ARG0 I-ARG0 I-ARG0 B-ARGM-TMP I-ARGM-TMP O
  DEP:         ←nsubj:pass ←root       ←──── obl:agent ────→       ←── obl:tmod ──→

Question asks "When" → look for ARGM-TMP or DATE entities → "1976"
```

The QA head doesn't need to learn that "1976" is a temporal expression —
NER already marked it as `B-DATE` and SRL marked it as `B-ARGM-TMP`.
The QA head just needs to learn to match question types to cascade labels.

**Expected advantage**: faster convergence, better performance on rare
question types, and more interpretable predictions.

---

## 3. Architecture

### 3.1 Input Encoding

Standard SQuAD-style pair encoding:

```
[CLS] question tokens [SEP] context tokens [SEP] [PAD...]
```

DeBERTa handles this natively via `tokenizer(question, context)`.

### 3.2 Feature Assembly

The QA head receives a rich feature vector per context token:

```
                    ┌──────────────────────────────────────┐
                    │     DeBERTa encoder (frozen)          │
                    │     output_hidden_states=True          │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  qa_scalar_mix(hidden_states)          │
                    │  → [B, S, 1024] QA hidden              │
                    └──────────────┬───────────────────────┘
                                   │
               ┌───────────────────┼───────────────────────┐
               │                   │                       │
        POS probs (17)      NER probs (37)          SRL probs (42)
        (detached)          (detached)              (detached, if
                                                     predicate given)
               │                   │                       │
               └───────────────────┼───────────────────────┘
                                   │ concatenate
                                   ▼
                    ┌──────────────────────────────────────┐
                    │  [qa_hidden, pos_p, ner_p, srl_p]     │
                    │  → [B, S, 1024 + 17 + 37 + 42]        │
                    │  = [B, S, 1120]                        │
                    └──────────────┬───────────────────────┘
                                   │
                    ┌──────────────▼───────────────────────┐
                    │  QA projection + start/end classifiers │
                    └──────────────────────────────────────┘
```

### 3.3 QA Head Design

```python
class ExtractiveQAHead(nn.Module):
    def __init__(self, hidden_size=1024, cascade_dim=96, dropout=0.1):
        super().__init__()
        # cascade_dim = pos(17) + ner(37) + srl(42) = 96
        in_dim = hidden_size + cascade_dim

        # Project cascade-enriched features
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Independent start/end classifiers
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)

        # Answerability classifier (for SQuAD 2.0)
        # Uses [CLS] token representation
        self.answerable = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, qa_hidden, cascade_features, attention_mask, cls_hidden):
        """
        Args:
            qa_hidden:        [B, S, H]  from QA ScalarMix
            cascade_features: [B, S, 96] concatenated POS+NER+SRL probs
            attention_mask:   [B, S]
            cls_hidden:       [B, H]     [CLS] token for answerability
        Returns:
            start_logits:  [B, S]
            end_logits:    [B, S]
            answerable:    [B, 1]  (logit for has_answer)
        """
        h = self.proj(torch.cat([qa_hidden, cascade_features], dim=-1))
        start_logits = self.start_classifier(h).squeeze(-1)  # [B, S]
        end_logits = self.end_classifier(h).squeeze(-1)       # [B, S]

        # Mask question tokens and padding
        start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
        end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)

        answerable = self.answerable(cls_hidden)  # [B, 1]

        return start_logits, end_logits, answerable
```

**Design choices**:

- **Independent start/end** (not joint): simpler, works well with DeBERTa,
  same approach as HuggingFace `AutoModelForQuestionAnswering`.
- **Answerability head**: separate classifier on [CLS] token for SQuAD 2.0
  unanswerable questions. Trained with binary cross-entropy.
- **Cascade as soft features**: POS/NER/SRL probabilities concatenated
  (not hard labels) — preserves uncertainty and is differentiable.

### 3.4 SRL Integration Strategy

SRL requires a predicate index, but QA contexts may have multiple predicates.
Three options:

**Option A: No SRL cascade for QA** (simplest)
- Drop SRL from cascade features, use only POS + NER (54-dim)
- Lose argument structure signal but avoid predicate selection complexity

**Option B: Multi-predicate pooling**
- Run SRL for each verb in the context, take max or mean probs per token
- Rich signal but computationally expensive (N forward passes per verb)

**Option C: Question-guided predicate selection** (recommended)
- Identify the main verb in the question
- Find its corresponding verb in the context (lexical/embedding match)
- Run SRL with that predicate only
- Focused signal: "Who founded Apple?" → predicate="founded" → SRL for "founded"

```python
def get_question_predicate(question_tokens, question_pos, context_tokens):
    """Find the question's main verb and match it in context."""
    q_verbs = [t for t, p in zip(question_tokens, question_pos) if p == "VERB"]
    if not q_verbs:
        return None
    # Find matching or closest verb in context
    main_verb = q_verbs[0]  # simplified — could use embedding similarity
    for i, tok in enumerate(context_tokens):
        if tok.lower() == main_verb.lower():
            return i
    return None  # fallback: no SRL cascade
```

### 3.5 Full Forward Pass

```python
def forward_qa(input_ids, attention_mask, token_type_ids=None):
    # 1. Encode question + context
    out = encoder(input_ids=input_ids, attention_mask=attention_mask,
                  output_hidden_states=True)

    # 2. QA ScalarMix (learns to read from optimal layers)
    qa_hidden = qa_scalar_mix(out.hidden_states)

    # 3. Get cascade features (POS + NER, detached)
    pos_h = pos_scalar_mix(out.hidden_states)
    pos_p = softmax(pos_head(pos_h), dim=-1).detach()

    ner_h = ner_scalar_mix(out.hidden_states)
    ner_adapted = ner_proj(ner_lstm(ner_h)[0]) + ner_h
    ner_p = softmax(ner_head(cat([ner_adapted, pos_p], -1)), -1).detach()

    cascade = cat([pos_p, ner_p], dim=-1)  # [B, S, 54]

    # 4. Optional: add SRL if predicate identified
    # srl_p = run_srl_for_predicate(...)
    # cascade = cat([cascade, srl_p], dim=-1)  # [B, S, 96]

    # 5. QA head
    cls_hidden = qa_hidden[:, 0, :]  # [CLS] token
    # Mask question tokens (only predict spans in context)
    context_mask = build_context_mask(attention_mask, token_type_ids)
    return qa_head(qa_hidden, cascade, context_mask, cls_hidden)
```

---

## 4. Training

### 4.1 Data

| Dataset | Size | License | Notes |
|---------|------|---------|-------|
| **SQuAD 2.0** | 130K answerable + 50K unanswerable | CC-BY-SA-4.0 | Primary training set |
| **Natural Questions** | 307K (short answer) | CC-BY-SA-3.0 | Larger, real Google queries |
| **TyDi QA (English)** | 11K | Apache 2.0 | Diverse question types |

All commercially licensed. SQuAD 2.0 is the primary training set;
Natural Questions can be added for volume.

### 4.2 Training Strategy

**Phase**: QA is trained in Phase 3 (after POS, NER, DEP, SRL, CLS).

**Encoder**: Frozen. The QA head learns to read from the already-tuned
encoder layers via its own ScalarMix.

**Alternative**: Unfreeze top layers (20-24) with very low LR (5e-6) to
allow the encoder to slightly adapt for QA-style cross-sentence reasoning.
Risk: may affect SRL quality (same top layers). Run forgetting check.

```python
# QA training setup
qa_scalar_mix = ScalarMix(num_layers, favor_range=(18, 24))  # top layers
qa_head = ExtractiveQAHead(hidden_size=H, cascade_dim=54)

# Frozen encoder — only train QA head
qa_params = list(qa_scalar_mix.parameters()) + list(qa_head.parameters())
optimizer = AdamW(qa_params, lr=1e-3)
epochs = 5  # SQuAD converges fast with a good encoder
```

### 4.3 Loss Function

Combined loss for span extraction + answerability:

```python
def qa_loss(start_logits, end_logits, answerable_logit,
            start_pos, end_pos, has_answer):
    """
    start_pos, end_pos: gold span token indices (-1 if unanswerable)
    has_answer: binary (1 if answerable, 0 if not)
    """
    # Span loss (only for answerable examples)
    answerable_mask = has_answer.bool()
    if answerable_mask.any():
        span_loss = (
            cross_entropy(start_logits[answerable_mask], start_pos[answerable_mask]) +
            cross_entropy(end_logits[answerable_mask], end_pos[answerable_mask])
        ) / 2
    else:
        span_loss = 0.0

    # Answerability loss (all examples)
    answer_loss = binary_cross_entropy_with_logits(
        answerable_logit.squeeze(-1), has_answer.float()
    )

    return span_loss + answer_loss
```

### 4.4 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | Predicted span exactly matches gold (after normalization) |
| **F1** | Token-level overlap between predicted and gold spans |
| **HasAns EM/F1** | EM/F1 on answerable questions only |
| **NoAns Accuracy** | Accuracy on unanswerable questions |

Standard SQuAD 2.0 evaluation script handles all of these.

**Benchmark targets** (DeBERTa-large on SQuAD 2.0):

| Model | EM | F1 |
|-------|-----|-----|
| DeBERTa-large (fine-tuned, published) | 87.7 | 90.7 |
| Our model (frozen encoder + cascade) | TBD | TBD |
| Our model (top layers unfrozen) | TBD | TBD |

Expected: frozen encoder with cascade features should approach fine-tuned
performance due to the rich NER+SRL signal, despite not fine-tuning the
encoder specifically for QA.

---

## 5. Inference

### 5.1 Span Decoding

```python
def decode_answer(start_logits, end_logits, answerable_logit,
                  input_ids, tokenizer, max_answer_length=30,
                  answerable_threshold=0.5):
    """Decode predicted answer span from logits."""

    # Check answerability
    if sigmoid(answerable_logit) < answerable_threshold:
        return {"answer": "", "score": 0.0, "answerable": False}

    # Find best valid span (start <= end, length <= max)
    start_probs = softmax(start_logits, dim=-1)
    end_probs = softmax(end_logits, dim=-1)

    best_score, best_start, best_end = -inf, 0, 0
    for s in topk(start_probs, k=20).indices:
        for e in topk(end_probs, k=20).indices:
            if e >= s and (e - s + 1) <= max_answer_length:
                score = start_probs[s] + end_probs[e]
                if score > best_score:
                    best_score = score
                    best_start, best_end = s, e

    # Decode tokens to text
    answer_ids = input_ids[best_start:best_end + 1]
    answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True)

    return {
        "answer": answer_text.strip(),
        "score": best_score.item(),
        "start": best_start,
        "end": best_end,
        "answerable": True,
    }
```

### 5.2 Cascade-Enriched Output

The QA head can return not just the answer span but the full cascade
analysis of the answer:

```python
{
    "answer": "Steve Jobs and Steve Wozniak",
    "score": 0.97,
    "start": 5, "end": 9,
    "answerable": True,
    "analysis": {
        "pos": ["PROPN", "PROPN", "CCONJ", "PROPN", "PROPN"],
        "ner": ["B-PERSON", "I-PERSON", "O", "B-PERSON", "I-PERSON"],
        "srl_role": "ARG0",  # agent of "founded"
        "dep_relation": "obl:agent",
    }
}
```

This is uniquely possible because our QA head sits on top of the cascade —
every answer span comes pre-analyzed with POS, NER, SRL, and DEP labels.
No other QA system provides this out of the box.

---

## 6. Integration with Reformulation

The QA head enables a natural path to **question reformulation** and
**query understanding**:

### 6.1 Question Type Detection

The cascade's CLS head can classify the question type:

```
"Who founded Apple?"     → CLS: question (subtype: entity-person)
"When was it founded?"   → CLS: question (subtype: temporal)
"Is Apple profitable?"   → CLS: question (subtype: yes/no)
```

Combined with SRL on the question: `{predicate: founded, ARG1: Apple, ARG0: ?}`

### 6.2 Answer-Informed Reformulation

```
Question:   "Who founded Apple?"
Answer:     "Steve Jobs and Steve Wozniak"
SRL frame:  founded(ARG0="Steve Jobs and Steve Wozniak", ARG1="Apple")

Reformulated: "Steve Jobs and Steve Wozniak founded Apple."
              ─── ARG0 (answer) ───────── predicate  ARG1 ──
```

This uses SRL frames to slot the extracted answer into a declarative
statement — no generation model needed.

### 6.3 Multi-Hop Decomposition

For complex questions, the cascade can decompose into sub-questions:

```
"What is the capital of the country where Apple was founded?"

Step 1 (SRL): founded(ARG1="Apple", ARGM-LOC=?)
Step 2 (QA):  "Where was Apple founded?" → "United States"
Step 3 (NER): "United States" = GPE
Step 4 (QA):  "What is the capital of the United States?" → "Washington, D.C."
```

This is a future capability that builds naturally on QA + cascade.

---

## 7. Experiment Plan

### Experiment A: Baseline (no cascade)
- QA ScalarMix + start/end classifier only
- No POS/NER/SRL cascade features
- Establishes how much the raw encoder provides

### Experiment B: POS + NER cascade
- Add POS (17) + NER (37) probs as features = 54 extra dims
- Tests whether entity type awareness helps answer extraction

### Experiment C: Full cascade (POS + NER + SRL)
- Add SRL (42) probs using question-guided predicate = 96 extra dims
- Tests whether argument structure helps

### Experiment D: Top-layer unfreezing
- Same as C but unfreeze layers 20-24 at LR=5e-6
- Tests whether encoder adaptation helps QA specifically
- Must check SRL/CLS forgetting

**Expected outcome**: B > A (NER helps find answer entities),
C > B (SRL helps match question type to argument role),
D >= C (marginal gain if encoder already well-tuned).

---

## 8. Dataset Preparation

### SQuAD 2.0 Format

```json
{
    "context": "Apple was founded by Steve Jobs in 1976.",
    "question": "Who founded Apple?",
    "answers": {
        "text": ["Steve Jobs"],
        "answer_start": [24]
    },
    "is_impossible": false
}
```

### Conversion to Our Format

```python
def prepare_squad_for_cascade(squad_example, tokenizer):
    """Convert SQuAD example to cascade training format."""
    question = squad_example["question"]
    context = squad_example["context"]

    # Pair encoding
    enc = tokenizer(question, context, return_offsets_mapping=True,
                    max_length=384, truncation="only_second",
                    stride=128, return_overflowing_tokens=True)

    # Map character-level answer_start to token positions
    answer_start_char = squad_example["answers"]["answer_start"][0]
    answer_text = squad_example["answers"]["text"][0]
    answer_end_char = answer_start_char + len(answer_text)

    # Find token positions using offset_mapping
    start_token, end_token = None, None
    for i, (offset_start, offset_end) in enumerate(enc["offset_mapping"]):
        if enc.sequence_ids()[i] != 1:  # skip question tokens
            continue
        if offset_start <= answer_start_char < offset_end:
            start_token = i
        if offset_start < answer_end_char <= offset_end:
            end_token = i

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "start_position": start_token,
        "end_position": end_token,
        "is_impossible": squad_example.get("is_impossible", False),
    }
```

### Long Context Handling

For contexts longer than 384 tokens, use sliding window with stride:

```
Context (600 tokens):
  Window 1: [CLS] question [SEP] context[0:384]   [SEP]
  Window 2: [CLS] question [SEP] context[256:640]  [SEP]
                                  ─── stride=128 ───
```

Take the window with the highest answer span score. This is standard
SQuAD practice and handled by HuggingFace's tokenizer with
`return_overflowing_tokens=True, stride=128`.

---

## 9. Timeline and Dependencies

```
Prerequisites (must be complete before QA training):
  ✓ POS head trained
  ✓ NER head trained
  ✓ SRL head trained (for Experiment C)
  ✓ DEP head trained

QA implementation:
  1. Prepare SQuAD 2.0 dataset             (1 day)
  2. Implement ExtractiveQAHead            (1 day)
  3. Implement QA dataset + data pipeline  (1 day)
  4. Run Experiments A-D on Colab          (1 day)
  5. Evaluate on SQuAD 2.0 dev set        (1 day)
  6. Integrate into leader checkpoint      (0.5 day)
  7. Add to evaluate_leader.py             (0.5 day)
```

### Checkpoint Integration

The QA head adds these keys to the leader state dict:

```python
state["qa_scalar_mix"] = qa_scalar_mix.state_dict()
state["qa_head"] = qa_head.state_dict()
# Total: ~1.2M additional params (negligible vs 434M encoder)
```

---

## 10. Open Questions

1. **SRL for QA**: Is question-guided predicate selection worth the
   complexity, or does POS+NER cascade capture enough? Experiments B vs C
   will answer this.

2. **Layer range**: Should QA ScalarMix favor top layers (18-24, like SRL)
   or read broadly (like CLS)? QA requires both semantic understanding
   (top) and token matching (low). Start with free initialization.

3. **Answer type classification**: Should we add an explicit answer type
   head (person/location/date/number/description) derived from the
   question? This could help constrain answer span search to tokens
   matching the expected NER type.

4. **Cross-lingual potential**: DeBERTa-v3-large is English-only, but
   the cascade architecture (ScalarMix + heads) could transfer to
   multilingual encoders (mDeBERTa, XLM-R) with the same head designs.
