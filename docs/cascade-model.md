# kniv-deberta-cascade — Multi-Task NLP Cascade Model

## Overview

A single DeBERTa-v3-large encoder serving multiple NLP task heads through
layer-selective training and cascaded predictions. Each head reads from
optimal encoder layers via learned ScalarMix weights, and downstream heads
receive soft predictions from upstream heads as additional features.

```
                        DeBERTa-v3-large (434M params)
                     ┌──────────────────────────────────┐
  Input tokens ───►  │  Embedding Layer (Layer 0)        │
                     │  Transformer Layer 1               │ ◄── POS reads here
                     │  ...                               │     (layers 0-8)
                     │  Transformer Layer 8               │
                     │  Transformer Layer 9               │ ◄── NER reads here
                     │  ...                               │     (layers 5-12)
                     │  Transformer Layer 12              │
                     │  ...                               │ ◄── DEP reads here
                     │  Transformer Layer 18              │     (layers 12-18)
                     │  ...                               │ ◄── SRL reads here
                     │  Transformer Layer 24              │     (layers 18-24)
                     └──────────┬───────────────────────┘
                                │ output_hidden_states (25 layers)
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ScalarMix    ScalarMix    ScalarMix    ...
              (per task)   (per task)   (per task)
                    │           │           │
                    ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐
              │   POS   │→│   NER   │→│   DEP   │
              │ Linear  │ │ BiLSTM  │ │Biaffine │
              └─────────┘ └─────────┘ └─────────┘
                               ▲           ▲
                            POS probs   POS+NER probs
                            (cascade)    (cascade)
```

## Architecture

### Encoder

- **Model**: `microsoft/deberta-v3-large` (24 layers, 1024 hidden, 434M params)
- **Precision**: float32 (`.float()` required for correct inference)
- **Library**: `transformers==5.0.0` (pinned — other versions produce incorrect outputs)
- **Predicate embedding**: `Embedding(2, 1024)` injected at embedding level for SRL

### ScalarMix (Per-Task Layer Selection)

Each task head has its own ScalarMix module that learns a weighted combination
of all 25 encoder hidden states (embedding output + 24 transformer layers).
This allows each task to discover its optimal encoder depth:

```python
class ScalarMix(nn.Module):
    """Learned weighted average of encoder layers (ELMo-style)."""
    def __init__(self, n_layers, favor_range=None):
        # favor_range biases initialization toward specific layers
        # e.g., favor_range=(0, 8) for POS → low layers
```

| Task | Initialized Favor Range | Empirically Learned Layers |
|------|------------------------|---------------------------|
| POS  | 0–8 (low)             | Morphosyntactic layers     |
| NER  | 5–10 (mid)            | Entity-aware layers        |
| DEP  | 12–18 (mid-high)      | Syntactic structure layers |
| SRL  | 18–24 (top)           | Semantic layers            |
| CLS  | Free (all)            | Learned across all layers  |

### Cascade Architecture

Predictions flow from simpler to more complex tasks:

```
POS ──► NER ──► DEP
                      (SRL is independent — uses pred_embedding)
                      (CLS is independent — uses AttentionPool)
```

**Cascade mechanism**: upstream task probabilities are concatenated to the
hidden state before the downstream head's classifier:

```python
# NER receives POS soft predictions
pos_probs = softmax(pos_head(pos_scalar_mix(hidden_states))).detach()
ner_input = cat([ner_hidden, pos_probs], dim=-1)  # [B, S, H+17]

# DEP receives POS + NER soft predictions
dep_input = cat([dep_hidden, pos_probs, ner_probs], dim=-1)  # [B, S, H+17+37]
```

The `.detach()` prevents gradients from flowing back through the cascade,
so each head trains independently while benefiting from upstream predictions.

---

## Task Heads

### POS — Part-of-Speech Tagging

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(0-8) → Linear(1024, 17) |
| Labels | 17 UPOS tags (UD standard) |
| Training data | UD English EWT (12.5K gold, CC-BY-SA-4.0) |
| Eval benchmark | UD EWT dev (standard) |
| Decoding | argmax |
| v3.2 score | 0.966 accuracy |
| v4 score | 0.979 accuracy (with layer-selective tuning) |
| SOTA reference | ~0.982 (fine-tuned DeBERTa + CRF) |

### NER — Named Entity Recognition

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(5-10) → BiLSTM(256) → residual → cat(POS) → MLP |
| Labels | 37 BIO tags (18 OntoNotes entity types) |
| Training data | 195K SpanMarker silver labels (generated, commercially OK) |
| Eval benchmark | SpanMarker dev (silver) — need CoNLL-2003 gold for benchmark |
| Decoding | Viterbi (constrained BIO transitions) |
| Cascade input | POS probabilities (17-dim) |
| v3.2 score | 0.860 F1 |
| SOTA reference | ~0.946 F1 on CoNLL-2003 (fine-tuned + doc context) |

### SRL — Semantic Role Labeling

| Property | Value |
|----------|-------|
| Architecture | pred_embedding → encoder.encoder() → BiLSTM(256) → residual → MLP |
| Labels | 42 BIO tags (ARG0-4, 16 ARGM types) |
| Training data | 200K AllenNLP silver (MIT) + 41K PropBank EWT gold (CC-BY-SA-4.0) |
| Eval benchmark | PropBank EWT gold dev (1.2K) |
| Decoding | Viterbi (constrained BIO transitions, V excluded from scoring) |
| Forward path | Uses `encoder.encoder(embed, mask)` directly (NOT `encoder(inputs_embeds=)`) |
| Predicate marking | Single token indicator: `Embedding(2, 1024)` added to predicate's first subword |
| v3.2 score | 0.802 F1 |
| SOTA reference | ~0.880 F1 on CoNLL-2012 (span-based + global inference) |

**SRL forward path** (unique — bypasses model.forward):
```python
emb = encoder.embeddings(input_ids)
indicator = zeros(B, S); indicator[arange(B), predicate_idx] = 1
emb = emb + pred_embedding(indicator)
hidden = encoder.encoder(emb, attention_mask).last_hidden_state
# → BiLSTM → residual → classifier
```

### DEP — Dependency Parsing (Biaffine)

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(12-18) → cat(POS, NER) → proj → BiaffineDEPHead |
| Labels | 53 UD deprels (Dozat & Manning biaffine arc + label scoring) |
| Training data | UD English EWT (12.5K gold, CC-BY-SA-4.0) |
| Eval benchmark | UD EWT dev (standard) |
| Decoding | argmax (arc head selection + label classification) |
| Cascade input | POS probs (17-dim) + NER probs (37-dim) |
| v3.2 score | 0.922 UAS |
| SOTA reference | ~0.960 UAS (fine-tuned RoBERTa + biaffine + CRF2o) |

**Biaffine scorer**:
```
BiaffineDEPHead:
  arc_dep/arc_head:   Linear(H, 512) + LN + GELU   → Biaffine(512, 1)    → [B, S, S]     arc scores
  label_dep/label_head: Linear(H, 128) + LN + GELU → Biaffine(128, 53)   → [B, S, S, 53] label scores
```

### CLS — Dialog Act Classification

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(free) → AttentionPool → MLP(1024, 512, 8) |
| Labels | 8 dialog acts: inform, request, question, confirm, reject, offer, social, status |
| Training data | 60K SGD (CC-BY-SA-4.0) + GPT-labeled corpus (ours) |
| Eval benchmark | Carved dev split (4K, seed=42) |
| Context | Previous utterance encoded as pair: `tokenizer(prev_text, text)` |
| Decoding | argmax |
| v3.2 score | 0.930 macro F1 |
| SOTA reference | ~0.85 acc on SwDA 42-label (not directly comparable) |

**AttentionPool** (learned token weighting — outperforms mean pool by +1.7 F1):
```python
scores = Linear(H, 1)(hidden).squeeze(-1)       # [B, S]
scores = masked_fill(~mask, -inf)
weights = softmax(scores, dim=-1)                # [B, S]
pooled = (hidden * weights.unsqueeze(-1)).sum(1)  # [B, H]
```

---

## Training Strategy

### v4: Bottom-Up Layer-Selective Training

Instead of the v3 approach (SRL-first, then freeze everything), v4 trains
bottom-up, respecting the natural layer hierarchy:

```
Phase 1: POS   — unfreeze layers 0-8    (syntactic foundation)
Phase 2: NER   — unfreeze layers 5-12   (entities, overlap zone at 5-8 gets low LR)
Phase 3: DEP   — unfreeze layers 12-18  (syntactic structure)
Phase 4: SRL   — unfreeze layers 18-24  (semantic roles + pred_embedding)
Phase 5: CLS   — frozen encoder         (reads from all tuned layers)
```

**Key principles**:
- Each phase only unfreezes the layers its task reads from
- Lower layers get shaped first, upper layers build on them
- Overlap zones use differentiated LR (5e-6 for already-tuned, 2e-5 for fresh)
- After each phase, forgetting check on all previous heads
- Encoder state saved at every phase for rollback

**Learning rates**:
| Component | LR | Rationale |
|-----------|-----|-----------|
| Fresh encoder layers | 2e-5 | Standard transformer fine-tuning |
| Previously-tuned overlap layers | 5e-6 | Preserve earlier task's representations |
| Task head (ScalarMix + classifier) | 1e-3 | Head learns fast on frozen/slow encoder |
| Predicate embedding | 2e-5 | Trained with SRL encoder layers |

### Data — All Commercially Licensed

| Data | License | Used For |
|------|---------|----------|
| UD English EWT | CC-BY-SA-4.0 | POS, DEP gold |
| SpanMarker NER silver | Generated (our model) | NER training |
| AllenNLP SRL silver | MIT (model) | SRL training (397K examples) |
| PropBank EWT | CC-BY-SA-4.0 | SRL gold (41K examples) |
| SGD (Google) | CC-BY-SA-4.0 | CLS training |
| GPT corpus labels | Ours | CLS (question/inform/social) |

---

## Checkpoint Format

The leader model is saved as a single `model.pt` file containing a dict
of state dicts, one per component:

```python
state = {
    # Shared encoder
    "deberta":          encoder.state_dict(),           # 434M params
    "pred_embedding":   pred_embedding.state_dict(),    # 2K params

    # POS head
    "pos_scalar_mix":   pos_scalar_mix.state_dict(),
    "pos_head":         pos_head.state_dict(),

    # NER head (BiLSTM version)
    "ner_scalar_mix":   ner_scalar_mix.state_dict(),
    "ner_lstm":         ner_lstm.state_dict(),
    "ner_proj":         ner_proj.state_dict(),
    "ner_head":         ner_head.state_dict(),

    # SRL head
    "srl_lstm":         srl_lstm.state_dict(),
    "srl_proj":         srl_proj.state_dict(),
    "classifier":       srl_classifier.state_dict(),

    # DEP head (biaffine)
    "dep_scalar_mix":   dep_scalar_mix.state_dict(),
    "dep_proj":         dep_proj.state_dict(),
    "dep_biaffine":     dep_biaffine.state_dict(),

    # CLS head
    "cls_scalar_mix":   cls_scalar_mix.state_dict(),
    "cls_pool":         cls_pool.state_dict(),
    "cls_head":         cls_head.state_dict(),
}
```

**Metadata** (`metadata.json`):
```json
{
  "heads": {
    "pos": {"score": 0.979, "metric": "accuracy"},
    "ner": {"score": 0.860, "metric": "F1"},
    "srl": {"score": 0.802, "metric": "F1"},
    "dep": {"score": 0.922, "metric": "UAS", "las": 0.899},
    "cls": {"score": 0.930, "metric": "macro_f1", "labels": ["inform", ...]}
  },
  "encoder": "DeBERTa-v3-large",
  "version": "v4.5",
  "transformers": "5.6.2"
}
```

---

## Roadmap

### Current Heads (v4)

| Head | Status | Score | Benchmark |
|------|--------|-------|-----------|
| POS  | Shipped | 0.979 | UD EWT (standard) |
| NER  | Shipped | 0.860 | SpanMarker silver dev |
| SRL  | Shipped | 0.802 | PropBank EWT gold |
| DEP  | Shipped | 0.922 | UD EWT (standard) |
| CLS  | Shipped | 0.930 | SGD/GPT dev |

### Phase 2: Low-Hanging Fruit

| Head | Architecture | Data | Effort |
|------|-------------|------|--------|
| **Lemma** | ScalarMix(low) + POS cascade → char-level head | UD EWT (already have) | Small — same data as POS |
| **Morph** | ScalarMix(low) + POS cascade → multi-label | UD EWT (already have) | Small — same data as POS |
| **Keyword** | ScalarMix(mid) + NER cascade → BIO | Silver from KeyBERT (MIT) on corpus | Medium — need to generate silver |
| **Sentiment** | ScalarMix + AttentionPool → 3-class | Silver-label corpus or Amazon reviews | Medium |

### Phase 3: Comprehension

| Head | Architecture | Data | Effort |
|------|-------------|------|--------|
| **Extractive QA** | ScalarMix(top) + NER+SRL cascade → start/end span | SQuAD 2.0 (CC-BY-SA-4.0) | Medium |
| **Coreference** | Mention detection + linking head | fastcoref silver (MIT) + LitBank (CC-BY-4.0) | Hard |
| **Relation Extraction** | NER+DEP cascade → entity pair classifier | DocRED (MIT) | Hard |

### Phase 4: Generation Integration

The cascade model serves as a **structured understanding layer** for
downstream generation:

```
  ┌─────────────────────────────────────────────────┐
  │              kniv-cascade encoder                 │
  │  POS + NER + SRL + DEP + CLS + QA + Coref       │
  └──────────────────────┬──────────────────────────┘
                         │ structured representations
                         ▼
  ┌─────────────────────────────────────────────────┐
  │         Downstream consumers                      │
  │                                                   │
  │  Option A: LLM integration                        │
  │    Structured output → LLM prompt → generation    │
  │    (reformulation, summarization, dialogue)       │
  │                                                   │
  │  Option B: Small decoder (2-4 transformer layers) │
  │    Encoder hidden states → decoder → text          │
  │    (paraphrase, simplification, query rewriting)  │
  │                                                   │
  │  Option C: Template-based reformulation            │
  │    SRL frames → deterministic restructuring        │
  │    (no hallucination, fully controllable)          │
  └─────────────────────────────────────────────────┘
```

**Extractive QA as a bridge**: QA is the natural stepping stone from
pure extraction to comprehension. The cascade provides NER (entity
awareness) and SRL (argument structure) as input features to the QA head,
giving it an advantage over standard BERT-for-SQuAD approaches.

**Reformulation paths**:

1. **Template-based** (available now): Use SRL frames to decompose and
   restructure sentences. E.g., active→passive, question→statement.
   Deterministic and controllable.

2. **LLM-augmented** (integration): Feed cascade outputs as structured
   context to an LLM. The cascade handles extraction and understanding;
   the LLM handles fluency and generation.

3. **Native decoder** (future): Add a lightweight decoder head trained on
   paraphrase data (PARANMT CC-BY, or silver from GPT). Makes the model
   self-contained for generation tasks.

---

## Dependency Versions

| Package | Required Version | Notes |
|---------|-----------------|-------|
| `transformers` | `5.0.0` (v3.2) or `5.6.2` (v4) | **Critical** — other versions produce incorrect DeBERTa outputs |
| `torch` | `>=2.0` | float32 required (`.float()` on encoder) |
| `seqeval` | any | F1 scoring for NER/SRL |
| `scikit-learn` | any | Macro F1 for CLS |

**Version pinning is mandatory.** DeBERTa's attention implementation changed
between transformers 4.x, 5.0.0, and 5.1.0+. Models trained on one version
show 5-10 F1 point drops when evaluated on another. Always check
`metadata.json → transformers` field and match the version.

---

## Evaluation

Run local evaluation with:

```bash
uv pip install transformers==5.0.0  # match model's training version
uv run python models/kniv-deberta-cascade-large-nlp-en/evaluate_leader.py \
    --model-dir models/kniv-v3.2 --batch-size 16
```

The eval script handles all 5 heads, including:
- SRL's special `encoder.encoder()` forward path with predicate indicator
- DEP's biaffine arc + label scoring
- NER/SRL Viterbi decoding
- CLS pair encoding with prev_text context
