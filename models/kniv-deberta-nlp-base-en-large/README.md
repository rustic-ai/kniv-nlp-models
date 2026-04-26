---
language: en
license: cc-by-sa-4.0
library_name: transformers
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
        metrics:
          - type: accuracy
            value: 0.977
            name: Accuracy
      - task:
          type: token-classification
          name: Named Entity Recognition
        dataset:
          name: OntoNotes 5.0
        metrics:
          - type: f1
            value: 0.889
            name: F1 (micro)
      - task:
          type: token-classification
          name: Dependency Parsing
        dataset:
          type: universal-dependencies/universal_dependencies
          name: UD English EWT
        metrics:
          - type: accuracy
            value: 0.944
            name: UAS
      - task:
          type: token-classification
          name: Semantic Role Labeling
        dataset:
          name: PropBank EWT
        metrics:
          - type: f1
            value: 0.843
            name: F1
      - task:
          type: text-classification
          name: Dialog Act Classification
        dataset:
          name: SGD + GPT
        metrics:
          - type: f1
            value: 0.951
            name: Macro F1
---

# kniv-deberta-nlp-base-en-large

A multi-task NLP model that performs 5 language analysis tasks from a single
DeBERTa-v3-large encoder pass: POS tagging, Named Entity Recognition,
Dependency Parsing, Semantic Role Labeling, and Dialog Act Classification.

Part of the [Rustic](https://rustic.ai) initiative by
[Dragonscale Industries Inc.](https://dragonscale.ai)

## Standard Benchmark Results

Evaluated on standard public test sets. All numbers are reproducible
using the included benchmark scripts.

### Primary Benchmarks

| Head | Score | Metric | Benchmark | Split |
|------|-------|--------|-----------|-------|
| POS | 0.977 | Accuracy | UD English EWT | test |
| NER | 0.889 | F1 (micro) | OntoNotes 5.0 | test |
| DEP | 0.944 / 0.923 | UAS / LAS | UD English EWT | test |
| SRL | 0.843 | F1 | PropBank EWT | test |
| CLS | 0.951 | Macro F1 | SGD + GPT (8 labels) | dev |

### Cross-Benchmark NER

The NER head outputs 18 OntoNotes entity types. When evaluated on benchmarks
with different tag sets, entity types are mapped:

| Benchmark | Score | Metric | Mapping |
|-----------|-------|--------|---------|
| OntoNotes 5.0 test | 0.889 | F1 | Direct (same 18 types) |
| CoNLL-2003 test | 0.794 | F1 | 18→4 types (PERSON→PER, GPE/LOC→LOC, etc.) |

The CoNLL-2003 score is lower because numeric entities (DATE, CARDINAL,
MONEY, etc.) have no CoNLL equivalent and are mapped to O.

### NER Per-Entity Performance (OntoNotes 5.0)

| Entity Type | Precision | Recall | F1 | Support |
|-------------|-----------|--------|-----|---------|
| PERSON | 0.952 | 0.948 | 0.950 | 1,988 |
| GPE | 0.965 | 0.946 | 0.955 | 2,240 |
| ORG | 0.900 | 0.896 | 0.898 | 1,795 |
| NORP | 0.910 | 0.930 | 0.920 | 841 |
| DATE | 0.823 | 0.869 | 0.846 | 1,601 |
| CARDINAL | 0.840 | 0.843 | 0.841 | 935 |
| MONEY | 0.883 | 0.892 | 0.887 | 314 |
| PERCENT | 0.863 | 0.865 | 0.864 | 349 |
| LOC | 0.776 | 0.793 | 0.785 | 179 |
| FAC | 0.773 | 0.807 | 0.790 | 135 |
| ORDINAL | 0.833 | 0.918 | 0.873 | 195 |
| PRODUCT | 0.760 | 0.750 | 0.755 | 76 |
| EVENT | 0.657 | 0.698 | 0.677 | 63 |
| TIME | 0.654 | 0.660 | 0.657 | 212 |
| QUANTITY | 0.750 | 0.714 | 0.732 | 105 |
| WORK_OF_ART | 0.636 | 0.620 | 0.628 | 166 |
| LAW | 0.733 | 0.550 | 0.629 | 40 |
| LANGUAGE | 0.867 | 0.542 | 0.667 | 24 |

### Cross-Benchmark CLS

The CLS head uses 8 dialog act labels. When mapped to DailyDialog's
4-label scheme:

| Benchmark | Score | Metric | Mapping |
|-----------|-------|--------|---------|
| SGD + GPT dev (internal) | 0.951 | Macro F1 | Direct (8 labels) |
| DailyDialog test | 0.613 | Accuracy | 8→4 labels (lossy) |

The DailyDialog score reflects the mapping loss — our 8-label scheme
does not align cleanly to DailyDialog's 4 labels.

## Architecture

### Encoder

DeBERTa-v3-large (434M parameters, 24 transformer layers, 1024 hidden
dimension). All tasks share this encoder. A single forward pass with
`output_hidden_states=True` produces 25 hidden state tensors (embedding
layer + 24 transformer layers) that the task heads read from.

### ScalarMix (Per-Task Layer Selection)

Each task head has a learned ScalarMix module — a softmax-weighted
combination of all 25 encoder hidden states. This allows each task to
discover its optimal encoder depth without manual layer assignment:

```python
class ScalarMix(nn.Module):
    """Weighted average of encoder layers (ELMo-style)."""
    def __init__(self, n_layers):
        self.weights = nn.Parameter(torch.zeros(n_layers))  # learned
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, hidden_states):
        w = torch.softmax(self.weights, dim=0)
        return self.scale * sum(wi * hi for wi, hi in zip(w, hidden_states))
```

In practice, POS converges to reading layers 0-8 (morphosyntax), NER
reads layers 5-10 (entity-level), DEP reads layers 12-18 (structural),
and SRL reads the top layers (semantic). CLS learns a free mixture across
all layers.

### Task Cascade

Simpler tasks feed soft predictions into more complex tasks:

- **NER** receives POS probabilities (17-dim) concatenated to its hidden state
- **DEP** receives both POS (17-dim) and NER (37-dim) probabilities

Upstream probabilities are detached (no gradient flow back through the
cascade), so each head trains independently while benefiting from
upstream predictions at inference.

### Head Architectures

**POS — Part-of-Speech Tagging**

```
ScalarMix(25 layers, biased toward 0-8)
    → Linear(1024, 17)
    → argmax
```

17 UPOS tags. Simplest head — a linear probe on the ScalarMix output.

**NER — Named Entity Recognition**

```
ScalarMix(25 layers, biased toward 5-10)
    → BiLSTM(1024, 256, bidirectional)
    → Linear(512, 1024) + residual connection
    → concat with POS probs (17-dim)
    → LayerNorm(1041) → Linear(1041, 1024) → GELU → Dropout(0.1) → Linear(1024, 37)
    → Viterbi decoding (constrained BIO transitions)
```

37 BIO tags covering 18 OntoNotes entity types (PERSON, ORG, GPE, DATE,
etc.). The BiLSTM adapter captures local sequential patterns. Viterbi
decoding enforces valid BIO sequences at inference (+0.5-1.0 F1).

**DEP — Dependency Parsing (Biaffine)**

```
ScalarMix(25 layers, biased toward 12-18)
    → concat with POS probs (17-dim) + NER probs (37-dim)
    → LayerNorm(1078) → Linear(1078, 1024) → GELU  [projection]
    → BiaffineDEPHead:
        arc_dep/arc_head:     Linear(1024, 512) + LN + GELU → Biaffine(512, 1)
        label_dep/label_head: Linear(1024, 128) + LN + GELU → Biaffine(128, 53)
    → arc scores: [batch, seq, seq]         (head selection)
    → label scores: [batch, seq, seq, 53]   (relation classification)
    → argmax
```

Dozat & Manning (2017) biaffine attention. The arc scorer selects which
token is the head of each token. The label scorer classifies the
dependency relation (53 UD relation types). POS and NER cascade features
provide syntactic and entity type information to the parser.

**SRL — Semantic Role Labeling**

```
Embedding layer:
    word_embeddings(input_ids)
    + pred_embedding(indicator)         [Embedding(2, 1024), marks predicate]
    → encoder.encoder(emb, mask)        [direct encoder call, not model.forward]
    → last_hidden_state

Head:
    → Dropout(0.1) → Linear(1024, 1024) → GELU → Dropout(0.1) → Linear(1024, 42)
    → Viterbi decoding (constrained BIO transitions, V excluded from scoring)
```

42 BIO tags (ARG0-4, 16 ARGM modifier types). The predicate is marked
by adding a learned embedding at the encoder's embedding level — before
all 24 attention layers — so every layer can condition on which token is
the predicate. SRL uses `encoder.encoder()` directly (not `model.forward()`)
because the predicate embedding is injected before the encoder call.

**CLS — Dialog Act Classification**

```
ScalarMix(25 layers, free initialization)
    → AttentionPool:
        Linear(1024, 1) → masked softmax → weighted sum over tokens
    → LayerNorm(1024) → Linear(1024, 512) → GELU → Dropout(0.1) → Linear(512, 8)
    → argmax
```

8 dialog act labels: inform, request, question, confirm, reject, offer,
social, status. AttentionPool learns per-token importance weights,
outperforming mean pooling (+1.7 F1) and [CLS] token. Supports optional
previous utterance context via pair encoding: `tokenizer(prev_text, text)`.

## Training

### Bottom-Up Layer-Selective Training

Rather than fine-tuning the entire encoder for one task and then freezing
it, the encoder is shaped progressively from lower to upper layers:

**Phase 1 — POS**: All 24 encoder layers unfrozen at LR=2e-5 alongside the
POS head (LR=1e-3). This serves as a syntactic warm-up for the full
encoder. 10 epochs on UD EWT (12.5K sentences).

**Phase 2 — NER**: Layers 5-12 unfrozen. Layers 5-8 (overlap with POS) use
a reduced LR of 2e-5; layers 9-12 (fresh for NER) use 5e-5. NER head at
1e-3. 15 epochs on 195K SpanMarker silver labels with confidence weighting.

**Phase 3 — DEP**: Layers 12-18 unfrozen at LR=2e-5. DEP head at 1e-3.
20 epochs on UD EWT gold (12.5K sentences). POS and NER cascade features
are detached — DEP training does not affect POS or NER layers.

**Phase 4 — SRL**: Initially layers 18-24 unfrozen at LR=2e-5 (5 epochs).
Then all layers at ultra-low LR=3e-6 for 3 additional epochs to allow
semantic adaptation across the full encoder. SRL BiLSTM adapter trained
for 5 epochs on frozen encoder. Data: 200K AllenNLP silver + 41K PropBank
EWT gold (4x upsampled).

**Phase 5 — CLS**: Fully frozen encoder. Only ScalarMix + AttentionPool +
MLP trained at LR=1e-3. 5 epochs on 60K SGD + GPT dialog act data.

**Recovery**: After all phases, POS, NER, and DEP heads are retrained for
3 epochs each on the frozen final encoder to recover from minor degradation
caused by later phases modifying shared encoder layers.

### Training Data

| Data | Size | Used For |
|------|------|----------|
| UD English EWT | 12.5K sentences | POS, DEP |
| SpanMarker NER silver | 195K sentences | NER |
| AllenNLP SRL silver | 200K predicate-argument pairs | SRL |
| PropBank EWT gold | 41K predicate-argument pairs | SRL |
| SGD (Google) + GPT labels | 60K utterances | CLS |

## Checkpoint Format

Single `model.pt` file containing a dict of state dicts:

```python
{
    "deberta":        ...,  # 434M params (shared encoder)
    "pred_embedding": ...,  # 2K params (SRL predicate marker)
    "pos_scalar_mix": ..., "pos_head": ...,
    "ner_scalar_mix": ..., "ner_lstm": ..., "ner_proj": ..., "ner_head": ...,
    "dep_scalar_mix": ..., "dep_proj": ..., "dep_biaffine": ...,
    "classifier": ...,     # SRL MLP classifier
    "cls_scalar_mix": ..., "cls_pool": ..., "cls_head": ...,
}
```

Total head parameters: ~16M (3.6% of encoder). Adding a head costs
1-5M parameters and ~5ms latency.

## Requirements

```
transformers==5.6.2
torch>=2.0
seqeval
```

The model requires `transformers==5.6.2`. DeBERTa's internal computation
differs across library versions, producing incorrect outputs if the
version does not match.

## Labels

**POS**: ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X

**NER**: BIO tags for PERSON, NORP, FAC, ORG, GPE, LOC, PRODUCT, EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL, CARDINAL

**DEP**: 53 UD relations (root, nsubj, obj, obl, amod, det, case, conj, cc, advmod, aux, mark, xcomp, ccomp, acl, advcl, etc.)

**SRL**: BIO tags for V, ARG0-4, ARGM-TMP, ARGM-LOC, ARGM-MNR, ARGM-CAU, ARGM-PRP, ARGM-NEG, ARGM-ADV, ARGM-DIR, ARGM-DIS, ARGM-EXT, ARGM-MOD, ARGM-PRD, ARGM-GOL, ARGM-COM, ARGM-REC

**CLS**: inform, request, question, confirm, reject, offer, social, status

## Benchmark Methodology

### Evaluation Protocol

All benchmarks use the standard test splits of public datasets. No
training data from these benchmarks was used during model training.
The model was trained on separate data sources (UD EWT for POS/DEP,
SpanMarker silver for NER, AllenNLP silver + PropBank gold for SRL,
SGD + GPT for CLS).

- **POS/DEP**: Evaluated on UD English EWT test set (2,077 sentences).
  Standard benchmark for English syntactic analysis.
- **NER**: Evaluated on OntoNotes 5.0 test set (8,262 sentences) via
  `tner/ontonotes5` on HuggingFace. Uses the same 18 entity types as
  our model. Also evaluated on CoNLL-2003 (3,453 sentences) with entity
  type mapping for cross-benchmark comparison.
- **SRL**: Evaluated on PropBank EWT test set (1,269 predicate-argument
  examples). Gold annotations. V tags excluded from F1 scoring.
- **CLS**: Internal evaluation on SGD+GPT dev split (4,000 utterances).
  Cross-benchmark evaluation on DailyDialog test (7,740 utterances)
  with 8→4 label mapping.

### Decoding

- **POS, DEP, CLS**: argmax decoding
- **NER, SRL**: Viterbi decoding with constrained BIO transitions
  (I-X can only follow B-X or I-X of the same type)
- **DEP**: Biaffine arc scoring (greedy argmax, no MST decoding)

### Reproducing Benchmarks

```bash
# Step 1: Download benchmark datasets (some require HuggingFace access)
uv run python models/kniv-deberta-nlp-base-en-large/download_benchmarks.py

# Step 2: Run full benchmark suite
uv run python models/kniv-deberta-nlp-base-en-large/benchmark_standard.py

# Step 3: Run internal evaluation (on training dev sets)
uv run python models/kniv-deberta-nlp-base-en-large/evaluate_v5.py
```

The benchmark scripts are self-contained — they load the model, build all
head architectures, and run evaluation. No external dependencies beyond
`transformers==5.6.2`, `torch`, `seqeval`, `scikit-learn`, and `datasets`.

### Data Licensing for Benchmarks

| Benchmark | License | Usage |
|-----------|---------|-------|
| UD English EWT | CC-BY-SA-4.0 | Training + evaluation |
| OntoNotes 5.0 (tner) | LDC | Evaluation only |
| CoNLL-2003 | Reuters | Evaluation only |
| PropBank EWT | CC-BY-SA-4.0 | Training + evaluation |
| DailyDialog | CC-BY-NC-SA-4.0 | Evaluation only |

## Model Family

| Model | Heads | Status |
|-------|-------|--------|
| **kniv-deberta-nlp-base-en-large** | POS, NER, DEP, SRL, CLS | Released |
| kniv-deberta-nlp-tier1-en-large | +Lemma, Morph, Keyword | Planned |
| kniv-deberta-nlp-tier2-en-large | +Sentiment, Intent, Punct, Truecase | Planned |
| kniv-deberta-nlp-tier3-en-large | +QA, NLI, RelEx, Events, STS | Planned |
| kniv-deberta-nlp-base-en-base | Distilled student (DeBERTa-base) | Planned |

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
