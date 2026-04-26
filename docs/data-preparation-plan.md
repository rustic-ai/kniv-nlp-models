# Data Preparation Plan — kniv-cascade NLP Platform

## Overview

This document defines the data preparation pipeline for all current and
planned task heads. Every dataset must be **commercially licensed** — no
LDC-only, research-only, or NonCommercial data for training. Evaluation
on restricted benchmarks is permitted under fair use.

---

## Part I: License Audit

### Commercially Safe for Training

| Dataset | License | Verified | Used For |
|---------|---------|----------|----------|
| UD English EWT | CC-BY-SA-4.0 | Yes | POS, DEP, Lemma, Morph |
| PropBank EWT | CC-BY-SA-4.0 | Yes | SRL gold |
| SGD (Google) | CC-BY-SA-4.0 | Yes | CLS |
| SQuAD 2.0 | CC-BY-SA-4.0 | Yes | QA |
| Natural Questions | CC-BY-SA-3.0 | Yes | QA |
| SNLI | CC-BY-SA-4.0 | Yes | NLI |
| MNLI | OANC (permissive) | Yes | NLI |
| DocRED | CC0-1.0 | Yes | Relation Extraction |
| LitBank | CC-BY-4.0 | Yes | Coreference gold |
| SNIPS | CC0-1.0 | Yes | Intent classification |
| CLINC150 | CC-BY-3.0 | Yes | Intent classification |
| Banking77 | CC-BY-4.0 | Yes | Intent classification |
| CANARD | CC-BY-SA-4.0 | Yes | Query reformulation eval |
| QReCC | CC-BY-SA-3.0 | Yes | Query reformulation eval |

### Safe Tools for Silver Labeling

| Tool | License | Use | Notes |
|------|---------|-----|-------|
| SpanMarker | Apache-2.0 | NER silver labels | Model outputs are ours |
| AllenNLP SRL | Apache-2.0 | SRL silver labels | Inference OK; don't redistribute model training data |
| KeyBERT | MIT | Keyword silver labels | |
| fastcoref | MIT | Coreference silver labels | Inference OK |
| Presidio | MIT | PII regex patterns | |
| TextBlob | MIT | Sentiment silver labels | Low quality — supplement with better approach |

### Blocked for Commercial Training

| Dataset | License | Issue | Alternative |
|---------|---------|-------|-------------|
| Amazon Reviews (McAuley) | Amazon TOS | Commercial use prohibited | Silver-label our corpus |
| Yelp Open Dataset | Academic-only | Paid commercial license required | Silver-label our corpus |
| SemEval ABSA | CC-BY-NC-SA-3.0 | NonCommercial | Silver-label aspects with NER + sentiment |
| ACE 2005 | LDC ($3K+) | Paid membership required | Silver-label events from SRL |
| CoNLL-2003 (training) | Reuters copyright | Text not redistributable | Already have SpanMarker silver |
| CoNLL-2012 (training) | LDC | Not freely available | Already have AllenNLP silver |

### Requires Caution

| Dataset | License | Issue | Recommendation |
|---------|---------|-------|---------------|
| YAKE | GPLv3 | Copyleft — use as external tool only | Never bundle in product; use KeyBERT (MIT) instead |
| STS-B | Mixed | Scores CC-BY-SA, but source text includes Reuters/AP | Use scores for eval only; train on SNLI/MNLI pairs instead |
| FEVER | Likely CC-BY-SA-3.0 | Some sources claim CC-BY-NC | Verify from official fever.ai before training |
| SST-2 | Custom (Stanford) | Rotten Tomatoes text, unclear commercial status | Eval only |

### Evaluation-Only Benchmarks

These are used for benchmark reporting, not training:

| Benchmark | License | Task | Access |
|-----------|---------|------|--------|
| CoNLL-2003 test | Reuters (research eval OK) | NER | NIST distribution |
| CoNLL-2012 test | LDC (eval with membership) | SRL, Coref | LDC catalog |
| SST-2 | Custom | Sentiment | HuggingFace `datasets` |
| UD EWT test | CC-BY-SA-4.0 | POS, DEP | Same as training |
| SQuAD 2.0 dev | CC-BY-SA-4.0 | QA | HuggingFace `datasets` |
| MNLI matched/mismatched | OANC | NLI | HuggingFace `datasets` |

---

## Part II: Current Data Inventory

### What We Have

| File | Examples | Size | Used For | Status |
|------|----------|------|----------|--------|
| `ud_train.json` | 12,544 | 10.9 MB | POS, DEP | Ready |
| `ud_dev.json` | 2,001 | 1.4 MB | POS, DEP eval | Ready |
| `ner_spanmarker_train.json` | 195,120 | 132 MB | NER (with ner_weights) | Ready |
| `ner_spanmarker_dev.json` | 84,495 | 58 MB | NER eval | Ready |
| `srl_allennlp_silver.json` | 396,876 | 246 MB | SRL silver | Ready |
| `srl_train.json` | 40,872 | 22 MB | SRL gold | Ready |
| `srl_dev.json` | 1,236 | 0.6 MB | SRL eval | Ready |
| `cls_sgd_mwoz_train.json` | 64,000 | 10.9 MB | CLS | Ready |
| `label_vocabs.json` | — | — | Label mappings | Ready |

### What's Missing

| File | For | Source | Blocked By |
|------|-----|--------|-----------|
| Lemma annotations in `ud_train.json` | Lemma head | UD EWT CoNLL-U (CC-BY-SA-4.0) | Need to re-extract |
| Morph annotations in `ud_train.json` | Morph head | UD EWT CoNLL-U (CC-BY-SA-4.0) | Need to re-extract |
| Keyword BIO labels | Keyword head | KeyBERT (MIT) on our corpus | Need to generate |
| SQuAD 2.0 converted | QA head | HuggingFace (CC-BY-SA-4.0) | Need to download + convert |
| Coreference silver labels | Coref head | fastcoref (MIT) on our corpus | Need to generate |
| Sentiment silver labels | Sentiment head | Silver-label our corpus | Need to generate |
| Intent data (SNIPS+CLINC+Banking77) | Intent head | HuggingFace (CC0/CC-BY) | Need to download + merge |
| NLI data (SNLI+MNLI) | NLI head | HuggingFace (CC-BY-SA) | Need to download + convert |
| DocRED converted | RelEx head | HuggingFace (CC0) | Need to download + convert |
| CANARD/QReCC converted | Reformulation eval | HuggingFace (CC-BY-SA) | Need to download + convert |

---

## Part III: Preparation Scripts

### Script 1: Extend UD EWT with Lemma + Morph

**Purpose**: Add `lemmas` and `morph_features` fields to existing `ud_train.json`
and `ud_dev.json` by re-parsing the original CoNLL-U files.

**Source**: `/data/ud-english-ewt/en_ewt-ud-{train,dev,test}.conllu`
**License**: CC-BY-SA-4.0
**Output**: Updated `ud_train.json`, `ud_dev.json` with additional fields

```
CoNLL-U columns:
  1  ID       → (index)
  2  FORM     → words (already extracted)
  3  LEMMA    → lemmas (NEW)
  4  UPOS     → pos_tags (already extracted)
  5  XPOS     → (skip)
  6  FEATS    → morph_features (NEW)
  7  HEAD     → heads (already extracted)
  8  DEPREL   → deprels (already extracted)
  9  DEPS     → (skip)
  10 MISC     → (skip)
```

**New fields per example**:
```json
{
    "words": ["The", "companies", "were", "running"],
    "lemmas": ["the", "company", "be", "run"],
    "morph_features": [
        {"Definite": "Def", "PronType": "Art"},
        {"Number": "Plur"},
        {"Mood": "Ind", "Number": "Plur", "Tense": "Past", "VerbForm": "Fin"},
        {"Tense": "Pres", "VerbForm": "Part"}
    ],
    "pos_tags": ["DET", "NOUN", "AUX", "VERB"],
    ...existing fields...
}
```

**Lemma rule extraction**: Also generate a lemma rule vocabulary:
```python
# For each (word, lemma) pair, compute the edit rule
# e.g., ("companies", "company") → rule: "-ies+y"
# e.g., ("running", "run") → rule: "-ning+∅" or lookup
# Build rule vocabulary: ~200-300 rules cover 99%+ of English
```

**Script**: `models/kniv-deberta-cascade-large-nlp-en/prepare_ud_extended.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_ud_extended.py \
    --conllu-dir data/ud-english-ewt \
    --output-dir data/prepared/kniv-deberta-cascade \
    --extract-lemma-rules
```

**Effort**: 2-3 hours
**Dependencies**: `conllu` parser (already installed)

---

### Script 2: Keyword Silver Labels

**Purpose**: Generate BIO keyword/keyphrase labels for our corpus using KeyBERT.

**Source**: `corpus/output/annotated/{domain}/annotated.jsonl`
**Tool**: KeyBERT (MIT license)
**Output**: `keyword_train.json`, `keyword_dev.json`

**Pipeline**:
```
For each sentence in corpus:
  1. Run KeyBERT with keyphrase_ngram_range=(1, 3)
  2. Get top-k keyphrases with scores
  3. Match keyphrases back to token positions
  4. Generate BIO tags: B-KEY, I-KEY, O
  5. Store confidence scores as weights
```

**Output format**:
```json
{
    "words": ["The", "Federal", "Reserve", "announced", "a", "rate", "cut"],
    "keyword_tags": ["O", "B-KEY", "I-KEY", "O", "O", "B-KEY", "I-KEY"],
    "keyword_weights": [1.0, 0.92, 0.92, 1.0, 1.0, 0.87, 0.87]
}
```

**Filtering**:
- Minimum keyphrase confidence: 0.3
- Maximum 5 keyphrases per sentence
- Skip sentences with < 5 words
- Target: 200K examples from mixed domains

**Script**: `models/kniv-deberta-cascade-large-nlp-en/label_keywords.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/label_keywords.py \
    --corpus-dir corpus/output/annotated \
    --output data/prepared/kniv-deberta-cascade/keyword_train.json \
    --max-sentences 200000
```

**Effort**: 4-6 hours (mostly GPU inference for KeyBERT)
**Dependencies**: `pip install keybert`

---

### Script 3: Sentiment Silver Labels

**Purpose**: Generate sentiment labels for our corpus. Since Amazon/Yelp/SemEval
are all blocked for commercial use, we must silver-label our own data.

**Strategy**: Use a commercially-safe sentiment model as teacher:

| Option | Model | License | Quality |
|--------|-------|---------|---------|
| A | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Apache-2.0 | Good for short text |
| B | `nlptown/bert-base-multilingual-uncased-sentiment` | MIT | 5-star rating → 3-class |
| C | TextBlob rule-based | MIT | Low quality, but free |

**Recommended**: Option A (Cardiff RoBERTa, Apache-2.0) as primary teacher,
with TextBlob as a cross-check filter. Keep only examples where both agree.

**Output format**:
```json
{
    "text": "The camera quality is amazing but battery is terrible",
    "sentiment": "mixed",
    "confidence": 0.78,
    "sentence_sentiments": [
        {"span": "camera quality is amazing", "label": "positive", "confidence": 0.95},
        {"span": "battery is terrible", "label": "negative", "confidence": 0.92}
    ]
}
```

**For sentence-level sentiment**:
- Labels: positive, negative, neutral (3-class)
- Source: our corpus sentences scored by teacher model
- Filter: confidence > 0.8
- Target: 100K examples, balanced across classes

**For aspect-level sentiment** (future):
- Use our NER head to find aspect terms
- Score each aspect's surrounding context with teacher model
- Requires NER to be trained first

**Script**: `models/kniv-deberta-cascade-large-nlp-en/label_sentiment.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/label_sentiment.py \
    --corpus-dir corpus/output/annotated \
    --teacher cardiffnlp/twitter-roberta-base-sentiment-latest \
    --output data/prepared/kniv-deberta-cascade/sentiment_train.json \
    --max-sentences 100000 --min-confidence 0.8
```

**Effort**: 4-6 hours
**Dependencies**: Teacher model download (~500 MB)

---

### Script 4: SQuAD 2.0 Conversion

**Purpose**: Download SQuAD 2.0 and convert to our training format with
token-level span positions.

**Source**: HuggingFace `datasets` library → `rajpurkar/squad_v2`
**License**: CC-BY-SA-4.0

**Output format**:
```json
{
    "question": "Who founded Apple?",
    "context_words": ["Apple", "was", "founded", "by", "Steve", "Jobs", "..."],
    "start_position": 4,
    "end_position": 5,
    "is_impossible": false,
    "answer_text": "Steve Jobs"
}
```

**Processing**:
- Tokenize with DeBERTa tokenizer
- Map character-level `answer_start` to token positions
- Handle long contexts with sliding window (max_length=384, stride=128)
- Split unanswerable questions (is_impossible=True)
- Train/dev split: use official SQuAD split

**Script**: `models/kniv-deberta-cascade-large-nlp-en/prepare_squad.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_squad.py \
    --output-dir data/prepared/kniv-deberta-cascade
```

**Effort**: 1-2 hours
**Dependencies**: `datasets` library

---

### Script 5: Intent Classification Data Merge

**Purpose**: Download and merge SNIPS + CLINC150 + Banking77 into a unified
intent classification dataset.

**Sources**:
- SNIPS: `snips_built_in_intents` on HuggingFace (CC0)
- CLINC150: `clinc_oos` on HuggingFace (CC-BY-3.0)
- Banking77: `banking77` on HuggingFace (CC-BY-4.0)

**Label strategy**: Hierarchical intent taxonomy:
```
Level 1 (domain):  travel / banking / general / ...
Level 2 (intent):  book_flight / check_balance / weather / ...
```

Map each dataset's intents into a unified taxonomy.

**Output format**:
```json
{
    "text": "Can you change my seat to a window seat?",
    "domain": "travel",
    "intent": "change_seat",
    "source": "snips"
}
```

**Script**: `models/kniv-deberta-cascade-large-nlp-en/prepare_intent.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_intent.py \
    --output data/prepared/kniv-deberta-cascade/intent_train.json
```

**Effort**: 2-3 hours
**Dependencies**: `datasets` library

---

### Script 6: Coreference Silver Labels

**Purpose**: Generate coreference chains for our corpus using fastcoref.

**Source**: `corpus/output/annotated/{domain}/annotated.jsonl`
**Tool**: fastcoref (MIT)
**Gold supplement**: LitBank (CC-BY-4.0, 210K tokens, 100 documents)

**Output format**:
```json
{
    "words": ["John", "went", "to", "the", "store", ".", "He", "bought", "milk", "."],
    "coref_clusters": [
        [[0, 0], [6, 6]],
    ],
    "mention_spans": [[0, 0], [4, 4], [6, 6], [8, 8]]
}
```

Where `[start, end]` are inclusive token indices and each cluster is a list
of mention spans that refer to the same entity.

**Pipeline**:
```
1. Run fastcoref on each document (multi-sentence)
2. Map character offsets to token positions
3. Filter: keep only clusters with >= 2 mentions
4. Filter: keep only documents with >= 1 cluster
5. Add LitBank gold annotations (100 documents)
```

**Script**: `models/kniv-deberta-cascade-large-nlp-en/label_coref.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/label_coref.py \
    --corpus-dir corpus/output/annotated \
    --litbank-dir data/litbank \
    --output data/prepared/kniv-deberta-cascade/coref_train.json \
    --max-documents 50000
```

**Effort**: 6-8 hours (fastcoref is slow on long documents)
**Dependencies**: `pip install fastcoref`

---

### Script 7: NLI Data Download + Conversion

**Purpose**: Download SNLI + MNLI and convert to our format.

**Sources**:
- SNLI: HuggingFace `stanfordnlp/snli` (CC-BY-SA-4.0)
- MNLI: HuggingFace `nyu-mll/multi_nli` (OANC, permissive)

**Output format**:
```json
{
    "premise": "A man is playing guitar on stage.",
    "hypothesis": "A musician is performing.",
    "label": "entailment",
    "source": "snli"
}
```

**Processing**:
- Filter out examples with label=-1 (no gold label)
- Merge SNLI train + MNLI train
- Keep separate dev sets for evaluation
- Labels: entailment, contradiction, neutral

**Script**: `models/kniv-deberta-cascade-large-nlp-en/prepare_nli.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_nli.py \
    --output-dir data/prepared/kniv-deberta-cascade
```

**Effort**: 1-2 hours
**Dependencies**: `datasets` library

---

### Script 8: DocRED Relation Extraction Conversion

**Purpose**: Download DocRED and convert to entity-pair classification format.

**Source**: HuggingFace `docred` (CC0-1.0)
**License**: CC0 — fully public domain

**Output format**:
```json
{
    "words": ["Steve", "Jobs", "co-founded", "Apple", "in", "Cupertino", "."],
    "entities": [
        {"start": 0, "end": 1, "type": "PERSON", "text": "Steve Jobs"},
        {"start": 3, "end": 3, "type": "ORG", "text": "Apple"},
        {"start": 5, "end": 5, "type": "GPE", "text": "Cupertino"}
    ],
    "relations": [
        {"head": 0, "tail": 1, "relation": "founder"},
        {"head": 1, "tail": 2, "relation": "headquarters_location"}
    ]
}
```

**Processing**:
- Convert document-level annotations to sentence-level
- Map entity spans to token positions
- Filter: keep only intra-sentence relations (cross-sentence requires coref)
- 96 relation types in original DocRED

**Script**: `models/kniv-deberta-cascade-large-nlp-en/prepare_docred.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_docred.py \
    --output data/prepared/kniv-deberta-cascade/docred_train.json
```

**Effort**: 2-3 hours
**Dependencies**: `datasets` library

---

### Script 9: Punctuation + Truecasing (Self-Supervised)

**Purpose**: Generate training data by degrading our corpus and training
to restore. No external dataset needed.

**Source**: `corpus/output/annotated/{domain}/annotated.jsonl`
**License**: Ours — fully commercial

**Punctuation pipeline**:
```
Original:  "Hello, how are you? I'm fine."
Degraded:  "hello how are you i'm fine"
Labels:    [NONE, COMMA, NONE, NONE, QUESTION, NONE, PERIOD]
           (insert after each token)
```

**Truecasing pipeline**:
```
Original:  "Steve Jobs founded Apple in Cupertino."
Degraded:  "steve jobs founded apple in cupertino."
Labels:    [UPPER, UPPER, LOWER, UPPER, LOWER, UPPER, LOWER]
           (first character case of each token)
```

**Script**: `models/kniv-deberta-cascade-large-nlp-en/prepare_punct_truecase.py`

```bash
uv run python models/kniv-deberta-cascade-large-nlp-en/prepare_punct_truecase.py \
    --corpus-dir corpus/output/annotated \
    --output-dir data/prepared/kniv-deberta-cascade \
    --max-sentences 200000
```

**Effort**: 1-2 hours (CPU only, no model needed)
**Dependencies**: None

---

## Part IV: Preparation Schedule

### Priority Order (follows training roadmap)

```
Week 1 — Foundation training (POS, NER, DEP, SRL, CLS)
  Data: All ready ✓

Week 2 — Tier 1 prep
  [1] prepare_ud_extended.py      Lemma + Morph from CoNLL-U        2 hours
  [2] label_keywords.py           Keyword silver via KeyBERT         4 hours

Week 3 — Tier 2 prep
  [3] label_sentiment.py          Sentiment silver via Cardiff       4 hours
  [4] prepare_intent.py           SNIPS + CLINC + Banking77 merge    2 hours
  [5] prepare_punct_truecase.py   Self-supervised punct + truecase   1 hour

Week 4 — Tier 3 prep
  [6] prepare_squad.py            SQuAD 2.0 conversion               1 hour
  [7] prepare_nli.py              SNLI + MNLI download + merge        1 hour
  [8] label_coref.py              Coreference silver via fastcoref    6 hours
  [9] prepare_docred.py           DocRED relation extraction          2 hours

Total prep time: ~23 hours
```

### Data Size Estimates

| Dataset | Est. Examples | Est. Size | Prep Script |
|---------|-------------|-----------|-------------|
| UD EWT + Lemma/Morph | 12,544 | 15 MB | prepare_ud_extended.py |
| Keyword silver | 200,000 | 120 MB | label_keywords.py |
| Sentiment silver | 100,000 | 40 MB | label_sentiment.py |
| Intent (merged) | 50,000 | 15 MB | prepare_intent.py |
| Punctuation self-sup | 200,000 | 60 MB | prepare_punct_truecase.py |
| Truecasing self-sup | 200,000 | 60 MB | prepare_punct_truecase.py |
| SQuAD 2.0 | 130,000 | 80 MB | prepare_squad.py |
| NLI (SNLI+MNLI) | 940,000 | 200 MB | prepare_nli.py |
| Coreference silver | 50,000 docs | 150 MB | label_coref.py |
| DocRED relations | 56,000 | 50 MB | prepare_docred.py |

### Upload Plan

All prepared data should be uploaded to HuggingFace under
`dragonscale-ai/kniv-corpus-en` in the `prepared/kniv-deberta-cascade/`
directory, following existing conventions:

```
prepared/kniv-deberta-cascade/
  ├── ud_train.json            (existing, to be updated with lemma/morph)
  ├── ud_dev.json              (existing, to be updated)
  ├── keyword_train.json       (new)
  ├── keyword_dev.json         (new)
  ├── sentiment_train.json     (new)
  ├── sentiment_dev.json       (new)
  ├── intent_train.json        (new)
  ├── intent_dev.json          (new)
  ├── punct_train.json         (new)
  ├── truecase_train.json      (new)
  ├── squad_train.json         (new)
  ├── squad_dev.json           (new)
  ├── nli_train.json           (new)
  ├── nli_dev.json             (new)
  ├── coref_train.json         (new)
  ├── coref_dev.json           (new)
  ├── docred_train.json        (new)
  ├── docred_dev.json          (new)
  └── label_vocabs.json        (to be updated with new label sets)
```

---

## Part V: Label Vocabulary Extensions

The `label_vocabs.json` file needs to be extended with new label sets:

```json
{
    "pos_labels": [...],
    "ner_labels": [...],
    "dep_labels": [...],
    "cls_labels": [...],
    "srl_labels": [...],

    "lemma_rules": ["", "-s", "-ed", "-ing", "-ies+y", ...],
    "morph_features": ["Definite=Def", "Definite=Ind", "Gender=Fem", ...],
    "keyword_labels": ["O", "B-KEY", "I-KEY"],
    "sentiment_labels": ["positive", "negative", "neutral"],
    "intent_labels": {
        "domains": ["travel", "banking", "general", ...],
        "intents": ["book_flight", "check_balance", "weather", ...]
    },
    "punct_labels": ["NONE", "COMMA", "PERIOD", "QUESTION", "EXCLAIM", "COLON", "SEMICOLON"],
    "truecase_labels": ["LOWER", "UPPER", "ALL_CAPS", "MIXED"],
    "nli_labels": ["entailment", "contradiction", "neutral"],
    "coref_mention_labels": ["O", "B-MENTION", "I-MENTION"],
    "docred_relation_labels": ["no_relation", "founder", "headquarters_location", ...]
}
```

---

## Part VI: Quality Assurance

### Silver Label Quality Checks

For every silver-labeled dataset, run quality checks before training:

1. **Sample review**: Manually review 100 random examples per dataset
2. **Distribution check**: Verify label distribution is reasonable
   (not 99% O tags, not 50% one class)
3. **Cross-validation**: Compare silver labels against any available gold
   on overlapping data
4. **Confidence calibration**: Plot confidence histograms, verify filtering
   threshold removes noisy examples

### Reproducibility

Every prep script must:
- Set `random.seed(42)` for reproducibility
- Log dataset statistics (size, label distribution, domain breakdown)
- Save metadata alongside data (source, tool version, timestamp)
- Be idempotent (running twice produces same output)

### Data Versioning

Use a version suffix on HuggingFace when updating existing files:
```
v1: Original preparation (current)
v2: Added lemma/morph to UD data
v3: Added keyword/sentiment/intent data
```

Tag each upload with git commit hash of the prep script used.
