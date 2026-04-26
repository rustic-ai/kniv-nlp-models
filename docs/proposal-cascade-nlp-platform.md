# Proposal: kniv-cascade — A Multi-Task NLP Understanding Platform

## Executive Summary

kniv-cascade is a single DeBERTa-v3-large encoder serving multiple NLP
task heads through layer-selective training and cascaded predictions.
Unlike single-purpose models, the cascade architecture produces rich,
structured analysis from a single inference pass — enabling downstream
applications from extraction to QA to reformulation without additional
models.

This document proposes the full evolution of kniv-cascade from its
current 5-head model into a comprehensive NLP understanding platform
spanning 20+ language tasks across 4 tiers.

```
                        One encoder. Many heads. One pass.

   ┌──────────────────────────────────────────────────────────────┐
   │                    DeBERTa-v3-large                           │
   │                    434M shared params                         │
   │                                                               │
   │  Layer 0-8 ──► POS, Lemma, Morph                             │
   │  Layer 5-12 ─► NER, PII, Keyword, Chunking                   │
   │  Layer 12-18 ► DEP, Discourse                                 │
   │  Layer 18-24 ► SRL, Event Detection                           │
   │  All layers ─► CLS, Sentiment, Intent, QA, Similarity, NLI   │
   └──────────────────────────────────────────────────────────────┘
                              │
                 Structured understanding
                              │
              ┌─────────��─────┼───────────────┐
              ▼               ▼               ▼
        Extraction     Comprehension     Generation
        (NER, SRL,     (QA, Fact         (Reformulation,
         Slots)         Verification)     Summarization)
```

---

## Part I: Foundation (Current — v4)

### 1.1 Shipped Heads

These heads are trained, evaluated, and production-ready.

| # | Head | Architecture | Score | Benchmark | Data License |
|---|------|-------------|-------|-----------|-------------|
| 1 | **POS** | ScalarMix(0-8) → Linear | 0.979 | UD EWT (standard) | CC-BY-SA-4.0 |
| 2 | **NER** | ScalarMix(5-10) → BiLSTM → MLP + POS cascade | 0.860 | SpanMarker silver | Generated |
| 3 | **DEP** | ScalarMix(12-18) → Biaffine + POS/NER cascade | 0.922 | UD EWT (standard) | CC-BY-SA-4.0 |
| 4 | **SRL** | pred_embedding → BiLSTM → MLP | 0.802 | PropBank EWT gold | CC-BY-SA-4.0 |
| 5 | **CLS** | ScalarMix → AttentionPool → MLP | 0.930 | SGD/GPT dev | CC-BY-SA-4.0 |

### 1.2 Core Architecture Principles

**ScalarMix**: Each task learns which encoder layers to read from.
Lower layers carry syntax (POS), middle layers carry entities (NER),
upper layers carry semantics (SRL). No manual layer assignment needed —
the model discovers optimal depths.

**Cascade**: Simpler tasks feed predictions to complex tasks.
POS → NER → DEP is a cascade where each head receives soft probabilities
from upstream heads. This provides explicit linguistic features that the
frozen encoder cannot adapt to provide.

**Layer-selective training**: Each task unfreezes only the encoder layers
it reads from. This preserves earlier tasks' representations while
allowing each layer range to specialize.

**Single-pass inference**: All heads share one encoder forward pass.
Adding a head costs ~1-5M params and ~5ms latency, not another 434M
params and ~50ms.

---

## Part II: Linguistic Foundation (Tier 1)

Low-effort additions that complete the core linguistic analysis pipeline.
These use existing training data and proven head architectures.

### 2.1 Lemmatization

```
Input:  "The companies were running multiple operations"
Output: "the company be run multiple operation"
```

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(0-6) + POS cascade → token classifier |
| Approach | Classification: predict edit operation per token (suffix rules) |
| Data | UD English EWT (12.5K, CC-BY-SA-4.0) — lemmas already annotated |
| Cascade input | POS (critical — lemma depends on part of speech) |
| Effort | Small — same data as POS, simple head |
| Params | ~50K |

**Design**: Rather than generating lemmas character-by-character, classify
each token into a lemma rule: `{-ing→∅, -s→∅, -ed→∅, -ies→-y, ...}`.
With ~200 rules covering 99% of English morphology, this becomes a
200-class classifier conditioned on POS.

```python
# Example lemma rules derived from UD EWT
LEMMA_RULES = {
    0: "",           # no change (already lemma)
    1: "-s",         # remove trailing s
    2: "-ed",        # remove trailing ed
    3: "-ing",       # remove trailing ing
    4: "-ies+y",     # companies → company
    5: "-er",        # bigger → big (+ rule)
    ...
}
# ~200 rules cover 99.2% of UD EWT tokens
# Remaining 0.8% handled by lookup table (irregular verbs etc.)
```

### 2.2 Morphological Features

```
Input:  "She runs"
Output: {"She": {PronType=Prs|Case=Nom|Person=3|Number=Sing|Gender=Fem},
         "runs": {VerbForm=Fin|Tense=Pres|Person=3|Number=Sing|Mood=Ind}}
```

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(0-6) + POS cascade → multi-label classifier |
| Data | UD English EWT (12.5K, CC-BY-SA-4.0) — morph features annotated |
| Cascade input | POS (morphology is POS-dependent) |
| Approach | Multi-label: predict each feature independently |
| Effort | Small — same data, similar to POS head |
| Params | ~100K |

**Feature inventory** (from UD English):
```
Definite  = {Def, Ind}
Gender    = {Fem, Masc, Neut}
Mood      = {Imp, Ind, Sub}
Number    = {Plur, Sing}
Person    = {1, 2, 3}
PronType  = {Art, Dem, Int, Prs, Rel}
Tense     = {Past, Pres}
VerbForm  = {Fin, Ger, Inf, Part}
...
# ~20 features, each with 2-6 possible values
```

**Design**: One binary classifier per feature value. Output is a set of
active features per token. Total ~80 binary outputs, but sparse (most
tokens have 2-4 active features).

### 2.3 Keyword / Keyphrase Extraction

```
Input:  "The Federal Reserve announced a rate cut affecting global markets"
Output: ["Federal Reserve", "rate cut", "global markets"]
```

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(8-14) + NER cascade �� BIO tagger |
| Data | Silver-label with KeyBERT (MIT) or YAKE (GPLv3 — use for labeling only) |
| Cascade input | POS (nouns/adjectives are keywords) + NER (entities are keywords) |
| Labels | BIO: B-KEY, I-KEY, O |
| Effort | Medium — need to generate silver labels on our corpus |
| Params | ~500K |

**Silver labeling pipeline**:
```python
from keybert import KeyBERT  # MIT license
kw_model = KeyBERT("all-MiniLM-L6-v2")

for doc in corpus:
    keywords = kw_model.extract_keywords(doc["text"], keyphrase_ngram_range=(1, 3))
    # Match keywords back to token positions → BIO tags
    bio_tags = align_keywords_to_tokens(doc["tokens"], keywords)
```

### 2.4 Summary: Tier 1

| Head | Data Ready | New Params | Effort | Depends On |
|------|-----------|------------|--------|-----------|
| Lemma | Yes (UD EWT) | ~50K | 1-2 days | POS |
| Morph | Yes (UD EWT) | ~100K | 1-2 days | POS |
| Keyword | Need silver | ~500K | 3-5 days | POS, NER |

**Total new params**: ~650K (0.15% of encoder)
**Completion**: ~1 week

---

## Part III: Production Heads (Tier 2)

High commercial value tasks. Each addresses a real production need.

### 3.1 PII Detection / De-identification

```
Input:  "Contact John Smith at john@example.com or 555-0123"
Output: "Contact [PERSON] at [EMAIL] or [PHONE]"
```

| Property | Value |
|----------|-------|
| Architecture | NER variant with PII-specific entity types |
| Labels | PERSON, EMAIL, PHONE, SSN, ADDRESS, DOB, CREDIT_CARD, IP_ADDRESS, etc. |
| Data | Presidio PII dataset (MIT), or silver-label with regex + NER |
| Cascade input | NER (PERSON entities overlap), POS |
| Effort | Medium — extend NER or train separate head |
| Params | ~1M (same as NER) |
| Commercial value | Very high — GDPR/CCPA compliance |

**Approach**: Two options:

A. **Extend NER**: Add PII types (EMAIL, PHONE, SSN, etc.) to the existing
   NER label set. Retrain NER with combined entity + PII data. Simpler but
   couples PII to general NER.

B. **Separate PII head**: Dedicated BIO tagger with its own ScalarMix.
   Can be trained independently, deployed separately, different update
   cadence from NER. Recommended for production.

**Regex bootstrapping**: Many PII types (email, phone, SSN, credit card)
have reliable regex patterns. Use these to auto-label the corpus, then
train a neural head that also catches edge cases regex misses.

### 3.2 Sentiment Analysis

```
Input:  "The camera quality is amazing but the battery life is terrible"
Output: {overall: negative, aspects: [{camera: positive}, {battery: negative}]}
```

| Property | Value |
|----------|-------|
| Architecture | Two sub-heads: sentence-level + aspect-level |
| Sentence-level | ScalarMix → AttentionPool → Linear(3) [pos/neg/neutral] |
| Aspect-level | NER-style aspect extraction + per-aspect CLS |
| Data | Amazon reviews (open, millions), or silver-label corpus |
| Cascade input | NER (aspect entities), CLS (utterance-level signal) |
| Effort | Medium (sentence-level: easy, aspect-level: more work) |
| Params | ~600K (sentence) + ~1.5M (aspect) |
| Commercial value | Very high — product analytics, brand monitoring |

**Data sources** (commercially usable):

| Dataset | Size | License | Notes |
|---------|------|---------|-------|
| Amazon reviews (McAuley) | 233M | Open (research + commercial) | Sentence-level sentiment |
| Yelp Open Dataset | 7M reviews | Open (Yelp terms) | Rich aspect annotations |
| SemEval ABSA (restaurants/laptops) | 6K | CC-BY-4.0 | Gold aspect-level annotations |
| Silver from TextBlob/VADER on corpus | Unlimited | MIT (tools) | Free but noisy |

### 3.3 Intent Classification (Fine-Grained)

```
Input:  "Can you change my seat to a window seat?"
Output: {intent: "change_seat", confidence: 0.94,
         slots: {seat_type: "window"}}
```

| Property | Value |
|----------|-------|
| Architecture | CLS variant with domain-specific label sets |
| Data | SNIPS (CC0), CLINC150 (CC-BY-3.0), Banking77 (CC-BY-4.0) |
| Cascade input | CLS (coarse act type), NER (slot entities) |
| Approach | Hierarchical: CLS→domain→fine_intent |
| Effort | Medium — multiple domain-specific heads |
| Params | ~500K per domain head |
| Commercial value | Very high — chatbots, voice assistants |

**Hierarchical intent**:
```
CLS (coarse):    request / question / inform / ...
Domain:          travel / banking / support / ...
Fine intent:     book_flight / cancel_reservation / check_balance / ...
```

The cascade's CLS head provides the coarse type; domain and fine intent
are additional heads or fine-tuned variants.

### 3.4 Punctuation Restoration

```
Input:  "so i went to the store and i bought some milk you know"  (ASR output)
Output: "So I went to the store, and I bought some milk, you know."
```

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(0-8) + POS+DEP cascade → 4-class per-token |
| Labels | NONE, COMMA, PERIOD, QUESTION (insert after token) |
| Data | Silver-label: strip punctuation from corpus, train to restore |
| Cascade input | POS (clause boundaries), DEP (sentence structure) |
| Effort | Small — silver labeling is trivial (remove + retrain) |
| Params | ~50K |
| Commercial value | High — ASR post-processing, transcription |

### 3.5 Truecasing

```
Input:  "the president of the united states met with apple ceo tim cook"
Output: "The President of the United States met with Apple CEO Tim Cook"
```

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(0-8) + NER cascade → binary per-token (upper/lower) |
| Data | Silver-label: lowercase corpus text, train to restore case |
| Cascade input | NER (proper nouns), POS (sentence boundaries) |
| Effort | Small — same pattern as punctuation restoration |
| Params | ~20K |
| Commercial value | Medium — ASR, OCR, text normalization |

### 3.6 Summary: Tier 2

| Head | Data Source | New Params | Effort | Commercial Value |
|------|-----------|------------|--------|-----------------|
| PII Detection | Presidio (MIT) + regex | ~1M | 1 week | Very High |
| Sentiment | Amazon/Yelp + SemEval | ~2M | 1-2 weeks | Very High |
| Intent (fine-grained) | SNIPS/CLINC/Banking77 | ~500K/domain | 1 week/domain | Very High |
| Punctuation | Self-supervised | ~50K | 2-3 days | High |
| Truecasing | Self-supervised | ~20K | 2-3 days | Medium |

**Total new params**: ~4M (0.9% of encoder)
**Completion**: ~4-6 weeks

---

## Part IV: Comprehension Heads (Tier 3)

Tasks that build on the foundation to enable deeper understanding.

### 4.1 Extractive Question Answering

```
Question: "Who founded Apple?"
Context:  "Apple was founded by Steve Jobs and Steve Wozniak in 1976."
Answer:   "Steve Jobs and Steve Wozniak"
```

| Property | Value |
|----------|-------|
| Architecture | ScalarMix(top) + NER+SRL cascade → start/end span |
| Data | SQuAD 2.0 (CC-BY-SA-4.0), Natural Questions (CC-BY-SA-3.0) |
| Cascade advantage | NER tells QA where entities are, SRL tells it argument roles |
| Effort | 1 week |
| Params | ~1.2M |

See `docs/design-extractive-qa.md` for full design.

### 4.2 Coreference Resolution

```
Input:  "John went to the store. He bought milk there."
Output: [("He", "John"), ("there", "the store")]
```

| Property | Value |
|----------|-------|
| Architecture | Mention detection (BIO) + pairwise linking (bilinear scorer) |
| Data | fastcoref silver (MIT) + LitBank gold (CC-BY-4.0) |
| Cascade input | NER (mention candidates), DEP (syntactic constraints) |
| Effort | 2-3 weeks (hardest head) |
| Params | ~3M |

**Architecture**:
```
Step 1: Mention detection — BIO tagger identifies candidate mentions
Step 2: Mention representation — pool span tokens via attention
Step 3: Pairwise scoring — bilinear(mention_i, mention_j) → coref score
Step 4: Clustering — greedy left-to-right antecedent selection
```

### 4.3 Relation Extraction

```
Input:  "Steve Jobs co-founded Apple in Cupertino."
Output: [(Steve Jobs, founded, Apple), (Apple, located_in, Cupertino)]
```

| Property | Value |
|----------|-------|
| Architecture | NER (entities) → entity pair classifier |
| Data | DocRED (MIT, 5K documents, 96 relation types) |
| Cascade input | NER (entity spans) + DEP (syntactic path between entities) |
| Effort | 2 weeks |
| Params | ~2M |

**Approach**: For each pair of NER entities in a sentence, classify the
relation type. The classifier receives:
- Entity pair representations (from ScalarMix)
- DEP path between entities (shortest dependency path)
- POS/NER types of both entities

### 4.4 Event Detection

```
Input:  "An earthquake struck central Turkey on Monday, killing 50 people."
Output: Event(trigger="struck", type=DISASTER,
              Place="central Turkey", Time="Monday",
              Casualties="50 people")
```

| Property | Value |
|----------|-------|
| Architecture | SRL variant — event trigger detection + argument extraction |
| Data | ACE 2005 (LDC — eval only) or silver from SRL frames |
| Cascade input | SRL (event = predicate, arguments = event roles) |
| Effort | 1-2 weeks |
| Params | ~1M |

**Insight**: Event detection is largely SRL with typed predicates. Our SRL
head already extracts predicate-argument structures. Event detection adds
event TYPE classification to the predicate and maps SRL roles (ARG0, ARG1,
ARGM-LOC) to event-specific roles (Agent, Patient, Location).

### 4.5 Natural Language Inference

```
Premise:    "A man is playing guitar on stage."
Hypothesis: "A musician is performing."
Label:      entailment
```

| Property | Value |
|----------|-------|
| Architecture | Sentence-pair → ScalarMix → AttentionPool both → MLP(3) |
| Data | MNLI (GLOML license, 433K), SNLI (CC-BY-SA-4.0, 570K) |
| Cascade input | SRL frames of both sentences (structural comparison) |
| Effort | 1 week |
| Params | ~1M |

### 4.6 Semantic Textual Similarity

```
Sentence A: "A cat is sitting on a mat."
Sentence B: "A feline rests on a rug."
Score:      4.2 / 5.0
```

| Property | Value |
|----------|-------|
| Architecture | Sentence-pair → ScalarMix → AttentionPool both → cosine or regression |
| Data | STS-B (CC-BY-SA-4.0, 8.6K), SICK (CC-BY-NC — eval only) |
| Effort | 3-5 days |
| Params | ~500K |

### 4.7 Fact Verification

```
Claim:    "The Eiffel Tower is in London."
Evidence: "The Eiffel Tower is a landmark in Paris, France."
Verdict:  REFUTED
```

| Property | Value |
|----------|-------|
| Architecture | NLI variant + NER entity matching |
| Data | FEVER (CC-BY-SA-3.0, 185K claims) |
| Cascade input | NER (entity alignment), SRL (claim decomposition) |
| Effort | 1-2 weeks |
| Params | ~1.5M |

### 4.8 Summary: Tier 3

| Head | Data License | New Params | Effort | Depends On |
|------|-------------|------------|--------|-----------|
| QA | CC-BY-SA-4.0 | ~1.2M | 1 week | NER, SRL |
| Coreference | MIT + CC-BY-4.0 | ~3M | 2-3 weeks | NER, DEP |
| Relation Extraction | MIT | ~2M | 2 weeks | NER, DEP |
| Event Detection | Silver from SRL | ~1M | 1-2 weeks | SRL |
| NLI | CC-BY-SA-4.0 | ~1M | 1 week | SRL (optional) |
| STS | CC-BY-SA-4.0 | ~500K | 3-5 days | — |
| Fact Verification | CC-BY-SA-3.0 | ~1.5M | 1-2 weeks | NER, NLI |

**Total new params**: ~10M (2.3% of encoder)
**Completion**: ~8-12 weeks

---

## Part V: Application Layer (Tier 4)

These are not model heads — they are **orchestration systems** that
compose outputs from multiple heads. No additional model weights needed.

### 5.1 Query Reformulation

Deterministic restructuring using SRL frames + templates.

See `docs/design-query-reformulation.md` for full design.

| Engine | Approach | Needs LLM | Available |
|--------|----------|-----------|-----------|
| SRL Templates | Structural transforms | No | With current heads |
| Structured Extraction | Intent + slot filling | No | With current heads |
| Query Decomposition | Multi-hop splitting | No | With QA head |
| LLM Enhanced | Cascade → structured LLM prompt | Yes | Anytime |

### 5.2 Extractive Summarization

```
Document (10 paragraphs) → Top 3 most important sentences
```

**Approach**: Score each sentence using cascade features:
```python
def sentence_importance(sentence_cascade):
    score = 0
    score += len(sentence_cascade["ner_entities"]) * 2    # entity density
    score += len(sentence_cascade["srl_frames"]) * 3      # event density
    score += (1 if sentence_cascade["cls"] == "inform" else 0)  # informative
    score += sentence_cascade["dep"]["tree_depth"] * 0.5  # structural complexity
    return score
```

No new model needed — pure scoring function on cascade outputs.

### 5.3 Information Extraction Pipeline

```
Document → {entities, relations, events, temporal ordering}
```

**Combines**: NER + Coref + Relation Extraction + Event Detection + SRL

```python
def extract_information(document):
    # 1. Run cascade on each sentence
    sentences = [cascade.predict(s) for s in document.sentences]

    # 2. Resolve coreference across sentences
    coref_chains = cascade.coref(document.text)

    # 3. Extract entities (deduplicated via coref)
    entities = merge_entities(sentences, coref_chains)

    # 4. Extract relations between entities
    relations = cascade.relations(sentences, entities)

    # 5. Extract events with arguments
    events = cascade.events(sentences)

    # 6. Order events temporally
    timeline = order_events(events)  # using ARGM-TMP

    return {
        "entities": entities,
        "relations": relations,
        "events": events,
        "timeline": timeline,
    }
```

### 5.4 Conversational Understanding

```
User:  "I booked a flight to Paris last week but I need to change it"
Agent: [intent: modify_booking, entities: {dest: Paris, timeframe: last week},
        dialog_act: request, sentiment: neutral,
        srl: change(ARG0=I, ARG1=it→flight to Paris)]
```

**Combines**: CLS (dialog act) + NER (entities) + SRL (actions) +
Coref (pronoun resolution) + Intent (fine-grained) + Sentiment

This is the full-stack conversational AI understanding layer.

### 5.5 Document Intelligence

```
Contract → {parties, obligations, dates, amounts, clauses}
```

**Combines**: NER (parties, dates, amounts) + CLS (clause classification) +
SRL (who must do what by when) + Relation Extraction (party→obligation)

### 5.6 Summary: Tier 4

| Application | Heads Used | New Params | Effort |
|-------------|-----------|------------|--------|
| Query Reformulation | POS+NER+SRL+DEP+CLS | 0 | 1-2 weeks (code) |
| Extractive Summarization | NER+SRL+CLS+DEP | 0 | 3-5 days (code) |
| Information Extraction | NER+Coref+RelEx+Events | 0 | 2-3 weeks (code) |
| Conversational Understanding | All heads | 0 | 2-3 weeks (code) |
| Document Intelligence | NER+CLS+SRL+RelEx | 0 | 2-4 weeks (code) |

**Total new params**: 0 (orchestration only)

---

## Part VI: Platform Architecture

### 6.1 Model Size Budget

```
Component                  Params        % of Total
─────────────────────────────────────────────────
DeBERTa-v3-large encoder   434,000,000   96.5%
pred_embedding                    2,048    0.0%

Tier 1 (Lemma+Morph+Key)       650,000    0.1%
Tier 2 (PII+Sent+Intent+...)  4,000,000    0.9%
Tier 3 (QA+Coref+RelEx+...)  10,200,000    2.3%
Current heads (v4)             16,000,000    0.4%
─────────────────────────────────────────────────
TOTAL                        ~450,000,000   100%
```

All 20+ heads add only ~3.5% to model size. The encoder dominates.

### 6.2 Inference Architecture

```
                    ┌─────────────────────���
   Input text ───►  │  Encoder (1 pass)    │  ~50ms on GPU
                    └──────────┬──────────┘
                               │ hidden_states (cached)
                    ┌──────────┴──────────┐
                    │  Head dispatcher      │
                    │                       │
                    │  Requested:           │
                    │  [pos, ner, srl]      │
                    │                       │
                    │  Auto-added cascade:  │
                    │  pos → ner (needs pos)│
                    │  srl (needs encoder)  │
                    └──────────┬──────────┘
                               │
              ��────────────────┼────────────────┐
              ▼                ▼                ▼
         POS head         NER head         SRL head
          ~1ms             ~2ms             ~3ms
              │                │                │
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Structured output    │
                    │  {pos, ner, srl}      │
                    └──────────────────────┘

   Total: ~56ms for 3 heads (vs ~150ms for 3 separate models)
```

**Head selection**: Users request specific heads. The dispatcher
automatically adds cascade dependencies (requesting NER auto-adds POS).

### 6.3 API Design

```python
from kniv import CascadeModel

model = CascadeModel.load("kniv-v4")

# Select specific heads
result = model.predict("Steve Jobs founded Apple in 1976.",
                       heads=["ner", "srl", "sentiment"])

# result.tokens = ["Steve", "Jobs", "founded", "Apple", "in", "1976", "."]
# result.ner    = ["B-PER", "I-PER", "O", "B-ORG", "O", "B-DATE", "O"]
# result.srl    = [{"predicate": "founded", "ARG0": "Steve Jobs",
#                   "ARG1": "Apple", "ARGM-TMP": "in 1976"}]
# result.sentiment = "neutral"

# All heads
result = model.predict(text, heads="all")

# Application-level (Tier 4)
from kniv import Reformulator, QA, InformationExtractor

reform = Reformulator(model)
reform.to_question("Steve Jobs founded Apple", target="who")
# → "Who founded Apple?"

qa = QA(model)
qa.answer("Who founded Apple?", context="Apple was founded by Steve Jobs...")
# → {"answer": "Steve Jobs", "score": 0.97}

ie = InformationExtractor(model)
ie.extract("Steve Jobs founded Apple in Cupertino in 1976.")
# → {"entities": [...], "relations": [...], "events": [...]}
```

---

## Part VII: Data Strategy

### 7.1 Commercially Licensed Data

All training data must be commercially usable. Current inventory:

| Data | License | Used For |
|------|---------|----------|
| UD English EWT | CC-BY-SA-4.0 | POS, DEP, Lemma, Morph |
| SpanMarker NER silver | Generated | NER |
| AllenNLP SRL silver | MIT | SRL |
| PropBank EWT | CC-BY-SA-4.0 | SRL gold |
| SGD (Google) | CC-BY-SA-4.0 | CLS |
| SQuAD 2.0 | CC-BY-SA-4.0 | QA |
| Natural Questions | CC-BY-SA-3.0 | QA |
| SNLI | CC-BY-SA-4.0 | NLI |
| DocRED | MIT | Relation Extraction |
| FEVER | CC-BY-SA-3.0 | Fact Verification |
| LitBank | CC-BY-4.0 | Coreference gold |
| fastcoref | MIT | Coreference silver |
| SemEval ABSA | CC-BY-4.0 | Aspect sentiment |
| SNIPS | CC0 | Intent classification |
| Banking77 | CC-BY-4.0 | Intent classification |
| CLINC150 | CC-BY-3.0 | Intent classification |
| Presidio PII | MIT | PII detection |
| STS-B | CC-BY-SA-4.0 | Semantic similarity |
| CANARD | CC-BY-SA-4.0 | Query reformulation |
| QReCC | MIT | Query reformulation |

### 7.2 Silver Labeling Strategy

For tasks without sufficient gold data, use teacher models to label
our corpus:

| Task | Teacher Model | License | Expected Quality |
|------|--------------|---------|-----------------|
| NER | SpanMarker RoBERTa-large | MIT | ~91 F1 on OntoNotes |
| SRL | AllenNLP BERT SRL | MIT | ~86 F1 on CoNLL-2012 |
| Keyword | KeyBERT | MIT | ~80% precision |
| Sentiment | TextBlob / VADER | MIT | ~75% accuracy |
| Coreference | fastcoref | MIT | ~80 F1 on OntoNotes |
| Punctuation | Self-supervised (remove + restore) | N/A | ~98% accuracy |
| Truecasing | Self-supervised (lowercase + restore) | N/A | ~99% accuracy |

### 7.3 Evaluation Benchmarks

| Task | Benchmark | License | Notes |
|------|-----------|---------|-------|
| POS | UD EWT test | CC-BY-SA-4.0 | Standard benchmark |
| NER | CoNLL-2003 test | DGfS (eval OK) | Most-cited NER benchmark |
| SRL | CoNLL-2012 test | CC-BY-NC-ND (eval OK) | Most-cited SRL benchmark |
| DEP | UD EWT test | CC-BY-SA-4.0 | Standard benchmark |
| QA | SQuAD 2.0 dev | CC-BY-SA-4.0 | Standard benchmark |
| NLI | MNLI matched/mismatched | GLOML | Standard benchmark |
| STS | STS-B test | CC-BY-SA-4.0 | Standard benchmark |
| Sentiment | SST-2 | Stanford (eval OK) | Standard benchmark |

---

## Part VIII: Implementation Roadmap

### Phase 1: Foundation (Current — Weeks 1-2)

```
[■■■■■■■■□□] v4 training on transformers 5.6.2
  ✓ POS  (0.979)
  ▶ NER  (training...)
  ○ DEP
  ○ SRL
  ○ CLS
```

### Phase 2: Linguistic Completion (Weeks 3-4)

```
  ○ Lemma          (UD EWT data ready)
  ○ Morph          (UD EWT data ready)
  ○ Keyword        (silver labeling needed)
  ○ Eval on standard benchmarks (CoNLL-2003, CoNLL-2012)
```

### Phase 3: Production Heads (Weeks 5-8)

```
  ○ PII Detection  (Presidio + regex)
  ○ Sentiment      (sentence + aspect)
  ○ Punctuation    (self-supervised)
  ○ Truecasing     (self-supervised)
  ○ Intent         (SNIPS/CLINC/Banking77)
```

### Phase 4: Comprehension (Weeks 9-14)

```
  ○ Extractive QA     (SQuAD 2.0)
  ○ Coreference       (fastcoref silver + LitBank gold)
  ○ NLI               (SNLI/MNLI)
  ○ Relation Extract   (DocRED)
  ○ Event Detection    (silver from SRL)
  ○ STS               (STS-B)
  ○ Fact Verification  (FEVER)
```

### Phase 5: Applications (Weeks 15-18)

```
  ○ Query Reformulation (SRL templates + structured extraction)
  ○ Extractive Summarization
  ○ Information Extraction pipeline
  ○ Conversational Understanding
  ○ Python SDK + API
```

### Phase 6: Optimization + Release (Weeks 19-22)

```
  ○ ONNX export + quantization (INT8)
  ○ Selective head loading (only load requested heads)
  ○ Benchmark all heads on standard test sets
  ○ Documentation + examples
  ○ HuggingFace Hub publication
  ○ API service deployment
```

---

## Part IX: Competitive Landscape

### 9.1 Existing Multi-Task NLP Systems

| System | Tasks | Architecture | Limitations |
|--------|-------|-------------|-------------|
| **spaCy** | POS, NER, DEP, Lemma | Pipeline (separate models) | No SRL, no CLS, no cascade |
| **Stanza** | POS, NER, DEP, Lemma, Morph | Pipeline | No SRL, no CLS, no QA |
| **Flair** | POS, NER, CLS | Stacked embeddings | No DEP, no SRL, no cascade |
| **AllenNLP** | Individual task models | Separate models | No shared encoder, deprecated |
| **Trankit** | POS, NER, DEP, Lemma | XLM-R pipeline | No SRL, no CLS, no cascade |
| **kniv-cascade** | POS, NER, DEP, SRL, CLS + 15 more | **Shared encoder + cascade** | English only (currently) |

### 9.2 Our Differentiators

1. **Single encoder, all tasks**: One forward pass produces POS + NER +
   DEP + SRL + CLS + more. Competitors run separate models for each task.

2. **Cascade architecture**: NER knows POS. DEP knows NER + POS.
   This inter-task communication improves each head beyond what isolated
   training achieves.

3. **Layer-selective training**: Each task tunes its optimal encoder layers.
   Lower layers specialize for syntax, upper for semantics. No task
   compromises another.

4. **SRL as a first-class citizen**: Most NLP toolkits skip SRL entirely.
   Our SRL head enables query reformulation, event detection, and
   structured extraction that competitors cannot do.

5. **Commercially licensed**: All training data is CC-BY-SA or MIT.
   spaCy models use OntoNotes (LDC, not redistributable) for NER.

6. **Structured output for downstream AI**: The cascade's output is
   designed to feed into LLMs, RAG pipelines, and dialog systems —
   not just for linguistic annotation.

---

## Part X: Resource Requirements

### 10.1 Training Compute

| Phase | GPU Hours | GPU Type | Est. Cost |
|-------|-----------|----------|-----------|
| Foundation (5 heads) | ~8 hours | A100/L4 | $8-16 (Colab) |
| Tier 1 (Lemma+Morph+Key) | ~2 hours | A100/L4 | $2-4 |
| Tier 2 (PII+Sent+Intent+Punct+True) | ~10 hours | A100/L4 | $10-20 |
| Tier 3 (QA+Coref+RelEx+Events+NLI+STS+Fact) | ~20 hours | A100/L4 | $20-40 |
| Total | ~40 hours | | ~$40-80 |

### 10.2 Model Size

| Configuration | Size (fp32) | Size (INT8) | Inference Speed |
|--------------|------------|------------|-----------------|
| Encoder only | 1.7 GB | 450 MB | ~50ms/sentence |
| + All Tier 1-3 heads | 1.8 GB | 470 MB | ~60ms/sentence |
| ONNX quantized (INT8) | — | ~450 MB | ~25ms/sentence |

### 10.3 Team

| Role | Effort | Notes |
|------|--------|-------|
| ML Engineer | 16-20 weeks | Training, evaluation, optimization |
| Data Engineer | 4-6 weeks | Silver labeling, data pipelines |
| Backend Engineer | 4-6 weeks | API, SDK, deployment |
| Total | ~22-28 person-weeks | |

---

## Appendix A: Complete Head Inventory

| # | Head | Type | Cascade From | Layer Range | Params | Status |
|---|------|------|-------------|-------------|--------|--------|
| 1 | POS | Token | — | 0-8 | 17K | Shipped |
| 2 | NER | Token | POS | 5-12 | 4.3M | Shipped |
| 3 | DEP | Token-pair | POS, NER | 12-18 | 5.9M | Shipped |
| 4 | SRL | Token | pred_embedding | 18-24 | 4.2M | Shipped |
| 5 | CLS | Sentence | — | All | 530K | Shipped |
| 6 | Lemma | Token | POS | 0-6 | 50K | Tier 1 |
| 7 | Morph | Token | POS | 0-6 | 100K | Tier 1 |
| 8 | Keyword | Token | POS, NER | 8-14 | 500K | Tier 1 |
| 9 | PII | Token | NER | 5-12 | 1M | Tier 2 |
| 10 | Sentiment | Sentence | CLS | All | 600K | Tier 2 |
| 11 | Aspect Sentiment | Token+Sent | NER, CLS | All | 1.5M | Tier 2 |
| 12 | Intent | Sentence | CLS | All | 500K | Tier 2 |
| 13 | Punctuation | Token | POS, DEP | 0-8 | 50K | Tier 2 |
| 14 | Truecasing | Token | NER, POS | 0-8 | 20K | Tier 2 |
| 15 | QA | Span | NER, SRL | 18-24 | 1.2M | Tier 3 |
| 16 | Coreference | Span-pair | NER, DEP | All | 3M | Tier 3 |
| 17 | Relation Extraction | Entity-pair | NER, DEP | 12-18 | 2M | Tier 3 |
| 18 | Event Detection | Token | SRL | 18-24 | 1M | Tier 3 |
| 19 | NLI | Sentence-pair | SRL | All | 1M | Tier 3 |
| 20 | STS | Sentence-pair | — | All | 500K | Tier 3 |
| 21 | Fact Verification | Sentence-pair | NER, NLI | All | 1.5M | Tier 3 |

**Total heads**: 21
**Total head params**: ~28M (6% of encoder)
**Encoder**: 434M (shared across all heads)
