# Application-Layer vs Model-Layer Capabilities

A guide to deciding which retrieval/NLP capabilities belong in the cascade
model versus the application code that consumes its outputs.

## TL;DR

Several feature requests from downstream consumers (e.g. uniko) fit more
cleanly in the **application layer** than as cascade model heads. The cascade
model is best used for **per-token classification tasks that share contextual
encoder representations**. Other capabilities — structured generation,
reference-time-dependent operations, cross-document state, sentence-pair
reasoning — are more cleanly handled where they live: in the app code that
consumes our model's outputs.

This isn't a "no" to those capabilities; it's a "yes, but architecturally
elsewhere." Splitting the workload this way makes both layers cleaner, faster
to iterate on, and easier to maintain.

## The decision principle

A capability belongs in the **cascade model** when:

- It's a per-token or per-span **classification or regression** decision.
- It benefits from **contextual encoder representations** shared across other
  tasks.
- It runs on a single sentence or utterance (≤128 tokens).
- It's deterministic-ish given the encoder context.

A capability belongs in the **application layer** when:

- It produces **structured output** (dates, graphs, links) rather than discrete
  labels.
- It depends on **runtime context** the model can't see (timestamps,
  conversation history, user state).
- It operates **cross-sentence** or **cross-document**.
- It's a **sentence-pair** task with a different input shape (NLI).
- A **mature off-the-shelf component** (rule-based or pretrained) outperforms
  what we'd train.

## Specific recommendations

### TIMEX3 normalization → application layer

The cascade model **already gives you what's needed**: NER tags `DATE`/`TIME`
and the SRL tag `ARGM-TMP`. These mark the *spans* of temporal expressions in
the text.

What's missing — turning "yesterday" into `2026-04-28T00:00:00` — is
**structured value generation that depends on a reference timestamp**, which
the model itself doesn't have.

**Use SUTime, HeidelTime, or dateparser at the application level.** These are
mature, well-tested libraries that handle the full TIMEX3 spec (including SET
expressions like "every Sunday"). They expect `(text, reference_time)` and
produce a normalized value.

```python
# Application-layer flow
spans = cascade_model.extract_temporal_spans(message)  # uses NER + SRL
for span in spans:
    normalized = sutime.parse(span.text, reference=message.timestamp)
    observation.timex3 = normalized
```

**Why not in the model**: A small ML model could memorize common forms but
would be brittle on edge cases ("the Tuesday before last", "two weeks from this
Friday"). SUTime handles these via robust grammar rules. ML models for
temporal normalization typically *defer to* rule-based libs anyway.

**Effort**: hours, not weeks. Ship by integrating SUTime as a post-processor on
cascade outputs.

### NLI / contradiction detection → standalone fine-tuned model

NLI is fundamentally a **sentence-pair classification task**: input is two
sentences, output is one of three labels (entailment, contradiction, neutral).
This shape doesn't fit our cascade — our heads operate on *single utterances*
with shared encoder representations.

**Recommendation**: Use a separate `deberta-v3-base-mnli` (or similar)
off-the-shelf and call it as a service when contradiction checking is needed.

```python
# Application-layer flow
new_obs = "I think Caroline started a dance studio"
existing = "Caroline doesn't run any business"
result = nli_model.classify(new_obs, existing)  # → contradiction
```

**Why standalone**: Contradiction checks happen rarely (only when comparing a
new statement against existing observations), so loading a dedicated model is
more efficient than carrying NLI capacity in every cascade forward pass.
Off-the-shelf models like `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` are
already excellent — fine-tuning may not even be needed.

**Effort**: 1-2 days to integrate, possibly zero if off-the-shelf is good
enough.

### Cross-sentence coreference → application-layer service (fastcoref)

The cascade model can handle **within-sentence coreference** as a head (see
"Stays in the cascade model" below). But cross-sentence and document-level
coref needs:

- Context windows >128 tokens.
- Mention detection + clustering algorithms different from per-token
  classification.
- Domain-specific training data we don't have.

**Recommendation**: Use `fastcoref` (or `LongCoref` for very long contexts) as
an external service.

```python
# Application-layer flow
within_sentence_coref = cascade_model.coref_clusters(message)  # in-sentence
doc_level_coref = fastcoref.predict(conversation_text)         # cross-message
merged = merge_cluster_ids(within_sentence_coref, doc_level_coref)
```

**Why not in cascade**: fastcoref is trained on OntoNotes (300K+ coref
clusters). We can't realistically replicate that data. Even if we could,
document-level architecture is fundamentally different from a token-tagging
cascade.

**Effort**: 1-2 weeks to integrate properly with the conversational pipeline
(entity resolution across messages is the hard part, not coref itself).

### Cross-document event linking → application-layer state

Detecting that "I have a doctor's appointment tomorrow" and "the doctor said my
levels are fine" reference the **same event** is fundamentally a
**cross-document state-management** problem.

**Cascade can help** by extracting:

- Event triggers (per-token detection — fits as a head).
- Event arguments (already have via SRL).
- Event types (per-token classification).

**Application layer must handle**:

- Linking event mentions across messages.
- Resolving event identity over time.
- Storing events as graph nodes.

**Recommendation**: Build event extraction *into* the cascade. Build event
linking as application logic operating on extracted events + entities.

### Aspect-based sentiment → split

The hard part of "how does X feel about Y" isn't the polarity
(positive/negative) — it's identifying the **target** (Y) and **holder** (X).
The polarity classification could be a cascade head; the target/holder
identification is essentially a relation extraction problem.

**Recommendation**: Add per-token sentiment polarity as a cascade head
(cheap), but rely on existing SRL + coreference + RelEx to identify
target/holder. Compose at the application layer.

```python
# Application-layer flow
polarity_per_token = cascade_model.sentiment(message)
relations = cascade_model.relations(message)
for rel in relations:
    if rel.type == "FEEL_ABOUT":
        sentiment = avg_polarity(polarity_per_token, rel.span)
        store(holder=rel.subject, target=rel.object, polarity=sentiment)
```

## What stays in the cascade model

These are good cascade head candidates because they fit the per-token
classification pattern:

| Head | Output | Why fits cascade |
|------|--------|------------------|
| **Factuality / modality** | per-token: factual / hedged / future / negated | well-defined labels, contextual decision, FactBank data exists |
| **Within-sentence coref** | per-token: cluster ID | mention-pair scoring uses encoder hidden states efficiently |
| **Lemma** | per-token: lemmatized form (or rule index) | token transformation, cheap, useful for retrieval |
| **Morph features** | per-token: case/number/tense/etc. | classification per feature, trivial extension |
| **Keyword salience** | per-token: importance score (0-1) | regression, benefits from context |
| **Sentiment polarity** | per-token: pos/neg/neutral | classification with context |
| **Relation extraction** | per-entity-pair: relation type | uses NER + DEP + SRL features as cascade |
| **Event triggers** | per-token: event type | classification with context |

## Implementation pattern

Establish a clean separation in the pipeline:

```
INCOMING MESSAGE
    ↓
[Cascade model] — single forward pass produces:
    POS, NER, DEP, SRL, CLS, factuality, lemma, keyword,
    coref-within, sentiment, relations, events
    ↓
[Application layer]:
    SUTime normalizes temporal spans → TIMEX3 values
    fastcoref resolves cross-message pronouns
    NLI service checks contradictions
    Event linker resolves cross-message identity
    Graph builder constructs entity/event/relation nodes
    ↓
STORED OBSERVATION + GRAPH UPDATES
```

The cascade is the **structured perception layer**. The application layer is
the **reasoning + state-management layer**. Don't conflate them.

## What this means for roadmap priorities

- **TIMEX3**: ship in days (SUTime integration), no model retrain needed.
  Closes the 34% date-anchored failure category immediately.
- **Cross-sentence coref**: 1-2 week integration of fastcoref. Cleans up
  observation noise.
- **NLI**: 1-2 day integration of off-the-shelf MNLI model. Replaces
  rule-based contradiction.
- **Cascade head additions** (factuality, coref-within, lemma, keyword,
  sentiment, RelEx, events): batch into the next teacher retrain. ~2-3 weeks
  for the full set, then ~2 days to re-distill students.

Most of the application-side wins land **before** the next big model
retrain. The retrain then captures the genuinely-cascade-shaped capabilities.

## Reference: when in doubt, ask

1. *Can this be expressed as a per-token label or per-span score?*
   → cascade head.
2. *Does it need a runtime value (timestamp, user ID, prior conversation) the
   model has never seen during training?* → application layer.
3. *Does an off-the-shelf model or rule-based library already solve this
   robustly?* → application layer; integrate, don't reinvent.
4. *Does it operate on more than one sentence or one document?*
   → application layer (the cascade only sees one utterance at a time).
5. *Is the output structured (a date value, a graph, a link)?*
   → application layer composes these from cascade primitives.
