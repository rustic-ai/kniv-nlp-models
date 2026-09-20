# v6 training dataset — specification and preparation plan

One corpus. One record per window. Every head supervised on the same
tokens.

This document is the authoritative spec for the v6 training data: what it
is, how it is structured, how it is produced, and in what order. Annotator
selection is settled and evidenced separately in [DECISIONS.md](DECISIONS.md);
the CLS annotation contract is [CLS_TAXONOMY.md](CLS_TAXONOMY.md). This
file supersedes `docs/data-preparation-plan.md`, which describes the v5
five-dataset design.

---

## 1. Proposal

### 1.1 What changes from v5

| | v5 | v6 |
|---|---|---|
| datasets | 5 disjoint, one per head | 1, all heads on every row |
| unit | sentence (≤128 tok) | 512-token window |
| labels | each example supervises 1 head | every example supervises 7 heads |
| POS/DEP source | spaCy silver | kniv-v5 (measured best of 7) |
| NER source | SpanMarker silver | kniv-v5 |
| lemma / morph | absent | Stanza |
| coref | absent | LingMess |
| CLS | 8 single labels, 3 near-empty | 6 ISO-derived multi-labels |
| provenance | claimed in docs | recorded per row, per layer |

### 1.2 Why one corpus

v5's heads each learned their own dataset's domain as well as their task.
Nothing forced them to be consistent with each other, and nothing let a
head benefit from another head's supervision on the same sentence. A
single corpus with all layers on the same tokens gives:

- **Shared encoder gradient from every example** instead of 1/5 of them.
- **Consistency by construction** — the DEP tree and the SRL frames refer
  to the same tokenization, so they can be checked against each other.
- **One domain distribution** across all heads, so per-domain evaluation
  means the same thing for each.

### 1.3 Why windows, not sentences

- **Coreference is impossible below the window.** Most chains cross
  sentence boundaries; within-sentence coref is close to vacuous.
- **It makes SRL cheaper.** SRL needs one predicate-conditioned forward
  pass per predicate — 4.82 verbs/sentence measured
  (`data/bench_srl_fanout.json`), so ~5.8 encoder passes per sentence. One
  512-token window scoring all predicates jointly is ~4.3× cheaper for the
  same text, because the fan-out multiplier dominates the quadratic
  attention term at these lengths.
- **CLS needs context.** The v5 head reads 0.951 in-domain and 0.613 in
  the wild, partly because it sees one utterance plus at most one
  predecessor.
- **It removes the sentence splitter** from the inference path — an error
  source and a spaCy dependency.
- **512 is free.** `microsoft/deberta-v3-large` has
  `max_position_embeddings: 512`, `position_buckets: 256`. No position
  surgery, no architecture change. Beyond 512 is extrapolation.

At the corpus's measured 20.9 tokens/sentence, a 512-token window is ~24
sentences — a whole message or a short exchange.

**Two layers stay sentence-scoped inside the window:** DEP (a tree is
per-sentence; a window is a forest) and CLS (a label set is per utterance).
The record carries `sentence_spans`, and those layers index against it.

### 1.4 Why LLM annotation is not the default

It was the original premise, and the bake-off contradicted it. Measured on
public gold, per-layer, same-sample: specialists win six of seven layers,
and the LLM ensemble wins none. Frontier LLMs are closest on the semantic
end and furthest on tasks with a closed label set (NER −10.0, morph −8.7,
DEP −5.3). The corpus is still distilled knowledge — just from the model
that is demonstrably best at each layer rather than from one family
everywhere. LLMs take the three layers where no inventory and no gold
exist: CLS, sentiment, keyword.

---

## 2. Corpus source

### 2.1 Domains and licences

All sources are already collected by `corpus/domains/*/collect.py` into
`corpus/output/raw/<domain>/<source>.jsonl` as **whole documents**.

| domain | sources | licences |
|--------|---------|----------|
| conversation | OpenAssistant/oasst1, MultiWOZ 2.2, ABCD, AirDialogue, Glaive function-calling, Taskmaster | CC-BY-4.0, Apache-2.0, MIT |
| narrative | Project Gutenberg | public domain (US) |
| technical | Wikipedia, Python docs | CC-BY-SA-3.0, PSF |
| news | Wikinews, Wikipedia | CC-BY-2.5, CC-BY-SA-3.0 |
| encyclopedic | Wikipedia | CC-BY-SA-3.0 |
| business | SEC EDGAR, Enron, OpenStax, Wikipedia, CUAD | public domain, CC-BY-4.0, CC-BY-SA-3.0, ODC-BY |

Two licence consequences, neither affecting the trained model:

- **CC-BY-SA text is in the corpus.** If the *dataset* is published, the
  share-alike terms apply to it. The model is not a derivative work of its
  training data — settled, and not re-opened here.
- **Attribution is required** for CC-BY and CC-BY-SA material, so
  `source` and `source_url` are mandatory per-row fields, not optional
  provenance.

### 2.2 Personal data

The business domain includes the **Enron email corpus**, which contains
real names, addresses and phone numbers of real people who did not consent.
It is public and widely used, but it is personal data, and it is going into
a corpus we may publish and into weights we will publish.

**Gate before any business-domain window is annotated:** run PII detection
over Enron documents and either redact or drop. This is a hard gate in the
QA list (§7), not a recommendation.

### 2.3 Document structure — the upstream blocker

The *published* corpus artifacts cannot support windows. Measured on
`corpus/gold/test.parquet` (65,732 rows):

| signal | value |
|---|---|
| rows whose `prev_text` equals the previous row's `text` | **0 / 65,731** |
| rows with any `prev_text` | 11.9% |
| `sent_id` sequential within a domain | no — a gap at every pair |
| `# newdoc` / `# newpar` in `test.conllu` | 0 / 0 |

Both artifacts are shuffled sentence samples with document membership and
ordering discarded. It is lost in `corpus/domains/*/preprocess.py`, which
emits `{"text", "source", "domain"}` per sentence with no document id and
no index within the document.

**It survives in `corpus/output/raw/`,** which holds whole documents — that
is what the phi4 LoRA pipeline already trains on.

**Resolution: build windows directly from `raw/`,** skipping the sentence
flattening entirely. Split into sentences only to record `sentence_spans`.
This keeps the sentence splitter out of the training path, which was the
point. It needs a per-domain document adapter, because `raw/` shapes
differ by domain:

| domain | raw record | document key | ordering key |
|--------|-----------|--------------|--------------|
| conversation | one utterance | `conv_id` | `turn_idx` |
| narrative | plain `.txt` per book | filename | byte offset |
| technical (wiki) / news / encyclopedic | one article | `title` | paragraph index |
| technical (python_docs) | one page | `path` | paragraph index |
| business (sec_edgar) | one filing | accession | paragraph index |
| business (enron) | one email | message id | paragraph index |
| business (openstax) | one section | section id | paragraph index |

### 2.4 Volume

The domain targets in `corpus/domains/domain_config.yaml` are stated in
**sentences** and were sized for sentence-level training:

| domain | target sentences |
|---|---|
| business | 200,000 |
| conversation | 30,000 |
| narrative | 20,000 |
| technical | 15,000 |
| news | 15,000 |
| encyclopedic | 10,000 |
| **total** | **290,000** |

At 20.9 tokens/sentence that is ~6.1M tokens, which packs into ~11,800
windows of 512. Row count drops ~24×, but **token count is what matters**
— every one of those tokens now carries seven layers instead of one.

For scale reference from v5: NER trained on 195K sentences, SRL on ~397K
verb-level examples, distillation on 500K sentences (~10.5M tokens).

**Target: 10M tokens ≈ 20,000 windows**, which matches v5's distillation
token budget while supervising every head on all of it. The domain mix
should be rebalanced at that point — business at 69% of the corpus is an
artifact of a target set for a different purpose, and the uniko use case
is conversational.

**Restate the domain targets in tokens before collection resumes.** This
is a decision for §9 sequencing, not an assumption baked in here.

---

## 3. Record schema

One row per window, Parquet. Sentence-scoped layers index against
`sentence_spans`.

### 3.1 Identity and provenance

| field | type | notes |
|---|---|---|
| `window_id` | `string` | stable hash of `(doc_id, first_sent_idx)` |
| `doc_id` | `string` | document key from §2.3; **splits are made on this** |
| `domain` | `string` | one of the six |
| `source` | `string` | dataset/source name, for attribution |
| `source_url` | `string` | required where the licence requires attribution |
| `licence` | `string` | SPDX-ish string from §2.1 |
| `sent_idx_start` | `int32` | index of the first sentence within the document |

### 3.2 Text

| field | type | notes |
|---|---|---|
| `tokens` | `list<string>` | canonical, immutable, ≤512 |
| `sentence_spans` | `list<list<int32>>` | `[[start, end), …]`, covering all tokens |
| `n_tokens` | `int32` | |

### 3.3 Token-level layers

| field | type | length | annotator |
|---|---|---|---|
| `pos` | `list<string>` | `n_tokens` | kniv-v5 |
| `lemma` | `list<string>` | `n_tokens` | Stanza |
| `morph` | `list<string>` | `n_tokens` | Stanza |
| `ner` | `list<string>` | `n_tokens` | kniv-v5 |
| `dep_heads` | `list<int32>` | `n_tokens` | kniv-v5 |
| `dep_rels` | `list<string>` | `n_tokens` | kniv-v5 |

`dep_heads` are **window-absolute token indices**, with the root of each
sentence pointing at `-1`. Storing them window-absolute rather than
sentence-relative means no re-basing at load time; the constraint that a
head lies inside the token's own sentence span is a validation rule (§7).

### 3.4 Structured layers

| field | type | notes |
|---|---|---|
| `srl_frames` | `list<struct{predicate_idx: int32, tags: list<string>}>` | one entry per predicate in the window; `tags` is length `n_tokens` |
| `coref` | `list<list<list<int32>>>` | clusters → mentions → `[start, end)` |
| `cls` | `list<list<string>>` | one label **set** per entry in `sentence_spans` |
| `sentiment` | `list<string>` | one per sentence span |
| `keywords` | `list<string>` | window-level |
| `relations` | `list<struct{...}>` | typed triples over `coref` clusters — see §3A.2 |

### 3.5 Supervision control

| field | type | notes |
|---|---|---|
| `loss_mask` | `map<string, list<bool>>` | layer → per-token mask; absent layer means "fully supervised" |
| `provenance` | `map<string, string>` | layer → annotator id + version |
| `agreement` | `map<string, float>` | layer → inter-annotator agreement, where more than one ran |

`provenance` is a **field, not a doc claim**. NER, DEP and
`corpus/gold/test.parquet` have each been documented as gold while
actually being silver; recording it per row makes that impossible to
repeat.

### 3.6 Label inventories

Fixed, and validated on write (`v6/schemas.py`):

| layer | size | inventory |
|---|---|---|
| POS | 17 | UPOS |
| DEP | 53 | UD EWT deprels incl. subtypes |
| NER | 37 | BIO over 18 OntoNotes types |
| SRL | 42 | BIO over PropBank ARG0–4 + 15 ARGM + `V` |
| morph | open | UD FEATS strings; validated against the 293 `(UPOS, FEATS)` pairs attested in UD EWT train |
| CLS | 6 | `Question`, `Inform`, `Directive`, `Commissive`, `Feedback`, `Social` — multi-label, empty set legal |
| relations | 96 | Re-DocRED Wikidata properties, multi-label per pair (§3A.1) |
| sentiment | 3 | `positive`, `negative`, `neutral` |

---

## 3A. The relation layer

Knowledge-graph extraction is a v6 goal, so relations are annotated in this
pass. The head may ship later; the corpus cannot be re-annotated cheaply,
and adding relations afterwards means a second full pipeline run over every
window.

### 3A.1 Inventory

**Re-DocRED's 96 Wikidata properties**, taken whole rather than hand-picked.
An inventory invented for this project would not be evaluable against any
published gold, and mapping error would then be baked into every number.

Relation names are resolved from the Wikidata API rather than a copied
lookup table, so the mapping is checkable against a primary source and an
unresolvable code fails loudly instead of becoming its own id.

The inventory is taken over test ∪ dev (96), not test alone (95).
Restricting the label set to what the evaluation split happens to contain
would leak which relations are in it.

**Re-TACRED cannot be used.** It publishes only label patches — id → revised
label — and the underlying sentences are TACRED, which is LDC-gated
(LDC2018T24). We have the corrections but not the text. Same shape as the
OntoNotes-coref restriction that pushed coref evaluation to LitBank.

### 3A.2 Schema

```
relations: list<struct{
    head_cluster: int32,           // index into `coref` clusters
    tail_cluster: int32,
    labels: list<string>,          // multi-label; empty = no relation
    evidence_sents: list<int32>,
    temporal_span: list<int32> | null,   // [start, end) in the window
    polarity: "positive" | "negated"
}>
```

**Multi-label is not optional.** Measured on Re-DocRED test: of 13,627
entity pairs that hold any relation, **3,621 (26.6%) hold more than one**. A
single-label field would silently discard a quarter of the graph.

Arguments are **coref clusters, not spans** — relations link entities, and
`Caroline` and `she` are one argument. This makes relation quality bounded
by coref quality, which is the layer's main structural risk: LingMess
measured 0.695 CoNLL-F1 on LitBank.

`temporal_span` and `polarity` are links into the text, not resolved values.
Normalising "last March" to an interval needs a document date, and deciding
whether a fact still holds needs graph state — both are app concerns, the
same line drawn for CLS and for graph assembly. Both attributes are grounded
in layers already annotated (NER `DATE`/`TIME`, SRL `ARGM-TMP`,
`ARGM-NEG`), so they cost no new supervision and can be cross-checked.

`modality` (asserted / hypothetical / desired) was considered and deferred:
sparse attributes are what made v5's CLS unmeasurable. Its frequency will be
counted during CLS gold adjudication and decided on evidence.

### 3A.3 Annotation: two stages

```
NER clusters + coref   →  entity list          deterministic, from the cascade
                              ↓
window + entity list + inventory  →  (head, tail, relation) triples   annotated
```

Stage 1 is code, not a model task, so **recall is not the annotator's
problem** — open-ended relation extraction is where LLMs invent entity
boundaries and drift on what counts as a mention. Supplying the entity list
turns it into bounded classification and makes every answer mechanically
checkable: the cluster index must exist.

**Emit triples; do not enumerate pairs.** At Re-DocRED's mean of 19.6
entities a document has ~397 ordered pairs against ~35 true triples. Asking
for a verdict per pair is two orders of magnitude of wasted output on a task
whose answer is sparse.

Cost control for the corpus run, in order of effect:

1. **Type constraints.** `employed_by` is PERSON × ORG. Most candidate pairs
   are type-impossible and drop out before any model sees them — free, since
   NER already supplies the types.
2. **Locality.** Pairs whose mentions never occur within N sentences.
3. **One call per window**, all surviving pairs against the shared context.

### 3A.4 Validation gates

Each is checkable against another layer annotated in the same pass — the
concrete payoff of doing relations now rather than later.

- both cluster indices exist in the window
- head ≠ tail (a self-relation is not a fact)
- duplicate triples collapse rather than counting twice
- the NER types are compatible with the predicted relation
- each cited evidence sentence contains a mention of both arguments
- `temporal_span`, where present, overlaps a `DATE`/`TIME` span or an
  `ARGM-TMP`

### 3A.5 Annotator selection

Bake-off on **Re-DocRED test** — 500 documents, 9,779 entities, 17,448 gold
triples, all 500 documents under the 512-token window budget so none is
truncated. Micro-F1 over `(head, tail, relation)` triples.

Measured over the 472 documents all seven annotators covered:

| annotator / combination | F1 | P | R |
|---|---|---|---|
| **astra + grok, union** | **0.503** | 0.520 | 0.487 |
| astra | 0.478 | 0.619 | 0.389 |
| grok | 0.470 | 0.532 | 0.421 |
| sol | 0.326 | 0.581 | 0.226 |
| luna | 0.275 | 0.499 | 0.189 |
| deepseek | 0.239 | 0.292 | 0.202 |
| mistral | 0.230 | 0.294 | 0.188 |
| mai | 0.110 | 0.153 | 0.085 |
| all seven, majority | 0.310 | 0.655 | 0.203 |

**Superseded — see below.** The union of astra and grok was the first
decision at 0.503; the layer was then improved to **0.566** by unioning the
LLM with a *supervised* model. Original reasoning retained because the
mechanism still holds:

**Decision: the union of astra and grok.** It is the only combination in
this project that beat its best member — relations are set-valued and
recall-bound, so majority voting *intersects* the annotators (recall 0.323)
while union *adds* (0.487). The guardrails are narrow: a third member is
neutral at best (sol −0.4) or harmful (mai −9.1), and all seven together
cost 16.8.

### Revision: add a supervised model

Error decomposition showed 74.7% of misses were entity pairs never
proposed, not misclassified. Two fixes were measured:

| system | F1 | P | R | cost |
|---|---|---|---|---|
| ATLOP (supervised, local) | 0.459 | **0.952** | 0.302 | free, 90s |
| astra PAIRS (pairs enumerated) | 0.545 | 0.576 | 0.518 | 5.4x API |
| union astra+grok (first decision) | 0.502 | 0.519 | 0.487 | 2x API |
| **union ATLOP + astra** | **0.566** | 0.664 | 0.493 | **1x API** |
| union ATLOP + PAIRS | 0.586 | 0.596 | 0.577 | 5.4x API |

ATLOP's near-perfect precision is an artifact of training on the original
DocRED (systematic false negatives -> a conservative model).

**Final: retrain ATLOP on Re-DocRED and annotate with it alone.** Measured
on Re-DocRED test (473 docs): **F1 0.790, P 0.901, R 0.704** — +24 over the
best LLM single model and +20 over the best LLM combination. Every union
with an LLM *lowers* F1, so the relation layer needs **no API calls**: 500
documents infer in ~90 seconds on a laptop.

At 0.901 precision, intersecting with an LLM is no longer worth it: it buys
~6 precision points for ~28 recall points and reintroduces the API calls
the supervised model removed. **Annotate with the model alone** and mask
uncertain pairs rather than labelling them `no_relation`.

**Also correct §3A.3:** the type-constraint prune described there does not
work. Measured on Re-DocRED, type constraints remove only 2.6% of ordered
pairs, and locality pruning discards 52% of gold at a one-sentence window.
Any pair-enumeration costing must assume ~397 pairs per document.

**Domain risk:** ATLOP is trained on Wikipedia and scored there. Our corpus
is conversation, business and narrative, with no gold. Verify with the
agreement probe before the corpus run.

Schema validation did its job completely: across 3,500 calls there were
**zero parse, length or range failures** — no malformed JSON, no invented
relation names, no out-of-range entity ids. Every failure was transport or
content filter. Structured outputs guarantee shape, not content.

**These numbers are not comparable to published DocRED scores.** Those come
from models trained on DocRED's own train split; every annotator here is
zero-shot over a supplied entity list. Supervised document-level RE tops out
near 66–67 F1, and that is the field's ceiling with in-domain training, not
a target for zero-shot annotation.

There is no usable specialist for this layer. GLiFormer is the only open
schema-conditioned encoder that claims joint relation extraction and reports
12.78 micro-F1 on DocRED — included in the bake-off as a floor, not a
candidate. So relations go to the LLM ensemble, and unlike POS or morph that
is defensible: the bake-off showed LLMs are closest to specialists on the
semantic end, and here there is no specialist to lose to.

Expect relations to be the **weakest head in the cascade**, against 0.84–0.99
for the others. That is the task, not our shortfall.

### 3A.6 What relations do not cover

Two structural limits, worth stating before the knowledge-graph design
assumes otherwise:

1. **OntoNotes NER has no span for roles or occupations.** "Caroline is a
   nurse" has no tail entity. Occupation, title and role facts are
   unrepresentable by this head under any inventory.
2. **Most conversational memory is not entity-entity.** "Caroline is
   researching adoption agencies" is predicate-argument structure, and the
   **SRL head already covers it**. Relations complement SRL over named
   entities; they do not replace it.


## 4. Production pipeline

```
corpus/output/raw/<domain>/<source>.jsonl        whole documents
        │
        │  [NEW] window builder
        │    · per-domain document adapter (§2.3)
        │    · PII gate on business/enron (§2.2)
        │    · sentence split → spans only, never a unit of training
        │    · pack to ≤512 tokens, no sentence split across windows
        │    · canonical tokenization fixed HERE, once, for all annotators
        ▼
   v6 windows  (data/v6-corpus/windows/)
        │
        ├── kniv-v5   → pos, ner, dep, srl        own weights, no new provenance
        ├── Stanza    → lemma, morph              Apache-2.0
        ├── LingMess  → coref                     MIT
        ├── LLM ×5    → cls, sentiment, keywords  Azure; consensus + adjudication
        └── LLM ×N    → relations                 after NER + coref (§3A.3)
        │
        ▼  assemble: provenance, per-token masks, agreement
   v6 training corpus  (data/v6-corpus/corpus/, parquet shards)
```

### 4.1 Rule zero

**Tokenization is fixed before any annotator sees the text**, and every
annotator returns exactly one entry per token index. Silent
re-tokenization is the dominant failure mode of LLM token-level
annotation; here a length or range mismatch is a **recorded failure, never
padded**.

### 4.2 Environment isolation

The annotators need mutually incompatible dependency stacks — v5 pins
`transformers==5.6.2`, Stanza runs current, fastcoref needs
`transformers==4.36` + `numpy<2` + `pyarrow<15` on Python 3.10. **The
response cache is the seam:** each annotator runs in its own venv and
writes into a shared cache keyed by
`(annotator, layer, prompt_version, item)`; assembly reads only from the
cache. Proven by the bake-off, which ran seven annotators across three
environments into one report.

This also gives resumability for free — a re-run replays cached items
instantly and only issues calls for the gaps.

### 4.3 LLM annotation budget

CLS, sentiment and keywords, five families, one call per window per layer:

```
20,000 windows × 3 layers × 5 annotators = 300,000 calls
```

At ~3 s/call and concurrency 16 that is ~15.6 hours wall-clock. **Grok is
excluded from the bulk pass** — measured at ~129 s/call it alone would be
~9 days for its share. Use grok as an adjudicator on disagreements only,
where its measured strength on semantic layers pays for its latency.

Note also that luna, sol and terra are all `gpt-5.6` variants — one
lineage, not three. Measured same-family minus cross-family agreement:
+0.007 (lemma), +0.015 (POS), +0.069 (morph), **+0.164 (DEP)**. Their
three votes are worth roughly one where the task is hard. DeepSeek, Grok,
Mistral and MAI are the genuinely independent families, and the ensemble
should be composed on that basis rather than on model count.

### 4.4 Mixed provenance and the morph mask

POS comes from v5, morph from Stanza — and Stanza's morph is conditioned
on Stanza's *own* POS. Measured on UD EWT test:

| | |
|---|---|
| v5 / Stanza POS agreement | 97.72% |
| …morph impossible under v5's POS | 42% of disagreements |
| **impossible pairs, all tokens** | **0.96%** |
| on disagreements, v5 correct vs gold | 72% (Stanza 25%) |

Failures are systematic — participial adjectives (`fledged`, `United`,
`committed`) where Stanza reads a past participle and emits verbal
features under an `ADJ` that v5 gets right.

**Mitigation: per-token morph masking.** A `(UPOS, FEATS)` pair absent
from the 293 attested in UD EWT train is masked, not taught. Implemented
in `build_corpus.py`; measured at 1.98% on a technical-domain slice.
**Computed per window, never assumed** — disagreement roughly doubles on
the worst domain (2.28% → 4.95%).

---

## 5. On-disk layout

```
data/v6-corpus/
├── MANIFEST.json                    build id, git sha, annotator versions,
│                                    row/token counts, per-layer mask rates
├── DATA_CARD.md                     sources, licences, attribution, PII
│                                    treatment, known limitations
├── windows/                         pre-annotation, canonical tokenization
│   └── domain=<d>/part-*.parquet
├── annotations/                     one tree per annotator; cache-backed
│   └── <annotator>/<layer>/part-*.parquet
├── corpus/                          assembled, this is what training reads
│   └── split=<train|dev|test>/domain=<d>/part-*.parquet
└── gold/
    └── cls/                         adjudicated CLS gold + human overlap
```

Shard size ~50–100 MB. `windows/` is kept after assembly so annotators can
be re-run or added without rebuilding tokenization.

### 5.1 Splits

**Split on `doc_id`, never on `window_id`.** Windows from the same
document share entities, coref chains and topic; splitting on windows
leaks. Stratify by domain so each split has the same mix, and hold
`dev`/`test` at 5% each.

The v6 `test` split exists for corpus QA and ablation only. **Headline
numbers are reported on public gold** (§6) — training and testing on our
own annotations would measure agreement with our annotators, not accuracy.

---

## 6. Evaluation

Public benchmarks are **evaluation-only and never enter training**.

| head | benchmark | metric |
|------|-----------|--------|
| POS, lemma, morph, DEP | UD English EWT test | acc / UFeats / UAS / LAS |
| NER | OntoNotes 5.0 test, rebuilt via `v6.gold.build_ner_gold` | span F1 |
| SRL | PropBank EWT test | span F1 |
| coref | LitBank | MUC / B³ / CEAF-e / CoNLL-F1 |
| relations | Re-DocRED test | micro-F1 over (head, tail, relation) |
| CLS | adjudicated gold + human ceiling | per-label F1, micro-F1, exact-set-match, κ |

Two rules carried over from the bake-off, both of which changed a
conclusion when applied:

- **Same-sample comparison.** Published toolkit scores are end-to-end from
  raw text and include their own tokenizer's error. Feeding gold tokens
  moved Stanza's DEP from a published 86.22 UAS to a measured 91.7 (+5.5),
  POS 95.40 → 97.7 (+2.3), UFeats 96.11 → 96.1 (+0.0).
- **Sample size.** On 300 sentences Stanza appeared to beat v5 on NER
  (0.931 vs 0.925); on the full 8,261 it reversed (0.882 vs 0.889). A
  sub-1-point gap at n=300 is noise.

CLS evaluation is specified in full in [CLS_TAXONOMY.md](CLS_TAXONOMY.md).

### 6.1 Domain transfer

Benchmarks are not our domains, so agreement was measured on the corpus
itself (`v6/probe_corpus.py`, 600 sentences, six domains):

| | v5 vs Stanza | v5 vs spaCy | Stanza vs spaCy |
|---|---|---|---|
| all six domains | **0.9640** | 0.87–0.92 | 0.87–0.92 |
| UD EWT reference | 0.9772 | | |

Agreement degrades 1.3 points with a 2.2-point spread — mild and uniform,
so the bake-off rankings transfer.

The third column matters more: **two independent modern taggers agree with
each other 96% of the time and with the corpus's existing spaCy labels
only 87–92%.** `corpus/gold/test.parquet` is silver from one 2020-era tool
despite `gold` in its path. Re-annotation is a real quality gain, and any
v5 head trained on it inherited spaCy's errors.

---

## 7. Quality gates

Every gate is a build failure, not a warning, unless marked *report*.

**Windows**
1. `sentence_spans` tile `[0, n_tokens)` exactly — no gap, no overlap.
2. No sentence split across a window boundary.
3. `n_tokens ≤ 512` under the DeBERTa-v3 tokenizer, not a whitespace proxy.
4. PII scan clean on business/enron (§2.2).
5. Near-duplicate rate across documents — *report*, with a dedup threshold
   set from the observed distribution.

**Per-layer**
6. Every per-token list has length `n_tokens`. Mismatch = recorded
   annotator failure, never padded.
7. Every label is in its inventory (§3.6).
8. NER and SRL BIO sequences are well-formed (`I-X` only after `B-X`/`I-X`).
9. Each sentence span induces exactly one tree: single root, no cycle, no
   head outside the span.
10. Coref mentions lie within `[0, n_tokens)` and do not partially overlap
    a token.
11. `cls` has one entry per sentence span; labels from the six.
12. Every relation passes the §3A.4 gates: cluster indices exist, head ≠ tail,
    argument NER types compatible, evidence sentence contains both arguments.

**Corpus**
13. Coverage per layer ≥ 98% of tokens after masking — *report* below that,
    fail below 90%.
14. Morph mask rate per domain — *report*; a domain above ~5% needs
    investigation, not acceptance.
15. Label distribution per layer per domain — *report*, against the v5
    distribution where one exists.
16. No `doc_id` appears in more than one split.
17. `MANIFEST.json` records annotator versions and git sha for every layer.

---

## 8. Known defects to fix

| defect | where | status |
|--------|-------|--------|
| `benchmarks/ontonotes5_test.json` tag-id map broken — 2,205/152,723 tokens (1.44%) mislabelled, six types at 0.000 F1, ~10 F1 understatement | kniv-corpus-en dataset repo | worked around by `v6/gold/build_ner_gold.py`; **source not fixed** |
| `download_benchmarks.py` `TNER_TAGS` has 36 entries where tner has 37 — index 35 is `I-ORDINAL` not `I-LANGUAGE`, index 36 missing and silently mapped to `O` | `models/kniv-deberta-nlp-base-en-large/` | **not fixed** |
| `label_vocabs.json` in the published model repos lists 9 CLS labels against 8 output units | HF repos | **not fixed** |
| four model cards relicensed Apache-2.0 locally; HuggingFace still shows CC-BY-SA | HF repos | **not pushed** |
| NER and DEP documented as gold-trained when trained on SpanMarker / spaCy silver | docs, model cards | **not corrected** |

---

## 9. Sequencing

| # | step | blocks | status |
|---|------|--------|--------|
| 1 | Restate domain targets in **tokens**, rebalance the mix away from 69% business (§2.4) | 2 | open decision |
| 2 | **Window builder** over `corpus/output/raw/` with per-domain adapters (§2.3) | everything | not started |
| 3 | PII gate on Enron (§2.2) | annotation of business | not started |
| 4 | Extend `build_corpus.py` sentence → window: `sentence_spans`, multi-predicate SRL, per-sentence CLS | 5 | partial |
| 5 | Annotate: v5, Stanza, LingMess over windows; LLM ensemble for CLS/sentiment/keywords | 6 | harness proven on bake-off |
| 5b | Annotate relations — **after** NER and coref are final, since arguments are clusters | 6 | bake-off harness built (§3A.5) |
| 6 | Assemble + run the §7 gates; publish `MANIFEST.json` and `DATA_CARD.md` | training | not started |
| 7 | Build CLS adjudicated gold (400–600 items) and measure the human ceiling | CLS eval | `locomo50_gold_labels.csv` at n=50 |
| 8 | Fix §8 defects — cheap, and they actively mislead | — | open |

Steps 1 and 2 are the critical path. Nothing downstream can start until
documents have structure again.
