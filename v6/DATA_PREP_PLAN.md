# v6 data preparation plan

One corpus, every head, one record. This is the plan for producing it, the
evidence behind each choice, and the two decisions still blocking execution.

Annotator selection is settled and measured — see `DECISIONS.md`. This
document covers everything else: the unit of training, where the corpus
comes from, the record schema, and sequencing.

## 1. The unit is a window, not a sentence

v5 trained on sentences from five disjoint datasets. Each example supervised
one head, and each head learned its dataset's domain. v6 trains on
**512-token windows** with every layer on the same tokens.

Why the window rather than the sentence:

- **Coreference is impossible below it.** Most chains cross sentence
  boundaries; within-sentence coref is close to vacuous.
- **It makes SRL cheaper, not dearer.** SRL currently needs one
  predicate-conditioned forward pass per predicate — 4.82 verbs/sentence
  measured (`data/bench_srl_fanout.json`), so ~5.8 encoder passes per
  sentence. One 512-token window with all predicates scored jointly is
  ~4.3x cheaper for the same text, because the fan-out multiplier dominates
  the quadratic attention term at these lengths.
- **CLS needs context.** The head reads 0.951 in-domain and 0.613 in the
  wild partly because it sees one utterance plus at most one predecessor.
- **It removes the sentence splitter** — an error source and a spaCy
  dependency at inference time.
- **512 is free.** `microsoft/deberta-v3-large` has
  `max_position_embeddings: 512` and `position_buckets: 256`. No position
  surgery, no architecture change. Beyond 512 is extrapolation.

At the corpus's measured 20.9 tokens/sentence, a 512-token window is ~24
sentences — a whole message or a short exchange.

**Layers that stay sentence-scoped inside the window:** DEP (a tree is
per-sentence; a window is a forest) and CLS (a label is per utterance).
The record therefore carries `sentence_spans`, and those layers are indexed
against it.

## 2. BLOCKER: document structure must be reintroduced upstream

The published corpus cannot support windows. Measured on
`corpus/gold/test.parquet` (65,732 rows):

| signal | value |
|---|---|
| rows whose `prev_text` equals the previous row's `text` | **0 / 65,731** |
| rows with any `prev_text` | 11.9% |
| `sent_id` sequential within a domain | no — a gap at every pair |
| `# newdoc` / `# newpar` in `test.conllu` | 0 / 0 |

Both artifacts are shuffled sentence samples with document membership and
ordering discarded.

**Where it is lost:** `corpus/pipeline/…` is not at fault —
`corpus/domains/*/preprocess.py` emits `{"text", "source", "domain"}` per
sentence, with no document id and no index within the document. `source` is
company- or title-granular at best.

**Where it survives:** `corpus/output/raw/<domain>/<src>.jsonl` holds whole
documents — that is what the phi4 LoRA pipeline trains on ("full documents,
not sentence-split").

**Fix, in preference order:**

1. **Window directly from `raw/`.** Skip the sentence flattening entirely
   for v6: read documents, split into sentences only to record
   `sentence_spans`, and emit 512-token windows. Cleanest, and it keeps the
   sentence splitter out of the training path exactly as intended.
2. Amend `preprocess.py` to emit `doc_id` + `sent_idx`, then regroup. Keeps
   the existing pipeline shape but requires re-running preprocess for all
   six domains.

Option 1 is preferred. Either way this is a corpus-pipeline change, and no
v6 corpus can be built before it lands.

## 3. RESOLVED: the CLS taxonomy

Settled in `CLS_TAXONOMY.md`: five multi-label memory actions — `EXTRACT`,
`UPDATE`, `QUERY`, `COMMIT`, `SKIP` — replacing all three of the label sets
previously in the repo. Rationale in brief; the full guide is the annotation
contract.

### Why it was undecided

Three incompatible label sets exist in the repo, none a subset of another:

| source | labels |
|--------|--------|
| shipped model (`label_maps.json`) | inform, request, question, confirm, reject, offer, social, status |
| `corpus/pipeline/classify.py` | inform, correction, agreement, question, plan_commit, request, feedback, social, filler |
| `eval/cls_annotation_guide.md` | statement, question, question_fact, command, greeting, filler, acknowledgment |

The shipped taxonomy is partly dead in production: across 500 LongMemEval
sentences (`data/longmemeval_summary.json`) `reject` fired once and `offer`
once; three labels cover 90% of traffic. A macro-F1 over eight labels where
three are near-empty measures very little, which is part of why 0.951
in-domain and 0.613 in the wild diverge.

### Resolution

The head is a dispatch table, so granularity follows the number of distinct
downstream operations — five. Three of the shipped eight labels were
unmeasurable (`status` 8, `reject` 1, `offer` 1 per 500) and collapse into
those five. Single-label forces choices the data does not support — the
`rationale` column in the LoCoMo gold exists to record what the single label
discarded. The rate at which utterances carry two labels is **not yet
measured**; the adjudicated gold set will establish it.

The `plan_commit` vs `request` actor distinction is dropped from CLS and left
to the SRL head, which already recovers it from `ARG0`. A nested dialog-act
tier was considered and rejected — it reintroduces the sparse classes that
made v5's CLS unmeasurable.

## 4. Pipeline

```
corpus/output/raw/<domain>/*.jsonl        documents (already exist)
        |
        |  [NEW] window builder: split to sentences for spans only,
        |        pack into <=512-token windows, keep doc_id + sent_idx
        v
v6 windows (canonical tokenization fixed here, once, for all annotators)
        |
        +-- kniv-v5    -> POS, NER, DEP, SRL      (own weights, no new provenance)
        +-- Stanza     -> lemma, morph            (Apache-2.0)
        +-- LingMess   -> coref                   (MIT)
        +-- LLM x5     -> CLS, sentiment, keyword (Azure; consensus + adjudication)
        |
        v  assemble: provenance, per-token masks, agreement
v6 training corpus (parquet shards)
```

**Rule zero: tokenization is fixed before any annotator sees the text**, and
every annotator returns one entry per token index. Silent re-tokenization is
the dominant failure mode of LLM token-level annotation; here a length or
range mismatch is a recorded failure, never padded.

**Environment isolation.** The annotators need mutually incompatible
dependency stacks (v5 pins `transformers==5.6.2`; Stanza runs current;
fastcoref needs `transformers==4.36` + `numpy<2` + `pyarrow<15` on Python
3.10). The response cache is the seam: each annotator runs in its own venv
and writes to a shared cache, and assembly reads from it. Proven by the
bake-off, which ran seven annotators across three environments into one
report.

## 5. Record schema

One row per window. Sentence-scoped layers index against `sentence_spans`.

| field | notes |
|-------|-------|
| `window_id`, `doc_id`, `domain`, `source` | provenance of the text itself |
| `tokens` | canonical, immutable, <=512 |
| `sentence_spans` | `[[start,end], ...]` — metadata, not a preprocessing step |
| `pos`, `lemma`, `morph`, `ner` | per-token |
| `morph_token_mask` | per-token; see below |
| `dep_heads`, `dep_rels` | per-token, heads within their own sentence span |
| `srl_frames` | `[{predicate_idx, tags}, ...]` — all predicates in the window |
| `coref` | clusters of `[start, end]` spans over the window |
| `cls_per_sentence` | one label (or label set) per sentence span |
| `loss_mask` | per layer, and per token where needed |
| `provenance` | layer -> annotator, recorded in the row |
| `agreement` | per layer, where multiple annotators ran |

`provenance` as a field rather than a doc claim is deliberate: NER, DEP and
`corpus/gold/test.parquet` have each been documented as gold while actually
being silver.

## 6. Mixed provenance: measured and mitigated

POS comes from v5, morph from Stanza — and Stanza's morph is conditioned on
Stanza's *own* POS. Measured on UD EWT test:

| | |
|---|---|
| v5 / Stanza POS agreement | 97.72% |
| ...morph impossible under v5's POS | 42% of disagreements |
| **impossible pairs, all tokens** | **0.96%** |
| on disagreements, v5 correct vs gold | 72% (Stanza 25%) |

Failures are systematic — participial adjectives (`fledged`, `United`,
`committed`) where Stanza reads a past participle and emits verbal features
under an `ADJ` that v5 gets right.

**Mitigation: per-token morph masking.** A `(UPOS, FEATS)` pair absent from
the 293 combinations attested in UD EWT train is masked rather than taught.
Implemented in `build_corpus.py`; measured at 1.98% on a technical-domain
slice, and it must be computed per window, not assumed — disagreement
roughly doubles on the worst domain (2.28% -> 4.95%).

## 7. Domain transfer

Benchmarks are not our domains, so agreement was measured on the corpus
itself (`v6/probe_corpus.py`, 600 sentences, six domains):

| | v5 vs Stanza | v5 vs spaCy | Stanza vs spaCy |
|---|---|---|---|
| all six domains | **0.9640** | 0.87-0.92 | 0.87-0.92 |
| UD EWT reference | 0.9772 | | |

Agreement degrades 1.3 points with a 2.2-point spread — mild and uniform, so
the bake-off rankings transfer.

The third column matters more: two independent modern taggers agree with
each other 96% of the time and with the corpus's existing spaCy labels only
87-92%. `corpus/gold/test.parquet` is silver from one 2020-era tool despite
`gold` in its path. Re-annotation is a genuine quality gain, and any v5 head
trained on it inherited spaCy's errors.

## 8. Evaluation

**Training on our corpus, reporting on public gold** is the strongest
position available and the only falsifiable one — training and testing on
our own annotations would measure agreement with our annotators, not
accuracy. Public benchmarks are evaluation-only and never enter training.

| head | benchmark |
|------|-----------|
| POS, lemma, morph, DEP | UD English EWT test |
| NER | OntoNotes 5.0 test (rebuild via `v6.gold.build_ner_gold`) |
| SRL | PropBank EWT test |
| coref | LitBank |
| CLS | adjudicated gold + human ceiling (below) |

**CLS is the exception** — no public gold exists for any of the three
taxonomies. Plan:

1. Fix the taxonomy; treat the annotation guide as its definition.
2. Build gold by **adjudication**, not from scratch: five independent LLM
   families label N items; unanimous items get a 10% human spot-audit;
   disagreements get human adjudication with a recorded rationale. Roughly
   5-10x cheaper than full manual labelling, and disagreement is a good
   sampler for hard cases. `data/locomo50_gold_labels.csv` is already this
   shape at n=50 — too small (+/-14 points of 95% CI). Target 400-600.
3. Report **Cohen's kappa against humans, with the human-human ceiling
   beside it.** Two annotators on a 100-item overlap. If human-human kappa
   is below ~0.6 the taxonomy is underspecified and no model will fix it.
4. **Slice, never average**: per domain, per label, and the confusion pairs
   — checked against the guide's own edge cases. The aggregate is what hid
   the problem last time.
5. Public dialog-act sets are a domain-shift probe, never a headline.
   DailyDialog is CC-BY-NC-SA — eval only, and confirm before publishing.

## 9. Synthetic data for CLS / sentiment / keyword

These layers have no gold and no incumbent, so LLM annotation is the only
option — and the one place consensus is the right mechanism, because there
is nothing to select on.

- **Diversity over volume.** NuNER's result came from 200K unique concepts;
  the authors attribute performance to type and domain diversity. Sweep
  domains, registers, speaker counts, and turn positions, with deliberate
  coverage of rare labels a natural sample would starve.
- **Structured outputs, not `json_object`.** Removes the parse-failure path
  in `corpus/pipeline/validate.py`, which currently records a JSON decode
  error as "annotation was correct".
- **Generate with one model, adjudicate with another.** Disagreement becomes
  a routing signal rather than training noise.
- **Mind the family structure.** luna, sol and terra are all `gpt-5.6`
  variants — one lineage. Measured same-family minus cross-family agreement:
  +0.007 (lemma), +0.015 (POS), +0.069 (morph), **+0.164 (DEP)**. On easy
  layers lineage is invisible; on hard layers the three converge sharply.
  Their three votes are worth roughly one where it matters. DeepSeek, Grok,
  Mistral and MAI are the genuinely independent families.
- **Do not use consensus where gold exists.** Measured across seven layers,
  the ensemble never beat its best single member — by up to 5.7 points on
  SRL — at six times the API calls.

## 10. Known defects to fix

| defect | where | status |
|--------|-------|--------|
| `benchmarks/ontonotes5_test.json` tag-id map broken — 2,205/152,723 tokens (1.44%) mislabelled, six types at 0.000 F1, ~10 F1 understatement | kniv-corpus-en dataset repo | worked around locally by `v6/gold/build_ner_gold.py`; **source not fixed** |
| `download_benchmarks.py` `TNER_TAGS` has 36 entries where tner has 37 — index 35 is `I-ORDINAL` not `I-LANGUAGE`, index 36 missing and silently mapped to `O` | `models/kniv-deberta-nlp-base-en-large/` | **not fixed** |
| Four model cards relicensed Apache-2.0 locally; HuggingFace still shows CC-BY-SA | HF repos | **not pushed** |
| NER and DEP documented as gold-trained when trained on SpanMarker / spaCy silver | docs, model cards | **not corrected** |

## 11. Sequencing

1. **Reintroduce document structure** — window builder over `corpus/output/raw/`. Blocks everything.
2. ~~Settle the CLS taxonomy~~ — done, see `CLS_TAXONOMY.md`.
3. Extend `build_corpus.py` from sentences to windows; add `sentence_spans`, multi-predicate SRL frames, per-sentence CLS.
4. Annotate: v5, Stanza, LingMess over the windows; LLM ensemble for CLS/sentiment/keyword.
5. Build the CLS adjudicated gold set and measure the human ceiling.
6. Assemble, then audit: coverage, mask rates, per-domain agreement.
7. Fix the defects in section 10 — they are cheap and they mislead.
