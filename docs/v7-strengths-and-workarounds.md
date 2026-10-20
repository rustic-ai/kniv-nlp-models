# kniv v7: strengths, weaknesses and workarounds

A guide for people building on the two kniv v7 models:
- [`kniv-v7-ultrabert-380m-en`](https://huggingface.co/dragonscale-ai/kniv-v7-ultrabert-380m-en) (UltraBERT below);
- [`kniv-v7-deberta-v3-large-en`](https://huggingface.co/dragonscale-ai/kniv-v7-deberta-v3-large-en) (DeBERTa below).

For each capability it covers:
- what each model does well;
- where the evaluations found it failing;
- what you can do in your own code to avoid those failures.

**Where the numbers come from.** Unless marked otherwise, numbers are from
standard test sets. Error breakdowns come from the error analyses recorded
in [v7-stage-b.md](v7-stage-b.md).

**Code.** The examples use the `kniv_v7` runtime shipped in each repository.

## 1. Which model to use

| your workload | choose | why |
|---|---|---|
| new relation or entity types (zero-shot), agent memory, conversations | **UltraBERT** | better zero-shot relations and entity types, more answers reach the memory store, better negation |
| fixed linguistic analysis (parsing, NER, SRL), Wikipedia-style relations | **DeBERTa** | ahead by 0.2-1.4 points on these, fewer false relations |
| long documents | **UltraBERT** | reads 1,024 subwords per window against DeBERTa's 512 |
| CPU throughput | **UltraBERT INT8** | 0.47 s per 200 words on an Apple M2 Max (DeBERTa INT8: 0.58 s) |
| highest relation quality, cost no object | **both** | averaging their relation scores adds +2.7 F1 on Re-DocRED (section 3.7) |

### At a glance (test sets)

| capability | UltraBERT | DeBERTa | better |
|---|---|---|---|
| POS / LAS (UD EWT) | 0.978 / 0.927 | 0.982 / 0.929 | ~tie |
| NER (OntoNotes) | 0.860 | 0.870 | DeBERTa |
| SRL end to end | 0.797 | 0.810 | DeBERTa |
| dialogue acts (SGD) | 0.761 | 0.758 | tie |
| coreference (LitBank) | 0.707 | 0.692 | UltraBERT |
| relations, gold entities / end to end (Re-DocRED) | 0.728 / 0.616 | 0.733 / 0.627 | DeBERTa (slightly) |
| zero-shot relations (FewRel m = 5 / 10 / 15) | 0.729 / 0.675 / 0.550 | 0.731 / 0.620 / 0.602 | mixed |
| held-out relations from our annotation | 0.888 | 0.852 | UltraBERT |
| zero-shot entity types (CrossNER) | 0.580 | 0.533 | UltraBERT |
| negation of a fact (teacher dev F1) | 0.80 | 0.70 | UltraBERT |
| modality / holder / time (teacher dev accuracy) | 0.80 / 0.94 / 0.81 | 0.77 / 0.93 / 0.81 | UltraBERT (slightly) |
| LongMemEval: answer in entities / facts | 0.693 / 0.480 | 0.573 / 0.400 | UltraBERT |
| LongMemEval: questions with facts about the user | 0.913 | 0.967 | DeBERTa |

The attribute scores agree with our own annotators on held-out windows; they
are not human-labelled test sets.

## 2. Settings you can change

Every decoding setting is an attribute of the `Kniv` object, loaded from
`calibration.json`. You can change any of them at run time:

| attribute | default (UB / DeB) | what it does | to get more precision |
|---|---|---|---|
| `fact["margin"]` | -- | each fact's distance above its threshold | keep facts with `margin >= t`, e.g. 1.0 |
| `kniv.rel_offsets[name]` | Re-DocRED relations only | per-relation leniency: a fact is kept when its raw margin > `-offset`. You may set it for your own relation names too | lower it |
| `kniv.unseen_offset` | -1.5 / -1.5 | every relation without its own offset: kept when raw margin > `unseen_offset` | raise it toward 0 |
| `kniv.top` | 4 | at most this many facts per entity pair | lower it (1 = only the best relation per pair) |
| `kniv.coref_bias` | 2.0 / 1.5 | bias against linking a mention to an antecedent | raise it |
| `kniv.entity_coref` | `"none"` | whether coreference merges entities for relation extraction: `"none"`, `"compatible"` (only name-compatible links) or `"all"` | keep `"none"` |
| `kniv.span_offset` | 0 | span typer (`entity_types` given): keeps spans with margin > offset | raise it |
| `kniv.ent_o_bias` | 0 | DocRED entity head (no `entity_types`): bias toward "not an entity" | raise it |

```python
from kniv_v7 import Kniv
kniv = Kniv(path)
kniv.unseen_offset = -0.5                  # stricter on your own relations
kniv.rel_offsets["works for"] = -1.0       # stricter still, for one relation
out = kniv.extract(text, relations=schema)
facts = [f for w in out for f in w["facts"] if f["margin"] >= 1.0]
```

## 3. Relations

**Strengths.**
- Any relation you can name and describe, with no retraining.
- Direction is almost never reversed (0.1% of false facts).
- Facts several sentences apart are found nearly as well as facts within one sentence:

  | F1 by distance | same sentence | adjacent sentences | 2+ sentences apart |
  |---|---|---|---|
  | UltraBERT | 0.761 | 0.694 | 0.702 |
  | DeBERTa | 0.749 | 0.701 | 0.732 |

### 3.1 Missed facts are mostly borderline yes/no decisions

**Evidence.** About 29% of Re-DocRED gold facts are missed with gold entities
(UltraBERT 29.2%, DeBERTa 28.5%). Of the misses:

| cause | UltraBERT | DeBERTa |
|---|---|---|
| right relation ranked first, but under the threshold | 52% | 55% |
| another relation predicted | 26% | 20% |
| nothing predicted for the pair | 22% | 25% |

**Workaround.** Recall is a dial: lower `unseen_offset`, or raise
`rel_offsets[name]` for the relations you care about. Then rank or filter by
`margin` downstream, rather than treating the model's default yes/no as final.

### 3.2 Most false facts are on pairs with no relation at all

**Evidence.** 74% (DeBERTa) and 80% (UltraBERT) of false facts are on entity
pairs with no gold relation. Most of the rest are confusions on a related pair.

**Workaround.** Check each relation's type signature against the entity types.
This is cheap and removes a large share of these errors:

```python
SIGNATURE = {"employer": ({"PER"}, {"ORG"}), "headquarters location": ({"ORG"}, {"LOC"})}

def typed(window):
    ents = window["entities"]
    return [f for f in window["facts"]
            if f["relation"] not in SIGNATURE
            or (ents[f["head"]]["type"] in SIGNATURE[f["relation"]][0]
                and ents[f["tail"]]["type"] in SIGNATURE[f["relation"]][1])]
```

With your own `entity_types`, the types are your names. Without them, they are
DocRED's PER, ORG, LOC, TIME, NUM, MISC; speakers have type `speaker`.

### 3.3 Near-synonym relations get confused

**Evidence.** The most frequent confusions are between near-synonyms and
granularities:
- "member of" → "member of sports team" / "member of political party";
- "located in" → "headquarters location" / "country of origin";
- "contains" → "capital";
- "publication date" → "point in time";
- "part of" ↔ "has part".

The largest error volumes are "located in the administrative territorial
entity" (misses) and "country of citizenship" / "country of origin" (false facts).

**Workarounds.**
- **Avoid near-synonyms in one schema.** If you only need "is located in",
  don't also ask for "headquarters location"; merge them downstream if you need both.
- **Write the distinction into the descriptions.** For example: "the city
  where the head organization has its main office" against "any place that
  contains the head". The model reads them.
- **Keep the head and the tail clear.** Descriptions that name both ("the
  head ... the tail ...") work best; this is how the training schemas read.
- **Ask for "part of" or "has part", not both,** and derive the inverse yourself.

### 3.4 Entities the model finds are the end-to-end bottleneck

**Evidence.** With gold entities, Re-DocRED F1 is 0.73. With the model's own
entities it is 0.62.
- The gap is mostly entity grouping, not finding mentions: mention F1 is about 0.93.
- Partial names ("Obama" / "Barack Obama") are not merged.
- Coreference is not used to merge entities by default, because on Re-DocRED
  it fused distinct entities more often than it helped.

**Workaround: bring your own entities.** If you already have entities from
your own NER or entity linking, or the model's mentions merged by your alias
rules, pass them in. Relation quality then moves toward the gold-entity score.
Spans are `[start, end)` word indices in the window:

```python
from kniv_v7 import windows
w = windows(text)[0]
spans = [tuple(s) for s in w["sentence_spans"]]
ents = [{"type": "PER", "mentions": [(0, 2)]},                # Marta Ruiz
        {"type": "ORG", "mentions": [(3, 5), (8, 10)]},       # Acme Robotics, "The company"
        {"type": "PER", "mentions": [(13, 15)]},              # Leo Park
        {"type": "LOC", "mentions": [(19, 20)]}]              # Austin
nodes, facts = kniv.relations(w, w["tokens"], spans, ents, schema, kniv.top)
```

Here is the example above run on UltraBERT INT8. Merging "The company" into
"Acme Robotics" raised the margins of the facts it supports:

| fact | margin, model's own entities | margin, with the merge |
|---|---|---|
| Acme Robotics, founded by, Leo Park | 0.3 | 7.6 |
| Acme Robotics, headquarters location, Austin | 2.4 | 5.2 |

### 3.5 Weak entity types: MISC

**Evidence.** By entity-type pair:
- worst: MISC → LOC (0.56-0.57) and MISC → MISC (0.59-0.62);
- best: LOC → LOC (0.73-0.77).

**Workaround.** Pass `entity_types` with descriptions for your domain (for
example "product", "law", "software"), so "MISC" is never the type. This makes
the type checks of 3.2 more useful too.

### 3.6 Facts are per window

**Evidence.** Each window is up to 512 words, and nothing is linked across
windows or documents.

**Workaround.** Key facts by the entities' normalised text (or your entity
IDs) across windows. Keep the highest margin per (head, relation, tail), and
count how many windows support it: a fact seen in several windows is more
reliable.

### 3.7 Highest quality: use both models

Averaging the two models' relation margins on the same entities scored 0.748
on Re-DocRED dev, against 0.720-0.721 for either alone (+2.7). Most of the
errors cancel: only 40% of either model's false facts are the other's too.

Distilling DeBERTa into UltraBERT did not capture this, so the gain needs both models at inference time:
1. Run one model for entities.
2. Call `relations(...)` on both models with those entities.
3. Average the margins, then apply your threshold.

This roughly doubles the relation pass's cost.

## 4. Zero-shot relations (your own relation names)

**Strengths.**
- UltraBERT ranks the right relation first for 89% of facts of relations
  held out of training.
- On FewRel's unseen relations it reaches 0.73 macro F1 among 5 candidates.

### 4.1 Generic labels attract specific instances ("label sink")

**Evidence.** On FewRel, "sport" absorbs instances of:
- "competition class" (F1 0.00);
- "taxon rank" (0.05);
- "voice type" (0.13);
- part of "position played on team" (0.43).

**Workarounds.**
- Avoid broad, short names next to specific ones.
- Give each relation a description that says what kind of tail it takes ("the
  weight class or category the head competed in").
- If you need a broad relation, ask for it in a separate call.

### 4.2 Many candidates make it harder

**Evidence.** FewRel macro F1 falls as candidates grow:

| m (candidates) | UltraBERT | DeBERTa |
|---|---|---|
| 5 | 0.73 | 0.73 |
| 10 | 0.68 | 0.62 |
| 15 | 0.55 | 0.60 |

**Workaround.** Keep each schema to the relations relevant to the text. If
you have many, group them by domain and run the group that matches the
document type. UltraBERT holds up better at 10 candidates, DeBERTa at 15.

### 4.3 Thresholds for your relations are generic

**Evidence.** Your relations use one offset (`unseen_offset`, -1.5), chosen
on held-out relations from our corpus. On those, thresholded F1 is low (UltraBERT
0.42, DeBERTa 0.31), although ranking is good.

**Workaround.** Label 50-100 sentences of your own data. Then set
`kniv.rel_offsets[name]` per relation to the value that gives you the
precision you need. This is the single most effective tuning step for a new
schema.

## 5. Entities and entity types

**Strengths.**
- Mentions are found reliably: about 0.93 mention F1 on Re-DocRED.
- The span typer reads your type names at inference time.
- Few-NERD fine types: 0.52 (UltraBERT). This is supervised: kniv trained on
  Few-NERD's training split, which GLiNER (0.32 on the same sample) did not.

### 5.1 Zero-shot entity types are mid-range

**Evidence.**
- CrossNER: UltraBERT 0.580, DeBERTa 0.533, against GLiNER-large's 0.613.
- AI and science domains are the hardest. For example, UltraBERT's AI domain is 0.42.

**Workarounds.**
- **Use UltraBERT** for custom entity types.
- **Give each type a description** ("programming language: a language used to
  write software, such as Python or Rust"). The span typer was trained on
  descriptions, and the runtime passes them through.
- **Tune `span_offset`** on a small sample of your data.

### 5.2 Grouping mentions into entities

**Evidence.** Grouping is rule-based: identical names, plus
coreference only if you enable it. Partial names, acronyms and nominal
mentions ("the company") stay separate entities.

**Workarounds.**
- Merge with your own alias table, or an entity linker.
- Or set `kniv.entity_coref = "compatible"`, which lets coreference join
  mentions with compatible names. Check it on your data: it can also fuse
  distinct entities.

## 6. Coreference

**Strengths.**
- Pronouns are resolved well: about 95% of the gold pronoun mentions the model finds are linked correctly.
- Retraining on GUM cut spurious pronoun links from 12% to 4-5%.

**Evidence of weakness** (LitBank):
- 23-25% of name mentions are missed, often because of boundary
  disagreements. Before the GUM retraining, about 40% of nominal mentions
  ("the old man") were missed too.
- 18-20% of predicted entities fuse two real ones.

**Workarounds.**
- **Raise `kniv.coref_bias`** (for example 3.0) for fewer, safer links.
- **Use clusters to resolve pronouns.** Be careful merging two named
  entities only because coreference linked them.
- **Map clusters by overlap.** Clusters are `[start, end]` inclusive spans.
  Map them to your entities by overlap, not exact match, to absorb boundary
  differences.

## 7. Linguistic layer

**Strengths.**
- POS and lemmas at about 98%.
- LAS about 0.93, at the level of the best UD English EWT parsers.
- SRL end to end about 0.80.
- Each sentence gets exactly one well-formed dependency tree.

| failure (evidence) | workaround |
|---|---|
| **NER follows spaCy-style conventions** (trained on kniv v5's spaCy-labelled data). Against OntoNotes' own conventions: 0.86-0.87 | If you need OntoNotes conventions exactly, map labels in post-processing. For domain types, use `entity_types` (section 5) instead of the fixed NER |
| **Nominal and adjectival predicates in SRL** are weaker: end to end, nouns 0.54 (UltraBERT) / 0.59 (DeBERTa), adjectives 0.70 / 0.76, verbs about 0.80 | Rely on verbal frames for event extraction. Treat noun frames as hints |
| **Dialogue acts** about 0.76 micro-F1, best with the preceding turns in the same window | Pass whole conversations as `turns`, not single utterances |

## 8. Conversations and agent memory

**Strengths.**
- Speakers are entities; first-person statements are attributed to the speaker.
- Facts about the user are found in 91-97% of LongMemEval questions.
- Single-session questions about what the assistant said reach the facts
  87% of the time (UltraBERT).

### 8.1 Multi-session and temporal questions

**Evidence.** LongMemEval, answer found in the extracted facts:

| question type | UltraBERT | DeBERTa |
|---|---|---|
| multi-session | 0.20 | 0.10 |
| temporal reasoning | 0.43 | 0.43 |
| knowledge update | 0.47 | 0.40 |

The model sees one window at a time and has no notion of session dates.

**Workarounds.**
- **Aggregate across sessions yourself** (section 3.6), keyed on the speaker
  plus normalised entity text.
- **Stamp each fact with its session date** in your store. The model's `time`
  attribute only points at time expressions inside the window.
- **Handle updates in the store.** For "knowledge update" questions, keep every
  version of a fact with its session date and let your store's policy decide
  which holds (latest wins, or cardinality one for relations such as "lives in").

### 8.2 "you" from the assistant's turns

**Evidence.** Seen in the release's quick start: UltraBERT made the
assistant's "you" an entity and produced "you, lives in, Lisbon" (planned).
DeBERTa attributed the same facts to the user correctly.

**Workaround.** In two-party conversations, either drop pronoun-only entities,
or remap them to the speaker they refer to ("you" in an assistant turn means
the user). Use speaker names consistently across turns, so speaker entities
merge.

### 8.3 Negation and modality

**Evidence.**
- Negation F1 against our annotators: 0.80 (UltraBERT), 0.70 (DeBERTa).
- Modality accuracy (asserted, hypothetical, planned, desired, questioned,
  reported): 0.80 and 0.77.

**Workarounds.**
- **Check negation against SRL.** Cross-check `polarity` with the SRL frames
  of the same sentence: an `ARGM-NEG` argument on the predicate between the
  head and the tail is strong evidence of a negative fact.
- **Choose UltraBERT** if negation matters.
- **Store questioned and hypothetical facts as low confidence,** not as truth.

## 9. Long inputs

**Evidence.** A window holds up to 512 words, but the encoder reads at most
1,024 subwords (UltraBERT) or 512 (DeBERTa). Words past that are not analysed;
each result's `analysed_words` says how many were. Dense text (numbers, rare
names, code) uses more subwords per word, so DeBERTa can stop well short of
512 words.

**Workarounds.**
- With DeBERTa, split long paragraphs into chunks of about 300 words (at
  paragraph or sentence boundaries) before calling `analyse`.
- Check `analysed_words` against `len(tokens)` to detect truncation.
- UltraBERT rarely needs this.

## 10. Precision variants

| failure (evidence) | workaround |
|---|---|
| **INT8 changes borderline decisions.** No task drops more than 0.7 points (UltraBERT) / 0.2 (DeBERTa), but individual facts near the threshold can flip: 6 / 3 of 14 test texts decode identically to fp32 | Calibrate thresholds (section 4.3) on the variant you deploy. Compare variants on your own sample, not on single examples |
| **FP16 is not faster on CPU** (DeBERTa is ~25% slower) | Use FP16 for download size or GPUs; use INT8 for CPU speed |

## 11. Checklist before production

1. **Pick the model** from section 1 for your workload.
2. **Write the schema well:**
   - descriptions for every relation and entity type;
   - no near-synonyms in one schema;
   - specific names, not generic ones.
3. **Label a small sample** (50-100 sentences) and set per-relation offsets
   (`kniv.rel_offsets`) to reach your target precision.
4. **Post-filter facts:**
   - by relation type signatures;
   - by `margin`;
   - by agreement across windows.
5. **Bring your own entities** where you have an entity linker, or merge aliases before relation extraction.
6. **Conversations:**
   - consistent speaker names;
   - drop or remap the assistant's "you";
   - keep session dates in your store.
7. **Long text:** with DeBERTa, chunk to about 300 words; check `analysed_words`.
8. **Variant:** calibrate on the variant you will run (fp32, FP16 or INT8).
