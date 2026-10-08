# v6 corpus quality audit — 823-window manual sample

Sample: 823 windows, 15,424 sentences, stratified by domain, `random.seed(20260928)`.
Window ids in `v6/audit/sample_ids.json`, so the sample is re-drawable and the findings
are checkable. Gates cover structure; this audit is about whether the
annotations are **right**, which no gate can answer.

Method: automated plausibility probes over the whole sample, then reading
complete annotated windows and 33 labelled sentences by hand.

## Verdict

Usable for training. One design issue worth deciding on (SRL over auxiliaries),
and a set of small defects that are recorded rather than alarming.

## By layer

| layer | finding | rate |
|---|---|---|
| structure | zero integrity issues on a trainer-style load of 10,813 rows | 0% |
| POS | `X` (unknown) tag | 1.32% overall; conversation 2.70%, news 0.08% |
| POS | bare punctuation tagged `PUNCT` | 98.9% |
| NER | single-character `PERSON`/`ORG`/`GPE` | 1.3% of windows |
| CLS | sentences with no function, on fragments and headings | 6.12% |
| CLS | `?` sentence without `Question`/`Directive` | 2.3% of windows |
| SRL | **VERB** predicate marked `V` | **91.2%** |
| SRL | **AUX** predicate marked `V` | **13.9%** |
| relations | entity pairs carrying >1 relation | 30.1% (2), 1.1% (3) |
| text | token longer than 40 chars (URLs) | 0.9% of windows |
| sentences | split at an abbreviation period | 0.40% |

POS's UPOS distribution is normal English — NOUN 21.6%, PUNCT 14.9%, VERB 10.1%,
ADP 9.2%, DET 8.0% — which is the check that the tagger has not collapsed.

## The one design issue: SRL over auxiliaries

`SRL_PREDICATE_TAGS` is `{VERB, AUX}`, and AUX produces **14,488 of 43,572
frames in the sample — a third of the layer** — of which only 13.9% mark the
conditioned predicate as `V`, against 91.2% for VERB.

This is not a model failure. It is PropBank convention: an auxiliary is
`ARGM-MOD` of the main verb rather than a predicate in its own right, and
`B-ARGM-MOD` appears 1,571 times at `predicate_idx`. The head is behaving
correctly and our predicate selection is asking it the wrong question.

The consequence is that the SRL layer mixes two conventions, and a third of it
teaches "this token is the predicate, do not tag it `V`". **Decision needed:**
drop AUX from predicate selection — which removes a third of SRL frames and a
third of SRL compute — or keep it and document the split convention. AUX frames
are not empty (1.91 args mean, 11.5% with no arguments), so they are not
worthless, only inconsistent.

## Smaller findings

**Sentence splitting on abbreviations** (0.40%, narrative 1.16%). `Mr .`,
`No .`, `Dr .`, `Stephen G .` become their own sentences. This is the root
cause of two other observations: the single-character `PERSON` entities (`H`,
`J`, `T` are name initials orphaned by the split) and the `I-PERSON` opening a
sentence that failed gate 8 before masking. Worth fixing in `split_sentences`
if the corpus is ever rebuilt; not worth a rebuild on its own.

**Contractions.** The canonical tokenizer splits at the apostrophe, so `I'd`
becomes `I ’ d` and the bare apostrophe carries a tag: `PUNCT` 410, `PART` 366
(correct for possessive), `AUX` 133. This differs from UD, which keeps `'s` as
one token. It is a consequence of the deliberately dependency-free tokenizer
(§4.1), it is consistent across every layer, and it is not corruption — but a
model trained here will see contractions differently from a UD-trained one.

**Reference and boilerplate text** reaches the corpus: bibliography entries,
`= = Sources = =` headings, table titles. These are what the 15.2% empty-CLS
rate in business and encyclopedic is made of, and the annotator handles them
correctly by assigning no function. `looks_like_prose` does not reject them.

**Relations** are multi-label as DocRED allows, and 53% are geographic
(`located in the administrative territorial entity` 674, `country` 588,
`continent` 170). That matches the inventory rather than indicating
over-prediction: a city is legitimately in a region and a country and a
continent.

## What the manual read showed

33 sentences read across conversation, business and narrative. CLS agreed with
my own reading on roughly nine in ten, and the disagreements were not scattered
— they concentrated on the accept/offer boundary already open in
CLS_TAXONOMY.md ("That sounds perfect." as `Inform`; "Would you like to purchase
tickets?" as `Question` rather than `Commissive`). Documentation imperatives are
correctly `Directive`, compound sentences correctly carry two labels, and
sentiment tracked the text.

Spot checks on the structural layers read correctly too: SEC filings give
`MONEY: $ 386 million`, `ORG: the Financial Accounting Standards Board`,
`LAW: ASU`; Wikinews gives `WORK_OF_ART: Twilight`, `PERSON: Stephenie Meyer 's`.
NER spans include leading determiners and trailing possessives, which is
OntoNotes convention rather than error.
