# v6 annotator decisions

Which annotator produces each layer of the v6 training corpus, and the
measurement behind each choice. Every number here was produced by
`v6.bakeoff` against public gold on this repo — none is quoted from a paper
or a model card.

Runs live in `v6/runs/<run_id>/`. Predictions are cached, so any row can be
re-scored without re-running inference.

## Decisions

| layer | annotator | score | runner-up | margin |
|-------|-----------|-------|-----------|--------|
| POS | **kniv-v5** | 0.987 acc | Stanza 0.977 | +1.0 |
| lemma | **Stanza** | 0.980 acc | grok 0.971 | +0.9 |
| morph | **Stanza** | 0.961 UFeats | sol/grok 0.874 | +8.7 |
| DEP | **kniv-v5** | 0.949 UAS / 0.925 LAS | Stanza 0.917 / 0.888 | +3.2 |
| NER | **kniv-v5** | 0.889 F1 | Stanza 0.882 | +0.7 |
| SRL | **kniv-v5** | 0.843 F1 | grok 0.815 | +2.8 |
| coref | **LingMess** | 0.695 CoNLL-F1 | grok 0.681 | +1.4 |
| CLS, sentiment, keyword | LLM ensemble | — | — | no gold exists |

**kniv-v5 takes four layers, Stanza two, LingMess one. The LLM ensemble takes
none.**

## Three findings that generalise

**1. Consensus never beat best-of-breed — 7 layers, 7 times.**

| layer | ENSEMBLE | best single | delta |
|-------|----------|-------------|-------|
| POS | 0.953 | 0.960 (grok) | −0.7 |
| lemma | 0.971 | 0.971 (grok) | 0.0 |
| morph | 0.900 | 0.918 (sol) | −1.8 |
| DEP | 0.868 | 0.896 (grok) | −2.8 |
| NER | 0.778 | 0.789 (grok) | −1.1 |
| SRL | 0.758 | 0.815 (grok) | −5.7 |

Majority voting drags the strongest annotator toward the weakest. Per-layer
model selection wins on every layer *and* costs 1/6 the calls. Consensus
remains the right mechanism where no gold exists to select on — but it is a
fallback, not an improvement.

**2. The specialist advantage tracks how structural the task is.**

| task | specialist margin over best LLM |
|------|-------------------------------|
| NER (closed 18-type inventory) | +10.0 |
| morph (closed feature inventory) | +8.7 |
| DEP (tree constraint) | +5.3 |
| SRL | +2.8 |
| POS | +2.7 |
| coref (mention clustering) | +1.4 |
| lemma | +0.9 |

Frontier LLMs are closest on the semantic/discourse end and furthest on tasks
with a closed label set a supervised model can exploit. This is why CLS,
sentiment and keyword go to the LLMs — not because LLMs are better in
general, but because those layers have no inventory and no gold.

**3. Same-sample comparison is not optional.**

Published toolkit scores are end-to-end from raw text; Stanza's own tokenizer
and sentence splitter score 99.01 / 81.13 on English EWT, and that error is
inside its published numbers. Feeding gold tokens moved Stanza's DEP from a
published 86.22 UAS to a measured 91.7 (+5.5), POS 95.40 → 97.7 (+2.3), and
UFeats 96.11 → 96.1 (+0.0). The gradient is itself informative: DEP is highly
sensitive to sentence segmentation, POS moderately, morph not at all.

Sample size matters just as much. On 300 sentences Stanza appeared to beat v5
on NER (0.931 vs 0.925); on the full 8,261 the result reversed (0.882 vs
0.889). A sub-1-point gap at n=300 is noise.

## Mixed provenance: measured, and affordable

The decisions above take POS/NER/DEP/SRL from v5 and lemma/morph from Stanza.
Stanza's morph is conditioned on Stanza's *own* POS, so a corpus recording
v5's POS beside Stanza's morph teaches a mapping neither model implements.

Measured on UD EWT test (`v6/runs/_cache`, no new inference needed):

| | |
|---|---|
| v5 / Stanza POS agreement | 97.72% |
| disagreements | 2.28% |
| ...Stanza morph impossible given v5's POS | 42% of those |
| **impossible pairs, share of all tokens** | **0.96%** |
| on disagreements: v5 correct vs gold | 72% (Stanza 25%) |

"Impossible" means a `(UPOS, FEATS)` combination that never occurs among the
293 attested in UD EWT train. The failures are systematic, not random —
participial adjectives (`fledged`, `United`, `committed`), where Stanza reads
a past participle and emits `Tense=Past|VerbForm=Part|Voice=Pass` under an
`ADJ` that v5 gets right.

**Resolution: per-token loss masking.** Where the morph is incompatible with
the recorded POS, mask that token's morph label rather than teaching the
contradiction. Costs 0.96% of morph supervision; avoids falling back to a
coherent-bundle compromise that would cost 3.2 DEP points.

## Domain transfer

The bake-off scores each annotator on its benchmark's domain. The v6 corpus
is not those domains. `v6/probe_corpus.py` measures inter-annotator agreement
on 600 corpus sentences across all six domains — agreement, not accuracy,
since no gold exists there.

| domain | v5 vs Stanza | v5 vs spaCy | Stanza vs spaCy |
|--------|--------------|-------------|-----------------|
| conversation | 0.9726 | 0.8965 | 0.8900 |
| narrative | 0.9688 | 0.9218 | 0.9211 |
| news | 0.9670 | 0.9005 | 0.8971 |
| technical | 0.9668 | 0.8913 | 0.8815 |
| business | 0.9595 | 0.8705 | 0.8662 |
| encyclopedic | 0.9505 | 0.8943 | 0.8921 |
| **all** | **0.9640** | | |
| *UD EWT reference* | *0.9772* | | |

Agreement degrades 1.3 points with a 2.2-point spread — mild and uniform, so
the rankings transfer. But disagreement roughly doubles in the worst domain
(2.28% -> 4.95%), so the morph mask rate is not constant and must be computed
per sentence rather than assumed.

The third column is the more consequential finding: **two independent modern
taggers agree with each other 96% of the time and with the corpus's existing
spaCy labels only 87-92%.** `corpus/gold/test.parquet` is silver from a
single 2020-era tool despite the `gold` in its path. Re-annotating with the
bake-off winners is a real quality gain, not just a schema change — and any
v5 head trained on it inherited spaCy's errors.

## Verifications

**v5's published headline numbers both reproduce exactly.**

| head | model card | measured | n | delta |
|------|-----------|----------|---|-------|
| NER (OntoNotes 5.0) | 0.889 | 0.8888 | 8,261 | −0.0002 |
| SRL (PropBank EWT) | 0.843 | 0.8429 | 1,269 | −0.0001 |

The NER figure had no reproducing script in the repo — `download_benchmarks.py`
fetches OntoNotes but `benchmark_standard.py` only ever scored the mapped
CoNLL-03 protocol. It is now reproducible via
`v6.bakeoff --annotators kniv-v5 --layers ner --limit 9000`.

## Data defects found

**`benchmarks/ontonotes5_test.json` (kniv-corpus-en) has a broken tag-id map.**
2,205 / 152,723 tokens (1.44%) are mislabelled — `Europe` tagged `I-TIME`,
`1/4` tagged `B-TIME`. Six entity types score exactly 0.000 against it, and
any model scored on it is understated by ~10 F1 (v5: 0.782 vs 0.889).
`v6/gold/build_ner_gold.py` rebuilds the gold from `tner/ontonotes5`'s own
`label.json`, verifying the upstream tagset before writing. The original is
preserved as `ontonotes5_shipped_test.json`.

**`download_benchmarks.py`'s `TNER_TAGS` has 36 entries; tner has 37.**
Index 35 is mapped to `I-LANGUAGE` but is actually `I-ORDINAL`, and index 36
(`I-LANGUAGE`) is missing, so `.get(t, "O")` silently converts it to `O`.
This is a *different* bug from the one in the shipped file — that file uses
`tokens`/`tags` where this script writes `words`/`ner_tags`, so its
provenance is unknown.

Both live in the dataset repo and are not fixed by anything in `v6/`.

## Annotators evaluated but rejected

| candidate | reason |
|-----------|--------|
| **Trankit** | Model weights are served only from `nlp.uoregon.edu`, which is unreachable. Also requires Python 3.10 + `transformers==4.36`. Unmeasurable, and an unacceptable supply chain for production corpus generation. |
| **Maverick** | CC-BY-NC-SA-4.0. Best coref system available (87.4 CoNLL-F1) and unusable commercially — the licence restricts *use*, independent of what happens to outputs. |
| **MAI-Thinking-1** | Capacity 125 vs 2,500–7,500 elsewhere; ~0.1 items/sec. Mid-pack quality at 20–60× the wall-clock. Reserve for layers where reasoning plausibly helps. |
| **mistral** | Worst score on every layer and the largest failure source (47 of 70 `length` failures in the four-layer sweep; coverage down to 0.90 on NER). |

## Ensemble composition

luna, sol and terra are all `gpt-5.6` variants — one lineage, not three.
Measured same-family vs cross-family agreement:

| layer | delta |
|-------|-------|
| lemma | +0.007 |
| POS | +0.015 |
| morph | +0.069 |
| DEP | +0.164 |

On easy layers lineage is invisible; on hard layers the three converge
sharply on each other. Their three votes are worth roughly one where it
matters most. DeepSeek, Grok, Mistral and MAI are the genuinely independent
families.

grok is the strongest LLM overall — best on NER, SRL and coref, second on
POS — at ~129 s per call versus ~2–5 s for the others.

## Reproducing

```bash
./data/download_ud.sh
uv run python -m v6.gold.build_ner_gold
cp v6/annotators.example.yaml v6/annotators.yaml   # then fill in deployments

uv run python -m v6.bakeoff --layers pos,lemma,morph,dep --limit 300
uv run python -m v6.bakeoff --annotators kniv-v5 --layers ner,srl --limit 9000
uv run python -m v6.bakeoff --annotators fcoref,lingmess --layers coref --limit 200
```

Stanza, Trankit and fastcoref each need their own virtualenv — see
`v6/README.md`. The response cache is the integration point, so runs from
different environments land in the same report.
