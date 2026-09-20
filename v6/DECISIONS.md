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
| coref | **LingMess** | 0.699 CoNLL-F1 | astra 0.701 | −0.2 (confirmed) |
| rel | **ATLOP retrained on Re-DocRED** | 0.790 triple-F1 | astra PAIRS 0.545 | +24.5 |
| CLS, sentiment, keyword | LLM ensemble | — | — | no gold exists |

**kniv-v5 takes four layers, Stanza two, LingMess one. Relations go to a
supervised model unioned with an LLM.**

The relation layer was re-opened after the first result (astra+grok union,
0.503) and improved to **0.566** — see "Improving the relation layer" below.
The original conclusion that relations had "no specialist to lose to" was
wrong: ATLOP exists, its released checkpoint fits our inventory exactly
because we adopted Re-DocRED's own 96 types, and it runs the full test set
in 90 seconds on a laptop.

Two decisions are open after the gpt-6 (`astra`) round:

* ~~**coref**~~ — **resolved: LingMess keeps the layer.** astra led by +1.4
  at n=200, which collapsed to **+0.16 on the full 513-window LitBank**
  (0.7006 vs 0.6990). Astra retains a real B³ edge (0.7254 vs 0.7109), so
  the two cluster differently, but not by enough to show in the headline.
  A 0.16-point difference cannot justify replacing a local MIT model with a
  hosted API across 20K windows — astra brings per-call cost, content-filter
  rejections and network failure modes that LingMess does not have.
* **NER** — astra 0.849 vs Stanza 0.848 at n=300 is inside the noise band
  that reversed last time; neither is near v5. Needs the full test set.

## Three findings that generalise

**1. Consensus beat best-of-breed on exactly one layer of eight — and only
under the right combination rule.**

The original finding was "7 layers, 7 times, majority voting loses". It
held, but it was stated too broadly: all seven were **per-token labelling**
tasks, where majority voting is the natural rule and drags the strongest
annotator toward the weakest.

Relations are **set-valued and recall-bound**, and there the rule matters
more than the idea:

| combination | rule | F1 | P | R |
|---|---|---|---|---|
| astra | — | 0.478 | 0.619 | 0.389 |
| grok | — | 0.470 | 0.532 | 0.421 |
| **astra + grok** | **union** | **0.503** | 0.520 | 0.487 |
| astra + grok + sol | union | 0.499 | 0.508 | 0.490 |
| astra + grok | majority | 0.435 | 0.668 | 0.323 |
| astra + grok + mai | union | 0.412 | 0.353 | 0.493 |
| all seven | union | 0.335 | 0.244 | 0.532 |
| all seven | majority | 0.310 | 0.655 | 0.203 |

Measured on the 472 Re-DocRED test documents all seven annotators covered
(16,293 gold triples). Per annotator:

| annotator | coverage | F1 | P | R |
|---|---|---|---|---|
| **astra** | 0.992 | **0.478** | 0.619 | 0.389 |
| **grok** | 0.952 | **0.470** | 0.532 | 0.421 |
| sol | 0.992 | 0.326 | 0.581 | 0.226 |
| luna | 0.992 | 0.275 | 0.499 | 0.189 |
| deepseek | 0.996 | 0.239 | 0.292 | 0.202 |
| mistral | 0.998 | 0.230 | 0.294 | 0.188 |
| mai | 0.996 | 0.110 | 0.153 | 0.085 |

grok's 0.952 coverage is the outlier: 23 documents lost to
`APIConnectionError` *after four retries each*, at a measured median of 507s
per call. It matches astra on quality and is far more expensive and less
reliable to run.

Majority *intersects* the annotators and destroys recall; union *adds* and
recovers it. The guardrails are narrow: only the top two members help
(adding sol costs 0.4, adding all six costs 11.5), and only where recall is
the binding constraint. On a per-token layer union would simply accumulate
errors.

**1b. Majority voting still loses on every per-token layer — 7 of 7.**

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

Sample size matters just as much, and this has now happened twice:

| layer | small sample | full sample |
|---|---|---|
| NER | Stanza 0.931 > v5 0.925 (n=300) | **reversed** — v5 0.889 > Stanza 0.882 (n=8,261) |
| coref | astra +1.4 over LingMess (n=200) | **evaporated** — +0.16 (n=513) |

A sub-1-point gap at small n is noise. Treat it as a prompt to enlarge the
sample, never as a result.

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

## Improving the relation layer

The first relation result was 0.503 (astra+grok union). Error decomposition
on the cached predictions showed the loss was **not** under-generation —
32.4 triples emitted per document against 34.5 gold — but pair *selection*:

| | share of misses |
|---|---|
| pair never proposed at all | **74.7%** |
| pair found, wrong relation | 24.9% |
| direction reversed | 0.3% |

and 81.4% of false positives fell on pairs holding no gold relation. Three
interventions were measured against that diagnosis.

| system | F1 | P | R | cost |
|---|---|---|---|---|
| ATLOP (supervised, local) | 0.459 | **0.952** | 0.302 | free, 90s total |
| astra | 0.477 | 0.619 | 0.388 | 1x API |
| astra + few-shot | — | — | — | +1.2 solo, **0.0 in union** |
| astra PAIRS (enumerated pairs) | 0.545 | 0.576 | 0.518 | 5.4x API |
| union astra+grok *(first decision)* | 0.502 | 0.519 | 0.487 | 2x API |
| **union ATLOP + astra** | **0.566** | 0.664 | 0.493 | **1x API** |
| **union ATLOP + PAIRS** | **0.586** | 0.596 | 0.577 | 5.4x API |
| union ATLOP + PAIRS + grok | 0.564 | 0.522 | 0.614 | 7x API |

**1. A supervised model is the single biggest gain, and it is free.**
ATLOP's released roberta-large checkpoint scores 0.952 precision at 0.302
recall against Re-DocRED. That profile is a *training artifact*: it was
trained on the ORIGINAL DocRED, whose systematic false negatives taught it
to be conservative; Re-DocRED restored the missing triples, which ATLOP
never proposes. Its errors are therefore near-complementary to the LLMs',
which over-propose — and the union exploits that for +8.9 F1 over astra at
**no additional API cost**.

**2. Pair enumeration works, at 5.4x the calls.** Handing the model the
candidate pairs instead of asking it to find them lifts astra 0.477 ->
0.545, and raises triples emitted per document from 23.7 to 32.4. Neither
prune helps: type constraints keep 97.4% of ordered pairs (DocRED has six
coarse entity types, most combinations admitting 40+ relations) and
locality pruning costs too much gold (same-sentence keeps 47.6%). So it is
~397 pairs per document, chunked 80 per call.

**3. Few-shot is not worth it.** +1.2 F1 solo and **0.0 in the union** — the
examples recover what the second annotator was already contributing.

**Superseded: retrain the supervised model.** See "Retraining ATLOP" below.
Everything above measures the RELEASED checkpoint, which was trained on the
original DocRED. Retrained on Re-DocRED it reaches **0.755 test F1**, beats
every LLM combination by 17+ points, and makes the LLMs redundant on this
layer — every union with an LLM now *lowers* F1.

**The open risk is domain.** ATLOP is trained on Wikipedia and its 0.952
precision is measured there. Our corpus is conversation, business and
narrative, where we have no gold. That precision may not survive the shift,
and it must be checked with the agreement probe (`v6/probe_corpus.py`)
before the corpus run, not assumed.

Reproduce: `v6/experiments/rel_variants.py` (few-shot, pairs) and
`v6/experiments/atlop_runner.py` (setup in its docstring).

## Retraining ATLOP: the decision for the relation layer

The released checkpoint's 0.952 precision / 0.302 recall was a training
artifact, not a property of the architecture: it learned from the ORIGINAL
DocRED, whose systematic false negatives taught it to under-propose.
Retraining the same architecture on **Re-DocRED train** (3,053 documents,
MIT) changes the regime entirely.

Measured on Re-DocRED **test**, 473 documents common to every system:

| system | F1 | P | R |
|---|---|---|---|
| **ATLOP retrained on Re-DocRED (10 epochs)** | **0.790** | **0.901** | 0.704 |
| ATLOP retrained, epoch 2 only | 0.755 | 0.816 | 0.703 |
| ATLOP released (DocRED-trained) | 0.459 | 0.952 | 0.302 |
| astra PAIRS | 0.545 | 0.576 | 0.518 |
| astra | 0.477 | 0.619 | 0.388 |
| grok | 0.469 | 0.530 | 0.421 |
| union retrained + astra | 0.721 | 0.672 | 0.778 |
| union retrained + PAIRS | 0.695 | 0.612 | 0.806 |

**The LLMs are redundant here.** Every union lowers F1 — they add more
false positives than they recover in recall. The relation layer needs no
API calls at all: 500 documents infer in ~90 seconds on a laptop.

Training: 10 epochs, best at the final epoch, dev F1 0.7799 / test 0.790.
roberta-large, batch 4, lr 3e-5, classifier lr 1e-4. The late epochs buy
**precision**: 0.7775 at epoch 0 rising to 0.8871 at epoch 9 while recall
holds near 0.70 — which is exactly what relation labels need.

An earlier partial run stopped at epoch 2 (test 0.755) and the gap between
epochs 2 and 4 was only +0.3 F1, from which this document previously
concluded the curve had flattened. **That was wrong** — epochs 4 to 9 added
a further 2.4 dev F1 and 3.4 precision. A plateau inferred from two
adjacent points in the middle of a curve is not a plateau.

### Precision knob for training labels

Relation labels are training data, where a wrong triple teaches an error
and a missing one only costs supervision. Intersecting with an LLM raises
precision sharply:

| rule | F1 | **P** | R |
|---|---|---|---|
| **complete model alone** | **0.790** | **0.901** | 0.704 |
| complete AND PAIRS | 0.588 | 0.964 | 0.423 |
| complete AND astra | 0.482 | 0.966 | 0.321 |

**Use the model alone.** At 0.901 precision the intersections are no longer
worth their cost: they buy ~6 precision points and give up ~28 recall
points, and they reintroduce the API calls the supervised model removed.
Uncertain pairs are masked rather than labelled `no_relation`, as for
morph.

ATLOP's adaptive threshold remains available as a cheaper precision knob
than an LLM intersection if 0.901 is not enough for a given layer.

### Operational lessons (five VMs lost, then a clean run)

Five A100s were reclaimed at 1.5h, 1h, 22min, 25min and 28min before the
sixth attempt completed 10 epochs without incident.

**Root cause: the job ran detached from the Jupyter kernel.** Training was
launched as a background OS process, so the GPU was busy but the *kernel*
was idle — and Colab reclaims on kernel idleness, which keep-alive does not
address. `colab status` reported `IDLE` throughout every failed run and
that was dismissed as cosmetic. Running the same job *inside* the kernel
via `colab exec` fixed it outright.

Two things this ruled out along the way: the auth token carries the
`colaboratory` scope, and the keep-alive daemon does spawn and does
survive. Neither was the problem.

Note that a kernel-resident job cannot be monitored with `colab exec` — the
kernel is single-threaded, so probes queue behind the training cell. Use
the contents API (`colab download`) instead, which is what the checkpoint
puller does.

Four further failures were ours, and each has a fix now in the code:

| failure | cause | fix |
|---|---|---|
| lost a 0.7374 checkpoint entirely | nothing copied weights off the VM | puller downloads `best.pt` on every improvement |
| 40 minutes of "progress" after the VM died | puller downloaded *into* the destination file, so a failed download silently left the stale copy | download to a temp path; treat a >15 min stale heartbeat as a dead run |
| **overwrote a 0.7161 checkpoint with a 0.6278 one** | no comparison before writing | keep a `best.f1` sidecar; only replace on a higher score |
| restarting from scratch three times | only `best.pt` was pulled — no optimizer state | pull `latest.pt` too, so a new VM resumes |
| lost a 0.7511 checkpoint when the VM died mid-pull | the 4.1 GB resume point was fetched *before* the 1.4 GB weights | weights first, resume point second |

The general shape is the one that keeps recurring in this project:
**silence and staleness look identical to progress unless something
explicitly checks.**

## Environment fragility found while re-running coref

`.venv-tools` had been upgraded to transformers 5.x at some point after the
original coref run, and **LingMess stopped loading entirely** — two separate
breakages, both silent until a run needed an uncached window:

| symptom | cause | fix |
|---|---|---|
| `LongformerModel does not support ... scaled_dot_product_attention` | transformers now defaults to SDPA; LingMess is Longformer-based and fastcoref exposes no `attn_implementation` | inject `attn_implementation="eager"` for the duration of the load only |
| `'LingMessModel' object has no attribute 'all_tied_weights_keys'` | transformers 5.x reads an attribute its own `post_init` sets; fastcoref's custom classes never call it | class-attribute fallback (not a property — `post_init` *assigns* to it on models that do call it) |

Both live in `v6/annotate/coref.py` as a narrow context manager rather than
process-wide settings, so FCoref and every other model path are untouched.

The wider point: the cached predictions stayed valid and replayed fine, so
nothing looked wrong until a larger sample was requested. **A decided layer
had become unreproducible without anyone noticing.** Pinning the annotator
environments is a prerequisite for the corpus run, not a tidiness exercise.

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
