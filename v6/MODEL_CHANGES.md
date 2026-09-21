# v6 model changes

Architecture changes for the v6 cascade. The corpus is specified in
[DATASET_SPEC.md](DATASET_SPEC.md); this file covers what the model does
differently, and why.

Each entry states what it costs as well as what it buys. Where a change
carries a quality risk, the experiment that settles it is named.

---

## 1. SRL: score every predicate in one pass

### What v5 does

The SRL head is **predicate-conditioned**. A marker is injected at the
embedding layer, so every attention layer sees which token is the
predicate:

```python
emb = encoder.embeddings(ids)
indicator[0, predicate_subtoken] = 1
emb = emb + pred_embedding(indicator)
hidden = encoder.encoder(emb, mask)
```

Two consequences:

* **The caller must supply the predicate.** In practice that means running
  POS or DEP first and passing verb indices in — the model cannot be used
  on raw text alone.
* **One encoder pass per predicate.** Measured at 4.82 predicates per
  sentence (`data/bench_srl_fanout.json`), so SRL costs ~4.8x every other
  layer. On the v6 corpus build that is 1,974,952 passes against 409,741
  for each of POS/NER/DEP — 56% of the total annotation time.

### What v6 should do

Two heads over a single encoding:

| head | shape | job |
|---|---|---|
| predicate | `[S, 2]` | is this token a predicate? |
| role | `[S, S, 42]` | for a (predicate, token) pair, which role? |

The role head is the **same biaffine machinery the DEP head already
uses**, which produces `[S, S, 53]` over (token, candidate-head, relation).
SRL becomes that with a different label set, which is an argument that it
fits the cascade rather than bolting onto it.

### What it buys

* **One encoder pass per sentence instead of 4.82.**
* **No predicate argument in the API.** The head reads raw text; the
  caller stops needing a POS pass and stops needing to know what a
  predicate is. This is the larger win — it is an ease-of-use change, not
  only a speed change.
* Consistency with DEP: one decode path, one biaffine implementation.

### What it risks

**It may cost F1.** v5's conditioning is not incidental — injecting the
marker at the embedding layer specialises the *entire encoder* to one
predicate. A biaffine head reading a single unconditioned encoding has to
serve every predicate at once, and end-to-end SRL models have historically
traded accuracy for that. The bet is that the shared multi-task encoder
compensates.

**Settle it by measurement, not argument:** train both heads on the same
v6 corpus and score on PropBank EWT against v5's published 0.843. If the
joint head loses more than ~1 F1, the fallback is a hybrid — predict
predicates jointly, then run the conditioned head only for the predicates
found, which still removes the API burden while keeping v5's accuracy.

### What it does not do

**It does not reduce the corpus build cost.** The corpus is annotated by
*v5*, which still needs its per-predicate passes. The saving is at v6
inference time.

### Corpus impact: none

`DATASET_SPEC.md` already stores SRL as
`srl_frames: [{predicate_idx, tags}, ...]` — a list of frames per window
rather than one tagged sequence per example. A joint head trains from that
by reshaping, and a conditioned head trains from it directly. No corpus
change is needed either way, which is what makes this decidable later
rather than now.
