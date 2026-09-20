# kniv v6

Next-generation pipeline. Deliberately separate from the v5 work under
`models/`, `shared/` and `scripts/` — nothing here imports from those. v5 is
referenced only as a *teacher*, never as a code dependency.

## Documents

| file | what it is |
|------|-----------|
| [DATASET_SPEC.md](DATASET_SPEC.md) | the v6 dataset: proposal, record schema, pipeline, QA gates, sequencing |
| [DECISIONS.md](DECISIONS.md) | which annotator produces each layer, and the measurement behind it |
| [CLS_TAXONOMY.md](CLS_TAXONOMY.md) | the CLS annotation contract |

## Annotator bake-off

The v6 corpus is annotated by a committee: frontier LLMs for the semantic and
pragmatic layers, the kniv v5 teacher for the structural ones. The bake-off
decides where that line falls, per layer, with a measurement.

It matters because **consensus measures agreement, not accuracy**. Where every
model in the ensemble is weak at a task, they agree on wrong answers and the
agreement score reports the corpus as clean. Published evidence puts zero-shot
LLM dependency parsing at roughly 40% UAS against a 34% left-branching trivial
baseline, and zero-shot SRL near 44 F1 — against v5's 0.944 UAS and 0.843 F1.
The bake-off checks whether that holds for *your* annotators rather than
assuming it.

Public treebanks are used for **evaluation only**. They are not v6 training
data. Training on our own corpus while reporting on public gold is the
strongest position available, and it is what keeps the numbers falsifiable.

### Setup

```bash
./data/download_ud.sh                      # UD English EWT (eval use)
cp v6/annotators.example.yaml v6/annotators.yaml
$EDITOR v6/annotators.yaml                 # deployments; secrets via ${ENV}
uv pip install -e ".[v6]"
```

SRL gold comes from `data/prepared/kniv-deberta-cascade/srl_test.json`
(PropBank EWT test) — fetch it from the `kniv-corpus-en` dataset repo or pass
`--srl-path`.

### Run

```bash
uv run python -m v6.bakeoff --dry-run                        # calls + cost, spends nothing
uv run python -m v6.bakeoff --annotators luna --layers pos --limit 25
uv run python -m v6.bakeoff --limit 300                      # the real thing
```

Output lands in `v6/runs/<run_id>/`: `report.md`, `report.json`, and
`events.jsonl` with one line per item.

### Reading the table

| column | meaning |
|---|---|
| `primary` | accuracy (POS, lemma), feat-F1 (morph), UAS (dep), span-F1 (SRL) |
| `secondary` | LAS for dep, exact-match for morph |
| `cover` | fraction of items the annotator returned a usable answer for — accuracy is over covered items only, so failures cannot inflate the score |
| `tree-ok` | dep only: fraction of parses that are actually single rooted trees |
| `vs v5` | delta against the published v5 teacher on the same test set |

A layer moves to the LLM ensemble only if it clears `vs v5`. Otherwise it
stays with the teacher.

The `ENSEMBLE` row is a per-token majority vote across annotators, included so
consensus is measured rather than assumed. Watch its `tree-ok` value in
particular: a per-token majority over dependency arcs is not guaranteed to be
a tree, and the rate quantifies how often voting produces something
structurally unusable.

## Operational contract

Carried forward from the v5 training code:

- **Resume** — every response is cached at `v6/runs/_cache/` keyed by
  `(annotator, layer, prompt_version, item)`. Writes are atomic; a corrupt
  entry from an interrupted write is discarded rather than read as empty.
  Kill a run and restart it: only the missing calls are issued.
- **Observability** — `events.jsonl` carries latency, token counts, repair
  count and failure *kind* per item. Console progress is unbuffered
  (`flush=True`) with a running rate, ETA and cost.
- **No silent failures** — malformed responses are counted by kind
  (`api`, `parse`, `length`, `range`) and surfaced. Nothing is padded,
  truncated, or coerced to a default label to make it fit. This is the
  failure mode that quietly poisoned earlier corpora, and it is designed out
  rather than handled.
- **Control** — annotators, layers, sample size, concurrency and repair
  budget are all CLI-selectable; `--dry-run` prices a run before it costs
  anything.
- **Provenance** — `prompt_version` is part of the cache key, so editing a
  prompt re-fetches instead of silently reusing stale responses.

## Layout

```
v6/
  config.py        annotator registry (${ENV} secrets), paths, layer list
  schemas.py       strict JSON schemas — one entry per token, enforced
  prompts.py       per-layer annotation guidelines; PROMPT_VERSION
  gold/            UD EWT + PropBank loaders (evaluation only)
  annotate/        cache, retry, validation, cost accounting, LLM client
  score/           per-layer metrics, all reported with coverage
  bakeoff.py       the driver
```

## Design rule: the annotator never re-tokenizes

Tokens are fixed before any annotator sees the text, and every response is
one entry per token index. Silent re-tokenization is the dominant failure
mode of LLM token-level annotation; here a length or range mismatch is a
recorded failure, and the model gets at most `--max-repairs` corrective
turns before the item is reported as failed.
