# v6 CLS taxonomy

## Purpose

The CLS head is a **dispatch table**, not a linguistic classification. Its
job is to decide what the downstream memory system does with an utterance.
`corpus/pipeline/classify.py` already said so — every label there carries an
arrow to a memory operation.

That fixes the design rule: **granularity is set by the number of distinct
downstream actions.** Two labels triggering the same operation are one class
with two names. A label triggering no operation is not a class.

## The labels

Five actions. Multi-label: every action that applies fires.

| label | memory operation | fires when |
|-------|------------------|-----------|
| `EXTRACT` | create an observation | the utterance asserts something about the world that could be stored |
| `UPDATE` | revise or reinforce an existing observation | it corrects, contradicts, confirms or agrees with something already said |
| `QUERY` | record a knowledge gap | it seeks information |
| `COMMIT` | create a goal/task node | it creates an obligation or intention to act |
| `SKIP` | nothing | none of the above fired |

## Decision procedure

Evaluate all four content labels independently, then `SKIP` if none fired.

1. **EXTRACT** — does it assert a storable fact about the world, a person, or
   a state of affairs? Include facts embedded in other acts.
2. **UPDATE** — does it revise, contradict, confirm, or agree with prior
   content? Requires discourse context; the marker is usually explicit
   (*actually, no, I meant, yes that's right, exactly*).
3. **QUERY** — does it seek information the speaker does not have?
4. **COMMIT** — does it create an obligation or intention to act, by either
   party? (*I'll send it* / *please restart the server* / *shall we meet?*)
5. **SKIP** — assign only if 1–4 all failed. `SKIP` is exclusive by
   construction, which removes the annotator's hardest judgement call.

## Multi-label is the point

Single-label forces a choice the data does not support. The evidence here is
illustrative, not quantified — **nobody has yet annotated anything under this
scheme**, so the rate at which utterances carry two labels is unknown and is
one of the things the adjudicated gold set exists to measure.

What is checkable today: the `rationale` column in
`data/locomo50_gold_labels.csv` exists because annotators needed somewhere to
record what the single label discarded — *"opening greeting + phatic"*,
*"reaction then info-seeking question"*, *"thanks + new question about the
painting"*.

And the cost is concrete. `data/locomo50_with_prev_multilabel.csv` row 1:

> *"Hey Mel! Good to see you! How have you been?"* — gold `social`, model
> predicted `question` at 0.83, **scored wrong.**

(`data/locomo50_with_prev_multilabel.csv`, row 1.)

The model was right. It is a greeting **and** a question: `SKIP` + `QUERY`
under this scheme — except `SKIP` is exclusive, so it is simply `QUERY`.
Forced single-label choices like this inflate the apparent error rate and are
part of why CLS reads 0.951 in-domain and 0.613 in the wild.

Worked examples:

| utterance | labels |
|-----------|--------|
| "Caroline works at the hospital downtown." | `EXTRACT` |
| "Did you know Caroline moved to Paris?" | `EXTRACT` + `QUERY` |
| "Actually, she moved to Paris, not London." | `EXTRACT` + `UPDATE` |
| "I'll send the report tomorrow." | `EXTRACT` + `COMMIT` |
| "Please restart the server." | `COMMIT` |
| "Where does she work now?" | `QUERY` |
| "Yes, that's right." | `UPDATE` |
| "Hey! How have you been?" | `QUERY` |
| "lol" / "Thanks!" / "Good morning." | `SKIP` |

Implementation: per-label sigmoid with a threshold, not a softmax. `SKIP` is
predicted as the complement — if no content label clears threshold, emit
`SKIP`.

## What was deliberately dropped

**The sparse classes.** This is the load-bearing argument for the collapse,
and it rests on measured counts rather than interpretation. In 500 wild
sentences (`longmemeval_summary.json`)
`status` fired 8 times, `reject` once, `offer` once. A 500-item gold set
yields 1–8 examples of each — not enough to estimate an F1, let alone
compare annotators. Three of the shipped eight labels were unmeasurable.
They collapse into `EXTRACT`, `UPDATE` and `COMMIT`.

**The actor distinction.** `plan_commit` (speaker will act) versus `request`
(addressee should act) is a real difference, but splitting `COMMIT` puts both
halves near 2% — back in unmeasurable territory. The actor is already
recoverable from the SRL head: *I'll send it* has `ARG0 = I`, *please send
it* is imperative with an implicit addressee. **Let the cascade carry it**
rather than paying for two sparse CLS classes.

**A second level.** A nested dialog-act tier (`inform`/`status` under
EXTRACT, `correction`/`agreement` under UPDATE, etc.) was considered and
rejected for v6. It reintroduces exactly the sparse classes that made v5's
CLS unmeasurable, doubles the annotation burden, and has no consumer today.
Annotators may record a finer act as free metadata; no head predicts it and
no report includes it.

## Expected distribution

Consistent across two independent samples, and stable under the collapse:

| label | 500 wild (LongMemEval) | 50 gold (LoCoMo) |
|-------|------------------------|------------------|
| `EXTRACT` | 41.4% | 42% |
| `QUERY` | 29.6% | 34% |
| `SKIP` | 21.2% | 20% |
| `COMMIT` | 4.0% | 0%* |
| `UPDATE` | 3.8% | 4% |

\* LoCoMo is casual conversation with no task commitments; business and
technical domains carry `COMMIT`.

Two consequences to plan for:

- **`COMMIT` and `UPDATE` need deliberate oversampling** in the synthetic
  corpus. A natural 500-item sample gives ~20 examples each. This is the
  concrete form of "diversity over volume".
- **Multi-label raises the effective count per label**, since an utterance
  can carry two — so the gold set supports these five better than a
  single-label set of the same size would.

## Context requirement

`UPDATE` cannot be decided from an utterance alone — *"Yes, that's right"*
reinforces something, and which something matters. It needs the preceding
turns. This is the same requirement that drove the 512-token window, so the
taxonomy and the encoder contract reinforce each other: CLS is predicted per
sentence **within** a window that supplies its context.

## Migration

| from | to |
|------|----|
| `inform`, `status`, `statement` | `EXTRACT` |
| `confirm`, `reject`, `correction`, `agreement` | `UPDATE` |
| `question` | `QUERY` (+ `EXTRACT` where a fact is embedded) |
| `request`, `offer`, `plan_commit`, `command` | `COMMIT` |
| `social`, `feedback`, `filler`, `greeting`, `acknowledgment` | `SKIP` |

Existing single-label data (corpus parquet in the 9-label set, `locomo50`
gold in the 8-label set) maps forward mechanically, but **the mapping is
lossy in one direction only**: it cannot recover the second label a
multi-label scheme would assign. Migrated data is usable for training and
**not** usable as evaluation gold — the adjudicated gold set must be
annotated natively under this scheme.

## Evaluation

- **Macro-F1 over the five labels**, plus per-label P/R.
- **Cohen's kappa against human gold, with the human–human ceiling beside
  it.** Two annotators on a 100-item overlap. Below ~0.6 human–human means
  this document is underspecified and no model will fix it.
- **Sliced by domain**, never averaged alone — the aggregate is what hid the
  in-domain/wild gap last time.
- Public dialog-act sets are a domain-shift probe with the mapping stated,
  never a headline. DailyDialog is CC-BY-NC-SA: eval only.
