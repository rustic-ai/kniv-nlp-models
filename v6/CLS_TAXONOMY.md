# v6 CLS taxonomy

Six labels, multi-label, named after ISO 24617-2 general-purpose
communicative functions. This file is the annotation contract: it defines
the labels, the decision procedure, and the edge cases. Where it and a
prompt disagree, this file wins.

## Provenance

ISO 24617-2 (the Dialogue Act Markup Language standard, DiAML) defines 56
communicative functions across 9 dimensions. Its **general-purpose**
functions form this hierarchy:

```
General-Purpose Communicative Functions
├─ Information-Transfer
│  ├─ Information-Seeking → Question
│  │     Propositional Q · Set Q · Choice Q · Check Q · Test Q
│  └─ Information-Providing
│        Inform  → Agreement, Disagreement
│        Answer  → Confirm, Disconfirm
└─ Action-Discussion
   ├─ Commissives  Offer · Promise · Address/Accept/Reject Request
   └─ Directives   Request · Instruct · Suggestion ·
                   Address/Accept/Reject Suggestion ·
                   Address/Accept/Reject Offer
```

Two things this taxonomy is **not**:

1. **It is not full ISO.** Full ISO assigns each segment a *dimension* as
   well as a function, and permits labelling at any depth of the tree.
   Doing that properly needs a hierarchical classifier with one output
   layer per level; the one published attempt trains on DialogBank, which
   is far too small to be a teacher corpus. We take ISO's top level only.
2. **It is not purely general-purpose.** `Feedback` and `Social` are
   dimension-specific in ISO — `autoPositive`/`autoNegative` belong to the
   Auto-Feedback dimension, greeting/goodbye/thanking/apology to Social
   Obligations Management. A flat head has to flatten across dimensions to
   represent them, and both are far too frequent in conversational text to
   drop.

So: **ISO-derived, not ISO-compliant.** Documented that way deliberately,
because claiming standards compliance we do not have is how the previous
taxonomies ended up undefendable.

## Why not finer

ISO's own annotated corpora are extremely long-tailed: `inform` 53.8%,
`autoPositive` 20.7%, then `propositionalQuestion` 2.6%, `setQuestion`
0.97%, `checkQuestion` 0.66%. Splitting `Question` into ISO's five subtypes
reproduces exactly the failure mode of the shipped v5 head, where three of
eight labels were near-empty and macro-F1 measured almost nothing.

Reference points for what is achievable: SwDA with 42 tags is ~85.5%
accuracy with a human inter-annotator ceiling of 84% / κ=0.80; MRDA
collapsed to 5 tags is ~92.2%. Coarse labels are not a compromise, they are
where the reliable signal is.

## The labels

Multi-label. Every function that applies fires. The empty set is legal and
means none applied (filler, stalling, fragments) — there is no `SKIP` class.

| label | ISO origin | fires when |
|-------|-----------|-----------|
| `Question` | Information-Seeking | the speaker seeks information they do not have |
| `Inform` | Information-Providing (incl. Answer, Agreement, Disagreement) | the speaker asserts, answers, agrees or disagrees with propositional content |
| `Directive` | Action-Discussion / Directives | the speaker tries to get the *addressee* to act — request, instruct, suggest |
| `Commissive` | Action-Discussion / Commissives | the *speaker* commits to act — offer, promise, accept/reject a request |
| `Feedback` | Auto-Feedback dimension | the speaker signals their own processing of what was said — acknowledgement, backchannel, non-understanding |
| `Social` | Social Obligations Management | greeting, goodbye, thanking, apology, congratulation |

## Decision procedure

Evaluate every label independently against the utterance. Do not pick "the
best one".

1. **Question** — is there a genuine information request? Rhetorical
   questions and *checking* questions both count (ISO's `checkQuestion` is
   Information-Seeking). Tag questions on an assertion fire both `Question`
   and `Inform`.
2. **Inform** — is propositional content asserted? Answers are `Inform`
   (ISO's `Answer` is under Information-Providing). Agreement and
   disagreement about content are `Inform`, **not** `Feedback`.
3. **Directive** — is the addressee being asked, told or advised to act?
   Includes imperatives, polite requests, and suggestions.
4. **Commissive** — is the speaker undertaking to act? Includes accepting
   or refusing someone else's request.
5. **Feedback** — is the speaker reporting their own uptake? *mm-hm, ok,
   right, sorry what?* Distinguishing rule: `Feedback` is about the
   **communication**; `Inform` (agreement) is about the **content**.
   "Right." as a backchannel is `Feedback`; "Right, it shipped Tuesday" is
   `Inform`.
6. **Social** — is a social obligation being discharged? Greetings,
   thanks, apologies, farewells.

## Known edge cases

| utterance | labels | why |
|---|---|---|
| "Can you send the report?" | `Directive` | interrogative form, directive function |
| "Do you know when it ships?" | `Question` | genuine information-seeking |
| "It ships Tuesday, right?" | `Question` + `Inform` | check question over asserted content |
| "Sure, I'll do it." | `Commissive` | speaker commits |
| "Sure." (after a request) | `Commissive` | accept-request, even with no content |
| "Got it, thanks." | `Feedback` + `Social` | uptake plus thanking |
| "No, it was Wednesday." | `Inform` | disagreement about content |
| "Sorry, what?" | `Feedback` | negative auto-feedback, not `Social` apology |
| "um, so, yeah" | ∅ | stalling; ISO Time Management, out of scope |

## External benchmark: SGD gold dialogue acts

CLS had **no gold at all** — produced by an LLM, evaluated against nothing,
so "CLS quality" was a claim rather than a number. Schema-Guided Dialogue
carries human dialogue-act annotation (CC-BY-SA-4.0) whose 18 acts map onto
these six functions, giving the one external reference available without
commissioning annotation. `v6/gold/sgd_cls.py` builds it; 55,377 corpus
sentences match an SGD turn exactly.

| label | P | R | F1 | gold | predicted |
|---|---|---|---|---|---|
| Social | 0.929 | 0.844 | **0.884** | 10,296 | 9,351 |
| Question | 0.761 | 0.844 | **0.801** | 14,189 | 15,738 |
| Inform | 0.833 | 0.759 | **0.794** | 28,547 | 26,013 |
| Directive | 0.436 | 0.577 | 0.496 | 6,113 | 8,093 |
| Commissive | 0.724 | **0.180** | 0.289 | 9,625 | 2,395 |
| Feedback | — | — | — | 0 | 2,176 |

Exact set match 60.1%, mean Jaccard 0.701, **micro F1 0.730** excluding
Feedback.

**Feedback is unscorable here.** No SGD act maps to it: task-oriented dialogue
annotated for slot-filling does not mark backchannels. Its 2,176 predictions
are counted as false positives however correct they are, which is why it is
excluded from the aggregate rather than reported as 0.000. The thinnest class
in the corpus is the one this benchmark cannot measure.

**The mapping is the dominant source of error, not the annotator.** A first
pass scored micro F1 0.618 and Commissive F1 0.131; both were artifacts of
mapping SGD's `OFFER` — 38,270 acts, third most common — onto Commissive.
SGD's `OFFER` presents an *entity* ("I found a good restaurant in Milpitas"),
which is Inform; the taxonomy's Commissive "offer" is offering to *act*, which
is SGD's `OFFER_INTENT`. Reading utterances per act instead of trusting act
names moved micro F1 to 0.730. Anyone extending this mapping should read
examples first.

**Commissive recall 0.180 is a taxonomy-boundary question, not a settled
defect.** It concentrates almost entirely in `SELECT` (n=3,814, recall
**0.017**), where SGD marks accepting an offered option — "Yes, it seems good
for me" — and the annotator says `Inform`. Three things are established: the
annotator had the full window as context (0 of 3,814 fell beyond the
2,000-character cut), the prompt states the rule verbatim ("A bare 'Sure.'
after a request is Commissive"), and it still does not fire.

That leaves a genuine boundary dispute rather than a bug. The taxonomy's
example accepts a request to *act*; SGD's `SELECT` accepts a *choice*, which
is arguably agreement with an evaluation and therefore `Inform`. Whether the
corpus is under-labelling Commissive or this benchmark over-claims it **is
exactly what the adjudicated gold set should settle**, and it says where to
spend that sample: accept-type utterances, not a uniform draw.

This matters because Commissive is one of the two thin classes Taskmaster-2
and SGD were added to raise. If the annotator does not label accepts, adding
accept-dense data does not raise it.

**What this does not measure.** SGD is task-oriented human/assistant dialogue:
dense in Directive, Commissive and Social, with no argumentative or narrative
text. It is evidence about CLS on task-oriented dialogue — a large part of the
conversation domain and none of the other four — and it is a benchmark, not
the human ceiling. An annotator and a human can agree with each other and both
differ from SGD's conventions.

## Relationship to the three label sets in the repo

None of the three is a subset of another; all three are superseded.

| old label | source | v6 |
|---|---|---|
| inform, status | shipped model | `Inform` |
| question, question_fact | shipped / eval guide | `Question` |
| request, command | shipped / eval guide | `Directive` |
| offer | shipped | `Commissive` |
| confirm, acknowledgment, agreement, feedback | all three | `Feedback` or `Inform` by the rule in step 5 |
| reject | shipped | `Commissive` (reject-request) or `Inform` (disagreement) |
| correction | `classify.py` | `Inform` |
| plan_commit | `classify.py` | `Commissive` |
| social, greeting | all three | `Social` |
| filler | `classify.py` / eval guide | ∅ |
| statement | eval guide | `Inform` |

`plan_commit` vs `request` was an actor distinction (who acts). That is now
carried by `Commissive` vs `Directive`, which is the same distinction under
its standard name, and the SRL head recovers the actor independently
via `ARG0`.

## Measured distribution

Annotated over 427,450 corpus sentences, then re-measured after adding two
spoken-dialogue sources (Taskmaster-2, SGD):

| label | first corpus | Taskmaster-2 + SGD | blended (projected) |
|---|---|---|---|
| Inform | 73.7% | 50.5% | ~69% |
| Directive | 11.0% | 16.5% | ~12% |
| Question | 7.7% | 27.9% | ~12% |
| Social | 3.7% | 11.8% | ~5% |
| Commissive | 2.5% | 4.8% | ~3.0% |
| **Feedback** | **1.9%** | **4.1%** | **~2.4%** |
| *empty (no function)* | 6.5% | 0.1% | ~5% |

**Multi-label rate: 6.6%** of sentences carry more than one label. This is
the measured figure; an earlier draft of this document quoted 44% from a
substring heuristic, which was not a measurement.

### Feedback stays thin, and that is accepted

Adding spoken dialogue raised every minority label — Feedback 2.2x,
Question 3.6x, Social 3.2x — and Feedback is still only ~2.4% blended. The
reason is structural: backchannels are dense in telephone and meeting
speech (Switchboard, Fisher, MRDA) and every one of those is LDC-gated,
the same wall that blocked Re-TACRED.

**Decision: accept the distribution and evaluate per class.** Two reasons.
The absolute count is large — 4.1% of 3.68M conversational tokens is ample
supervision, and v5's actual failure was `reject` firing once per 500
sentences, which is a different problem. And reweighting the corpus toward
conversation would narrow domain coverage for every other head sharing the
encoder.

The consequence is a reporting rule, not a training change: **per-class F1,
never macro.** A macro average over six classes where two sit near 2%
measures variance more than skill, and that averaging is what hid v5's
problem.

## Evaluation

There is no public gold for this scheme on our domains, so:

- Gold is built by **adjudication** — five independent LLM families label
  N items, unanimous items get a 10% human spot-audit, disagreements go to
  human adjudication with a recorded rationale. Target 400–600 items
  (`data/locomo50_gold_labels.csv` is this shape at n=50, which is ±14
  points of 95% CI — too small).
- Report **Cohen's κ against humans with the human–human ceiling beside
  it**, from a 100-item two-annotator overlap. If human–human κ is below
  ~0.6 the guide above is underspecified and no model will fix it.
- Multi-label metrics: per-label F1, micro-F1, and exact-set-match. Never
  a single macro average — the aggregate is what hid the problem last time.
- Slice per domain and per label, and inspect the confusion pairs the edge
  case table predicts (`Feedback`/`Inform`, `Question`/`Directive`).
- Public dialog-act sets (SwDA, DailyDialog) are a domain-shift probe only,
  never a headline, and DailyDialog is CC-BY-NC-SA — evaluation only.
- **SGD is the most promising external benchmark.** It is CC-BY-SA-4.0 and
  ships gold dialogue acts — AFFIRM, NEGATE, OFFER, REQUEST, THANK_YOU,
  GOODBYE, CONFIRM, SELECT, NOTIFY_SUCCESS/FAILURE — which map onto these
  six labels. That would give CLS a human-annotated external yardstick
  rather than LLM adjudication alone, which is the weakest part of the plan
  above. The text is already collected; the acts are not yet extracted.
