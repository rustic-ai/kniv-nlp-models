# v6 status

**Where we are:** the training corpus is **built, gated and audited**. The
model is **designed but not written** — `v6/train/` does not exist. The next
unit of work is the training code, not more data.

Corpus build id: git `fecf5ae`. Figures below are read from
`data/v6-corpus/MANIFEST.json` and `v6/gates.py`, not from notes.

---

## 1. The three v6 goals

| goal | status |
|---|---|
| **Improve CLS quality** | Corpus layer complete (686,759 sentences). First external measurement exists: **micro F1 0.730** against SGD gold. One taxonomy decision open (§5.1) that changes what `Commissive` means on ~9,600 sentences. |
| **Single pass over a whole message or document** | Corpus supports it: every window carries all 11 layers with per-layer masks. The single-pass SRL head is designed (`MODEL_CHANGES.md` §1) and unbuilt. |
| **Relation graph builder** | Corpus layer complete: **145,596 triples** over 30,284 documents. Relations are a **model head**, not a pipeline — ATLOP was the annotator, not the deliverable. Needs coref as a head too, since relations are defined over coref-merged clusters. |

---

## 2. Corpus

**36,699 windows · 12,538,722 tokens · 686,759 sentences · 20,263 documents · 19 shards · 11 layers**

Target was 10M tokens ≈ 20,000 windows (`DATASET_SPEC` §2.4). Exceeded on both.

| domain | windows | tokens | share | documents | sources |
|---|---|---|---|---|---|
| conversation | 16,173 | 3,671,038 | 29.3% | 15,076 | oasst, multiwoz, glaive, sgd, taskmaster, taskmaster2 |
| technical | 6,459 | 2,847,420 | 22.7% | 1,513 | python_docs, wikipedia |
| business | 6,519 | 2,688,203 | 21.4% | 1,827 | sec_edgar, openstax, odoo, wikipedia, cuad, s2orc |
| narrative | 3,008 | 1,385,519 | 11.1% | 475 | gutenberg (10 books) |
| news | 2,788 | 1,173,758 | 9.4% | 955 | wikinews, wikipedia |
| encyclopedic | 1,752 | 772,784 | 6.2% | 417 | wikipedia |

All six domains of the spec are present. **Enron is deliberately excluded** —
it was the only source carrying personal data about people who did not consent,
and it was 15,000 of 185,000 business sentences. What was lost is a *register*
(informal workplace correspondence), not volume. No PII gate is therefore
required, and gate 4 keys off the Enron **source** so re-adding it re-arms the
gate automatically.

**Splits:** train 33,037 (90.0%) · dev 1,831 (5.0%) · test 1,831 (5.0%), on
`doc_id`, stratified by domain, exactly 5.0% per domain. No document spans
splits. Exact-duplicate windows (193) are dropped *before* the split is
assigned, which also closes a content leak that splitting on `doc_id` cannot
see: two different documents holding identical text.

**Layers and provenance** — every row carries `provenance` and `loss_mask`:

| layer | annotator | version |
|---|---|---|
| pos, ner, dep, srl | kniv-v5 | local checkpoint, identified by size + mtime |
| lemma, morph | stanza | 1.14.0, processors `tokenize,pos,lemma` |
| coref | lingmess | fastcoref 2.1.6 |
| cls, sentiment, keywords | astra | gpt-6-astra |
| relations | atlop-redocred | checkpoint + `dev_f1=0.7799` |

**Coverage:** pos/ner/dep/lemma/morph/coref 100%; srl_frames 99.8%;
cls/sentiment/keywords 99.6%. Relations on 33.7% of windows, masked elsewhere.

---

## 3. Quality gates

**20 gates: 19 pass, 0 fail, 1 not checked.** Gate 2 (no sentence split across
a window boundary) reports *not checked* rather than passing — it is a property
of the builder and is not reconstructible from the shards. Not-checked and
passed must not look the same.

Selected reported values:

- **Encoder overflow:** 13,502/36,699 windows (36.79%) carry a tail past 512
  subwords — 429,755/12,538,722 word positions (**3.43%**). All recorded per
  row as `encoder_word_limit`. **Training must mask `[encoder_word_limit,
  n_tokens)`.**
- **DEP:** 49,497/686,759 sentences (7.21%) are not well-formed trees. All
  masked, none unmasked.
- **CLS:** 6.12% of sentences carry no dialogue-act function. Verified by hand
  to be fragments, headings and bibliography entries — correct behaviour.
- **Relation yield per window:** news 22.52 · encyclopedic 9.17 · business 4.70
  · technical 2.45 · conversation 1.14 · narrative 0.61.
- **CLS distribution is strongly domain-dependent:** encyclopedic 99% Inform,
  news 98%, technical 90%, business 86%, narrative 78%, conversation 46%.
- **Attribution:** all 36,699 rows carry `source_url` and `license`. Eight
  licences present, including CC-BY-SA — **share-alike applies to the dataset
  if it is published.**

---

## 4. Independent quality evidence

**Sampled audit** (`v6/AUDIT.md`): 823 windows / 15,424 sentences, stratified,
fixed seed, ids committed. Automated probes plus reading complete windows and
33 labelled sentences by hand. Verdict: **usable for training.**

- POS `X` rate 1.32%; punctuation→`PUNCT` 98.9%; UPOS distribution normal for
  English (NOUN 21.6, PUNCT 14.9, VERB 10.1, ADP 9.2, DET 8.0).
- CLS agreed with a manual read on roughly **nine sentences in ten**, and the
  disagreements were *concentrated*, not scattered — all on the accept/offer
  boundary of §5.1.
- Sentence splitting 99.6% clean; 0.40% break on abbreviation periods.
- Zero integrity issues on a trainer-style load of 10,813 rows.

**External CLS benchmark** (`v6/gold/sgd_cls.py`): CLS had no gold at all.
SGD's human dialogue acts give one. Over 55,377 corpus sentences matching an
SGD turn exactly:

| label | P | R | F1 |
|---|---|---|---|
| Social | 0.929 | 0.844 | **0.884** |
| Question | 0.761 | 0.844 | **0.801** |
| Inform | 0.833 | 0.759 | **0.794** |
| Directive | 0.436 | 0.577 | 0.496 |
| Commissive | 0.724 | 0.180 | 0.289 |

**Micro F1 0.730**, exact set match 60.1%, excluding `Feedback` — no SGD act
maps to it, so the corpus's thinnest class is the one this benchmark cannot
measure.

---

## 5. Open decisions

### 5.1 The Commissive/accept boundary — *the one that changes the data*

Does accepting an offered **option** (as distinct from accepting a request to
**act**) count as `Commissive` or `Inform`?

Three independent signals converge here:

1. SGD marks it as its own act (`SELECT`); our annotator says `Inform`
   (recall 0.017 on n=3,814).
2. **MultiWOZ, annotated independently, has no user-side accept act at all.**
   Across 3,021 acceptance-shaped user turns its gold is `Inform` 48.3%,
   `thank` 24.8%, `Request` 19.2%. Our annotator agrees with MultiWOZ.
3. The hand audit's CLS disagreements land on exactly this boundary.

Ruled out: missing context (all 3,814 had the full window; 0 beyond the
2,000-character cut) and a missing prompt rule (the prompt states it verbatim).
So the rule is **underdetermined**, not violated. The taxonomy should decide
explicitly and say so in the label table.

This decides whether adding Taskmaster-2 and SGD achieved what they were added
for: if accepting an option is `Commissive`, the corpus under-labels it on
thousands of sentences; if `Inform`, the class is genuinely rare in
task-oriented dialogue.

### 5.2 SRL over auxiliaries — *affects a third of one layer*

`SRL_PREDICATE_TAGS` includes `AUX`, which produces **14,488 of 43,572 frames
in the audit sample** — a third of the layer — of which only **13.9%** mark the
conditioned predicate as `V`, against **91.2%** for VERB.

The head is right and the question is wrong: PropBank treats an auxiliary as
`ARGM-MOD` of the main verb, not a predicate (`B-ARGM-MOD` appears 1,571 times
at `predicate_idx`). The layer therefore mixes two conventions. Dropping `AUX`
removes a third of the frames and a third of SRL compute. AUX frames are not
empty (1.91 args mean), only inconsistent.

### 5.3 Window repack — *recommended against*

Packing by subwords would recover the masked 3.43%. It needs a tokenizer inside
a builder that §4.1 keeps deliberately dependency-free, and re-packing
invalidates every window-keyed annotation. Not worth it for 3.43%; the masking
is correct either way.

### 5.4 Published HuggingFace artifacts — *outward-facing, unasked*

- `label_vocabs.json` still present and still mislabelling CLS for downloaders.
- Four model cards relicensed Apache-2.0 locally; HF still shows CC-BY-SA.
- NER and DEP documented as gold-trained when trained on silver.

### 5.5 Git

The merge is **already on the remote** — `origin/main` is at `f631d82`, pushed
externally. Only one commit is unpushed locally (`e321797`, the
`label_vocabs.json` removal).

Two consequences worth recording. The v6 branch history was rewritten to purge
a 116 MB corpus file, which lost its shared ancestry with `main`; the merge was
therefore completed with `--allow-unrelated-histories` and 13 conflicts
resolved by file ownership rather than three-way merge. That rewritten history
is now **public**. For anyone holding the previous `main` this is still a plain
fast-forward — the old tip `2b9a179` is an ancestor of the pushed merge — so no
re-clone is needed. A durable backup of all refs is at `~/kniv-git-backups/`;
an earlier backup was lost to session-scoped temp storage.

---

## 6. What is built, and what is not

**Built:** window builder with per-domain adapters and per-source token
budgets · annotation drivers for kniv-v5, Stanza, LingMess and the LLM layers ·
entity assembly · ATLOP relation runner with checkpoint/resume · assembly into
partitioned parquet · 20 quality gates · generated data card · SGD CLS
benchmark · sampled audit · training design. ~6,600 lines of `v6/` code,
~2,400 lines of `v6/` documentation.

**Not built — `v6/train/` does not exist:**

1. `data.py` — parquet → tensors: subword alignment, the per-layer mask
   contract, the `encoder_word_limit` cut. Needs a round-trip test that
   reconstructs labels out of a batch and compares them to the corpus; this is
   where silent corruption would live.
2. `model.py` — encoder + 13 heads, including single-pass SRL and the
   NER→coref→clusters→relations chain.
3. `loop.py` — per-head loss weighting, eval, and the per-head loss readout
   (a head flat from step zero is invisible in a total).
4. Lemma edit-script and morph FEATS vocabularies, derived from the corpus.
5. HF private-repo checkpointing: `latest.pt`, `best.pt`, `best.metric`.
6. Colab A100 orchestration, in-kernel.
7. **An A100 40GB memory fit check** — batch size, gradient checkpointing and
   bf16 settings are assumed, not measured.

`v6/experiments/atlop_runstate.py` is reusable for 5 — resume, heartbeat,
atomic writes, SIGTERM→checkpoint are written and proven — but is ATLOP-shaped
and needs generalising.

---

## 7. Operational lessons already paid for

These are in `TRAINING.md` §7 because each cost real time:

- **Run in-kernel.** A detached process leaves the Colab kernel IDLE and the VM
  is reclaimed within the hour regardless of keep-alive. **Five VMs** lost.
- **Checkpoint to HF, not Drive.** `colab drivemount` needs a human at the
  terminal, so it cannot be part of an unattended restart.
- **Never overwrite a checkpoint without comparing the score it was selected
  on.** A puller once replaced a 0.7161 checkpoint with a 0.6278 one.
- **A stale heartbeat means dead, not slow** (>15 min).
- **Instrument memory.** ATLOP stalled for eleven hours holding ~47 GB while
  reporting 108 MB RSS, because RSS does not count swapped pages. Fixed with
  per-batch result writes, bounded accumulation and an MPS cache flush; a
  2-hour run then completed with peak RSS flat at 8–11 GB.
- **Verify, do not assume, that a command succeeded.** Several wrong
  conclusions this cycle came from commands whose stderr was discarded.

---

## 8. Notable defects found and fixed this cycle

Ordered by what they would have cost had they shipped.

| defect | would have shipped as | found by |
|---|---|---|
| LLM layers never assembled | corpus with **no CLS column at all** | reading `stage_assemble` |
| Coref batch misalignment | one window's clusters attached to another, **silently** | a `KeyError` on the last item |
| Coref type-mixed entities | 4.71% of entities merging `('2010' TIME, 'Toyota' ORG)`; **9.51% of relation triples** built on them | inspecting a relation's vertexSet |
| 33% of windows overflow the encoder | supervising positions the encoder never produced | gate 3, on its first run |
| Content-blind cache keys | 29.7% of cached entries attached to different text | measuring after a cleaning change |
| One 727-window document broke the splits | business at 12.9% test instead of 5% | gate 16 after adding business |
| Cross-split duplicate content | 64 windows of identical text in train **and** test | gate 5 trend, then a corpus-wide scan |
| Token cap collected the wrong corpus | business with **zero SEC filings** | checking source mix after windowing |
| SEC filings unattributable | `filings-0`, CIK dropped, CC-BY unsatisfiable | gate 19 before annotating |
| OntoNotes tag map off by one | every `I-ORDINAL` relabelled, every `I-LANGUAGE` → `O` | verifying against the dataset's own `label2id` |
| ATLOP attention under transformers 5 | plausible relations pooled from the **wrong tensor** | an `IndexError` that was luck |

Two general shapes recur and are worth stating plainly:

**Libraries signal problems by omission.** fastcoref drops an over-long
document and returns a short list; it returns `None` for an unresolvable
mention. Neither raises. Both produce silently wrong data unless the count is
checked.

**Shape gates cannot catch wrong-but-plausible values.** Scanning all 7,423
coref entries for out-of-range mentions found zero *while misaligned data was
present*. That load is carried by content-addressed cache keys and by refusing
a batch whose result count is short — not by gates 6 and 10, which are
annotated in `gates.py` as structurally unable to see it.

---

## 9. Next steps

1. **Decide §5.1** (Commissive boundary) — it changes the labels, so it should
   land before training rather than after.
2. **Decide §5.2** (SRL over AUX) — a ten-minute edit plus a re-assemble; the
   cache keeps every annotation.
3. **Write `v6/train/data.py`** with its round-trip verification.
4. **Measure the A100 fit** before committing to a long run.
5. **Short smoke run** — a few hundred steps, checkpoint, kill, resume — before
   any long run. Every Colab failure in this project so far has been
   operational, not mathematical.
6. Then the full run, with `status.json` polled from here.

## 10. What this corpus is not for

Every layer is model-annotated, and the NER and DEP teachers were themselves
trained on silver. The `test` split exists for corpus QA and ablation only:
training and testing on our own annotations measures agreement with our
annotators, not accuracy. **Headline numbers must come from public gold.**
No such evaluation has been run against this corpus yet.
