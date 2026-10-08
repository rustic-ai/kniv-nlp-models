# v6 training — design and operations

One model, one forward pass, every layer. The corpus exists so that a single
encoder is supervised on all of it at once rather than one model per task.

Target hardware is a Colab **A100 40GB** driven by the `colab` CLI.

## 1. Why one model, and what that forces

v5 trained per-task heads on per-task corpora and ran them as a cascade. v6
trains every head on the *same window*, so the encoder sees one text and
produces every layer. Two consequences the design has to honour:

**Relations are a head, not a pipeline.** ATLOP annotated the corpus; it is
not the deliverable. Relations are defined over entity clusters, and clusters
come from NER plus coreference, so the model needs **its own** NER, coref and
clustering to emit a graph at inference:

```
tokens → encoder → NER spans ─┐
                              ├→ entity clusters → entity pairs → relation
                   coref ─────┘
```

That chain is the reason coref is a head rather than a preprocessing step.
Training it as a pipeline over gold clusters and running it over predicted
clusters is the classic exposure mismatch; §4 says how that is handled.

**Every head trains on masked labels.** A layer absent for a row, or a
sentence whose parse is not a tree, or a window tail past the encoder, must
contribute no gradient. The corpus already records all three.

## 2. Heads

| head | shape | unit | loss |
|---|---|---|---|
| POS | `[T, 17]` | token | CE |
| morph | `[T, V_feats]` | token | CE |
| lemma | `[T, V_rules]` | token | CE over edit scripts (§3) |
| NER | `[T, 37]` | token | CE, Viterbi at inference |
| DEP arc | `[T, T]` | token→token | CE over candidate heads, per sentence |
| DEP label | `[T, T, 53]` | arc | CE on the gold arc |
| SRL predicate | `[T, 2]` | token | BCE |
| SRL role | `[T, T, 42]` | predicate→token | CE, **all predicates in one pass** |
| CLS | `[S, 6]` | sentence | BCE, multi-label, empty set legal |
| sentiment | `[S, 3]` | sentence | CE |
| coref mention | `[M, 1]` | span | BCE |
| coref antecedent | `[M, M+1]` | span→span | marginal log-likelihood |
| relation | `[E, E, 97]` | entity pair | ATLOP adaptive-threshold loss |

`T` tokens, `S` sentences, `M` candidate mentions, `E` entity clusters.

Scalar mixes stay per head, as in v5: a layer that wants surface form and a
layer that wants semantics should not be forced to share one encoder layer.

## 3. Lemma as classification, not generation

A lemma head cannot emit strings. The standard trick is to predict an **edit
script** — strip `k` characters from the end, append `s`, optionally lowercase
— and apply it to the token. The script vocabulary is derived from the corpus
and covers the overwhelming majority of tokens; anything uncovered is masked
out of training rather than approximated, and falls back to the token itself.

Vocabularies for morph and lemma come from the **corpus**, and the canonical
label sets for the existing heads come from `models/label_maps.json`, whose
list position *is* the class index.

Not from `label_vocabs.json`. That filename belongs to the dep2label
generation — a linear DEP head over ~1,411 `{offset}@{deprel}@{head_UPOS}`
composites and a 9-unit CLS head — and the v5 student it was found beside is
biaffine over 53 deprels with 8 CLS units. It was not a stale copy of our
labels but another architecture's, which is why it was deleted rather than
corrected; `scripts/verify_label_maps.py` now fails if it reappears
(DATASET_SPEC §8).

## 4. The exposure problem for relations

Relations are supervised over clusters that ATLOP saw, built from
**annotated** NER and coref. At inference the model builds clusters from its
**own** predictions. Training only on gold clusters produces a relation head
that has never seen a wrong entity.

The plan is staged, and honest about which stage is cheap:

1. **Teacher forcing first.** Train the relation head on corpus clusters. This
   is the cheap, stable stage and it is where most of the signal is.
2. **Then predicted clusters** for a final fraction of training: build clusters
   from the model's own NER and coref, align them to corpus clusters by span
   overlap, and supervise the pairs that align. Pairs with no gold counterpart
   are supervised as `Na` only when both entities align, never when alignment
   fails — an unaligned pair is unknown, not negative.

Stage 2 is the part most likely to be cut for time. If it is cut, the model is
a relation extractor over gold entities, and that must be stated rather than
implied.

## 5. Masking contract

Nothing is padded to fit. Each is already in the corpus:

| signal | source | effect |
|---|---|---|
| layer absent | `loss_mask.<layer> == true` | whole layer, this row |
| sentence not a tree | `loss_mask.dep_tokens[i]` | DEP, those tokens |
| NER BIO ill-formed | `loss_mask.ner_tokens[i]` | NER, those tokens |
| CLS/sentiment missing | `loss_mask.<layer>_sentences[j]` | that sentence |
| past the encoder | `encoder_word_limit` | **all** heads, tokens `>= limit` |
| non-first subword | tokenizer alignment | all token heads |

`encoder_word_limit` is not optional. 36.79% of windows carry a tail past 512
subwords — 3.43% of all positions — and supervising them trains against
representations the encoder never produced.

## 6. Loss weighting

Start uniform, then normalise per head by its own early-training loss scale so
one head does not dominate by having more classes or more tokens. Record the
per-head loss in `status.json` every interval: a head whose loss is flat from
step zero is not learning, and that is invisible in a single total.

## 7. Operations

Colab is hostile: the VM vanishes without warning, there is a wall-clock cap,
and the filesystem dies with the instance.

**Run in-kernel.** A detached process leaves the kernel IDLE and Colab
reclaims the VM within the hour regardless of keep-alive. That cost five VMs
during ATLOP training. `colab exec` runs in the kernel, which is correct;
never background the trainer inside the VM.

**Checkpoints go to a private HuggingFace repo**, not Drive: `colab drivemount`
needs a human at the terminal and so cannot be part of an unattended restart.

| artifact | when | holds |
|---|---|---|
| `latest.pt` | every `save_every` steps, and each epoch | model, optimizer, scheduler, scaler, step, epoch, best, RNG state |
| `best.pt` | when dev improves | weights only |
| `best.metric` | beside `best.pt` | the score it was selected on |
| `status.json` | every interval | step, loss per head, rate, ETA, best, heartbeat |
| `events.jsonl` | append | checkpoints, evals, signals, errors |

`best.metric` exists because a puller once overwrote a 0.7161 checkpoint with
a 0.6278 one. Never overwrite a checkpoint without comparing the score it was
selected on.

**Resume reads `latest.pt` and nothing else.** RNG state is included so a
resumed run is not a different sample order.

**SIGTERM and SIGINT checkpoint and exit**, so a preemption or a UI stop costs
at most `save_every` steps.

**A heartbeat older than 15 minutes means dead**, not slow. Downloads land on
a temp path and are renamed, so a half-pulled checkpoint is never mistaken for
a whole one.

**Memory is instrumented, not assumed.** ATLOP stalled for eleven hours
holding 47 GB while reporting 108 MB RSS, because RSS does not count swapped
pages. `status.json` records allocated and reserved GPU memory and host RSS
every interval.
