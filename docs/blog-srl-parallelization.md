# The Hidden Cost of Semantic Role Labeling — and How to Parallelize It Without Retraining

*Up to 3.3× SRL throughput on a cascade NLP model, measured on 9,000+ (sentence, verb) pairs from LongMemEval, across three model sizes. No model changes. No accuracy loss. Just the right batch shape.*

---

## The shape of the problem

There's a quiet asymmetry in multi-task NLP models that nobody warns you about until you profile them.

Run [kniv-cascade](https://huggingface.co/dragonscale-ai) on a sentence and a single forward pass gives you part-of-speech tags, named entities, dependency arcs, dialog-act class, and semantic role labels for one predicate. POS, NER, DEP, CLS — those four tasks each produce per-token output for the whole sentence in O(1) forwards. SRL is different. SRL produces argument labels *for one verb at a time*. A sentence with five verbs needs five forward passes.

This isn't a kniv quirk. It's the dominant SRL architecture since [He et al. (2017)](https://arxiv.org/abs/1707.06090): a predicate-indicator embedding tells the encoder which token is the verb, and the encoder's representation is shaped by that signal. Change the predicate, change the encoder's view of the sentence, change the argument labels. AllenNLP's BERT-SRL did it. The current SOTA span-based SRL models still do it. The signal has to enter at the embedding layer because that's where it has time to influence attention patterns through all the encoder's layers — the very mechanism that makes SRL good is also what forbids reusing one encoder pass across multiple verbs.

So you do *k* encoder passes for *k* verbs. On the average LongMemEval sentence, *k* is 4.7. The encoder is what dominates the cascade's compute. The math is unforgiving.

We did the measurement properly, and the answer turned out to be more nuanced than the math suggests.

---

## A short detour through the cascade

To make sense of what we're parallelizing, here's the relevant part of [`student_loader.py`](https://github.com/rustic-ai/kniv-nlp-models/blob/main/models/student_loader.py):

```python
def forward(self, input_ids, attention_mask, predicate_idx, ...):
    B, S = input_ids.size()
    emb = self.encoder.embeddings(input_ids)
    # The predicate-indicator embedding — a learned 2D vector — is added
    # to the token embeddings at exactly one position per row.
    indicator = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)
    indicator.scatter_(1, predicate_idx.unsqueeze(1), 1)
    emb = emb + self.pred_embedding(indicator)
    enc_out = self.encoder.encoder(emb, attention_mask, output_hidden_states=True)
    ...
```

The model is already batched: `input_ids` is `(B, S)` and `predicate_idx` is `(B,)`. Each row gets its own predicate. There's nothing stopping you from putting the same sentence in multiple rows with different `predicate_idx` values — but the question is whether you should.

There are two ways to get *k* predicate labels for one sentence:

**Option 1 — Per-sentence fan-out.** For a sentence with *k* verbs, replicate the input *k* times along the batch dimension, set a different `predicate_idx` per row, run one forward pass with batch size *k*. The encoder fires *k* times in parallel inside one call.

**Option 2 — Cross-sentence packing.** Flatten the entire corpus into (sentence, verb) pairs and batch them at a fixed batch size *B*. Each batch contains different sentences with different predicates. The encoder fires *B* times per call, regardless of how the verbs are distributed.

Both produce identical outputs. Both do the same number of encoder forward passes total. The only thing that changes is what each batch looks like — and that turns out to matter a lot.

---

## What LongMemEval actually looks like

We benchmarked on [LongMemEval](https://github.com/xiaowu0162/LongMemEval), the long-term memory benchmark for conversational agents. 500 questions, each anchored to ~50 multi-turn chat sessions for a total of about 200K turns across 19K unique sessions. After regex sentence splitting we get **1.21M sentences**. Roughly 95% are under 128 tokens — the kniv training length — so we can run the full cascade without out-of-distribution behavior from longer inputs.

A few stats we wrote down before benchmarking, because they shape the answer:

| metric | value |
|---|---|
| unique sentences | 1,213,737 |
| avg tokens / sentence (capped at 128) | 30 |
| truncation rate @ 128 | 2.8% |
| sentences with ≥ 1 verb | ~95% |
| **avg verbs / sentence** | **4.68** |
| verb-count distribution | long right tail; many k=1 or k=2 |

Splitting on sentence boundaries also flattens the CLS distribution out — at the turn level, 90% of LongMemEval gets classified as `inform` or `question` because the long ShareGPT-style prompts drown out everything else. At the sentence level, `social` (greetings, closings, acknowledgments) jumps to 21%, and `confirm` and `reject` finally appear at all. The cascade's eight-way classifier was always there; the unit of analysis just had to be small enough for it to show its teeth.

That long right tail on verbs-per-sentence is going to matter shortly.

---

## Option 1: per-sentence fan-out

The wrapper is five lines:

```python
def srl_per_sentence(model, input_ids, attention_mask, verb_positions):
    """input_ids: (S,)  verb_positions: list[int] of k token indices."""
    k = len(verb_positions)
    ids  = input_ids.unsqueeze(0).expand(k, -1)
    mask = attention_mask.unsqueeze(0).expand(k, -1)
    pidx = torch.tensor(verb_positions, dtype=torch.long, device=ids.device)
    *_, srl_logits, _ = model(ids, mask, pidx)
    return srl_logits  # (k, S, N_SRL)
```

The expand is a no-copy view, so memory-wise it's cheap. The encoder runs once per row, and rows fan out in parallel as far as the device's compute allows. For a sentence with *k* verbs, one call replaces *k* sequential calls. Pure win, right?

Almost. The batch size is *k*, which varies per sentence. About a fifth of LongMemEval sentences have *k=1*. Another fifth have *k=2*. These small batches are pure overhead — one kernel launch per matmul, dispatch latency, host-side Python — with no parallelism payoff. On a GPU with hundreds of compute units, a batch of 1 leaves 99% of the silicon idle. The 1.68× we're going to measure is almost entirely about the bottom of this distribution.

---

## Option 2: cross-sentence packing

Same total work, different batch shape. We pre-pass the corpus once to find verb positions (the cascade gives us POS in the same forward we'd run anyway), then build the SRL batch like this:

```python
# pairs: list of (input_ids, attention_mask, verb_token_idx) — flattened across all sentences
pairs.sort(key=lambda p: p[0].size(0))   # length-bucket to reduce pad waste

for i in range(0, len(pairs), B):
    chunk = pairs[i:i + B]
    L = max(p[0].size(0) for p in chunk)
    ids_b  = pad_to(L, [p[0] for p in chunk])
    mask_b = pad_to(L, [p[1] for p in chunk])
    pidx_b = torch.tensor([p[2] for p in chunk])
    *_, srl_logits, _ = model(ids_b, mask_b, pidx_b)
```

Every batch is *B* rows. Every batch is the same shape. BLAS kernels — both CPU MKL and GPU cuBLAS — are tuned for steady tensor shapes, and a 32×128 forward pass through DeBERTa-v3-base is one of the well-trodden ones. The variance is gone.

The pre-pass is the kind of cost that disappears in any real pipeline: kniv-cascade returns POS, NER, DEP, SRL, and CLS in one forward pass, so harvesting verb positions is free if you're going to run the cascade for any of those other tasks anyway. We time only the SRL phase.

---

## What the numbers say

Test setup: model is [kniv-deberta-nlp-base-en-base](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-base) (the 86M-param `base` student — there's also `xsmall`, `small`, and `large` in the same family). 2,000 LongMemEval sentences, 1,907 contained at least one verb, total **9,195 (sentence, verb) pairs**, mean *k = 4.82*. CPU: 8-thread x86 with MKL. GPU: a single 8 GB consumer card, batch=32 chosen after a small sweep. `torch.cuda.synchronize()` brackets every timed region. The smoke-test ran on CPU first because the GPU was occupied; we re-ran on GPU once it freed.

| device | Option 1 | Option 2 (B=32) | Option 2 advantage |
|---|---|---|---|
| CPU (8-thread x86, MKL) | 25.7 pairs/s | **32.6** pairs/s | **1.27×** |
| GPU (8 GB consumer) | 328 pairs/s | **551** pairs/s | **1.68×** |

The GPU advantage is bigger than the CPU advantage because GPUs hate small batches more than CPUs do. A batch of *k=1* on CPU is still a matmul with a non-trivial work share; on GPU it's a launch overhead with a rounding-error of actual compute. Option 2's fixed *B=32* keeps the GPU fed; Option 1's varying *k* leaves it half-empty for a fifth of the work.

Notice what doesn't show up here: any accuracy difference. Both options run the same model with the same inputs and produce the same logits. We're not trading quality for speed; we're trading nothing for speed.

### The speedup grows as the model shrinks

We re-ran the same benchmark on the smaller members of the kniv family. The pattern is sharper than we expected on GPU, and present-but-muted on CPU.

**GPU (8 GB consumer card, batch=32):**

| model | params | Opt1 pairs/s | Opt2 pairs/s | Opt2 advantage |
|---|---|---|---|---|
| [`kniv-deberta-nlp-base-en-xsmall`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall) | 22M | 435 | **1,446** | **3.32×** |
| [`kniv-deberta-nlp-base-en-small`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-small) | 44M | 544 | 977 | 1.79× |
| [`kniv-deberta-nlp-base-en-base`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-base) | 86M | 328 | 551 | 1.68× |

**CPU (8-thread x86, MKL, batch=32):**

| model | params | Opt1 pairs/s | Opt2 pairs/s | Opt2 advantage |
|---|---|---|---|---|
| xsmall | 22M | 64.9 | **104.9** | **1.62×** |
| small | 44M | 44.8 | 63.5 | 1.42× |
| base | 86M | 25.7 | 32.6 | 1.27× |

Three surprises worth sitting with:

**The Option-2 advantage gets bigger as the model gets smaller**, on both devices — 1.68× → 1.79× → 3.32× on GPU, 1.27× → 1.42× → 1.62× on CPU. The mechanism is the same: per-call overhead (Python dispatch, kernel launch, host-side tensor construction) is a fixed cost per forward, and as the encoder shrinks that cost becomes a larger fraction of total time. Option 1's variable-shape batches pay that cost in proportion to how many small-k sentences they hit; Option 2 pays it once per fixed batch of 32. On a small model, the fixed cost dominates and Option 2's amortization advantage compounds.

**GPU amplifies the effect, CPU mutes it.** The same architectural insight produces a 3.32× win on GPU-xsmall but only a 1.62× win on CPU-xsmall. GPUs hate small batches more than CPUs do — kernel launches and dispatch round-trips are pure overhead with no "tiny matmul" floor like there is on CPU. If you're moving a CPU pipeline to GPU, expect the parallelization payoff to roughly *double* on top of the raw hardware speedup.

**Option 1 on xsmall-GPU is slower than Option 1 on small-GPU** (435 < 544 pairs/s). The xsmall encoder is *half* the size of small but ends up slower under per-sentence fan-out. The variable-batch overhead exceeds whatever compute savings the smaller model gave us. Per-sentence fan-out is actively broken on small models in a way it isn't on base. CPU doesn't show this pathology — Option 1 throughput on CPU monotonically improves as models shrink — which is another tell that small batches are a *GPU-specific* tax.

The practical consequences cut across model selection. On GPU, **getting the batch shape right beats sizing down**: Option 2 on xsmall (1,446 pairs/s) is 2.7× faster than Option 1 on small (544 pairs/s) and 4.4× faster than Option 1 on base (328 pairs/s). On CPU, the speedups from batching and sizing-down are roughly the same magnitude (~1.5×), so both levers compound and neither dominates. And across devices: even the best-batched CPU run (xsmall, 105 pairs/s) is about 5× *slower* than the worst-batched GPU run (base, 328 pairs/s). For SRL throughput specifically, hardware beats every other lever — even the worst-batched GPU run beats the best-batched CPU run by a wide margin.

---

## Why the obvious tuning knob doesn't help

If steady batches are good, bigger batches should be better. They aren't.

| Option 2 batch size | pairs/s | speedup |
|---|---|---|
| **32** | **551** | 1.68× |
| 64 | 537 | 1.65× |
| 128 | 505 | 1.54× |

This took us a beat to understand. Mixed-length padding is the culprit. Even with bucketing, a batch of 128 sentences spans wider lengths than a batch of 32, and the longest sentence in the batch dictates compute for all 128 rows. That's pad waste — kernels you fire that produce results you'll mask out. At *B=32*, bucketing keeps within-batch length variance tight; at *B=128*, variance widens enough that the savings from extra batch parallelism are eaten by the extra padded tokens.

There's also a Python-side cost: the padding loop in Option 2 is `O(B)` per batch step. At *B=32* it's ~600 padding ops per second; at *B=128* it's ~150, but each is doing four times as much work. Small but measurable on the host side.

The lesson is that "bigger batch = faster" stops being true once length variance exists. The kniv encoder is well-tuned for shape-(B, 128) forwards; ask for shape-(B, L_max) where L_max wanders, and you give back some of what bucketing bought.

---

## Reproducing the numbers

The script we used lives at [`scripts/bench_srl_fanout.py`](https://github.com/rustic-ai/kniv-nlp-models/blob/main/scripts/bench_srl_fanout.py):

```bash
# 1. Get LongMemEval (or use any conversational corpus with `haystack_sessions`)
hf download xiaowu0162/LongMemEval --repo-type dataset \
  --local-dir data/longmemeval/

# 2. Run the benchmark
uv run python scripts/bench_srl_fanout.py \
    --input data/longmemeval/longmemeval_s.json \
    --model models/kniv-deberta-nlp-base-en-base \
    --n-sentences 2000 \
    --batch-size 32

# Output:
#   data/bench_srl_fanout.json   # full results JSON
```

Two phases happen inside. Phase A runs a single batched POS pre-pass over all sentences and harvests verb token positions — this is shared cost, not counted in either option's timing. Phase B times each strategy separately, with `torch.cuda.synchronize()` bracketing each region.

If you want to use Option 2 in your own inference code, the wrapper is small:

```python
def srl_corpus(model, tokenizer, sentences, verb_positions_per_sentence,
               device, batch_size=32):
    """sentences: list[str], verb_positions_per_sentence: list[list[int]]
       returns: dict (sentence_idx, verb_idx) -> srl_logits (S, N_SRL)."""
    # Build (sent_idx, ids, mask, verb_idx) tuples for every (sentence, verb) pair.
    pairs = []
    for s_idx, (sent, verbs) in enumerate(zip(sentences, verb_positions_per_sentence)):
        enc = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        for v in verbs:
            pairs.append((s_idx, enc["input_ids"][0], enc["attention_mask"][0], v))
    pairs.sort(key=lambda p: p[1].size(0))  # length bucket

    out = {}
    pad_id = tokenizer.pad_token_id or 0
    with torch.inference_mode():
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            B = len(chunk); L = max(p[1].size(0) for p in chunk)
            ids  = torch.full((B, L), pad_id, dtype=torch.long, device=device)
            mask = torch.zeros((B, L), dtype=torch.long, device=device)
            pidx = torch.zeros(B,      dtype=torch.long, device=device)
            for j, (s_idx, sid, sm, v) in enumerate(chunk):
                ids[j, :sid.size(0)]  = sid
                mask[j, :sm.size(0)] = sm
                pidx[j] = v
            *_, srl_logits, _ = model(ids, mask, pidx)
            for j, (s_idx, _, _, v) in enumerate(chunk):
                out[(s_idx, v)] = srl_logits[j].cpu()
    return out
```

A real production pipeline would stream results back instead of buffering them in a dict, but the idea is the same: build the unit of work as a (sentence, verb) pair, not a sentence, and let batching do its job at the corpus level.

---

## What we didn't try, and why it would help more

Option 1 and Option 2 are both *batch-shape* fixes. Neither one removes the fundamental cost: *k* encoder forwards per *k*-verb sentence. To break that asymmetry you have to change the model.

There's an architectural fix — call it Option 3 — where you move the predicate-indicator signal *out* of the embedding layer and into the SRL head. The forward becomes:

1. Run the encoder once. Get hidden states `h ∈ R^(B,S,H)`.
2. For each verb position `v` in each sentence, gather `h_v` and run the SRL head with `h` and `h_v` as inputs.

Step 2 is a fan-out, but it's at the head level, not the encoder level. The SRL head is a couple of LSTMs and linear layers — perhaps 5% of the encoder's FLOPs. So *k* verbs in one sentence cost `1 × encoder + k × head` instead of `k × (encoder + head)`. For *k=4.7*, that's roughly a 4× reduction on top of whatever batching wins you've already taken.

The catch is that this needs retraining. Predicate-at-the-embedding lets the verb signal influence attention from layer 0 onwards, which is part of why these models are good at long-range argument identification. Predicate-at-the-head loses that — the encoder produces a predicate-agnostic representation and the head has to do all the conditional reasoning post-hoc. Published comparisons typically show a 1–3 F1 drop on PropBank-style benchmarks for this kind of design. You can claw most of it back with a [FiLM layer](https://arxiv.org/abs/1709.07871) (predicate-conditional gamma/beta modulating the encoder output) or a cross-attention SRL head where the predicate query attends over the sentence; both of these have been explored in the literature, neither has supplanted the He-et-al design as the default. The training cost is real and the F1 cost is real, but a 4× speedup on top of 1.7× is a 6.8× total — worth it for a system where SRL is the critical-path bottleneck.

We're sitting on the bench-shape win for now because it was a half-day's work and we measured an honest 1.68×. Option 3 is on the roadmap once we have a clear use case where the remaining cost matters.

---

## What this means for production

Take a kniv-cascade deployment over the full LongMemEval corpus — 1.21M sentences, mean *k = 4.7* — and the SRL fan-out becomes about **5.7 million (sentence, verb) pairs**. The model choice and the batch strategy together span an order of magnitude in wall-clock time:

| | base | small | xsmall |
|---|---|---|---|
| Option 1 (per-sentence) | 4.8 h | 2.9 h | 3.6 h |
| **Option 2 (cross-sentence)** | **2.9 h** | **1.6 h** | **1.1 h** |

On CPU under Option 2 with the base model, the same workload is about 48 hours.

The lever you should pull depends entirely on where SRL fits in your pipeline. If you're labeling a one-time ingest of historical conversation data, Option 2 on a single GPU does it overnight and you move on. If you're running SRL inline on a streaming pipeline at high QPS, the 1.7× still matters but the architectural fix is where the real headroom lives. If you're on CPU because no GPU is available, the 1.27× is a free improvement but the math says the answer is "get to a GPU" — *17×* speedup from device alone dwarfs anything you can do with batch shape.

The most generalizable lesson, if we had to pick one, is this: when a model has built-in per-instance work (predicates, query types, span proposals — anything that says "we need to re-run for each of these"), the unit of batching should be the per-instance work, not the input. Building (input, instance) pairs and batching at corpus level recovers steady tensor shapes, which recovers BLAS throughput, which recovers GPU utilization. The cascade architecture wasn't designed around this, but it doesn't need to be — the right wrapper around the existing forward gets most of the way there.

The model is on Hugging Face. The benchmark script reproduces these numbers in a single command. If you find a regime where Option 1 wins — large *k*, short sentences, very fast device — we'd love to see the numbers.

---

## Links

- Model family (pick a size to match your latency budget):
  - [`kniv-deberta-nlp-base-en-xsmall`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall) — ~22M params, the fastest student
  - [`kniv-deberta-nlp-base-en-small`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-small) — ~44M params
  - [`kniv-deberta-nlp-base-en-base`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-base) — ~86M params (benchmarked here)
  - [`kniv-deberta-nlp-base-en-large`](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-large) — ~304M params, highest quality
  - [`kniv-deberta-v3-large-nlp-en`](https://huggingface.co/dragonscale-ai/kniv-deberta-v3-large-nlp-en) — the teacher used for distillation
- Training corpus: [`dragonscale-ai/kniv-corpus-en`](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en)
- Code: [rustic-ai/kniv-nlp-models](https://github.com/rustic-ai/kniv-nlp-models)
- Benchmark script: [`scripts/bench_srl_fanout.py`](https://github.com/rustic-ai/kniv-nlp-models/blob/main/scripts/bench_srl_fanout.py)
- Dataset: [LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al. 2024)
- Background reading: [He et al. 2017 — Deep Semantic Role Labeling](https://arxiv.org/abs/1707.06090) (the predicate-indicator design)
- FiLM (for Option 3): [Perez et al. 2017](https://arxiv.org/abs/1709.07871)
