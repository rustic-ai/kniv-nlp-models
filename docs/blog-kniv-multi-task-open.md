# Building a Multi-Task English NLP Cascade in the Open

There's an awkward gap between what NLP research papers describe and what ships on Hugging Face. Multi-task models are everywhere in the literature — joint POS+chunking+parsing models go back to Hashimoto et al. 2017, multi-task fine-tuning is at the heart of MT-DNN (Liu et al. 2019) and MUPPET (Aghajanyan et al. 2021), the BERT-rediscovers-classical-NLP probing work (Tenney et al. 2019; Jawahar et al. 2019) practically begs for a single encoder feeding multiple heads. In production, the same five tasks get five separate encoders. Stanza chains them. spaCy shares an encoder for POS/DEP/NER but stops there. AllenNLP — which produced the canonical SRL and biaffine-parser models — was archived in December 2022 without ever combining them. Trankit shares an encoder across POS/DEP/NER for many languages but doesn't cover SRL or dialog acts. Search the Hub for a multi-task English checkpoint that handles POS + NER + DEP + SRL + dialog-act classification in one forward pass and you find pipelines, not models.

This post is about a model that does — kniv-cascade — and the choices that produced it. Everything is on Hugging Face: the [training corpus](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en), the [teacher and three distilled students](https://huggingface.co/dragonscale-ai), and the architecture, training recipes, and benchmark scripts in [rustic-ai/kniv-nlp-models](https://github.com/rustic-ai/kniv-nlp-models). Two commands reproduce the whole pipeline end-to-end.

We are writing this less to announce the model and more because the path to it had four or five dead ends that we couldn't find written down anywhere, and the things that actually worked are the kind of engineering details a Hub model card has no place to put.

## The architecture took several attempts

The first cut used DistilRoBERTa with three heads — POS, NER, DEP — trained sequentially. Quality on individual tasks looked fine. Then we added more heads and the model started fighting itself. Training NER pulled the encoder away from what POS had learned. Adding SRL on top broke DEP.

We tried joint multi-task training with a shared encoder and equal loss weights. POS dominated — the easiest task converged fastest and its gradient swamped the others. We tried task-grouped curricula. We tried freezing the encoder after the first task and training subsequent heads in isolation on top of frozen representations. Quality cratered everywhere except whichever head had owned the encoder.

What converged was layered training. The intuition borrows from probing work showing that BERT's layers encode a linguistic hierarchy — surface features at the bottom, syntax in the middle, semantics at the top (Tenney et al. 2019; Jawahar et al. 2019). Rather than ignore that, train *with* it.

Five phases, each shaping a different layer range, with reduced learning rates on bands a previous phase already touched. POS first across all layers as a syntactic warm-up. NER on layers 5–12 with the overlap zone at half the LR. DEP on 12–18. SRL across all layers at an ultra-low rate to adapt the now-shaped foundation to verb-conditioned tagging. CLS last, on a frozen encoder, learning to read what the encoder already encodes. Aggregate gain over the previous top-down approach was 13.3 F1 points across the five heads — the biggest single methodology improvement in the project.

A small detail with outsized effect: each task head learns its own ScalarMix over all 25 encoder layers (embedding + 24 transformer layers), and those learned mixtures converge to different layer ranges per task — POS to the bottom band, NER to the middle, DEP higher, SRL across the top, CLS spread across all of them. The model rediscovers the linguistic hierarchy on its own, given the chance.

## DeBERTa-v3, specifically

The encoder is DeBERTa-v3 (He et al. 2021). The reasons are mechanical.

DeBERTa's disentangled attention (He et al. 2020) represents each token with two vectors — one for content, one for relative position — and computes attention as a sum of four matrices: content-to-content, content-to-position, position-to-content, position-to-position. Standard BERT and RoBERTa fold absolute position into the input embedding from layer 0, entangling content and position throughout the stack. DeBERTa keeps them separable. For an architecture where one head wants positional structure (a dependency parser building arc scores between tokens) and another wants content semantics (an SRL head deciding whether a token is an agent or a patient), separable streams are a useful prior.

This intuition isn't directly proven in the literature for our exact setup — no paper compares DeBERTa vs RoBERTa as a shared encoder across these five heads — but layer-probing work (Fayyaz et al. 2021) finds DeBERTa pushes syntactic information earlier in the stack than BERT does and spreads semantics differently across upper layers, which is consistent with the mechanism.

DeBERTa-v3 specifically replaces masked language modeling with ELECTRA-style replaced token detection and adds gradient-disentangled embedding sharing. Both yield better representation quality per parameter than v1 or v2; on CoNLL-2003 NER, DeBERTa-v3-large reports 93.8 F1 versus RoBERTa-large at ~92.4. On SuperGLUE, the v3 family was the first to surpass the human baseline.

The teacher we trained is DeBERTa-v3-large — 304M backbone parameters, 434M total once you include the 128K SentencePiece vocab embedding. Overkill at the inference latency target, ideal as a teacher.

## The five heads

Four of the five heads form a cascade — POS feeds NER, POS and NER feed DEP, POS and DEP-relations feed SRL — by concatenating upstream softmax probabilities to each head's input. CLS sits outside the cascade, reading the encoder directly. The skip connection from POS to SRL is intentional: POS information shouldn't have to survive the journey through NER and DEP to reach a head that wants it.

One implementation detail with outsized effect: those upstream probabilities are `.detach()`-ed before being passed downstream. No gradient flows backward through the cascade. NER doesn't pull POS toward what NER wants. DEP doesn't drag POS or NER. Each head trains against its own labels with whatever signal the upstream heads happen to provide at that step. Joint multi-task training where every head's gradient hits every parameter was the source of the "POS dominates" failure mode in the earlier section; detaching the cascade is what made the multi-head design tractable.

Each head also has its own `ScalarMix` — a learned softmax-weighted average over all 25 encoder outputs (embedding + 24 transformer layers). The mixture trains end-to-end with no manual layer assignment. After training, the converged weights show a pattern that mirrors the probing literature: POS pulls hardest from layers 0–8, NER from 5–10, DEP from 12–18, SRL from the top of the stack, CLS spreads its weight across the whole encoder. The model finds the layer hierarchy on its own — we don't impose it, we just give each head the freedom to look where it wants.

The heads themselves, briefly:

- **POS** — `Linear(H → 17)` over ScalarMix. Token-level, argmax decoding. The simplest head.
- **NER** — ScalarMix → BiLSTM (hidden H/4 each direction) → projection back to H → concat with detached POS probabilities → MLP → 37 BIO tags. Constrained Viterbi at inference enforces valid `B-X` / `I-X` transitions.
- **DEP** — ScalarMix → concat with POS and NER probabilities → BiLSTM → biaffine arc head (Dozat & Manning 2017) producing an `[S, S]` score matrix per sentence, plus a biaffine label head producing `[S, S, 53]` for relation labels at each candidate edge.
- **SRL** — ScalarMix → predicate-aware features (the per-token hidden, the predicate token's hidden, their elementwise product, their absolute difference, plus detached POS and DEP-relation probabilities) → projection → BiLSTM → MLP → 42 BIO tags. Constrained Viterbi at inference.
- **CLS** — ScalarMix → attention pooling over valid tokens → MLP → 8 dialog-act labels. No cascade input; CLS reads the encoder directly.

One detail in the DEP→SRL hop is worth flagging because it changes what SRL actually sees. The DEP label biaffine produces an `[S, S, 53]` tensor — every `(token, candidate-head, relation)` triple. SRL doesn't need all of that; what it needs is the most likely relation for each token's most likely parent. So the cascade does an argmax-and-gather: for each token, pick the predicted head from the arc biaffine, gather the 53-dim label vector at that specific arc, and softmax. The signal SRL receives is "for each token, what relation is it likely to have to its predicted syntactic parent" — a per-token 53-class probability, much smaller than the full biaffine tensor and more useful than passing arc scores along would be.

The predicate handling is worth flagging separately. Most SRL systems mark the target verb with a special token at the input — a single embedding added or concatenated at the verb's position. That signal then has to travel up through 24 attention layers before reaching the SRL head. We instead add a learned `Embedding(2, H)` (initialized to zero) directly to the input embeddings, indexed by whether each token is the predicate. The encoder's first attention layer already sees the predicate-conditioned input; every subsequent layer's hidden state is implicitly predicate-aware. At inference, passing `predicate_idx=0` is benign — the zero-initialized embedding adds nothing — so the same encoder serves both the predicate-aware SRL pass and the predicate-agnostic POS/NER/DEP/CLS pass in a single forward call.

The full stack:

```
DeBERTa-v3-large encoder + Embedding(2, 1024) predicate signal
│
│  cascade chain (probabilities flow downstream, all .detach()-ed):
├─ ScalarMix(25) → Linear(17)                              → POS  [argmax]
├─ ScalarMix(25) → BiLSTM → +POS_p → MLP(37)               → NER  [Viterbi]
├─ ScalarMix(25) → +POS_p +NER_p → BiLSTM → Biaffine arc + label
│                                                          → DEP  [argmax + relation gather]
├─ ScalarMix(25) → +POS_p +DEP_rel_p → BiLSTM → MLP(42)    → SRL  [Viterbi]
│  (POS_p reaches SRL via skip connection, not through NER/DEP)
│
│  independent (no cascade input):
└─ ScalarMix(25) → AttentionPool → MLP(8)                  → CLS  [argmax]
```

443M total parameters in the teacher: 304M encoder backbone, 130M vocab embedding, 9.5M across all five heads. The heads are cheap; the encoder does the work.

## A note on the corpus

The training corpus is [kniv-corpus-en](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en). It exists because, when we sat down to assemble training data covering five tasks, every commonly cited multi-task corpus turned out to be encumbered. OntoNotes 5.0 sits behind an LDC license. CoNLL-2003 NER annotations are free; the underlying Reuters text isn't. PropBank's frame *definitions* are Apache-2.0, but the annotated training data points at the Wall Street Journal, which is LDC-licensed under separate terms. DailyDialog is CC BY-NC-SA. Switchboard Dialog Acts depends on Switchboard audio, which is LDC. There are permissive single-task corpora for each task — Universal Dependencies for POS and DEP, Schema-Guided Dialogue and MultiWOZ for dialog acts, Few-NERD and WNUT-17 for NER, QA-SRL Bank for an alternative SRL formulation — but no permissively licensed corpus combines them at training scale.

So we built one. UD provides the syntactic spine. SGD and MultiWOZ cover dialog acts. SRL and NER coverage are filled with silver labels generated through a teacher pipeline running over permissively-sourced text, with PropBank frames and the OntoNotes type taxonomy used as schema definitions rather than training data. We are not aware of another openly published English multi-task corpus that covers these five tasks with licenses you can ship.

It isn't the centerpiece of the project. It was the thing we had to build before the project could start.

## The first distillation attempt fell short

The teacher trained, the layered recipe worked, the five heads on a 434M-parameter encoder hit benchmark targets we could write down. None of that was deployable. DeBERTa-v3-large at 1.74 GB and ~50ms per inference call on commodity CPU is not what most people mean by "local NLP."

So we did the obvious thing: soft-label distillation in the spirit of Hinton et al. 2015. Run the teacher on a corpus, save its logits per head, train a smaller student to match those logits with KL divergence at temperature 3. The literature is dense — DistilBERT did this at the pretraining stage (Sanh et al. 2019), MT-DNN distillation did it across nine GLUE tasks (Liu et al. 2019).

The xsmall student lost three to four F1 points across SRL and NER versus the teacher. Acceptable for a research demo, not for the deployment target.

The diagnosis came from the literature that followed Hinton: soft labels alone don't carry enough of what the teacher learned. The teacher's *internal* representations encode predicate-argument structure, syntactic features, and discourse signals that get collapsed into a logit vector at the output layer; matching only the logits matches the projection, not the source. TinyBERT (Jiao et al. 2019) made the same observation and added attention-matrix and hidden-state matching in a four-loss recipe. PKD (Sun et al. 2019) added patient hidden-state matching across multiple teacher layers. MiniLM (Wang et al. 2020) matched value-relation matrices instead. R-Drop (Liang et al. 2021) regularized against dropout-induced inconsistency.

For a five-head structured-prediction student, none of those recipes was sufficient on its own.

## Five concurrent losses, two staged passes

The recipe that closed the gap is in [`shared/student_train.py`](https://github.com/rustic-ai/kniv-nlp-models/blob/main/shared/student_train.py).

Stage 1 trains all five heads concurrently against the teacher with five complementary losses. KL divergence on logits with temperature 3 is the soft signal, applied per-head with task-specific weights. Hard cross-entropy on argmax labels anchors the student where the teacher is uncertain — soft labels alone drift on tokens where the teacher's distribution is flat. Layer-normalized MSE between projected student hidden states and teacher hidden states at layers 12, 18, and 24 carries the representational structure across; it's the contribution that made the largest single difference in our ablations. R-Drop consistency between two stochastic forward passes regularizes against dropout-induced inconsistency. A separate cross-entropy on dependency relation labels gathered at the predicted arc head — easy to miss, since arc loss alone tells you which token is the head but not what the relation is.

The CLS soft-KL term has one non-obvious correctness fix worth flagging. CLS lives at the sentence level, so its `batchmean` reduction divides by batch size, while POS/NER/SRL divide by valid token count. Without compensation the CLS signal vanishes in batches with long sentences. Multiplying by `valid.sum() / B` puts it back on the same scale as the token-level terms. This is the kind of detail that costs three days of debugging if you miss it.

Stage 2 takes the Stage-1 student and runs two surgical fine-tunes. Stage 2a fine-tunes SRL on PropBank gold mixed with teacher silver, encoder unfrozen and CLS frozen — predicate-aware representations the per-token soft signal couldn't transmit get learned directly from gold supervision. Stage 2b freezes everything — encoder in `eval()` mode, every non-CLS module in `eval()` mode, dropout disabled — and trains only the three CLS modules on dialog acts. The `eval()` calls matter; leaving frozen modules in `train()` mode keeps dropout firing during the forward pass, which destabilizes the CLS head with noise it can't fix.

Multi-task knowledge distillation across structured-prediction heads, as far as the literature we surveyed, has no direct precedent. MT-DNN distillation (Liu et al. 2019, arXiv:1904.09482) distills nine tasks into one student, but the tasks are GLUE-style sentence classification — heads sharing a `[CLS]` representation, not heads needing biaffine attention over arc scores or predicate-aware encoding for verb-conditioned tagging. Hashimoto's joint many-task model (2017) cascades token-level heads but was never published with a distillation variant. The honest framing is that this work extends multi-head KD into the cascaded structured-prediction regime, with the engineering details written down in the open so the next person doesn't have to rediscover them.

## What ships

Four checkpoints, all on Hugging Face under the same permissive terms as the corpus they were trained on:

| Model | Backbone | Total params | INT8 ONNX | Notes |
|---|---|---|---|---|
| [xsmall](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-xsmall) | DeBERTa-v3-xsmall | 75M | 92 MB | Edge / mobile target |
| [small](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-small) | DeBERTa-v3-small | 157M | 190 MB | Production sweet spot |
| [base](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-base) | DeBERTa-v3-base | 200M | 250 MB | Quality / latency middle |
| [large](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-large) | DeBERTa-v3-large | 443M | 654 MB | Teacher, reference |

The xsmall student lands within 1.4 F1 points of the teacher on every head at 5.9× compression. The small student closes most of the remaining gap at 2.8× compression. Latency on a single-GPU workstation runs in single-digit milliseconds at sequence length 128 with the FP32 ONNX path; INT8 CPU is ~22ms for xsmall, suitable for embedding into an application binary without a GPU dependency.

Five heads, one ONNX session, label maps bundled, every artifact public.

If you want to retrain from scratch:

```bash
# 1. Generate v2 distillation shards from the teacher
uv run python scripts/generate_distillation_shards.py \
    --teacher-dir models/kniv-deberta-nlp-base-en-large \
    --corpus corpus/output/annotated \
    --output data/distillation

# 2. Run the three-stage student pipeline
uv run python scripts/run_student_pipeline.py xsmall \
    --output models/kniv-deberta-nlp-base-en-xsmall-repro
```

The corpus, architecture, training recipe, and weights are all public. If you try to reproduce or extend any of it and find something missing or unclear, open an issue or reach out — we'll be glad to share whatever's needed.

---

*kniv-cascade is part of the [Rustic](https://rustic.ai) initiative by [Dragonscale Industries Inc.](https://dragonscale.ai)*
