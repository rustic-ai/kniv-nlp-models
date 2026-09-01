# One Encoder. Five Heads. Every Token Understood.

*Introducing kniv-cascade: a multi-task NLP model that extracts POS, NER, dependency structure, semantic roles, and dialog acts in a single pass — and why we built it for agent memory.*

---

There's a dirty secret in agent memory systems. Every one of them — Mem0, Zep, Graphiti, Letta — calls an LLM to understand what was said. Every message ingested, every fact extracted, every entity identified: an API call to GPT-4 or Claude, a dollar spent, a second burned, a dependency on someone else's server staying up. The LLM reads the sentence, extracts entities and facts, and the memory system stores them. Multiply by ten thousand messages across a hundred sessions and you have a system that works in demos and bankrupts in production.

The alternative is supposed to be local NER — spaCy, a distilled BERT, maybe a regex pipeline. These are fast and cheap, but they only answer one question: *what are the entities?* They don't tell you what the sentence *means*. They can't parse "I told her about the adoption agency" into a dependency tree that reveals who told whom about what. They can't classify "sounds great, let's do Thursday" as a plan commitment rather than a factual statement. They can't distinguish the subject of a sentence from its object.

When you're building a memory system that needs to extract structured knowledge from conversation — not just entities, but observations, speaker attribution, temporal references, argument structure — a named entity tagger is not enough, and an LLM is too much.

We needed something in between. So we built it.

[kniv-cascade](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-large) is a multi-task NLP model that runs five analysis tasks in a single forward pass: part-of-speech tagging, named entity recognition, dependency parsing, semantic role labeling, and dialog act classification. One DeBERTa-v3-large encoder. Five task heads. One call. Every token fully analyzed.

This post explains what it does, how we trained it, and why agent memory is the use case that demanded it.

---

## Why Agent Memory Needs More Than NER

To understand why we built a five-headed model instead of plugging in an off-the-shelf NER tagger, you need to understand what [uniko](https://github.com/rustic-ai/uniko) — our cognitive memory system for AI agents — does with language.

uniko ingests conversations between humans and agents, extracts structured knowledge, and organizes it into a typed knowledge graph. Messages go in. Entities, observations, facts, and relationships come out. The extraction pipeline converts raw conversation into the kind of structured memory that makes agents smarter over time — not by re-reading old messages, but by querying compiled knowledge.

The spec calls this "compile once, query forever." The LLM pays the extraction cost during ingestion; every subsequent query benefits for free. But that only works if the extraction is good. And extraction quality depends entirely on how well you understand the language.

Here's what the pipeline needs to do with every sentence:

**1. Find the entities.** "Caroline is researching adoption agencies in Portland." Extract: Caroline (Person), Portland (Location). This is standard NER.

**2. Classify the sentence.** Is this a factual statement, a question, a greeting, a plan? The spec says observations should only be extracted from factual statements — "not questions, greetings, or reactions." A sentence classifier gates the extraction pipeline: informative sentences proceed, everything else is filtered before extraction even begins.

**3. Parse the structure.** Who is doing what to whom? "Caroline is researching adoption agencies" has a subject (Caroline), a predicate (is researching), and an object (adoption agencies). This isn't entity extraction — it's syntactic structure. You need a dependency parse to get it right, especially when the sentence is complex: "The workshop she attended last March in Seattle covered techniques for managing anxiety."

**4. Resolve pronouns.** "She's been volunteering at the shelter since January." Who is "she"? If the previous sentence mentioned Caroline, "she" is Caroline. First-person pronouns ("I started a new job") need to resolve to the message sender. Second-person ("you should try it") to the other participant. This requires POS tags to identify pronouns and dependency arcs to track antecedents.

**5. Extract temporal references.** "I moved here three years ago" — when is "three years ago"? The observation needs an `observed_at` timestamp anchored to the message time, not just the current date.

A NER model handles step 1. Steps 2-5 require POS, dependency parsing, and sentence classification — tasks that are traditionally handled by separate models, each loading its own transformer, each running its own forward pass, each unaware of what the others found.

Or you call an LLM. Which works, until you try to ingest a LoCoMo benchmark conversation — 5,882 turns across 10 conversations — and realize you're making tens of thousands of API calls just to populate the graph. At $0.15 per million input tokens, the cost is manageable. At 500ms per call, the latency isn't. And when the API is down, your memory system is blind.

We needed all five capabilities, locally, in one call, at inference latency measured in milliseconds.

---

## The Architecture

kniv-cascade is built on two ideas: layer selection and task cascading.

### Layer Selection

Not every NLP task needs the same depth of understanding. [Tenney et al. (2019)](https://arxiv.org/abs/1905.05950) showed that BERT's layers encode a linguistic hierarchy: surface features in lower layers, syntax in the middle, semantics at the top. [Jawahar et al. (2019)](https://arxiv.org/abs/1903.09000) confirmed this with probing experiments on BERT: phrase-level information peaks at lower layers, syntactic features at middle layers, and semantic features at the top.

Each task head in kniv-cascade has a learned [ScalarMix](https://docs.allennlp.org/main/api/modules/scalar_mix/) — a softmax-weighted average over all 25 encoder layers (embedding + 24 transformer layers) — that discovers which depths are most useful for its task:

- **POS** converges to layers 0-8 (morphosyntax — word forms, inflection)
- **NER** reads layers 5-10 (entity recognition peaks at the interface of syntax and semantics)
- **DEP** draws from 12-18 (structural relationships between words)
- **SRL** reads the top (last hidden state — full semantic representation)
- **CLS** uses all layers via ScalarMix (sentence-level classification benefits from the full depth)

No manual assignment. The model finds what it needs. The result is that each head gets a representation optimized for its task, from the same encoder, at zero additional compute cost.

### Task Cascading

The heads don't just share an encoder — they share predictions. Information flows from simpler tasks to more complex ones:

```
Encoder → POS → [POS probs] → NER → [NER probs] → DEP
                                ↓
                          POS informs entity boundaries
                          NER informs syntactic roles
```

The NER head receives POS probabilities — 17 dimensions of syntactic signal — concatenated to its BiLSTM hidden state. A token tagged as PROPN (proper noun) by POS is more likely to be an entity. The dependency parser receives both POS and NER probabilities. A token tagged as B-PERSON by NER is more likely to be a syntactic subject.

Upstream outputs are detached during training — no gradient flows backward through the cascade. Each head trains independently, but benefits from upstream predictions at inference. No joint training instabilities. No task interference. Just structured information flowing downstream.

This is the key insight from [Hashimoto et al. (2017)](https://arxiv.org/abs/1611.01587) and [Sanh et al. (2019)](https://arxiv.org/abs/1902.10969): multi-task learning with structured information sharing outperforms both single-task models and naive multi-task models where heads share nothing but an encoder. The cascade gives downstream heads *features they would otherwise need to learn from scratch*.

### The Full Architecture

```
DeBERTa-v3-large (434M) + predicate_embedding(2, 1024)
│
├─ ScalarMix(25) → Linear(17)                           → POS    [argmax]
├─ ScalarMix(25) → BiLSTM(256,bd) → +POS → MLP(37)     → NER    [Viterbi]
├─ ScalarMix(25) → +POS/NER → Biaffine(arc:512,lbl:128) → DEP    [argmax]
├─ last_hidden   → MLP(H→H→42)                          → SRL    [Viterbi]
└─ ScalarMix(25) → AttentionPool → MLP(H→H/2→8)        → CLS    [argmax]
```

443M total parameters. 434M encoder, 9.5M across all five task heads. One forward pass. Six output tensors.

---

## How the Encoder Was Shaped

The training strategy matters as much as the architecture. We call it bottom-up layer-selective training, and it produced 13.3 aggregate points of improvement over our previous top-down approach.

### The Problem with Conventional Multi-Task Training

The standard approach fine-tunes the entire encoder for one task, then freezes it for everything else. The first task dominates the encoder's representations. Everything after inherits whatever that task left behind — representations shaped for POS that NER must work around, not with.

### Bottom-Up Layer-Selective Training

We reverse the order. Each phase shapes a different layer range, building progressively from morphosyntax at the bottom to semantics at the top:

| Phase | Task | Layers Unfrozen | Encoder LR | Epochs | Data |
|-------|------|----------------|------------|--------|------|
| 1 | POS | All (warm-up) | 2e-5 | 10 | 12.5K [UD EWT](https://universaldependencies.org/treebanks/en_ewt/) gold |
| 2 | NER | 5-12 | 2e-5 / 5e-5 | 15 | 195K [SpanMarker](https://huggingface.co/tomaarsen/span-marker-roberta-large-ontonotes5) silver |
| 3 | DEP | 12-18 | 2e-5 | 20 | 12.5K UD EWT gold |
| 4 | SRL | All | 3e-6 | 3 | 200K AllenNLP silver + 41K [PropBank](https://github.com/propbank/propbank-release) gold |
| 5 | CLS | Frozen | — | 5 | 60K [SGD](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) + GPT |

Phase one trains POS with all layers unfrozen — a syntactic warm-up. Every layer learns basic morphosyntactic patterns. Phase two trains NER, unfreezing only layers 5-12. The overlap zone (layers 5-8, already shaped by POS) gets a reduced learning rate to preserve what POS built. Fresh layers (9-12) get a higher rate to develop entity representations. Phase three trains dependency parsing on layers 12-18. Phase four trains SRL, then runs a final pass across all layers at ultra-low learning rate for semantic adaptation. Phase five trains CLS on the frozen encoder — the encoder is done being shaped; CLS learns to read it.

After all phases, a brief recovery step retrains POS/NER/DEP heads for three epochs on the frozen final encoder, adapting to the slight shifts caused by later phases.

### The Results

| Task | Previous (v3.2) | Bottom-up (v5) | Gain |
|------|----------------|----------------|------|
| POS | 0.966 | 0.977 | +1.1 |
| NER | 0.860 | 0.898 | +3.8 |
| DEP | 0.922 | 0.944 | +2.2 |
| SRL | 0.802 | 0.843 | +4.1 |
| CLS | 0.930 | 0.951 | +2.1 |

Every head improved. SRL gained the most (+4.1), which makes sense — it sits at the top of the layer stack and benefits most from the well-shaped foundations below it.

### Training Data

All training data is published as [dragonscale-ai/kniv-corpus-en](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en) on HuggingFace.

POS and DEP train on gold-standard Universal Dependencies — human-annotated treebanks. NER trains on silver labels from [SpanMarker](https://huggingface.co/tomaarsen/span-marker-roberta-large-ontonotes5), a RoBERTa-large span classifier fine-tuned on OntoNotes 5.0. SRL trains on a mix of AllenNLP silver predictions and PropBank gold annotations. CLS trains on Schema-Guided Dialogue (SGD) augmented with GPT-generated examples for underrepresented dialog acts.

The choice to use silver labels for NER and SRL is deliberate. Gold NER annotation at the 18-type OntoNotes granularity is expensive and scarce. SpanMarker's silver labels on 195K sentences provide the volume needed for robust generalization, at a quality level (SpanMarker itself scores 0.91+ F1 on OntoNotes) that's more than adequate as training signal for our NER head. The model learns from a strong teacher, not from noisy labels.

---

## Predicate-Aware Encoding for SRL

Semantic Role Labeling answers "who did what to whom" — but the answer depends on which verb you're asking about. "John told Mary to leave" has two predicates (*told*, *leave*), each with different argument structures.

Most SRL systems handle this by running the encoder separately for each predicate, marking the target verb with a special token. This means N encoder passes for N verbs.

We inject the predicate signal at the embedding level. A learned `Embedding(2, 1024)` — initialized to zero — adds to the token embedding of the predicate word before the encoder sees it. All 24 attention layers then condition on which word is the predicate. In the ONNX export, this integrates into the single forward pass. Pass `predicate_idx=0` when SRL isn't needed — the zero-initialized embedding adds nothing.

---

## The Numbers

Evaluated on standard public benchmarks. No benchmark data was used during training. All results are reproducible with the included [benchmark scripts](https://github.com/rustic-ai/kniv-nlp-models/tree/main/models/kniv-deberta-nlp-base-en-large).

| Task | Score | Metric | Benchmark |
|------|-------|--------|-----------|
| POS tagging | 0.977 | Accuracy | [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/) test |
| Named Entity Recognition | 0.889 | F1 (micro) | [OntoNotes 5.0](https://huggingface.co/datasets/tner/ontonotes5) test |
| Dependency Parsing | 0.944 / 0.923 | UAS / LAS | [UD English EWT](https://universaldependencies.org/treebanks/en_ewt/) test |
| Semantic Role Labeling | 0.843 | F1 | PropBank EWT test |
| Dialog Act Classification | 0.951 | Macro F1 | SGD + GPT dev |

POS and DEP use Universal Dependencies English Web Treebank — the standard syntactic evaluation. NER runs on OntoNotes 5.0, the 18-entity-type benchmark that includes persons, organizations, dates, monetary values, and 14 other types. SRL evaluates on PropBank gold annotations.

NER was cross-evaluated on [CoNLL-2003](https://huggingface.co/datasets/eriktks/conll2003) (F1 = 0.794) with entity type mapping — 18 OntoNotes types compressed to 4 CoNLL types. Numeric entities like DATE and CARDINAL have no CoNLL equivalent and map to O, so the CoNLL number understates the model's actual entity coverage. The OntoNotes number is the fairer comparison.

CLS was cross-evaluated on [DailyDialog](https://huggingface.co/datasets/daily_dialog) (accuracy = 0.613) with lossy 8-to-4 label mapping. CLS is optimized for conversational dialog — the domain where uniko uses it — not news or documents. Cross-domain degradation is expected and acceptable for our use case.

### Context: Where Does This Sit?

For reference, state-of-the-art single-task models on these benchmarks:

| Task | kniv-cascade | SOTA (single-task) | Notes |
|------|-------------|-------------------|-------|
| POS | 0.977 | ~0.98 | Within 0.3% of dedicated taggers |
| NER (OntoNotes) | 0.889 | ~0.91 | SpanMarker-large scores 0.91; we're 2 points behind with 5 heads from one encoder |
| DEP (UAS/LAS) | 0.944 / 0.923 | ~0.96 / 0.94 | Competitive with dedicated parsers |
| SRL | 0.843 | ~0.87 | Within 3% of specialized SRL models |
| CLS | 0.951 | ~0.96 | Strong for conversational classification |

The model is not the best at any single task. It is the only model that does all five at once. The multi-task penalty is small — typically 1-3% per head — and the compute savings are enormous: one encoder pass instead of five.

---

## How uniko Uses Every Output

This is where the model stops being an academic exercise and starts being infrastructure. Here's how uniko's extraction pipeline consumes each of the five heads.

### Dialog Act Classification → Sentence Gate

Before any extraction begins, the CLS head classifies each sentence. Only informative classes proceed:

- **Extraction proceeds:** `inform` (factual statements), `correction` (factual corrections), `plan_commit` ("I'm going to..."), `request` (requests containing implicit facts)
- **Filtered out:** `question`, `social` (greetings, farewells), `filler` ("okay", "right", "haha"), `agreement`, `feedback` ("that's great")

This gate runs per-sentence, not per-message. A message like "Hey! How are you? I just started a new job at the hospital." contains three sentences. The CLS head classifies the first two as social/question and the third as inform. Only the third produces an observation. Without this gate, the pipeline would attempt extraction on greetings and questions — producing noise that pollutes the knowledge graph.

### POS + Dependency Parsing → Structured Observation Extraction

This is the core of the pipeline. The dependency parser produces a tree structure that the extraction engine walks to produce self-contained observations.

Given: *"Caroline is researching adoption agencies in Portland"*

The dependency parse produces:

```
researching (VERB, root)
├── Caroline (PROPN, nsubj)
├── agencies (NOUN, obj)
│   ├── adoption (NOUN, compound)
│   └── Portland (PROPN, nmod)
└── is (AUX, aux)
```

The extraction engine:
1. Finds predicate tokens: VERB or copular ADJ/NOUN with a `cop` dependent
2. Collects `nsubj` dependents → subject ("Caroline")
3. Collects `obj` dependents → object, including compounds ("adoption agencies")
4. Collects `obl`/`advmod`/`xcomp` → modifiers ("in Portland")
5. Reconstructs a declarative observation: *"Caroline is researching adoption agencies in Portland"*

For copular sentences like "The workshop was inspiring," the parser detects the `cop` relation and the engine reconstructs "The workshop is inspiring" — injecting the copula rather than treating "inspiring" as a verb.

### POS Tags → Pronoun Resolution

POS tags identify pronouns (PRON), which trigger resolution against a session-scoped context:

```
"I"/"we"/"me"    → speaker name (resolved via SENT_BY edge to Participant node)
"you"/"your"     → other participant in the session
"it"/"this"      → last NOUN/PROPN seen in subject or object position
```

After each sentence, the engine records the most recent NOUN/PROPN tokens in subject and object positions — *excluding pronouns* — as antecedents for the next sentence.

**Concrete example from LoCoMo benchmark data:**

```
Message (speaker: Jon): "I'm thinking about starting a dance studio.
                         It could be something really special."

Sentence 1: CLS=inform → extract
  DEP: nsubj(thinking, I), xcomp(thinking, starting), obj(starting, studio)
  Pronoun: "I" → "Jon"
  Observation: "Jon is thinking about starting a dance studio"
  Context update: last_noun_subject = None, last_noun_object = "dance studio"

Sentence 2: CLS=inform → extract
  DEP: nsubj(be, It), xcomp(be, special)
  Pronoun: "It" → "dance studio" (from context)
  Observation: "dance studio could be something really special"
```

Without POS-driven pronoun resolution, the second observation would be "It could be something really special" — useless for knowledge extraction. With it, the observation correctly attributes the statement to "dance studio."

### Named Entity Recognition → Entity Graph

NER extracts 18 entity types from the OntoNotes taxonomy. These are mapped to uniko's entity type system and deduplicated against the knowledge graph:

| OntoNotes Type | uniko Entity Type | Example |
|---|---|---|
| PERSON | Person | "Caroline", "Jon" |
| ORG | Organization | "Google", "the hospital" |
| GPE, LOC, FAC | Location | "Portland", "Seattle" |
| DATE, TIME | Date | "last March", "Thursday" |
| MONEY, PERCENT, QUANTITY, CARDINAL, ORDINAL | Measurement | "$500", "three years" |
| EVENT, PRODUCT, WORK_OF_ART, NORP | Other | typed and stored |

Entities are linked to their source messages via MENTIONS edges, creating the graph structure that powers entity-scoped retrieval and multi-hop reasoning.

### SRL → Future: Structured Fact Extraction

The SRL head isn't consumed in uniko's current extraction pipeline — it's infrastructure for the next phase. When consolidation ships (P4), SRL frames will provide structured `(agent, predicate, patient, temporal, location)` tuples that feed directly into fact derivation:

```
"Caroline visited the adoption agency last Tuesday"
  ARG0: Caroline (agent)
  V:    visited  (predicate)
  ARG1: the adoption agency (patient)
  ARGM-TMP: last Tuesday (temporal)
```

This is richer than what dependency parsing alone provides. DEP gives syntactic structure; SRL gives semantic roles. The difference matters when the syntax is ambiguous but the semantics aren't.

### POS Tags → Recall Intent Construction

One more use, downstream from extraction: when a user queries the memory system, the intent profile is constructed by POS-filtering the query to content words only.

Query: *"What did Caroline research about adoption?"*

POS filter (keep NOUN, VERB, PROPN, ADJ, NUM): *"Caroline research adoption"*

These content words — not the original question — are embedded as the search vector. Questions contain interrogative structure ("What did...") that doesn't appear in the indexed content (declarative observations). Stripping to content words produces embeddings closer to the statements being searched. The improvement is measurable: entity-matched recall improves when the query vector is built from content words rather than the raw question.

---

## Deployment

The model exports to a single ONNX file. One call, six output tensors.

```python
session = ort.InferenceSession("cascade.onnx")
pos, ner, arc, label, srl, cls = session.run(None, {
    "input_ids": input_ids,
    "attention_mask": attention_mask,
    "predicate_idx": predicate_idx,
})
```

Three variants:

| Format | Size | Speed | Quality |
|--------|------|-------|---------|
| PyTorch checkpoint | 1.74 GB | baseline | reference |
| ONNX (FP32) | 1.78 GB | baseline | reference |
| ONNX (INT8 quantized) | 654 MB | ~2x faster | -0.2% POS accuracy |

The INT8 model runs approximately twice as fast on CPU with less than 0.3% degradation on POS accuracy. All outputs validated against the PyTorch reference (max difference < 0.001).

In uniko, the INT8 model runs via ONNX Runtime through the `ort` crate — pure Rust, no Python runtime, no PyO3 dependency. The tokenizer and label maps are embedded as compile-time assets (3.5MB tokenizer, 31KB labels). A single `NlpPipeline::analyze()` call returns words, POS indices, NER spans, dependency arcs, SRL frames, and sentence classification. Per-sentence analysis splits on punctuation boundaries and processes each sentence independently — avoiding problems where a greeting at the start of a message biases the CLS classification of a factual statement later in the same message.

### What This Means for Latency

uniko's spec requires NER extraction under 100ms (NF5) and observation extraction under 5 seconds (NF6). With the INT8 model on commodity hardware (M-series Mac or 8-core Linux):

- **Tokenization + ONNX inference + decoding:** ~15-40ms per message (varies with length)
- **NER span extraction + deduplication + graph upsert:** ~5-15ms
- **Observation extraction (DEP walk + pronoun resolution):** ~2-5ms
- **Total per-message pipeline:** ~25-60ms

This is 10-100x faster than an LLM API call, runs offline, and costs nothing per token.

---

## What's Next

This is the base model — five heads, one encoder. The architecture supports additive head training without retraining existing heads or modifying the encoder.

**Tier 1** adds lemmatization, morphological features, and keyword extraction — using data already prepared. Lemmatization is immediately useful for observation deduplication: "researching" and "researched" should resolve to the same predicate.

**Tier 2** adds sentiment analysis, fine-grained intent classification, punctuation restoration, and truecasing — production features for ASR-sourced conversations and noisy dialog.

**Tier 3** adds extractive question answering, natural language inference, and relation extraction — comprehension capabilities that build on the existing cascade. The QA head benefits from NER and SRL cascade features: it doesn't need to learn where entities and arguments are — it already knows. Relation extraction feeds directly into uniko's knowledge graph: `(entity_A, relation, entity_B)` tuples extracted without LLM calls.

### Student Models

The teacher model (443M parameters, 654MB INT8) is production-viable but large. Student models are in training via knowledge distillation: the teacher's soft predictions across all five heads are distilled into smaller encoders using KL divergence on the teacher's probability distributions.

| Student | Parameters | Target |
|---------|-----------|--------|
| DeBERTa-v3-base | 86M | General-purpose |
| DeBERTa-v3-xsmall | 22M | Edge / mobile |
| ModernBERT-base | 150M | Modern architecture |

All heads, single training session, temperature 3.0. The students inherit the cascade architecture — ScalarMix, BiLSTM NER, Biaffine DEP, Viterbi decoding — at a fraction of the size. Loss combines soft distillation (α=0.7) with hard cross-entropy on the teacher's predicted labels. Students see 500K sentences of teacher predictions stored as word-level logits in parquet shards.

The 22M DeBERTa-xsmall student is the target for uniko's default configuration: small enough to embed in the application binary, fast enough for real-time message processing, and accurate enough for the extraction pipeline. The 443M teacher serves as the high-quality reference for benchmarking and for users who can afford the memory footprint.

---

## Try It

Model and code on HuggingFace: [dragonscale-ai/kniv-deberta-nlp-base-en-large](https://huggingface.co/dragonscale-ai/kniv-deberta-nlp-base-en-large)

Training data: [dragonscale-ai/kniv-corpus-en](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en)

Source code on GitHub: [rustic-ai/kniv-nlp-models](https://github.com/rustic-ai/kniv-nlp-models)

```bash
pip install torch transformers==5.6.2 onnxruntime
python examples/cascade_demo.py
```

The demo is self-contained — it loads the model, runs all 5 heads, and prints POS tags, NER entities, the dependency tree, SRL frames, and dialog acts.

---

## Why This Matters

The conventional wisdom in agent memory is that you need an LLM for extraction and a vector store for retrieval. kniv-cascade challenges the first half of that equation. For the structured linguistic analysis that memory systems actually need — entity recognition, sentence classification, syntactic parsing, pronoun resolution — a purpose-built multi-task model at 654MB delivers what matters at a fraction of the cost.

The LLM isn't eliminated. It's repositioned. In uniko's architecture, the LLM handles what it's genuinely best at: complex coreference across paragraphs, implicit fact extraction, nuanced temporal reasoning, answer generation. The local NLP model handles the high-volume, latency-sensitive, structurally well-defined tasks that run on every message. The LLM is the specialist you call when needed. The NLP pipeline is the always-on analyst that processes everything.

One encoder. Five heads. Every token understood. And not an API call in sight.

---

*kniv-cascade is part of the [Rustic](https://rustic.ai) initiative by [Dragonscale Industries Inc.](https://dragonscale.ai)*
