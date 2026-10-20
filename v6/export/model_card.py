"""Write a release's model card (``README.md``) and examples.

    uv run python -m v6.export.model_card --model ultrabert-380m --out release/ultrabert-380m

Numbers are the test scorecard of release candidate 2 (``docs/v7-stage-b.md``)
and the LongMemEval probe.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORG = "dragonscale-ai"
GITHUB = "https://github.com/rustic-ai/kniv-nlp-models"

MODELS = {
    "ultrabert-380m": {
        "repo": "kniv-v7-ultrabert-380m-en", "run": "ultrabert-380m-v7b8ev",
        "base": "dmis-lab/UltraBERT-380M", "base_licence": "Apache-2.0", "params": "485M",
        "encoder_params": "378M", "window": "1,024", "other": "kniv-v7-deberta-v3-large-en",
        "scores": {
            "pos": "0.978", "lemma_morph": "0.976 / 0.977", "uas_las": "0.945 / 0.927",
            "ner": "0.860", "srl_root": "0.846", "srl_e2e": "0.797", "cls": "0.761",
            "coref": "0.707", "re_gold": "0.728", "re_e2e": "0.616",
            "fewrel": "0.729 / 0.675 / 0.550", "heldout": "0.888", "crossner": "0.580",
            "fewnerd": "0.523", "lme": ("0.693", "0.480", "0.913", "73")},
        "hn": "",
        "int8_speed": "about 2x faster than fp32 (Apple M2 Max: 0.47 s instead of 1.07 s for a "
                      "200-word window with the full analysis)",
        "int8_how": "both encoder passes in per-channel INT8 after SmoothQuant (alpha 0.75) moved "
                    "this encoder's activation outliers -- up to ~8,000 in the feed-forward "
                    "down-projections -- into the weights",
        "int8_drop": "0.7",
        "choose": "the better zero-shot model (new relation and entity types), the better "
                  "memory extractor, with longer windows (1,024 subwords) and about 50M fewer "
                  "parameters",
    },
    "deberta-v3-large": {
        "repo": "kniv-v7-deberta-v3-large-en", "run": "deberta-v3-large-v7b8ev",
        "base": "microsoft/deberta-v3-large", "base_licence": "MIT", "params": "538M",
        "encoder_params": "434M", "window": "512", "other": "kniv-v7-ultrabert-380m-en",
        "scores": {
            "pos": "0.982", "lemma_morph": "0.978 / 0.978", "uas_las": "0.946 / 0.929",
            "ner": "0.870", "srl_root": "0.855", "srl_e2e": "0.810", "cls": "0.758",
            "coref": "0.692", "re_gold": "0.733", "re_e2e": "0.627",
            "fewrel": "0.731 / 0.620 / 0.602", "heldout": "0.852", "crossner": "0.533",
            "fewnerd": "0.506", "lme": ("0.573", "0.400", "0.967", "82")},
        "int8_speed": "1.5-1.7x faster than fp32 (Apple M2 Max: 0.58 s instead of 0.97 s for a "
                      "200-word window with the full analysis)",
        "int8_how": "the plain pass in per-channel INT8 after SmoothQuant (alpha 0.85); the "
                    "relation pass, more sensitive, in per-channel INT8 with its feed-forward "
                    "down-projections kept fp32",
        "int8_drop": "0.2",
        "hn": " This model's relation layer was also trained with hard-negative relations "
              "(each true relation's nearest relations by label embedding in its batch).",
        "choose": "slightly ahead on the fixed benchmarks: Re-DocRED, SRL, NER and parsing",
    },
}

CARD = """---
license: apache-2.0
language:
- en
library_name: onnx
base_model: {base}
pipeline_tag: token-classification
tags:
- multi-task
- dependency-parsing
- semantic-role-labeling
- named-entity-recognition
- coreference-resolution
- relation-extraction
- zero-shot
- knowledge-graph
- onnx
---

# kniv v7 ({base_short})

One English encoder that gives a full linguistic analysis and extracts
knowledge against a schema you supply at inference time.

- **Linguistic layer** (fixed labels): part of speech, morphology, lemmas,
  named entities, dependency trees, semantic roles, dialogue act and sentiment
  per sentence, coreference.
- **Knowledge layer** (schema-conditioned): entities of the types you name,
  facts for the relations you name (each a name plus an optional description),
  and for every fact its truth polarity, modality, holder and time.
  Conversations are first-class: each speaker is an entity, so "I moved to
  Lisbon" yields `(user, lives in, Lisbon)`.

This is release candidate 2 of kniv v7 (`{run}`): {params} parameters on
[{base}](https://huggingface.co/{base}) ({encoder_params} encoder). A sibling
model, [{other}](https://huggingface.co/{org}/{other}), shares the design.
Choose this one if you want {choose}.

## Quick start (ONNX, no PyTorch)

```bash
pip install onnxruntime tokenizers numpy huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("{org}/{repo}")
sys.path.insert(0, path)
from kniv_v7 import Kniv

kniv = Kniv(path)
out = kniv.analyse(
    turns=[("user", "I moved to Lisbon last year and started at a bank."),
           ("assistant", "How are you finding it?")],
    relations=[{{"name": "lives in", "description": "the place where the head resides"}},
               {{"name": "employer", "description": "the organization the head works for"}}],
    entity_types=[{{"name": "location"}}, {{"name": "organization"}}, {{"name": "person"}}])

w = out[0]
for f in w["facts"]:
    print(w["entities"][f["head"]]["text"], f["relation"], w["entities"][f["tail"]]["text"],
          f["polarity"], f["modality"])
```

`analyse` returns one record per window (text is split into sentences and
packed into windows of up to 512 words):

| key | content |
|---|---|
| `tokens`, `sentences` | words; sentence spans (and speakers) |
| `words` | per word: `pos`, `morph`, `lemma`, `head` (0 = root, else 1-based), `deprel` |
| `ner` | `[type, start, end)` spans, OntoNotes types |
| `srl` | frames: `predicate` and `args` as `[role, start, end)` |
| `sentence_layers` | per sentence: dialogue acts (`cls`), `sentiment` |
| `coref` | clusters of `[start, end]` word spans (inclusive) |
| `entities` | type, text, mentions, spans (speakers first in a conversation) |
| `facts` | `head`, `relation`, `tail` (indices into `entities`), `margin`, `polarity`, `modality`, `holder`, `time` |

`kniv.extract(...)` returns entities and facts only. Without `entity_types`,
entities are DocRED's six coarse types (PER, ORG, LOC, TIME, NUM, MISC).
Facts are per window; linking them across windows or documents is left to you.

### PyTorch

`model.pt` holds the weights for the model code in [kniv-nlp-models]({github})
(`v6/train/model.py`). From a clone of that repository (run at its root,
dependencies from its `pyproject.toml`), the same pipeline runs on the
PyTorch graphs:

```python
from kniv_v7 import Kniv
from v6.export.torch_backend import TorchBackend

kniv = Kniv(path, backend=TorchBackend.from_release(path))
```

## How it works

One encoder runs up to three times per window:

1. **Plain pass.** The encoder as is. It feeds every linguistic head,
   coreference and the entity span typer.
2. **Label pass.** Your type and relation names, with an encoder copy that
   has a LoRA adapter merged in, become label vectors. These are cached per
   schema.
3. **Marker pass.** The window, with `*` around every entity mention, read by
   a second encoder copy with its own adapter. It scores every entity pair
   against every relation, then the attribute head reads the facts kept.

Decoding uses dev-tuned settings from `calibration.json`:
- relation thresholds: per relation for Re-DocRED's Wikidata relations, one offset for every other relation;
- the coreference link bias;
- the entity grouping rule.

`onnx/` holds eight graphs, one window per call:
- `plain`, `dep_rel`, `srl_roles`, `coref_spans` (the linguistic layer and coreference);
- `label`, `span`, `marker`, `attr` (the knowledge layer).

`kniv_v7/runtime.py` is the host code: windows, subword maps, marker text,
Viterbi, Chu-Liu-Edmonds trees, lemma scripts, clustering, thresholds. Each
graph matches the PyTorch model to within 2e-4. On the release checks, the
runtime's decoded output over ONNX equals its output over PyTorch on every
test text.

## Results (test sets)

| task (benchmark) | this model | reference |
|---|---|---|
| POS (UD English EWT) | {pos} | -- |
| lemma / morphology, exact (UD EWT) | {lemma_morph} | -- |
| UAS / LAS (UD EWT) | {uas_las} | kniv v5 / v6 LAS 0.923 |
| NER (OntoNotes 5, not trained on) | {ner} | kniv v5 0.889, v6 0.869 |
| SRL, root predicates | {srl_root} | kniv v5 0.843 |
| SRL, all predicates, end to end (PropBank EWT) | {srl_e2e} | -- |
| dialogue acts (SGD, micro-F1) | {cls} | kniv v6 0.768 |
| coreference (LitBank, CoNLL F1) | {coref} | LingMess 0.699 |
| relations, gold entities (Re-DocRED) | {re_gold} | fixed-schema ATLOP (kniv v6.2) 0.764 |
| relations, end to end (Re-DocRED) | {re_e2e} | kniv v6.2 0.566 |
| zero-shot relations (FewRel, m = 5 / 10 / 15 unseen) | {fewrel} | RE-Matching 0.926 / 0.829 (m = 5 / 10) |
| held-out relations from our own annotation (ranking) | {heldout} | kniv v6.2 0.710 |
| zero-shot entity types (CrossNER, mean of 4 domains) | {crossner} | GLiNER-large 0.613 |
| Few-NERD test | {fewnerd} | -- |

"End to end" means the model finds the entities or predicates itself.
FewRel's unseen relations are a fixed set removed from every training
source, so the protocol is comparable in task and metric, not in which
relations are unseen.

**FP16.** `onnx/*-fp16.onnx` are the three encoder graphs exported in half
precision (the heads stay fp32). Use them with `Kniv(path, variant="fp16")`.
On the release's dev subsets they match fp32 to within 0.001 (UD POS and LAS,
Re-DocRED relations, CrossNER spans), at half the size. They are for download
size and GPUs (`providers=["CUDAExecutionProvider"]`); on CPU they are no
faster than fp32. To download only what FP16 needs:

```python
path = snapshot_download("{org}/{repo}", ignore_patterns=[
    "model.pt", "onnx/plain.onnx", "onnx/label.onnx", "onnx/marker.onnx"])
kniv = Kniv(path, variant="fp16")
```

**INT8, for CPU speed.** `Kniv(path, int8=True)`: {int8_how}; the label
graph (run once per schema) is FP16. On the dev subsets no task drops more
than {int8_drop} points (UD POS and LAS, Re-DocRED relations, CrossNER
spans), and on CPU it runs {int8_speed}.

### Agent memory (LongMemEval, oracle sessions, 150 questions)

Facts are extracted from the sessions with one generic memory schema, and the
probe measures whether the answer reaches the extracted store (no LLM answers
the question):

| answer in extracted entities | answer in extracted facts | questions with facts about the user | facts per question |
|---|---|---|---|
| {lme0} | {lme1} | {lme2} | {lme3} |

## Training

The encoder was pretrained on knowledge first:
- T-REx with Wikidata types;
- DocRED distant supervision;
- Few-NERD;
- Re-DocRED;
- our own LLM-annotated corpus (6 domains: business, conversation, encyclopedic, narrative, news, technical);
- our own synthetic relation passages.

Then every head was trained jointly on the corpus, with gold linguistic data mixed in:
- UD English EWT with PropBank frames;
- 238k gold-filtered NER sentences from kniv v5.

The linguistic heads were then fine-tuned on a frozen encoder: an MLP
predicate head and a context transformer for SRL roles, and a BiLSTM for NER.
Coreference was trained on GUM's openly licensed genres. Knowledge-only passes,
with the encoder and the linguistic layer asserted frozen, then re-adapted the
knowledge layer; the last one weighted DREEAM's evidence loss up.{hn} 1,423 relations
and 1,559 entity types (names and descriptions) were sampled as schemas
during training.

Training data: Re-DocRED and DocRED (MIT), FewRel (MIT), Few-NERD (CC BY-SA
4.0), T-REx (CC BY-SA 4.0), Wikidata (CC0), UD English EWT and PropBank EWT
(CC BY-SA 4.0), GUM (CC BY 4.0, CC BY / BY-SA genres only), and text we
collected and annotated ourselves. Evaluation-only: OntoNotes, LitBank,
CrossNER, SGD, LongMemEval.

## Limitations

- **Document-level relations trail specialists.** About 29% of Re-DocRED gold
  facts are missed, mostly yes/no decisions on a pair (the right relation
  ranked first but under the threshold) and near-synonym confusions. The
  open schema costs about 3 points against a fixed-schema head.
- **Coreference** mostly loses on mention boundaries and on which noun phrases
  count as mentions; pronouns are resolved well.
- **Entity grouping** is rule-based (exact names, plus coreference where
  calibrated); partial names ("Obama" / "Barack Obama") are not merged.
- **Facts are per window** (up to 512 words); nothing is linked across
  windows. Windows longer than the encoder's subword limit ({window}
  subwords) are analysed up to the limit.
- **English only.** Entity and relation names in other languages are not supported.

## Licence

Apache-2.0 for the weights and the runtime code. The base encoder
([{base}](https://huggingface.co/{base})) is {base_licence}.
"""


def write(model: str, out: Path) -> None:
    m = MODELS[model]
    s = m["scores"]
    text = CARD.format(base=m["base"], base_short=m["base"].split("/")[-1], run=m["run"],
                       params=m["params"], encoder_params=m["encoder_params"], other=m["other"],
                       org=ORG, repo=m["repo"], choose=m["choose"], github=GITHUB,
                       window=m["window"], base_licence=m["base_licence"], hn=m["hn"], int8_speed=m["int8_speed"], int8_how=m["int8_how"], int8_drop=m["int8_drop"],
                       lme0=s["lme"][0], lme1=s["lme"][1],
                       lme2=s["lme"][2], lme3=s["lme"][3],
                       **{k: v for k, v in s.items() if k != "lme"})
    (out / "README.md").write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    write(args.model, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
