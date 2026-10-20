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
        "arch_tag": "ultrabert",
        "lr": "1e-4", "batch": "8 windows (token-budgeted to 2,048 subwords) x 8 accumulation",
        "steps_a": "10,000", "hours": "about 37",
        "int8_speed": "about 2x faster than fp32",
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
        "arch_tag": "deberta-v3",
        "lr": "8e-5", "batch": "4 windows x 8 accumulation",
        "steps_a": "12,000", "hours": "about 46",
        "int8_speed": "1.5-1.7x faster than fp32",
        "int8_how": "the plain pass in per-channel INT8 after SmoothQuant (alpha 0.85); the "
                    "relation pass, more sensitive, in per-channel INT8 with its feed-forward "
                    "down-projections kept fp32",
        "int8_drop": "0.2",
        "hn": " This model's relation layer was also trained with hard-negative relations "
              "(each true relation's nearest relations by label embedding in its batch).",
        "choose": "slightly better parsing, NER, semantic roles and encyclopedic relations",
    },
}

CARD = """---
license: apache-2.0
language:
- en
library_name: onnx
base_model: {base}
pipeline_tag: token-classification
metrics:
- f1
- accuracy
tags:
- multi-task
- nlp
- information-extraction
- named-entity-recognition
- ner
- entity-extraction
- relation-extraction
- zero-shot
- part-of-speech-tagging
- lemmatization
- morphology
- dependency-parsing
- semantic-role-labeling
- coreference-resolution
- sentiment-analysis
- dialogue-act-classification
- knowledge-graph
- conversational-memory
- agent-memory
- encoder
- {arch_tag}
- onnx
- fp16
- int8
- quantized
model-index:
{model_index}
---

# kniv v7 ({base_short})

One English encoder that gives a full linguistic analysis and extracts
knowledge against a schema you supply at inference time.

- **Linguistic layer:** part of speech, morphology, lemmas, named entities,
  dependency trees, semantic roles, dialogue act and sentiment per sentence,
  coreference.
- **Knowledge layer:** entities of the types you name, facts for the
  relations you name (a name plus an optional description), and for every
  fact its polarity, modality, holder and time. In a conversation each
  speaker is an entity, so "I moved to Lisbon" yields `(user, lives in, Lisbon)`.

A sibling model, [{other}](https://huggingface.co/{org}/{other}), shares the
design. Choose this one if you want {choose}.

**[GUIDE.md](GUIDE.md)** compares the two models and shows how to work
around their known failure modes in your own code.

## Model details

| | |
|---|---|
| developed by | Dragonscale ([{org}](https://huggingface.co/{org})) |
| model type | multi-task transformer encoder |
| version | kniv v7 (`{run}`), 2026-10-20 |
| base model | [{base}](https://huggingface.co/{base}) |
| parameters | {params} |
| input | up to 512 words per window ({window} subwords) |
| language | English |
| licence | Apache-2.0 |

## Uses

**Intended.** Analysing English prose and conversations; extracting entities
and facts for your own schema without fine-tuning, for agent memory,
knowledge graphs, search and document analytics.

**Out of scope.**
- Languages other than English.
- Acting on extracted facts without review where errors matter (medical,
  legal, financial or employment decisions about people).
- Profiling individuals without their consent.

## Installation

Everything needed to run the model is in this repository: the ONNX graphs,
the tokenizer, the settings, and the `kniv_v7` Python package that turns text
into results. There is nothing else to install from elsewhere: download the
repository and put it on your Python path.

```bash
pip install onnxruntime tokenizers numpy huggingface_hub
```

```python
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("{org}/{repo}")    # includes kniv_v7/
sys.path.insert(0, path)
from kniv_v7 import Kniv
```

## Quick start

```python
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

Pass text as `kniv.analyse(text, ...)` (paragraphs separated by blank lines)
or a conversation as `turns`. `analyse` returns one record per window:

| key | content |
|---|---|
| `tokens`, `sentences` | words; sentence spans (and speakers) |
| `words` | per word: `pos`, `morph`, `lemma`, `head` (0 = root, else 1-based), `deprel` |
| `ner` | `[type, start, end)` named-entity spans |
| `srl` | frames: `predicate` and `args` as `[role, start, end)` |
| `sentence_layers` | per sentence: dialogue acts (`cls`), `sentiment` |
| `coref` | clusters of `[start, end]` word spans (inclusive) |
| `entities` | type, text, mentions, spans (speakers first in a conversation) |
| `facts` | `head`, `relation`, `tail` (indices into `entities`), `margin`, `polarity`, `modality`, `holder`, `time` |

`kniv.extract(...)` returns entities and facts only. Without `entity_types`,
entities get six coarse types (PER, ORG, LOC, TIME, NUM, MISC).

### Variants

| variant | load | for |
|---|---|---|
| fp32 | `Kniv(path)` | reference |
| FP16 | `Kniv(path, variant="fp16")` | half the download; GPUs (`providers=["CUDAExecutionProvider"]`) |
| INT8 | `Kniv(path, int8=True)` | CPU speed: {int8_speed} |

FP16 matches fp32 within 0.001 on every task. INT8 stays within {int8_drop}
points. To download only the FP16 files:

```python
path = snapshot_download("{org}/{repo}", ignore_patterns=[
    "model.pt", "onnx/plain.onnx", "onnx/label.onnx", "onnx/marker.onnx"])
kniv = Kniv(path, variant="fp16")
```

### PyTorch

`model.pt` holds the PyTorch weights. They are used through the model code in
[kniv-nlp-models]({github}):

```python
from kniv_v7 import Kniv
from v6.export.torch_backend import TorchBackend

kniv = Kniv(path, backend=TorchBackend.from_release(path))
```

## How it works

One encoder runs up to three times per window:

1. **Plain pass.** Every linguistic head, coreference and the entity typer.
2. **Label pass.** Your type and relation names become vectors, cached per
   schema.
3. **Marker pass.** The window with every entity mention marked. Every entity
   pair is scored against every relation, then each kept fact's attributes.

Decoding thresholds are in `calibration.json`, and every one can be changed
at run time (see GUIDE.md).

## Results

| benchmark | score |
|---|---|
{results_rows}

## Bias, risks and limitations

- **Coverage.** People, places and organisations from well-covered regions
  are recognised and related more reliably than others.
- **Privacy.** The model extracts facts about people, speakers included.
  Store and use them with the consent and data-protection care that implies.
- **Facts are candidates.** Treat each fact as a candidate with a confidence
  (`margin`), not as verified truth. Keep a human in the loop where facts
  about people drive decisions.
- **Document-level relations.** About 29% of facts are missed, mostly
  borderline decisions. Lower the thresholds for recall (GUIDE.md).
- **Entity grouping** is rule-based. Partial names ("Obama" / "Barack
  Obama") are not merged.
- **Per-window facts.** Facts are per window; nothing is linked across
  windows or documents.
- **Long windows.** Text past {window} subwords in a window is not analysed.
- **English only.**

## Licence

Apache-2.0. The base encoder ([{base}](https://huggingface.co/{base})) is {base_licence}.

## Citation

```bibtex
@misc{{kniv_v7_2026,
  title        = {{kniv v7: one encoder for linguistic analysis and schema-conditioned knowledge extraction}},
  author       = {{{{Dragonscale}}}},
  year         = {{2026}},
  howpublished = {{\\url{{https://huggingface.co/{org}/{repo}}}}}
}}
```

## Contact

Questions and issues: [{github}/issues]({github}/issues).
"""


def results_rows(m: dict) -> str:
    sc = m["scores"]
    rows = [("UD English EWT, POS", sc["pos"]), ("UD English EWT, lemma / morphology", sc["lemma_morph"]),
            ("UD English EWT, UAS / LAS", sc["uas_las"]), ("OntoNotes 5, NER", sc["ner"]),
            ("PropBank, SRL", sc["srl_e2e"]), ("SGD, dialogue acts", sc["cls"]),
            ("LitBank, coreference", sc["coref"]),
            ("Re-DocRED, relations (gold / predicted entities)", f"{sc['re_gold']} / {sc['re_e2e']}"),
            ("FewRel, zero-shot relations (5 / 10 / 15-way)", sc["fewrel"]),
            ("CrossNER, zero-shot entity types", sc["crossner"]), ("Few-NERD, entity types", sc["fewnerd"]),
            ("LongMemEval, answer in extracted entities / facts", f"{sc['lme'][0]} / {sc['lme'][1]}")]
    return "\n".join(f"| {b} | {v} |" for b, v in rows)


def model_index(m: dict) -> str:
    """``model-index`` results for the Hub's evaluation widget."""
    sc = m["scores"]
    las = sc["uas_las"].split(" / ")[1]
    rows = [
        ("token-classification", "Part-of-speech tagging", "universal_dependencies", "UD English EWT",
         "test", [("accuracy", sc["pos"], "UPOS accuracy")]),
        ("token-classification", "Dependency parsing", "universal_dependencies", "UD English EWT",
         "test", [("accuracy", las, "LAS")]),
        ("token-classification", "Named entity recognition", "conll2012_ontonotesv5", "OntoNotes 5.0",
         "test", [("f1", sc["ner"], "span F1")]),
        ("token-classification", "Semantic role labeling", "propbank_ewt", "PropBank EWT",
         "test", [("f1", sc["srl_e2e"], "end-to-end argument F1")]),
        ("token-classification", "Coreference resolution", "litbank", "LitBank",
         "test", [("f1", sc["coref"], "CoNLL F1")]),
        ("token-classification", "Document-level relation extraction", "re-docred", "Re-DocRED",
         "test", [("f1", sc["re_gold"], "F1 (gold entities)"), ("f1", sc["re_e2e"], "F1 (end to end)")]),
        ("text-classification", "Zero-shot relation classification", "few_rel", "FewRel (15 unseen)",
         "train+validation", [("f1", sc["fewrel"].split(" / ")[0], "macro F1, m=5")]),
        ("token-classification", "Zero-shot named entity recognition", "crossner", "CrossNER",
         "test", [("f1", sc["crossner"], "span F1, mean of 4 domains")]),
    ]
    out = [f"- name: {m['repo']}", "  results:"]
    for task, tname, dtype, dname, split, metrics in rows:
        out += [f"  - task:", f"      type: {task}", f"      name: {tname}",
                f"    dataset:", f"      type: {dtype}", f"      name: {dname}", f"      split: {split}",
                f"    metrics:"]
        for mt, v, mn in metrics:
            out += [f"    - type: {mt}", f"      value: {v}", f"      name: {mn}"]
    return "\n".join(out)


def write(model: str, out: Path) -> None:
    m = MODELS[model]
    s = m["scores"]
    text = CARD.format(results_rows=results_rows(m), arch_tag=m["arch_tag"], model_index=model_index(m), lr=m["lr"], batch=m["batch"],
                       steps_a=m["steps_a"], hours=m["hours"], base=m["base"], base_short=m["base"].split("/")[-1], run=m["run"],
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
