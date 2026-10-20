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
        "lr": "1e-4", "batch": "8 windows (token-budgeted to 2,048 subwords) x 8 accumulation",
        "steps_a": "10,000", "hours": "about 37",
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
        "lr": "8e-5", "batch": "4 windows x 8 accumulation",
        "steps_a": "12,000", "hours": "about 46",
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
datasets:
- thunlp/docred
- DFKI-SLT/few-nerd
- dragonscale-ai/kniv-corpus-en
metrics:
- f1
- accuracy
tags:
- multi-task
- dependency-parsing
- semantic-role-labeling
- named-entity-recognition
- coreference-resolution
- relation-extraction
- zero-shot
- knowledge-graph
- conversational-memory
- onnx
model-index:
{model_index}
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

A sibling model, [{other}](https://huggingface.co/{org}/{other}), shares the
design. Choose this one if you want {choose}.

**[GUIDE.md](GUIDE.md)** compares the two models task by task, lists the
failure modes the evaluations found, and shows how to work around each one
in your own code (schema design, thresholds, entity handling,
conversations, long inputs).

## Model details

| | |
|---|---|
| developed by | Dragonscale ([{org}](https://huggingface.co/{org})) |
| model type | multi-task encoder: token, span, pair and sentence classification heads on one transformer encoder, two LoRA adapters merged into encoder copies |
| version | kniv v7, release candidate 2 (`{run}`), 2026-10-20 |
| base model | [{base}](https://huggingface.co/{base}) ({encoder_params} parameters, {base_licence}) |
| parameters | {params} in total (encoder, two merged adapters' deltas, heads) |
| window | up to 512 words per window; {window} subwords read by the encoder |
| language | English |
| licence | Apache-2.0 |
| code | [{github}]({github}) (training, export, evaluation) and `kniv_v7/` in this repository (inference) |

## Uses

**Direct use.** Analyse English prose or conversations: parse, tag, label
semantic roles, resolve coreference, and extract entities and facts for a
schema of your own (relation and entity-type names, optionally with
one-sentence descriptions), with no fine-tuning.

**Downstream use.** A first stage for agent memory stores, knowledge-graph
construction, search indexing and document analytics; the linguistic layer as
features for other models. The facts it returns are candidates with
calibrated margins, not verified statements.

**Out of scope.**
- Languages other than English.
- Treating extracted facts as verified truth, or acting on them without
  review where errors matter: medical, legal, financial or employment
  decisions about people.
- Profiling or surveilling individuals from their conversations or documents
  without their consent.
- Relation or entity schemas far from the training domains (encyclopedic,
  news, business, technical, narrative, conversation) without checking quality
  on your own data first.

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

kniv = Kniv(path)                  # fp32; variant="fp16" or int8=True below
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

### Variants

| variant | load | encoder graphs | what for |
|---|---|---|---|
| fp32 | `Kniv(path)` | `onnx/{{plain,label,marker}}.onnx` | reference |
| FP16 | `Kniv(path, variant="fp16")` | `onnx/*-fp16.onnx` | half the download; GPUs (`providers=["CUDAExecutionProvider"]`) |
| INT8 | `Kniv(path, int8=True)` | `onnx/{{plain,marker}}-int8.onnx`, FP16 label graph | CPU speed |

**FP16** is the three encoder graphs exported in half precision, with the heads in fp32.
On the dev subsets it matches fp32 to within 0.001 (UD POS and LAS, Re-DocRED
relations, CrossNER spans). On CPU it is no faster than fp32. To download only what it needs:

```python
path = snapshot_download("{org}/{repo}", ignore_patterns=[
    "model.pt", "onnx/plain.onnx", "onnx/label.onnx", "onnx/marker.onnx"])
kniv = Kniv(path, variant="fp16")
```

**INT8** is {int8_how}. The label graph runs once per schema, so it stays FP16.
On the dev subsets no task drops more than {int8_drop} points, and on CPU it
runs {int8_speed}.

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
fp32 graph matches the PyTorch model to within 2e-4. On the release checks, the
runtime's decoded output over ONNX equals its output over PyTorch on every
test text.

## Evaluation

**Test data and metrics.** These are standard test splits, never used for any choice.
Thresholds and decoding settings were chosen on dev splits.

| benchmark | split | metric |
|---|---|---|
| UD English EWT | test | POS accuracy; lemma and full-morphology exact match; UAS / LAS |
| OntoNotes 5 (not trained on) | test | exact span F1, 18 types |
| PropBank on EWT | test | argument F1; "end to end" with the model's own predicates, every predicate |
| SGD | test turns, with up to 6 preceding turns as context | dialogue-act micro-F1 |
| LitBank | the 400-word windows | CoNLL F1 (mean of MUC, B3, CEAF-e), averaged per window |
| Re-DocRED | test | micro F1 over the 96 relations: with gold entities, and end to end (the model's own entities, matched by mention) |
| FewRel | train + validation instances of the 15 unseen relations | macro F1, m-way classification (5 seeds for m = 5 and 10) |
| held-out teacher relations | our corpus, 40 relations removed from training | ranking accuracy |
| CrossNER | test, politics used for calibration | exact span F1, mean of the other 4 domains |
| Few-NERD | 3,000-sentence test sample | exact span F1, 66 types |

**Results.**

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

FewRel's unseen relations are a fixed set removed from every training
source. The protocol is comparable in task and metric to published results,
not in which relations are unseen.

**Agent memory (LongMemEval, oracle sessions, 150 questions).** Facts are
extracted from the sessions with one generic memory schema. The probe measures
whether the answer reaches the extracted store; no LLM answers the question.

| answer in extracted entities | answer in extracted facts | questions with facts about the user | facts per question |
|---|---|---|---|
| {lme0} | {lme1} | {lme2} | {lme3} |

## Training

**Data.**

| source | licence | used for |
|---|---|---|
| Re-DocRED, DocRED distant (`thunlp/docred`) | MIT; Wikipedia text CC BY-SA | relations, entities |
| FewRel train | MIT | relations |
| Few-NERD (`DFKI-SLT/few-nerd`) | CC BY-SA 4.0 | entity spans |
| T-REx, Wikidata types and labels | CC BY-SA 4.0, CC0 | relation and type pretraining, label texts |
| UD English EWT with PropBank EWT | CC BY-SA 4.0 | POS, morphology, lemma, dependencies, SRL |
| GUM (academic, bio, court, interview, news, voyage) | CC BY 4.0 (text CC BY / BY-SA) | coreference |
| kniv corpus windows (6 domains), labelled by our own LLM annotators | ours | every head (silver labels); relations and fact attributes |
| kniv v5 gold-filtered NER sentences ([`dragonscale-ai/kniv-corpus-en`](https://huggingface.co/datasets/dragonscale-ai/kniv-corpus-en)) | ours | NER |

Excluded on licence grounds:
- non-commercial sets (DialogRE, REBEL, ReLiK);
- LDC data (OntoNotes, TACRED);
- third-party GPT-generated sets (SynthIE, NuNER).

**Procedure.**
1. **Knowledge pretraining** ({steps_a} steps), on the knowledge sources and our annotated corpus.
2. **Every head jointly** (10,000 steps), with replay.
3. **Rebalance and gold data.** A rebalanced pass (6,000 steps), then
   4,000 steps with UD EWT, PropBank and the gold NER sentences in the mix,
   selected on gold dev sets.
4. **Head fine-tunes with the encoder frozen.**
   - POS, morphology, lemma and dependencies on UD EWT.
   - SRL: an MLP predicate head and a predicate-aware transformer for roles.
   - NER: a BiLSTM on 100k gold-filtered sentences.
5. **Knowledge-only passes** (1,500 steps each), with the encoder and the
   linguistic layer asserted bit-identical:
   - one re-adapting the knowledge layer;
   - one with DREEAM's evidence loss weighted up.{hn}
6. **Coreference** retrained on GUM (encoder frozen), between the two
   knowledge passes.

1,423 relations and 1,559 entity types (names and descriptions) were sampled
as schemas during training.

**Hyperparameters.**

| | |
|---|---|
| precision | bf16 autocast |
| optimiser | AdamW (betas 0.9 / 0.98), weight decay 0.01 |
| learning rate | encoder {lr}, heads 1e-4 (knowledge passes: heads and adapters 1e-4) |
| schedule | linear warm-up over 6% of steps, then linear decay |
| batch | {batch} |
| adapters | LoRA rank 64 (relation pass), rank 16 (label pass) |
| loss weighting | per-head losses normalised by their mean over the first 50 steps, then fixed weights |

**Compute.** Single NVIDIA A100 40GB GPUs (Google Colab), {hours} GPU-hours
for this model's final lineage. That is measured from the run logs where they
survive and estimated from step rates where not. Earlier experiments, the
encoder bake-off and the evaluation runs are not counted. Head fine-tunes ran
on an Apple M2 Max.

## Environmental impact

| | |
|---|---|
| hardware | NVIDIA A100 40GB |
| hours | {hours} (final lineage) |
| cloud | Google Colab |
| region / carbon | not recorded |

## Bias, risks and limitations

**Data biases.**
- Much of the training text is Wikipedia and Wikidata. People, places and
  organisations from well-covered regions are recognised and related more
  reliably than others.
- Silver labels from our LLM annotators carry those models' biases.

**People and privacy.** The model is built to extract facts about people,
speakers included, from conversations. Store and use them with the consent
and data-protection care that implies.

**Errors.** Outputs are probabilistic; treat facts as candidates (`margin` is
the decision's distance from the calibrated threshold).

**Known gaps.**
- **Document-level relations trail specialists.** About 29% of Re-DocRED
  gold facts are missed. Most misses are yes/no decisions on a pair (the right
  relation ranked first, but under the threshold) or near-synonym confusions.
  The open schema costs about 3 points against a fixed-schema head.
- **Coreference** loses mostly on mention boundaries and on which noun
  phrases count as mentions. Pronouns are resolved well.
- **Entity grouping** is rule-based (exact names, plus coreference where
  calibrated). Partial names ("Obama" / "Barack Obama") are not merged.
- **Facts are per window** (up to 512 words); nothing is linked across
  windows. Text past the encoder's subword limit ({window} subwords) in a
  window is not analysed.
- **English only.**

**Recommendations.**
- Check precision on a sample of your own data and schema before relying on
  it. Raise the relation offsets in `calibration.json` for more precision.
- Keep a human in the loop where facts about people drive decisions.

## Licence

Apache-2.0 for the weights and the runtime code. The base encoder
([{base}](https://huggingface.co/{base})) is {base_licence}.

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
    text = CARD.format(model_index=model_index(m), lr=m["lr"], batch=m["batch"],
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
