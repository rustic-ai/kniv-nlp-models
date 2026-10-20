"""The windows both teachers score for the v7 students (docs/v7-students-plan.md,
step 3): one fixed list, so the two teachers' targets line up exactly.

    uv run python -m v6.train.student_windows [--gold-ner 50000]

Each window: ``id``, ``source``, ``tokens``, ``sentence_spans``,
``speakers`` (per sentence, or none), ``entities`` (``{type, spans}`` with
``[start, end)`` word spans; empty: the window gets linguistic targets
only), ``relations`` and ``types`` (the schema the teachers score against).

Sources, training splits only (every dev and test item stays out):
- ``teacher``: our annotated corpus windows (the annotation's entities and
  schemas; a window annotated against two schemas gets both);
- ``corpus``: corpus windows outside the annotation, with their entity
  clusters, scored against the 96 calibrated relations;
- ``distant``: the 8,000 DocRED-distant documents no model trained on;
- ``redocred``: Re-DocRED train;
- ``ud``, ``propbank``, ``ner``: gold linguistic sentences (no entities).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from ..config import DATA_DIR

CORPUS = DATA_DIR / "v6-corpus"
OUT = CORPUS / "student_windows.jsonl"


def calibrated() -> list[str]:
    from ..label_bank import load as load_bank
    return [r["name"] for r in load_bank()["relations"] if r["source"] == "redocred"]


def teacher_windows() -> list[dict]:
    from .kdata import _is_dev
    items = json.loads((CORPUS / "kg_teacher.json").read_text())["items"]
    by: dict[str, dict] = {}
    for r in items:
        if _is_dev(r["id"]):
            continue
        wid = r["id"].split(":")[0]
        ents = [{"type": e["type"], "spans": [tuple(s) for s in e["spans"]]}
                for e in r["payload"].get("entities", [])]
        if wid not in by:
            by[wid] = {"id": wid, "source": "teacher", "domain": r["domain"], "tokens": r["tokens"],
                       "sentence_spans": r["sentence_spans"], "speakers": r.get("speakers") or [],
                       "entities": ents, "relations": list(r["relations"]), "types": list(r["types"])}
            continue
        w = by[wid]                                   # a second schema for the same window
        w["relations"] = list(dict.fromkeys(w["relations"] + r["relations"]))
        w["types"] = list(dict.fromkeys(w["types"] + r["types"]))
        taken = {s for e in w["entities"] for s in e["spans"]}
        w["entities"] += [e for e in ents if not set(e["spans"]) & taken]
    return list(by.values())


def corpus_windows(skip: set[str], rels: list[str]) -> list[dict]:
    """ATLOP-era entity clusters of corpus windows outside the annotation
    (training split by the corpus's own split)."""
    out = []
    train = set()
    for f in (CORPUS / "corpus" / "split=train").rglob("*.parquet"):
        import pandas as pd
        train |= set(pd.read_parquet(f, columns=["window_id"])["window_id"])
    for d in json.loads((CORPUS / "entities_docred.json").read_text()):
        if d["title"] in skip or d["title"] not in train:
            continue
        toks, spans = [], []
        for s in d["sents"]:
            spans.append([len(toks), len(toks) + len(s)])
            toks += s
        ents = [{"type": v[0]["type"], "spans": sorted({(spans[m["sent_id"]][0] + m["pos"][0],
                                                          spans[m["sent_id"]][0] + m["pos"][1]) for m in v})}
                for v in d["vertexSet"]]
        out.append({"id": d["title"], "source": "corpus", "tokens": toks, "sentence_spans": spans,
                    "speakers": [], "entities": ents, "relations": rels, "types": []})
    return out


def doc_windows(rels: list[str]) -> list[dict]:
    from .data import REDOCRED_REL
    out = []
    for i, d in enumerate(json.loads((CORPUS / "kd_distant.json").read_text())["train"]):
        out.append({"id": f"distant-{i}", "source": "distant", "tokens": d["tokens"],
                    "sentence_spans": d["sentence_spans"], "speakers": [],
                    "entities": [{"type": e["type"], "spans": [tuple(m) for m in e["mentions"]]}
                                 for e in d["entities"]], "relations": rels, "types": []})
    for i, d in enumerate(json.loads(REDOCRED_REL.read_text())["train"]):
        out.append({"id": f"redocred-{i}", "source": "redocred", "tokens": d["tokens"],
                    "sentence_spans": d["sentence_spans"], "speakers": [],
                    "entities": [{"type": e["type"], "spans": [(a, b + 1) for a, b in e["mentions"]]}
                                 for e in d["entities"]], "relations": rels, "types": []})
    return out


def gold_sentences(n_ner: int) -> list[dict]:
    from ..gold.ud_ewt import load_ud_items
    from .golddata import v5ner_rows
    out = []
    for i, it in enumerate(load_ud_items("train", max_tokens=200)):
        out.append({"id": f"ud-{i}", "source": "ud", "tokens": it.tokens,
                    "sentence_spans": [[0, len(it.tokens)]]})
    pb = json.loads((DATA_DIR / "prepared" / "propbank-full" / "train.json").read_text())
    for i, r in enumerate(pb):
        out.append({"id": f"propbank-{i}", "source": "propbank", "tokens": r["words"],
                    "sentence_spans": [[0, len(r["words"])]]})
    for r in v5ner_rows("train")[:n_ner]:
        out.append({"id": r["window_id"] if "window_id" in r else f"ner-{len(out)}", "source": "ner",
                    "tokens": r["tokens"], "sentence_spans": [[0, len(r["tokens"])]]})
    for w in out:
        w.update(speakers=[], entities=[], relations=[], types=[])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold-ner", type=int, default=50000)
    args = ap.parse_args()
    rels = calibrated()
    tw = teacher_windows()
    ws = tw + corpus_windows({w["id"] for w in tw}, rels) + doc_windows(rels) + gold_sentences(args.gold_ner)
    with OUT.open("w") as fh:
        for w in ws:
            fh.write(json.dumps(w) + "\n")
    c = Counter(w["source"] for w in ws)
    words = Counter()
    for w in ws:
        words[w["source"]] += len(w["tokens"])
    print(f"wrote {OUT}: " + ", ".join(f"{s} {c[s]:,} windows / {words[s]:,} words" for s in c), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
