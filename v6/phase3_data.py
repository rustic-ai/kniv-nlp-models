"""Phase 3 training sources from the Phase 2 labels (v8 plan r2, section 7).

    uv run python -m v6.phase3_data build      # writes the four files below
    uv run python -m v6.phase3_data show       # counts per file

Written beside the corpus (``data/v6-corpus/``), in the short-text export's
row format so ``kdata.teacher_features`` reads them unchanged:

- ``kg_short_d1.json`` (source ``short_d1``, D1): the 47,994 short training
  texts with their facts replaced by the judged ones. A teacher fact judged
  not supported is dropped; one judged negated takes negative polarity; one
  judged hedged keeps its relation but no attribute target (the judge's
  "hedged" spans four modalities). A fact only the model proposed, judged
  expressed, is added as a relation label without attributes. A fact with no
  verdict, and a text the judge never saw, keep the teacher's labels.
- ``kg_d2_concepts.json`` (source ``concepts``, D2): the business and legal
  passages written to order, with the teacher's blind facts plus the written
  facts it missed.
- ``label_para.json`` (D2): per bank relation id, the generated names not
  flagged (another bank relation's name, a P4 hand-written name, a CoNLL04,
  CrossRE or M2 relation name), described ones with their description.
- ``kg_d3_forms.json`` (source ``forms``, D3): the rewrites in six forms
  (denied, never, no longer, might, rumoured, expected), labels set by form.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

from .config import DATA_DIR, RUNS_DIR

CORPUS = DATA_DIR / "v6-corpus"
D1_LABELS = RUNS_DIR / "kg-judge" / "labels-d1-full-ultrabert-380m-v7b8ev.json"
D2_PARA = RUNS_DIR / "d2-labels" / "paraphrases-all.json"
D2_CONCEPTS = RUNS_DIR / "d2-labels" / "concepts-per40.json"
D3_FORMS = RUNS_DIR / "d3-forms" / "pairs.json"
OUT = {"short_d1": CORPUS / "kg_short_d1.json", "concepts": CORPUS / "kg_d2_concepts.json",
       "forms": CORPUS / "kg_d3_forms.json", "para": CORPUS / "label_para.json"}
EXPRESSED = ("supported", "negated", "hedged")


def d1_rows(c: Counter) -> list[dict]:
    rows = json.loads((CORPUS / "kg_short.json").read_text())["train"]
    labels = {r["id"]: r for r in json.loads(D1_LABELS.read_text())}
    assert len(labels) == len(rows) and set(labels) == {r["id"] for r in rows}, \
        "D1 labels are for another text set"
    out = []
    for r in rows:
        p, lab = r["payload"], labels[r["id"]]
        if not lab["judged"]:
            c["texts kept as the teacher's (not judged)"] += 1
            out.append(r)
            continue
        verdict = {(f["head"], f["relation"], f["tail"]): f["verdict"] for f in lab["facts"]}
        facts, have = [], set()
        for f in p["facts"]:
            k = (f["head"], f["relation"], f["tail"])
            v = verdict.get(k)
            if v == "not_supported":
                c["teacher facts dropped (not supported)"] += 1
                continue
            g = dict(f)
            if v == "negated" and g["polarity"] != "negative":
                g["polarity"] = "negative"
                c["teacher facts turned negative"] += 1
            if v == "hedged" and g["modality"] == "asserted":
                g["attr"] = False
                c["teacher facts hedged: no attribute target"] += 1
            if v is None:
                c["teacher facts without a verdict (kept)"] += 1
            facts.append(g)
            have.add(k)
        for f in lab["facts"]:
            k = (f["head"], f["relation"], f["tail"])
            if "teacher" in f["src"] or k in have or f["verdict"] not in EXPRESSED:
                continue
            facts.append({"head": k[0], "tail": k[2], "relation": k[1], "polarity": "positive",
                          "modality": "asserted", "holder": -1, "time": -1, "evidence": [],
                          "attr": False})
            have.add(k)
            c["model facts added (expressed)"] += 1
        c["texts judged"] += 1
        out.append({**r, "payload": {**p, "facts": facts}})
    return out


def para(c: Counter) -> dict:
    from .label_bank import load as load_bank
    bank = load_bank()
    by_name = {r["name"]: r for r in bank["relations"]}
    out = {}
    for name, names in json.loads(D2_PARA.read_text()).items():
        r = by_name.get(name)
        if r is None or not r.get("train", True):
            c["relations not trained (skipped)"] += 1
            continue
        keep = []
        for x in names:
            if x["flags"]:
                c["names flagged (kept out)"] += 1
                continue
            keep.append([x["name"], x["description"] if x["style"] == "described" else None])
        if keep:
            out[str(r["id"])] = keep
            c["names kept"] += len(keep)
    c["relations with names"] = len(out)
    return out


def build() -> int:
    c = Counter()
    d1 = d1_rows(c)
    OUT["short_d1"].write_text(json.dumps({"train": d1}))
    concepts = json.loads(D2_CONCEPTS.read_text())
    OUT["concepts"].write_text(json.dumps({"train": concepts}))
    c["concept passages"] = len(concepts)
    forms = json.loads(D3_FORMS.read_text())
    OUT["forms"].write_text(json.dumps({"train": forms}))
    c["form rewrites"] = len(forms)
    OUT["para"].write_text(json.dumps(para(c)))
    for k, v in sorted(c.items()):
        print(f"  {k}: {v:,}")
    for k, p in OUT.items():
        print(f"wrote {p} ({p.stat().st_size >> 10:,} KB)")
    return 0


def show() -> int:
    for k, p in OUT.items():
        d = json.loads(p.read_text())
        if k == "para":
            print(f"{k}: {len(d)} relations, {sum(len(v) for v in d.values())} names")
        else:
            rows = d["train"]
            print(f"{k}: {len(rows)} texts, {sum(len(r['payload']['facts']) for r in rows)} facts")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["build", "show"])
    args = ap.parse_args()
    return build() if args.cmd == "build" else show()


if __name__ == "__main__":
    raise SystemExit(main())
