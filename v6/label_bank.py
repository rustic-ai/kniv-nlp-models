"""The label bank: every relation and entity type the knowledge layer has a
text for, with stable ids.

    uv run python -m v6.label_bank          # writes data/v6-corpus/label_bank.json

The schema-conditioned heads read a label as text -- its name and
description -- so a caller can pass labels the model never trained on. This
file is the training side of that: the labels training sees, the texts it
reads them by, and the labels held out of training to measure exactly that
ability.

Ids are stable and append-only. Relations ``0..95`` are Re-DocRED's in the
vocabulary's order, so ``pair_labels`` built from the vocabulary index the
bank directly. Then FewRel's training relations Re-DocRED lacks, then
FewRel's validation relations no training source has -- those carry
``train: false`` and are only ever evaluated (true zero-shot). Then the
discovered general relations (``general_relations.json``) and the business
pack (``business_relations.json``), which the teacher annotates against; a
name already in the bank keeps its first entry. These are training
material, not a fixed ontology: a caller passes its own labels by text.

``holdout``: a fixed set of 20 Re-DocRED relations, chosen once by hash
among those with at least 20 test and 10 dev triples. They are excluded from
training only in the zero-shot diagnostic run (``--label-holdout``); every
other run trains on all 96.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from .config import DATA_DIR

BANK_PATH = DATA_DIR / "v6-corpus" / "label_bank.json"
VERSION = 4
N_HOLDOUT = 20

# DocRED's coarse entity types, described for the label pass.
DOCRED_TYPES = {
    "PER": ("person", "a human being, real or fictional, named or referred to"),
    "ORG": ("organization", "a company, institution, team, government body, band or other group"),
    "LOC": ("location", "a place: a country, city, region, building, natural feature or address"),
    "TIME": ("time", "a date, time, period or other temporal expression"),
    "NUM": ("number", "a quantity, amount of money, percentage or other numeric value"),
    "MISC": ("miscellaneous", "a named thing of another kind: a work, event, product, language, law or nationality"),
}


def _holdout(codes: list[str], counts: Counter) -> list[str]:
    eligible = [c for c in codes if counts[("test", c)] >= 20 and counts[("dev", c)] >= 10]
    eligible.sort(key=lambda c: hashlib.sha1(f"holdout:{c}".encode()).hexdigest())
    return sorted(eligible[:N_HOLDOUT])


def build() -> dict:
    from .gold.redocred import download, relation_inventory, resolve_relation_info
    names, by_name = relation_inventory()
    info = resolve_relation_info(sorted(by_name.values()))
    counts = Counter()
    for sp in ("dev", "test"):
        for d in json.loads(download(sp).read_text()):
            for lab in d["labels"]:
                counts[(sp, lab["r"])] += 1
    held = set(_holdout(sorted(by_name.values()), counts))
    from .gold import fewrel
    fr_train, fr_val = fewrel.codes("train"), fewrel.codes("val")
    relations = []
    for i, name in enumerate(names):            # vocabulary order: id == vocab index
        code = by_name[name]
        wd = info[code]
        aliases = list(wd["aliases"])
        if wd["label"] != name:                 # Wikidata relabelled it since
            aliases = [wd["label"]] + aliases
        relations.append({"id": i, "name": name, "description": wd["description"],
                          "aliases": aliases, "source": "redocred", "code": code,
                          "holdout": code in held, "train": True,
                          "fewrel": "train" if code in fr_train else
                                    "val" if code in fr_val else None})
    have = {r["code"] for r in relations}
    extra = [(c, "train") for c in fr_train if c not in have] + \
            [(c, "val") for c in fr_val if c not in have]
    finfo = resolve_relation_info([c for c, _ in extra])
    for code, split in extra:
        wd = finfo[code]
        relations.append({"id": len(relations), "name": wd["label"],
                          "description": wd["description"], "aliases": wd["aliases"],
                          "source": "fewrel", "code": code, "holdout": False,
                          "train": split == "train", "fewrel": split})
    if len({r["name"] for r in relations}) != len(relations):
        raise RuntimeError("two relations share a name")
    seen = {r["name"].lower() for r in relations}
    for fname, source in (("general_relations.json", "general"),
                          ("business_relations.json", "business")):
        path = DATA_DIR / "v6-corpus" / fname
        if not path.exists():
            continue
        for r in json.loads(path.read_text())["relations"]:
            if r["name"].lower() in seen:
                continue
            seen.add(r["name"].lower())
            relations.append({"id": len(relations), "name": r["name"],
                              "description": r["description"], "aliases": r.get("aliases") or [],
                              "source": source, "code": None, "holdout": False, "train": True,
                              "category": r.get("category") or "other",
                              "domains": r.get("domains") or {},
                              "inverse": r.get("inverse")})
    types = [{"id": i, "name": name, "description": desc, "aliases": [], "source": "docred",
              "code": code, "mergeable": code in ("PER", "ORG", "LOC", "MISC"),
              "temporal": code == "TIME"}
             for i, (code, (name, desc)) in enumerate(DOCRED_TYPES.items())]
    # Few-NERD's 66 fine types (named entities: mergeable by surface form).
    from .gold import fewnerd
    for code in fewnerd.fine_types():
        name, desc = fewnerd.type_text(code)
        types.append({"id": len(types), "name": name, "description": desc, "aliases": [],
                      "source": "fewnerd", "code": code, "mergeable": True, "temporal": False})
    if len({t["name"] for t in types}) != len(types):
        raise RuntimeError("two entity types share a name")
    bank = {"version": VERSION, "relations": relations, "entity_types": types}
    bank["hash"] = hashlib.sha1(json.dumps(bank, sort_keys=True).encode()).hexdigest()[:12]
    return bank


def load(path: Path = BANK_PATH) -> dict:
    if not path.exists():
        raise SystemExit(f"{path} missing; run python -m v6.label_bank")
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=BANK_PATH)
    args = ap.parse_args()
    bank = build()
    args.out.write_text(json.dumps(bank, indent=1))
    held = [r["name"] for r in bank["relations"] if r["holdout"]]
    zs = [r["name"] for r in bank["relations"] if not r["train"]]
    print(f"never trained (FewRel val, zero-shot) ({len(zs)}): {', '.join(zs)}")
    print(f"{len(bank['relations'])} relations, {len(bank['entity_types'])} entity types, "
          f"hash {bank['hash']}")
    print(f"held out ({len(held)}): {', '.join(held)}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
