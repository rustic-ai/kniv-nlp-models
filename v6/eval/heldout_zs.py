"""Held-out zero-shot relation extraction: CoNLL04 test and the short-text dev.

    uv run python -m v6.eval.heldout_zs --runs ultrabert-380m-v8e-w1b,ultrabert-380m-v7b8ev

Both sets score facts through ``Extractor`` with each run's deployed
calibration (``calibration.json``), in two modes:

- ``gold``: the gold entities are given, so only the relation step is scored;
  a fact is right when its head entity, relation and tail entity are.
- ``e2e``: the model finds the entities itself (typed by the set's entity
  types); a predicted fact is right when some mention of its head and of its
  tail equals (case-insensitive) a mention of a gold fact's head and tail,
  with the same relation. Each gold fact is matched once.

CoNLL04 is never trained on (its relations are not in the label bank, so all
five go through the new-relation path; ``kg_short`` drops training texts that
overlap it). The short-text dev is ``kg_short.json``'s ``dev`` split: 2,400
teacher-labelled texts from windows no training text shares, each with its
own schema.

Every candidate fact is kept with its margin, so the new-relation threshold
(``unseen_offset``) is also swept: ``--sweep`` prints F1 against the offset
per set, which is how the offset is refitted on the short-text dev (never on
CoNLL04). Two or more runs get a paired bootstrap per set and mode (texts
resampled, F1 difference against the first run's baseline, the last run).
"""
from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import numpy as np

from ..config import RUNS_DIR

CONLL04_RELATIONS = [
    {"name": "works for", "description": "a person is employed by or works for an organization"},
    {"name": "killed", "description": "a person killed another person"},
    {"name": "based in", "description": "an organization is based or headquartered in a location"},
    {"name": "lives in", "description": "a person lives or lived in a location"},
    {"name": "located in", "description": "a location is inside another location"},
]
CONLL04_REL = {"Work_For": "works for", "Kill": "killed", "OrgBased_In": "based in",
               "Live_In": "lives in", "Located_In": "located in"}
CONLL04_TYPES = [
    {"name": "person", "description": "a named person"},
    {"name": "organization", "description": "a company, agency, institution or other named group"},
    {"name": "location", "description": "a named place: country, city, region or landmark"},
]
CONLL04_TYPE = {"Peop": "person", "Org": "organization", "Loc": "location", "Other": "other"}
SHIFT = 20.0     # added to every offset so the extractor returns all candidates


def _lit(x):
    return ast.literal_eval(x) if isinstance(x, str) else x


def load_conll04(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    items = []
    for x in load_dataset("DFKI-SLT/conll04", split="test"):
        ents = [{"type": CONLL04_TYPE.get(e["type"], "other"), "spans": [[e["start"], e["end"]]]}
                for e in x["entities"]]
        facts = [{"head": r["head"], "tail": r["tail"], "relation": CONLL04_REL[r["type"]]}
                 for r in x["relations"]]
        items.append({"id": str(len(items)), "tokens": list(x["tokens"]), "participants": [],
                      "sentence_spans": [[0, len(x["tokens"])]], "entities": ents, "facts": facts,
                      "relations": CONLL04_RELATIONS, "types": CONLL04_TYPES})
        if limit and len(items) >= limit:
            break
    return items


def load_short_dev(limit: int | None = None) -> list[dict]:
    from ..label_bank import load as load_bank
    bank = load_bank()
    rel = {r["name"]: r for r in bank["relations"]}
    typ = {t["name"]: t for t in bank.get("entity_types", bank.get("types", []))}
    data = json.loads(Path("data/v6-corpus/kg_short.json").read_text())["dev"]
    items = []
    for x in data:
        p = _lit(x["payload"])
        items.append({
            "id": x["id"], "tokens": _lit(x["tokens"]), "sentence_spans": _lit(x["sentence_spans"]),
            "entities": p["entities"], "participants": list(p.get("participants") or []),
            "speakers": _lit(x.get("speakers")) or None,
            "facts": [{"head": f["head"], "tail": f["tail"], "relation": f["relation"]}
                      for f in p["facts"]],
            "relations": [{"name": n, "description": rel.get(n, {}).get("description", "")}
                          for n in _lit(x["relations"])],
            "types": [{"name": n, "description": typ.get(n, {}).get("description", "")}
                      for n in _lit(x["types"])],
            "domain": x.get("domain")})
        if limit and len(items) >= limit:
            break
    return items


def _mentions(it: dict, k: int) -> set[str]:
    """Fact argument ``k``: participants first, then the entities."""
    P = len(it.get("participants") or [])
    if k < P:
        return {it["participants"][k].lower()}
    return {" ".join(it["tokens"][a:b]).lower() for a, b in it["entities"][k - P]["spans"]}


def predict(ex, items: list[dict], mode: str) -> list[dict]:
    """Per item: candidate facts with their margin above the deployed bar
    (``m``: positive = predicted) and whether the relation takes the
    new-relation offset, plus the gold facts in the mode's terms."""
    known = set(ex.rel_offsets)
    out = []
    for it in items:
        w = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"]}
        if it.get("speakers"):
            w["sentence_speakers"] = it["speakers"]
        parts = it.get("participants") or []
        if mode == "gold":
            ents = [{"type": e["type"], "mentions": sorted(tuple(s) for s in e["spans"])}
                    for e in it["entities"]]
        else:
            ents = ex._entities(w, [t["name"] for t in it["types"]])
        nodes, facts = ex._relations(w, ents, list(it["relations"]), ex.top)
        first = {tuple(m): i for i, e in enumerate(ents) for m in e["mentions"]}

        def node_key(n: dict):
            if n.get("speaker"):
                return parts.index(n["text"]) if mode == "gold" and n["text"] in parts else \
                    frozenset([n["text"].lower()])
            if mode == "gold":
                return len(parts) + first[tuple(n["spans"][0])]
            return frozenset(m.lower() for m in n["mentions"])

        keys = [node_key(n) for n in nodes]
        cand = []
        for f in facts:
            cand.append({"key": (keys[f["head"]], f["relation"], keys[f["tail"]]),
                         "m": f["margin"] - SHIFT, "unseen": f["relation"] not in known})
        out.append({"cand": cand, "gold": it["facts"], "item": it})
    return out


def score_item(row: dict, mode: str, delta: float = 0.0) -> tuple[int, int, int]:
    """(true positives, predicted, gold) for one item. ``delta`` moves the
    new-relation bar: a candidate on that path is predicted when m > delta."""
    pred = [c for c in row["cand"] if c["m"] > (delta if c["unseen"] else 0.0)]
    gold = row["gold"]
    if mode == "gold":
        g = {(f["head"], f["relation"], f["tail"]) for f in gold}
        p = {c["key"] for c in pred}
        return len(p & g), len(p), len(g)
    it = row["item"]
    gm = [(_mentions(it, f["head"]), f["relation"], _mentions(it, f["tail"])) for f in gold]
    used, tp, seen = set(), 0, set()
    for c in pred:
        if c["key"] in seen:
            continue
        seen.add(c["key"])
        hs, r, ts = c["key"]
        for j, (gh, gr, gt) in enumerate(gm):
            if j not in used and gr == r and hs & gh and ts & gt:
                used.add(j)
                tp += 1
                break
    return tp, len(seen), len(gm)


def f1(c: np.ndarray) -> tuple[float, float, float]:
    tp, np_, ng = c.sum(0) if c.ndim == 2 else c
    p, r = tp / max(np_, 1), tp / max(ng, 1)
    return p, r, 2 * p * r / max(p + r, 1e-12)


def bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 13):
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        d.append(f1(a[i])[2] - f1(b[i])[2])
    d = np.array(d)
    return float(f1(a)[2] - f1(b)[2]), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, help="comma-separated; the last is the baseline")
    ap.add_argument("--sets", default="conll04,short_dev")
    ap.add_argument("--modes", default="gold,e2e")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device")
    ap.add_argument("--local", type=Path)
    ap.add_argument("--sweep", action="store_true", help="F1 against the new-relation offset")
    args = ap.parse_args()
    from ..infer import Extractor

    sets = {"conll04": load_conll04, "short_dev": load_short_dev}
    data = {s: sets[s](args.limit) for s in args.sets.split(",")}
    out_dir = RUNS_DIR / "heldout-zs" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict = {}
    summary: dict = {}
    for run in args.runs.split(","):
        ex = Extractor(run, device=args.device, local=args.local)
        dep = ex.unseen_offset
        ex.rel_offsets = {k: v + SHIFT for k, v in ex.rel_offsets.items()}
        ex.unseen_offset = dep - SHIFT
        ex.top = max(ex.top, 4)
        for s, items in data.items():
            for mode in args.modes.split(","):
                t0 = time.time()
                rows = predict(ex, items, mode)
                c = np.array([score_item(r, mode) for r in rows])
                counts[(run, s, mode)] = c
                p, r, f = f1(c)
                summary[f"{run}|{s}|{mode}"] = {"p": p, "r": r, "f1": f, "n": len(items),
                                                "unseen_offset": dep}
                print(f"{run} {s} {mode}: F1 {f:.3f} P {p:.3f} R {r:.3f} "
                      f"({len(items)} texts, {time.time() - t0:.0f}s)", flush=True)
                if args.sweep:
                    best = None
                    for d in np.round(np.arange(-4.0, 4.01, 0.25), 2):
                        cc = np.array([score_item(x, mode, d) for x in rows])
                        fd = f1(cc)[2]
                        best = max(best or (fd, d), (fd, d))
                        print(f"  unseen_offset {dep + d:+.2f}: F1 {fd:.3f}", flush=True)
                    summary[f"{run}|{s}|{mode}"]["sweep_best"] = {"f1": best[0],
                                                                  "unseen_offset": dep + best[1]}
        del ex
    runs = args.runs.split(",")
    if len(runs) > 1:
        base = runs[-1]
        print(f"\npaired bootstrap against {base} (F1 difference, 95% interval)")
        for run in runs[:-1]:
            for s in data:
                for mode in args.modes.split(","):
                    d, lo, hi = bootstrap(counts[(run, s, mode)], counts[(base, s, mode)])
                    summary[f"{run}-vs-{base}|{s}|{mode}"] = {"diff": d, "lo": lo, "hi": hi}
                    print(f"  {run} {s} {mode}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
