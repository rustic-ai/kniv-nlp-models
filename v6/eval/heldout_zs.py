"""Held-out zero-shot relation extraction: CoNLL04 test and the short-text dev.

    uv run python -m v6.eval.heldout_zs --runs ultrabert-380m-v8e-w1b,ultrabert-380m-v7b8ev

Both sets score facts through ``Extractor`` with each run's deployed
calibration (``calibration.json``), in two modes:

- ``gold``: the gold entities are given, so only the relation step is scored;
  a fact is right when its head entity, relation and tail entity are.
- ``e2e``: the model finds the entities itself (typed by the set's entity
  types, through the zero-shot span head); a predicted fact is right when some mention of its head and of its
  tail equals (case-insensitive) a mention of a gold fact's head and tail,
  with the same relation. Each gold fact is matched once.
- ``e2e_ent`` (diagnostic): as ``e2e`` but with the entity-mention head
  trained on Re-DocRED (no types), to tell the span head's misses from the
  relation step's.

End-to-end modes also print entity mention F1 and the share of gold facts
whose two arguments were both found, the relation step's ceiling.

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


def load_short_dev(limit: int | None = None, split: str = "dev", keep=None) -> list[dict]:
    from ..label_bank import load as load_bank
    bank = load_bank()
    rel = {r["name"]: r for r in bank["relations"]}
    typ = {t["name"]: t for t in bank.get("entity_types", bank.get("types", []))}
    data = json.loads(Path("data/v6-corpus/kg_short.json").read_text())[split]
    items = []
    for x in data:
        if keep is not None and not keep(x["id"]):
            continue
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
        elif mode == "e2e_ent":
            ents = ex._entities(w, None)
        else:
            ents = ex._entities(w, it["types"])
        nodes, facts = ex._relations(w, ents, list(it["relations"]), ex.top)
        if getattr(ex, "merge_variants", False):    # margins carry SHIFT here
            from ..infer import merge_variants
            facts = merge_variants(nodes, facts, bar=SHIFT)
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
                         "m": f["margin"] - SHIFT, "unseen": f["relation"] not in known,
                         "p": f.get("p")})
        out.append({"cand": cand, "gold": it["facts"], "item": it,
                    "ents": [(e.get("type"), [tuple(m) for m in e["mentions"]]) for e in ents]})
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


def entity_stats(rows: list[dict]) -> dict:
    """Where the end-to-end loss starts: the predicted entities against the
    gold ones, before any relation is decided. ``mention``: exact spans,
    type ignored; ``typed``: span and type; ``reach``: gold facts whose head
    and tail both have a mention among the predicted entities (case-
    insensitive text, as ``score_item`` matches), the relation step's ceiling."""
    m = np.zeros(3)
    t = np.zeros(3)
    reach = np.zeros(2)
    for r in rows:
        it = r["item"]
        gs = {(a, b) for e in it["entities"] for a, b in e["spans"]}
        gt = {(e["type"], a, b) for e in it["entities"] for a, b in e["spans"]}
        ps = {m_ for _, ms in r["ents"] for m_ in ms}
        pt = {(ty, a, b) for ty, ms in r["ents"] for a, b in ms}
        m += [len(ps & gs), len(ps), len(gs)]
        t += [len(pt & gt), len(pt), len(gt)]
        texts = {" ".join(it["tokens"][a:b]).lower() for a, b in ps}
        for f in r["gold"]:
            reach += [bool(_mentions(it, f["head"]) & texts and _mentions(it, f["tail"]) & texts), 1]
    return {"mention": f1(m), "typed": f1(t), "reach": reach[0] / max(reach[1], 1)}


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


def half(it: dict) -> int:
    """Short-dev half: 0 = the half the refitted offset is chosen on."""
    import hashlib
    return hashlib.sha1(it["id"].encode()).digest()[0] % 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True, help="comma-separated; the last is the baseline")
    ap.add_argument("--sets", default="conll04,short_dev")
    ap.add_argument("--modes", default="gold,e2e")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--device")
    ap.add_argument("--local", type=Path)
    ap.add_argument("--sweep", action="store_true", help="F1 against the new-relation offset")
    ap.add_argument("--refit", action="store_true",
                    help="also choose each run's new-relation offset on half the short dev "
                         "(gold entities) and score the other half and CoNLL04 at it")
    ap.add_argument("--cache", type=Path, help="candidate facts per run: read if there, else written")
    ap.add_argument("--span-boost", type=float,
                    help="typed entities: also propose the entity-mention head's spans, margin "
                         "raised by this (overrides calibration.json's span_boost)")
    args = ap.parse_args()
    import pickle
    from ..infer import Extractor

    sets = {"conll04": load_conll04, "short_dev": load_short_dev}
    data = {s: sets[s](args.limit) for s in args.sets.split(",")}
    out_dir = RUNS_DIR / "heldout-zs" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = args.runs.split(",")
    modes = args.modes.split(",")
    rows: dict = {}
    deployed: dict = {}
    for run in runs:
        cp = args.cache / f"{run}.pkl" if args.cache else None
        if cp and cp.exists():
            deployed[run], rows[run] = pickle.loads(cp.read_bytes())
            print(f"{run}: candidates from {cp}", flush=True)
            continue
        import os
        os.environ.pop("KNIV_UNSEEN_FROM", None)
        if "+e1u" in run:                           # "...+e1u": never-trained relations through E1b's relation step
            os.environ["KNIV_UNSEEN_FROM"] = "ultrabert-380m-v8e-w1half"
        base_run, _, ver = run.replace("+det", "").replace("+e1u", "").replace("+v1c", "").replace("+v2", "").partition("+ver@")   # "RUN+ver@T": RUN with its verifier at threshold T (plan r4 stage 3)
        ex = Extractor(base_run, device=args.device, local=args.local, span_boost=args.span_boost)
        ex.trim_det = "+det" in run                 # "...+det": leading articles trimmed from mentions
        deployed[run] = dep = ex.unseen_offset
        ex.rel_offsets = {k: v + SHIFT for k, v in ex.rel_offsets.items()}
        ex.unseen_offset = dep - SHIFT
        ex.top = max(ex.top, 4)
        if ver:
            vname = f"{base_run}-ver2" if "+v2" in run else f"{base_run}-ver1{'c' if '+v1c' in run else ''}"
            ex.use_verifier(vname, threshold=float(ver))   # "+v1c": V1c; "+v2": the shared-pass verifier
            ex.margin_shift = SHIFT
        rows[run] = {}
        for s, items in data.items():
            for mode in modes:
                t0 = time.time()
                rows[run][(s, mode)] = predict(ex, items, mode)
                print(f"{run} {s} {mode}: predicted ({len(items)} texts, "
                      f"{time.time() - t0:.0f}s)", flush=True)
        del ex
        if cp:
            cp.parent.mkdir(parents=True, exist_ok=True)
            cp.write_bytes(pickle.dumps((deployed[run], rows[run])))

    def counts(run, s, mode, delta=0.0, part=None):
        return np.array([score_item(r, mode, delta) for r in rows[run][(s, mode)]
                         if part is None or half(r["item"]) == part])

    summary: dict = {}
    for run in runs:
        dep = deployed[run]
        for s in data:
            for mode in modes:
                p, r, f = f1(counts(run, s, mode))
                summary[f"{run}|{s}|{mode}"] = {"p": p, "r": r, "f1": f, "unseen_offset": dep}
                print(f"{run} {s} {mode}: F1 {f:.3f} P {p:.3f} R {r:.3f} (offset {dep:+.2f})")
                if mode != "gold" and "ents" in rows[run][(s, mode)][0]:
                    es = entity_stats(rows[run][(s, mode)])
                    summary[f"{run}|{s}|{mode}|entities"] = es
                    print(f"  entities: mention F1 {es['mention'][2]:.3f} (P {es['mention'][0]:.3f} "
                          f"R {es['mention'][1]:.3f}), typed F1 {es['typed'][2]:.3f}, "
                          f"gold facts with both arguments found {es['reach']:.3f}")
                if args.sweep:
                    for d in np.round(np.arange(-4.0, 4.01, 0.25), 2):
                        print(f"  unseen_offset {dep + d:+.2f}: "
                              f"F1 {f1(counts(run, s, mode, d))[2]:.3f}")
    deltas = {r: 0.0 for r in runs}
    if args.refit and "short_dev" in data:
        print("\nrefit: offset chosen on short-dev half 0 (gold entities), scored on half 1 "
              "and CoNLL04")
        for run in runs:
            grid = np.round(np.arange(-4.0, 4.01, 0.25), 2)
            fs = [f1(counts(run, "short_dev", "gold", d, part=0))[2] for d in grid]
            deltas[run] = float(grid[int(np.argmax(fs))])
            off = deployed[run] + deltas[run]
            summary[f"{run}|refit"] = {"unseen_offset": off, "search_f1": max(fs)}
            print(f"{run}: unseen_offset {off:+.2f} (search half F1 {max(fs):.3f})")
            for s in data:
                for mode in modes:
                    part = 1 if s == "short_dev" else None
                    p, r, f = f1(counts(run, s, mode, deltas[run], part))
                    summary[f"{run}|{s}|{mode}|refit"] = {"p": p, "r": r, "f1": f}
                    print(f"  {s}{' half 1' if part else ''} {mode}: F1 {f:.3f} P {p:.3f} R {r:.3f}")
    if len(runs) > 1:
        base = runs[-1]
        for tag, use in (("deployed offsets", False), ("refitted offsets", args.refit)):
            if not use and tag != "deployed offsets":
                continue
            print(f"\npaired bootstrap against {base}, {tag} (F1 difference, 95% interval)")
            for run in runs[:-1]:
                for s in data:
                    for mode in modes:
                        part = 1 if use and s == "short_dev" else None
                        a = counts(run, s, mode, deltas[run] if use else 0.0, part)
                        b = counts(base, s, mode, deltas[base] if use else 0.0, part)
                        d, lo, hi = bootstrap(a, b)
                        summary[f"{run}-vs-{base}|{s}|{mode}|{tag}"] = {"diff": d, "lo": lo, "hi": hi}
                        print(f"  {run} {s}{' half 1' if part else ''} {mode}: "
                              f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
