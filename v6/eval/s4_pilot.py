"""Stage 4 pilot (v8 plan r4): where the margin and the verifier disagree,
on long windows across eight text types.

    uv run python -m v6.eval.s4_pilot select [--per 200]
    uv run python -m v6.eval.s4_pilot predict --part K [--size 50]
    uv run python -m v6.eval.s4_pilot judge [--cap 437] [--first N]
    uv run python -m v6.eval.s4_pilot read

``select``: windows of the teacher corpus (``kg_teacher.json``, each with
its schema) of 300 tokens or more, outside the verifier's held-out fifth and
documents any evaluation text comes from; windows no short text was cut
from first, then training windows short texts were cut from (the verifier
saw a few sentences of those, never the window); ``per`` per text type,
seed 7. The types are the corpus domains
(conversation, technical, narrative, news, encyclopedic) and, within
business, SEC filings, contracts (CUAD) and paper abstracts (S2ORC).

``predict``: d123's deployed flat decode with V1b at 0.6 (the verifier's
floor 3, as stage 3; of those candidates a random 40 per text are verified,
the rest left to the margin), logging every verified candidate's margin before the
verifier, its probability and its verdict. Slices of ``size`` texts, each in
its own process (MPS memory grows with distinct windows).

``judge``: a text goes to the judge (GPT-6, c3) when some verified
candidate is above the bar on one path and below on the other
(margin > 0 vs. p >= 0.6); a random 30 of its disagreeing candidates (all if fewer) go in one call.
At most ``cap`` texts per type, in the selection's order.

``read``: per type, texts with a disagreement, disagreements per text, and
who was right: the verifier's decision against the judge's on the
disagreeing candidates (0.5 = the verifier guesses there).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..config import RUNS_DIR, load_annotators

OUT = RUNS_DIR / "s4-pilot"
RUN, VER, THR = "ultrabert-380m-v8p3-d123", "ultrabert-380m-v8p3-d123-ver1", 0.6
VER_SAMPLE, JUDGE_MAX = 40, 30    # candidates verified per text, disagreements judged per call
TYPES = ["conversation", "technical", "narrative", "news", "encyclopedic", "filings", "legal", "science"]
BUSINESS = {"sec_edgar": "filings", "cuad": "legal", "s2orc": "science"}


def _lit(v):
    import ast
    return v if not isinstance(v, str) else ast.literal_eval(v)


def select(per: int) -> int:
    from ..label_bank import load as load_bank
    from .negation_pilot import heldout_base
    bank = load_bank()
    rel = {r["name"]: r for r in bank["relations"]}
    typ = {t["name"]: t for t in bank.get("entity_types", bank.get("types", []))}
    W = {}
    for line in open("data/v6-corpus/windows.jsonl"):
        w = json.loads(line)
        W[w["window_id"]] = w
    S = json.loads(Path("data/v6-corpus/kg_short.json").read_text())
    short_src = {e["id"].split("#")[0] for sp in ("train", "dev") for e in S[sp]}
    eval_docs = {W[e["id"].split("#")[0]]["doc_id"] for e in S["dev"] if e["id"].split("#")[0] in W}
    T = json.loads(Path("data/v6-corpus/kg_teacher.json").read_text())["items"]
    pools = defaultdict(list)
    for x in T:
        wid = x["id"].split(":")[0]
        w = W.get(wid)
        if w is None or w["doc_id"] in eval_docs or heldout_base(wid):
            continue
        tokens = _lit(x["tokens"])
        if len(tokens) < 300:
            continue
        kind = BUSINESS.get(w["source"]) if x["domain"] == "business" else x["domain"]
        if kind not in TYPES:
            continue
        pools[kind].append((wid in short_src, x, tokens))
    rng = random.Random(7)
    items = []
    for kind in TYPES:
        pool = pools[kind]
        rng.shuffle(pool)
        pool.sort(key=lambda z: z[0])       # windows no short text was cut from first
        for _, x, tokens in pool[:per]:
            items.append({"id": x["id"], "kind": kind, "domain": x["domain"], "tokens": tokens,
                          "sentence_spans": _lit(x["sentence_spans"]),
                          "speakers": _lit(x.get("speakers")) or None,
                          "participants": list(_lit(x.get("participants")) or []),
                          "relations": [{"name": n, "description": rel.get(n, {}).get("description", "")}
                                        for n in _lit(x["relations"])],
                          "types": [{"name": n, "description": typ.get(n, {}).get("description", "")}
                                    for n in _lit(x["types"])]})
        print(f"{kind}: {len(pool)} eligible ({sum(not z[0] for z in pool)} unused by short texts), "
              f"{min(per, len(pool))} taken")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "items.json").write_text(json.dumps(items))
    return 0


def predict(part: int, size: int, ver: str = VER, full: bool = False, tag: str = "") -> int:
    """``tag``: another verifier (plan r4 readiness A3) on the windows with
    judged disagreements only, written as ``ver-TAG-NNN.pkl``; ``full``:
    every candidate verified (the list verifier reads them all, as deployed),
    otherwise the same seeded sample as the pilot."""
    import torch
    from ..infer import Extractor
    items = json.loads((OUT / "items.json").read_text())
    if tag:
        judged = {t["id"] for t in json.loads((OUT / "labels-astra-c3.json").read_text()) if t["judged"]}
        items = [it for it in items if it["id"] in judged]
    items = items[part * size:(part + 1) * size]
    if not items:
        print("empty part")
        return 0
    ex = Extractor(RUN)
    ex.use_verifier(ver, threshold=THR)
    ex.margin_shift = 0.0
    orig = ex._verify
    cur = {}

    def sampled(w, nodes, spans, parts, cands):
        """A random ``VER_SAMPLE`` of the candidates (seeded by text): the rest
        get no probability, so the margin decides them and they cannot
        disagree. Long windows carry hundreds of candidates each."""
        rng = random.Random(cur["id"])
        pick = sorted(rng.sample(range(len(cands)), min(VER_SAMPLE, len(cands))))
        got = orig(w, nodes, spans, parts, [cands[j] for j in pick])
        verd = ex._last_verdicts
        out = [None] * len(cands)
        ex._last_verdicts = {}
        for i, j in enumerate(pick):
            out[j] = got[i]
            if i in verd:
                ex._last_verdicts[j] = verd[i]
        return out
    if not full:
        ex._verify = sampled
    rows = []
    for it in items:
        cur["id"] = it["id"]
        w = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"]}
        if it.get("speakers"):
            w["sentence_speakers"] = it["speakers"]
        ex.ver_log = []
        with torch.no_grad():
            ents = ex._entities(w, it["types"])
            nodes, facts = ex._relations(w, ents, list(it["relations"]), ex.top)
        names = [[n["text"]] if n.get("speaker") else n["mentions"] for n in nodes]
        cands = [{**c, "head_m": sorted(set(names[c["head"]])), "tail_m": sorted(set(names[c["tail"]]))}
                 for c in ex.ver_log]
        rows.append({"id": it["id"], "kind": it["kind"], "cands": cands, "n_facts": len(facts)})
        if ex.device.type == "mps":
            torch.mps.empty_cache()
    (OUT / (f"ver-{tag}-{part:03d}.pkl" if tag else f"pred-{part:03d}.pkl")).write_bytes(pickle.dumps(rows))
    print(f"part {part}: {len(rows)} texts, {sum(len(r['cands']) for r in rows)} verified candidates")
    return 0


def _rows():
    rows = []
    for p in sorted(OUT.glob("pred-*.pkl")):
        rows += pickle.loads(p.read_bytes())
    return rows


def _disagree(c) -> bool:
    return c["p"] is not None and (c["m"] > 0) != (c["p"] >= THR)


def judge(cap: int, first: int = 0) -> int:
    from ..annotate.base import CacheStore
    from ..annotate.kg import KGItem
    from ..kg_judge import CandJudge, _run
    from ..kg_teacher import CACHE
    from .m1_precision import PROMPTS, judge_prompt
    items = {it["id"]: it for it in json.loads((OUT / "items.json").read_text())}
    per, todo = Counter(), []
    for r in _rows():
        dis = [c for c in r["cands"] if _disagree(c)]
        if not dis or per[r["kind"]] >= cap:
            continue
        per[r["kind"]] += 1
        todo.append((items[r["id"]], dis))
    if first:
        todo = todo[:first]
    q_items = []
    todo = [(it, sorted(random.Random(it["id"]).sample(dis, min(JUDGE_MAX, len(dis))),
                        key=lambda c: (c["head"], c["tail"], c["relation"]))) for it, dis in todo]
    for it, dis in todo:
        facts = [{"key": (c["head_m"], c["relation"], c["tail_m"])} for c in dis]
        q = KGItem(id=f"{it['id']}:s4", tokens=it["tokens"], sentence_spans=it["sentence_spans"],
                   speakers=it.get("speakers"), participants=it["participants"], types=it["types"],
                   relations=it["relations"], domain=it.get("domain") or "")
        q.layers = {"judge_prompt": judge_prompt(it, facts)}
        q_items.append(q)
    print(f"{len(todo)} texts to judge ({dict(per)}), {sum(len(d) for _, d in todo)} candidates", flush=True)
    spec = load_annotators(["astra"])["astra"]
    ann = CandJudge(spec, CacheStore(CACHE, "c3"))
    ann.prompt = PROMPTS["c3"]
    res = asyncio.run(_run(ann, q_items, "s4-astra-c3"))
    out = []
    for it, dis in todo:
        r = res.get(f"{it['id']}:s4")
        v = r.payload if r is not None and r.ok else {}
        out.append({"id": it["id"], "kind": it["kind"], "judged": bool(v),
                    "cands": [{**c, "judge": v.get(n)} for n, c in enumerate(dis)]})
    tag = f"-first{first}" if first else ""
    (OUT / f"labels-astra-c3{tag}.json").write_text(json.dumps(out))
    print(Counter(c["judge"] for t in out for c in t["cands"]))
    return 0


def read() -> int:
    from .m1_precision import POSITIVE
    rows = _rows()
    L = json.loads((OUT / "labels-astra-c3.json").read_text())
    by_kind = defaultdict(list)
    for r in rows:
        by_kind[r["kind"]].append(r)
    lab = defaultdict(list)
    for t in L:
        if t["judged"]:
            lab[t["kind"]].append(t)
    print("| type | texts | verified / text | texts with a disagreement | disagreements / text | judged texts "
          "| verifier right | when it keeps (margin drops) | when it drops (margin keeps) |")
    print("|---|---|---|---|---|---|---|---|---|")
    tot = Counter()
    for kind in TYPES:
        rs = by_kind.get(kind, [])
        if not rs:
            continue
        nd = [sum(_disagree(c) for c in r["cands"]) for r in rs]
        c = Counter()
        for t in lab.get(kind, []):
            for x in t["cands"]:
                if x["judge"] is None:
                    continue
                keep = x["p"] >= THR
                right = keep == (x["judge"] in POSITIVE)
                c["n"] += 1
                c["right"] += right
                c[("keep", keep)] += 1
                c[("keep_right", keep)] += right
        tot.update(c)

        def rate(a, b):
            return f"{c[a] / c[b]:.2f} ({c[b]})" if c[b] else "-"
        print(f"| {kind} | {len(rs)} | {sum(len(r['cands']) for r in rs) / len(rs):.1f} | "
              f"{sum(x > 0 for x in nd) / len(rs):.2f} | {sum(nd) / max(sum(x > 0 for x in nd), 1):.1f} | "
              f"{len(lab.get(kind, []))} | {rate('right', 'n')} | "
              f"{rate(('keep_right', True), ('keep', True))} | {rate(('keep_right', False), ('keep', False))} |")
    if tot["n"]:
        print(f"\nall: verifier right on {tot['right'] / tot['n']:.3f} of {tot['n']} judged disagreements")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["select", "predict", "judge", "read"])
    ap.add_argument("--per", type=int, default=200)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--size", type=int, default=50)
    ap.add_argument("--cap", type=int, default=437)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--ver", default=VER, help="predict: the verifier")
    ap.add_argument("--full", action="store_true", help="predict: verify every candidate, not the pilot's sample")
    ap.add_argument("--tag", default="", help="predict: write ver-TAG-NNN.pkl, judged windows only")
    a = ap.parse_args()
    if a.cmd == "select":
        return select(a.per)
    if a.cmd == "predict":
        return predict(a.part, a.size, a.ver, a.full, a.tag)
    if a.cmd == "judge":
        return judge(a.cap, a.first)
    return read()


if __name__ == "__main__":
    raise SystemExit(main())
