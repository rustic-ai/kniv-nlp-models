"""D2 (v8 plan r2): open labels -- other names for bank relations, and
business and legal text for the concepts M4 found missing or unlearned.

    uv run python -m v6.d2_labels para [--limit 50]       # ten names per relation (P4's 50)
    uv run python -m v6.d2_labels para --all              # every trained bank relation
    uv run python -m v6.d2_labels p4file                  # generated names in P4's file format
    uv run python -m v6.d2_labels concepts --per 2        # passages written to order, annotated blind
    uv run python -m v6.d2_labels concepts --per 40       # the full set

**Names** (``para``). P4: a reworded name and definition costs 3.4 F1 on
short text, a terse name with no definition 9.6 (CoNLL04 16.7), and the loss
is in ranking, not the bar. Training with label texts drawn from several
names per relation is the D2 remedy. Per relation the teacher gives five
described names (a name and a reworded one-sentence definition) and five
terse names (what a user types into a schema, no definition), keeping the
head-to-tail direction. The pilot uses P4's 50 short-text relations, so
``p4file`` can put the first generated name of each style through
``label_robustness`` (``--para``) and show whether generated names are as
hard for v7.1 as hand-written ones. A generated name equal to another bank
relation's name, or to one of P4's hand-written names, is flagged (and kept
out of training: P4's names stay held out).

**Concepts** (``concepts``). M4: "sued" 0 of 22, "board member of" 0,
"holds stake in" 1 of 24 ranked first; "supplier of" and "priced at" rank
first but under the bar. Most of these concepts are in the bank but the
short-text data has almost no facts for them (median 0). Passages are
written to order as in ``kg_short.synth`` (a style, one to three relations
with definitions, random names), then annotated blind by the teacher against
the passage's relations plus distractors from the same list and the bank. A
written fact the teacher does not recover is reported; recovery per concept
says whether the passages state the facts clearly. Concepts not in the bank
(``NEW``) carry a definition written here; the bank decides what the model
is taught, so adding them is a reviewed step.

Both write to ``v6/runs/d2-labels/``. Generation and annotation use the
short-text teacher; responses are cached by layer and ``VERSION``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict

from . import kg_short as ks
from .annotate.kg import KGAnnotator, KGItem
from .config import DATA_DIR, RUNS_DIR
from .label_bank import load as load_bank
from .prompts import SYSTEM

OUT = RUNS_DIR / "d2-labels"
VERSION = "d2a"
P4 = DATA_DIR / "v6-eval" / "p4_paraphrases.json"

SYSTEM["kgpara"] = (
    "You help make an information-extraction model robust to how people name relations. Given "
    "one relation (a name and a definition; the head and tail are the two entities it links, in "
    "that order), give ten other names someone might use for exactly this relation in their own "
    "schema:\n"
    "- five 'described': a different name and a one-sentence definition in your own words "
    "(do not reuse the original definition's wording);\n"
    "- five 'terse': a short name of one to four words, the way a developer or analyst types "
    "it into a schema with no definition (mix forms: verb phrases, noun phrases, snake_case-free "
    "plain words).\n"
    "Every name must keep the same head-to-tail direction (never the inverse relation), mean the "
    "same relation rather than a broader or narrower one, and differ from the original name and "
    "from each other. Write names in lower case."
)

# Business and legal concepts for the passages: (name, definition). Names in
# the bank keep its definition; NEW ones are not in the bank.
CONCEPTS = [
    "board member of", "supplies to", "purchases from", "client of", "invests in", "has price",
    "competes with", "partnership with", "merged into", "acquires", "subsidiary of", "regulated by",
    "external auditor", "advises", "licenses", "settled", "subcontractor of", "distributed by",
    "convicted of", "employed by", "headquartered in", "owned by",
    # the full run adds these
    "officer of", "investor", "funder", "owner of", "issued by", "liable to", "owes", "payable to",
    "bound by", "must comply with", "grants license to", "licensed to", "defendant", "penalty",
    "industry", "brand",
]
NEW = {
    "sued": "The head person or organization brought a lawsuit or legal claim against the tail.",
    "fined by": "The head was ordered to pay a fine or monetary penalty by the tail authority or court.",
}
# "legal counsel of" was in the pilot; M2's clean set has Wikidata's "legal
# counsel" (P1593), so it is left out of D2 to keep M2 zero-shot.
STYLES = ["a news sentence", "a press release", "a work email", "an excerpt from an annual report",
          "a meeting note", "a court or regulatory filing summary", "a chat message between colleagues",
          "a short professional bio", "a forum post", "a financial news brief"]


class Para(KGAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        return [{"role": "system", "content": SYSTEM["kgpara"]},
                {"role": "user", "content": item.layers["prompt"] + self._json_hint(layer)}]


def m2_names() -> set[str]:
    """M2's relation names and Wikidata aliases (lower case): no D2 label may
    use one, or M2 stops being zero-shot."""
    from .eval import m2_unseen as m2
    s = RUNS_DIR / "m2-unseen" / "set.json"
    if not s.exists():
        return set()
    props = {p["code"]: p for p in json.loads(m2.PROPS.read_text())}
    out = set()
    for r in json.loads(s.read_text())["relations"]:
        p = props.get(r["code"], {})
        out |= {" ".join(x.lower().split()) for x in [r["name"], *(p.get("aliases") or [])]}
    return out


def _bank_names() -> dict[str, dict]:
    return {r["name"]: r for r in load_bank()["relations"]}


def para(limit: int | None, annotator: str, every: bool = False) -> int:
    """``every``: all trained bank relations (never-trained ones stay out, so
    the zero-shot holdouts keep their names unseen); else P4's 50."""
    from .eval.m2_unseen import CONLL04, CROSSRE
    bank = _bank_names()
    p4 = json.loads(P4.read_text())
    names = [n for n, r in bank.items() if r.get("train", True)] if every else list(p4["short"])
    names = names[:limit] if limit else names
    evalnames = {" ".join(x.lower().split()) for x in CONLL04 + CROSSRE}
    hand = set()
    for part in ("short", "conll04"):
        for v in p4[part].values():
            hand |= {v["a"]["name"].lower(), v["b"]["name"].lower()}
    m2 = m2_names()
    reqs = []
    for n in names:
        q = KGItem(id=f"para:{n}", tokens=[], sentence_spans=[], speakers=None, participants=[], types=[],
                   relations=[], domain="para",
                   layers={"prompt": f"Relation: {n}\nDefinition: {bank[n].get('description', '')}"})
        reqs.append(q)
    OUT.mkdir(parents=True, exist_ok=True)
    ks.OUT = OUT
    res = asyncio.run(ks._run(Para(ks.teacher_spec(annotator), ks.CacheStore(ks.CACHE, VERSION)),
                              "kgpara", reqs, f"para-{annotator}"))
    out, c = {}, Counter()
    for n, q in zip(names, reqs):
        r = res.get(q.id)
        if r is None or not r.ok:
            c["failed"] += 1
            continue
        rows = []
        for v in r.payload:
            flags = []
            if v["name"] == n:
                flags.append("original")
            elif v["name"] in bank:
                flags.append(f"bank name ({bank[v['name']].get('train', True) and 'trained' or 'held out'})")
            if v["name"] in hand:
                flags.append("P4 hand name")
            if v["name"] in m2:
                flags.append("M2 relation name")
            if v["name"] in evalnames:
                flags.append("CoNLL04 or CrossRE name")
            rows.append({**v, "flags": flags})
            c[v["style"]] += 1
            for f in flags:
                c[f.split(" (")[0]] += 1
        out[n] = rows
        c["relations"] += 1
        c["short of 5+5"] += sum(1 for s in ("described", "terse") if sum(v["style"] == s for v in rows) < 5)
    dest = OUT / ("paraphrases-all.json" if every else "paraphrases.json")
    dest.write_text(json.dumps(out, indent=1))
    print(dict(c))
    words = [len(v["name"].split()) for rows in out.values() for v in rows if v["style"] == "terse"]
    if words:
        print(f"terse names: mean {sum(words) / len(words):.1f} words")
    print(f"-> {dest}")
    return 0


def p4file() -> int:
    """The first unflagged generated name of each style per relation, in
    P4's format (``a`` described, ``b`` terse), for ``label_robustness --para``."""
    gen = json.loads((OUT / "paraphrases.json").read_text())
    p4 = json.loads(P4.read_text())
    short = {}
    for n in p4["short"]:
        rows = [v for v in gen.get(n, []) if not v["flags"]]
        a = next((v for v in rows if v["style"] == "described"), None)
        b = next((v for v in rows if v["style"] == "terse"), None)
        if a and b:
            short[n] = {"a": {"name": a["name"], "description": a["description"]},
                        "b": {"name": b["name"], "description": ""}}
    path = OUT / "p4_generated.json"
    path.write_text(json.dumps({"_note": "D2 pilot: generated names in P4's format; CoNLL04 part is P4's own.",
                                "short": short, "conll04": p4["conll04"]}, indent=1))
    print(f"{len(short)} of {len(p4['short'])} relations -> {path}")
    return 0


def concept_requests(per: int) -> tuple[list[KGItem], dict[str, str]]:
    bank = _bank_names()
    desc = {**{n: bank[n].get("description", "") for n in CONCEPTS}, **NEW}
    clash = [n for n in desc if n in m2_names()]
    assert not clash, f"concepts named like an M2 relation: {clash}"
    names = sorted(desc)
    reqs = []
    for i, n in enumerate(names):
        for k in range(per):
            rng = ks._rng(f"d2c:{n}:{k}")
            others = [x for x in names if x != n]
            chosen = [n] + list(rng.choice(others, int(rng.integers(0, 3)), replace=False))
            style = STYLES[int(rng.integers(len(STYLES)))]
            lines = [f"Style: {style}", "Relations (head -> tail):"]
            lines += [f"- {r}: {desc[r]}" for r in chosen]
            lines.append(ks.name_hints(ks._rng(f"d2hint:{n}:{k}")))
            reqs.append(KGItem(id=f"d2c:{n}:{k}", tokens=[], sentence_spans=[], speakers=None, participants=[],
                               types=[], relations=[], domain="business",
                               layers={"syn_prompt": "\n".join(lines), "relations": chosen, "style": style,
                                       "target": n}))
    return reqs, desc


def concepts(per: int, annotator: str) -> int:
    from .windows import tokenize
    bank = _bank_names()
    reqs, desc = concept_requests(per)
    # the cache is keyed by item id: the full run's schemas draw distractors
    # from a longer list than the pilot's, so it keeps its own responses
    spec, cache = ks.teacher_spec(annotator), ks.CacheStore(ks.CACHE, VERSION if per <= 2 else "d2b")
    OUT.mkdir(parents=True, exist_ok=True)
    ks.OUT = OUT
    gen = asyncio.run(ks._run(ks.Synthesizer(spec, cache), "kgsyn", reqs, f"concept-gen-{annotator}"))
    base = ks.SchemaSampler(load_bank())
    pool = [r["name"] for r in base.relations if r["name"] not in desc]
    items, c = [], Counter()
    for q in reqs:
        r = gen.get(q.id)
        if r is None or not r.ok:
            c["generation failed"] += 1
            continue
        text, facts = r.payload["text"], r.payload["facts"]
        want = set(q.layers["relations"])
        ok = [f for f in facts if f["relation"] in want and ks._contains(text, f["head"])
              and ks._contains(text, f["tail"])]
        c["written facts dropped (relation or mention)"] += len(facts) - len(ok)
        if not any(f["relation"] == q.layers["target"] for f in ok):
            c["no fact for the target concept"] += 1
            continue
        toks = tokenize(text)
        spans, s0 = [], 0
        for j, t in enumerate(toks):
            if t in {".", "!", "?"} and j + 1 < len(toks):
                spans.append([s0, j + 1]); s0 = j + 1
        spans.append([s0, len(toks)])
        rng = ks._rng(f"d2schema:{q.id}")
        same_list = [x for x in sorted(desc) if x not in want]
        distract = list(rng.choice(same_list, 3, replace=False)) + list(rng.choice(pool, 3, replace=False))
        schema = sorted(list(want) + distract, key=lambda x: ks._rng(f"o:{q.id}:{x}").random())
        rels = [{"name": x, "description": desc.get(x) or bank[x].get("description", "")} for x in schema]
        items.append(KGItem(id=f"{q.id}:ann", tokens=toks, sentence_spans=spans, speakers=None, participants=[],
                            types=ks.GENERAL_TYPES + base.always, relations=rels, domain="business",
                            layers={"gold": ok, "text": text, "style": q.layers["style"],
                                    "target": q.layers["target"]}))
    keep, ks.ROUND = ks.ROUND, 4
    try:
        ann = asyncio.run(ks._run(ks.ShortKG(spec, cache), "kg", items, f"concept-ann-{annotator}"))
    finally:
        ks.ROUND = keep
    per_c, rows, examples = defaultdict(Counter), [], []
    for it in items:
        r = ann.get(it.id)
        if r is None or not r.ok:
            c["annotation failed"] += 1
            continue
        p = r.payload
        d = {**it.__dict__}

        def name(i):
            a, b = p["entities"][i]["spans"][0]
            return " ".join(it.tokens[a:b])

        def same(x, y):
            return bool(x) and bool(y) and (ks._contains(x, y) or ks._contains(y, x))
        pred = [(name(f["head"]), f["relation"], name(f["tail"])) for f in p["facts"]]
        facts = list(p["facts"])
        have = {(f["head"], f["tail"], f["relation"]) for f in facts}
        for g in it.layers["gold"]:
            cc = per_c[g["relation"]]
            cc["written"] += 1
            found = any(rel == g["relation"] and same(h, g["head"]) and same(t, g["tail"]) for h, rel, t in pred)
            cc["recovered"] += found
            if not found and len(examples) < 10:
                examples.append(f"[{g['relation']}] {it.layers['text'][:220]}\n    missed: "
                                f"{g['head']} | {g['relation']} | {g['tail']}")
            h, t = ks._find_entity(d, p, g["head"]), ks._find_entity(d, p, g["tail"])
            if h < 0 or t < 0 or h == t:
                c["written fact not placed"] += 1
                continue
            if (h, t, g["relation"]) not in have:
                facts.append({"head": h, "tail": t, "relation": g["relation"], "polarity": "positive",
                              "modality": "asserted", "holder": -1, "time": -1, "evidence": []})
                have.add((h, t, g["relation"]))
        wanted = set(it.layers["gold"][i]["relation"] for i in range(len(it.layers["gold"])))
        c["teacher facts outside the written ones"] += sum(1 for _, rel, _ in pred if rel not in wanted)
        rows.append(ks._record(d, {**p, "facts": facts}, style=it.layers["style"], target=it.layers["target"]))
    dest = OUT / ("concepts.json" if per <= 2 else f"concepts-per{per}.json")
    dest.write_text(json.dumps(rows))
    print(f"{len(reqs)} requests, {len(items)} passages kept; {dict(c)}")
    print(f"{'relation':22s} {'written':>7s} {'recovered':>9s}  in bank")
    for n in sorted(per_c, key=lambda n: per_c[n]["recovered"] / max(per_c[n]["written"], 1)):
        cc = per_c[n]
        print(f"{n:22s} {cc['written']:7d} {cc['recovered'] / max(cc['written'], 1):9.0%}  "
              f"{'yes' if n in bank else 'NEW'}")
    tot = Counter()
    for cc in per_c.values():
        tot.update(cc)
    print(f"{'ALL':22s} {tot['written']:7d} {tot['recovered'] / max(tot['written'], 1):9.0%}")
    print("\n".join(examples))
    print(f"-> {dest} ({len(rows)} passages)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["para", "p4file", "concepts"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--per", type=int, default=2, help="concepts: passages per concept")
    ap.add_argument("--annotator", default="sol61")
    ap.add_argument("--all", action="store_true", help="para: every trained bank relation")
    a = ap.parse_args()
    if a.cmd == "para":
        return para(a.limit, a.annotator, a.all)
    if a.cmd == "p4file":
        return p4file()
    return concepts(a.per, a.annotator)


if __name__ == "__main__":
    raise SystemExit(main())
