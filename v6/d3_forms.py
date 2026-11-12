"""D3 (v8 plan r2): negation and hedging in the forms the 14k minimal pairs
do not cover.

    uv run python -m v6.d3_forms run --bases 52        # pilot: rewrite, judge, annotate
    uv run python -m v6.d3_forms run --bases 1000 --agents   # person / organisation heads only
    uv run python -m v6.d3_forms show --n 20           # rewrites to read by hand

The short-text minimal pairs (``kg_short variants``) rewrite one plain fact
six ways: negated, reported, planned, desired, questioned, hypothetical. M3
found the forms that fail are mostly others: denials, "never", "no longer",
"might", rumours, "expected to". This writes those six forms the same way:

- **bases**: round-4 short training texts with one plain asserted fact whose
  arguments are named in one sentence (``kg_short.bases``), never a base the
  attribute retrain held out (``negation_pilot.heldout_base``, the bases M3's
  pair set scores), never one the 14k pairs already used, never a CoNLL04
  overlap id;
- **rewrite** (the teacher), **judge** (one call per base over its six
  rewrites), **annotate** (the teacher on each kept rewrite, blind), as in
  ``kg_short variants``;
- **labels set by kind**, as ``kg_short.export_short`` does for the pairs:
  the target fact gets the kind's polarity and modality, added if the teacher
  missed it. A denial by the head (or its company) is negative and reported,
  the head holding it (decided 2026-11-12: the writer does not vouch for a
  denial); "never" / "no longer" are negative and asserted; "might" is
  hypothetical; a rumour is reported with no holder; "expected to" is planned.

Output: ``v6/runs/d3-forms/pairs.json`` in ``kg_short.json``'s pair record
format, and per-kind yields. An 8-word overlap check against M3's and M4's
hand texts drops any rewrite that shares a sequence with them.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from . import kg_short as ks
from .annotate.kg import KGAnnotator, KGItem
from .config import DATA_DIR, RUNS_DIR
from .eval.negation_pilot import heldout_base
from .prompts import SYSTEM
from .schemas import KGVAR4_KINDS

OUT = RUNS_DIR / "d3-forms"
# --agents: the head is a person or an organisation, the arguments the forms
# read naturally with (a denial, a rumour, "no longer"); the pilot's bases
# without it included table fragments and part-of facts.
AGENT = re.compile(r"person|organi[sz]ation|company|politician|athlete|actor|musician|writer|artist|"
                   r"scholar|scientist|government|party|team|band|director|soldier|business|firm|agency|"
                   r"corporation|club|university|school", re.I)
VERSION = "d3a"
EXPECT = {"denied": ("negative", "reported"), "never": ("negative", "asserted"),
          "no_longer": ("negative", "asserted"), "might": ("positive", "hypothetical"),
          "rumoured": ("positive", "reported"), "expected": ("positive", "planned")}
assert set(EXPECT) == set(KGVAR4_KINDS)

SYSTEM["kgvar4"] = (
    "You rewrite one sentence so that one fact in it changes status, changing as few words as "
    "possible. Keep the two entity mentions exactly as written (same spelling and case), keep "
    "the rest of the sentence natural and grammatical, and do not add other facts. Put the change "
    "on the clause that states the fact, never in a relative clause attached to one of the "
    "entities. If the fact is stated inside a longer sentence, you may shorten the sentence to the "
    "clause that states it, keeping both mentions exactly as written. Write one rewrite for each "
    "kind:\n"
    "- denied: the head itself (or, for a person, their company or spokesperson speaking for "
    "them) denies the fact, so the text says it does not hold ('X denied working for Y', 'X "
    "has denied that it supplies Y').\n"
    "- never: the fact never held ('X never worked for Y', 'X has never supplied Y').\n"
    "- no_longer: the fact held once but has ended ('X no longer works for Y', 'X stopped "
    "supplying Y last year').\n"
    "- might: the fact is possible, not asserted ('X might work for Y', 'X may supply Y').\n"
    "- rumoured: an unnamed or vague source claims it ('X is rumoured to work for Y', 'Reports "
    "suggest X supplies Y'); no named person or organisation makes the claim.\n"
    "- expected: it is expected or scheduled to become true ('X is expected to join Y', 'X is "
    "set to supply Y from next year').\n"
    "Leave 'source' empty for every kind."
)

SYSTEM["kgvarjudge4"] = (
    "You check rewrites of one sentence. Each rewrite should express the same fact (the same "
    "head, relation and tail) with a different status: denied (the head, or someone speaking for "
    "it, denies the fact), never (the fact never held), no_longer (it held once and has ended), "
    "might (it is possible, not asserted), rumoured (an unnamed or vague source claims it) or "
    "expected (it is expected or scheduled to become true). A rewrite is ok if it reads "
    "naturally, keeps the head and tail mentions, and a careful reader would see that exact fact "
    "with that exact status. It is not ok if the status attaches to a different part of the "
    "sentence, the fact itself changed, the sentence states the fact twice, once with and once "
    "without the status, or the status is placed in a relative or subordinate clause attached to "
    "one of the entities."
)


class Rewriter4(KGAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        return [{"role": "system", "content": SYSTEM["kgvar4"]},
                {"role": "user", "content": item.layers["var_prompt"] + self._json_hint(layer)}]


class RewriteJudge4(KGAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        return [{"role": "system", "content": SYSTEM["kgvarjudge4"]},
                {"role": "user", "content": item.layers["judge_prompt"] + self._json_hint(layer)}]


def _grams(text: str, n: int = 8) -> set[str]:
    w = ks._norm_words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def heldout_text_grams() -> set[str]:
    g = set()
    for f in ("m3_negation.json", "m4_business.json", "p5_negation.json"):
        d = json.loads((DATA_DIR / "v6-eval" / f).read_text())
        for it in d["items"] if isinstance(d, dict) and "items" in d else d:
            if isinstance(it, dict):
                t = it.get("text") or " ".join(it.get("tokens") or [])
                g |= _grams(t)
    return g


def pick_bases(n: int, annotator: str, seed: int = 4, agents: bool = False):
    ks.SPLIT = "train"
    ks.set_round(4)
    excl = set(json.loads((DATA_DIR / "v6-corpus" / "kg_short_exclude.json").read_text())["ids"])
    used = {vi["id"].split(":var:")[0] for vi in json.loads((ks.OUT / "variants.json").read_text())}
    teacher = ks._last_ok(ks.OUT / f"pilot-{annotator}")
    items = [KGItem(**d) for d in json.loads(ks.ITEMS.read_text())]
    items = [it for it in items if it.id in teacher and it.id not in excl and it.id not in used
             and not heldout_base(it.id)]
    random.Random(seed).shuffle(items)
    res = {it.id: SimpleNamespace(ok=True, payload=teacher[it.id]) for it in items}
    # kg_short.bases spreads over domains and takes the first per domain, so
    # hand it a shuffled pool a few times larger than needed
    if not agents:
        return ks.bases(items[: n * 40], res, n), len(used)
    out = []
    for it, f, head, tail, j in ks.bases(items, res, n * 4):
        k = len(it.participants)
        if AGENT.search(teacher[it.id]["entities"][f["head"] - k]["type"]):
            out.append((it, f, head, tail, j))
    return out[:n], len(used)


def run(n: int, annotator: str, judge: str, agents: bool = False) -> int:
    from .windows import tokenize
    OUT.mkdir(parents=True, exist_ok=True)
    picked, n_used = pick_bases(n, annotator, agents=agents)
    print(f"{len(picked)} bases (none held out, none of the {n_used:,} bases the pairs used)", flush=True)
    spec, cache = ks.teacher_spec(annotator), ks.CacheStore(ks.CACHE, VERSION)
    ks.OUT = OUT                    # run logs go here
    reqs = []
    for it, f, head, tail, j in picked:
        s, e = it.sentence_spans[j]
        sentence = " ".join(it.tokens[s:e])
        desc = next(r["description"] for r in it.relations if r["name"] == f["relation"])
        q = KGItem(**{**it.__dict__})
        q.id = f"{it.id}:var:{f['head']}-{f['tail']}-{f['relation']}"
        q.layers = {"var_prompt": ks.var_user(sentence, head, f["relation"], desc, tail),
                    "base": [sentence, head, f["relation"], tail]}
        reqs.append(q)
    rw = asyncio.run(ks._run(Rewriter4(spec, cache), "kgvar4", reqs, f"rewrite-{annotator}"))
    jspec, jq = ks.judge_spec(judge), []
    for q in reqs:
        r = rw.get(q.id)
        if r is None or not r.ok:
            continue
        sentence, head, rel, tail = q.layers["base"]
        lines = [f"Original: {sentence}", f"Fact: [{head}] {rel} [{tail}]", "", "Rewrites:"]
        lines += [f"- {k}: {r.payload[k]['sentence']}" for k in KGVAR4_KINDS if k in r.payload]
        jj = KGItem(**{**q.__dict__}); jj.id = f"{q.id}:vjudge"
        jj.layers = {"judge_prompt": "\n".join(lines), "base": q.id}
        jq.append(jj)
    jr = asyncio.run(ks._run(RewriteJudge4(jspec, cache), "kgvarjudge4", jq, f"rewrite-judge-{judge}"))
    verdict = {j.layers["base"]: (jr[j.id].payload if j.id in jr and jr[j.id].ok else {}) for j in jq}
    held = heldout_text_grams()
    var_items, c = [], Counter()
    for q in reqs:
        r = rw.get(q.id)
        if r is None or not r.ok:
            c["rewrite failed"] += 1
            continue
        sentence, head, rel, tail = q.layers["base"]
        for kind in KGVAR4_KINDS:
            v = r.payload.get(kind)
            c[f"{kind}: asked"] += 1
            if not v:
                c[f"{kind}: missing"] += 1
                continue
            if not (ks._contains(v["sentence"], head) and ks._contains(v["sentence"], tail)):
                c[f"{kind}: mention changed"] += 1
                continue
            if not verdict.get(q.id, {}).get(kind, False):
                c[f"{kind}: judged not ok"] += 1
                continue
            if _grams(v["sentence"]) & held:
                c[f"{kind}: overlaps a held-out text"] += 1
                continue
            toks = tokenize(v["sentence"])
            var_items.append(KGItem(id=f"{q.id}:{kind}", tokens=toks, sentence_spans=[[0, len(toks)]],
                                    speakers=None, participants=[], types=q.types, relations=q.relations,
                                    domain=q.domain, layers={"kind": kind, "head": head, "tail": tail,
                                                             "relation": rel, "base": sentence}))
    keep, ks.ROUND = ks.ROUND, 4
    try:
        ann = asyncio.run(ks._run(ks.ShortKG(spec, cache), "kg", var_items, f"annotate-{annotator}"))
    finally:
        ks.ROUND = keep
    rows = []
    for vi in var_items:
        L, kind = vi.layers, vi.layers["kind"]
        r = ann.get(vi.id)
        if r is None or not r.ok:
            c[f"{kind}: annotation failed"] += 1
            continue
        p, d = r.payload, {**vi.__dict__}
        h, t = ks._find_entity(d, p, L["head"]), ks._find_entity(d, p, L["tail"])
        if h < 0 or t < 0:
            c[f"{kind}: argument not among the entities"] += 1
            continue
        pol, mod = EXPECT[kind]
        facts, fixed = [], False
        for f in p["facts"]:
            if (f["head"], f["tail"], f["relation"]) == (h, t, L["relation"]):
                if fixed:
                    continue
                agree = (f["polarity"], f["modality"]) == (pol, mod)
                c[f"{kind}: teacher agrees" if agree else f"{kind}: label set by kind"] += 1
                holder = {"rumoured": -1, "denied": h}.get(kind, f["holder"])
                f = {**f, "polarity": pol, "modality": mod, "holder": holder}
                fixed = True
            facts.append(f)
        if not fixed:
            facts.append({"head": h, "tail": t, "relation": L["relation"], "polarity": pol, "modality": mod,
                          "holder": h if kind == "denied" else -1, "time": -1, "evidence": [0]})
            c[f"{kind}: fact added (teacher missed it)"] += 1
        c[f"{kind}: kept"] += 1
        rows.append(ks._record(d, {**p, "facts": facts}, kind=kind, base=L["base"]))
    (OUT / "pairs.json").write_text(json.dumps(rows))
    print(f"{'kind':10s} {'asked':>5s} {'judged ok':>9s} {'kept':>5s} {'teacher agrees':>14s} {'added':>5s}")
    for k in KGVAR4_KINDS:
        a = c[f"{k}: asked"]
        bad = c[f"{k}: missing"] + c[f"{k}: mention changed"] + c[f"{k}: judged not ok"]
        kept = c[f"{k}: kept"]
        print(f"{k:10s} {a:5d} {(a - bad) / max(a, 1):9.0%} {kept:5d} "
              f"{c[f'{k}: teacher agrees'] / max(kept, 1):14.0%} {c[f'{k}: fact added (teacher missed it)']:5d}")
    print({k: v for k, v in sorted(c.items()) if "overlap" in k or "failed" in k or "not among" in k})
    print(f"-> {OUT / 'pairs.json'} ({len(rows)} rewrites)")
    return 0


def show(n: int, seed: int = 0) -> int:
    rows = json.loads((OUT / "pairs.json").read_text())
    for r in random.Random(seed).sample(rows, min(n, len(rows))):
        print(f"[{r['kind']}] {r['base']}\n    -> {' '.join(r['tokens'])}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["run", "show"])
    ap.add_argument("--bases", type=int, default=50)
    ap.add_argument("--annotator", default="sol61")
    ap.add_argument("--judge", default="astra")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--agents", action="store_true", help="bases whose head is a person or organisation")
    a = ap.parse_args()
    return run(a.bases, a.annotator, a.judge, a.agents) if a.cmd == "run" else show(a.n)


if __name__ == "__main__":
    raise SystemExit(main())
