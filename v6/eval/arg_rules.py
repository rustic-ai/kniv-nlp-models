"""Argument boundaries, the rest after the article trim (plan r4 review
step 2): rules that drop facts with a broken argument, read for free.

    uv run python -m v6.eval.arg_rules k2                      # what the rules catch in K2's hand-sorted facts
    uv run python -m v6.eval.arg_rules m1 --configs v1c,d123   # M1 against the fixed reference file, three judges

Rules, from the frozen linguistic layer of the released xlarge-u (parse,
POS, lemmas, coreference; ``k-checks/layers-m1p3a-n500.pkl``), on an
argument's syntactic head (the token of the mention whose parent lies
outside it):

- ``verbal`` (not a thing): the head is a verb, adjective, adverb, auxiliary
  or function word ("terminated", "unsuccessful", "maintaining");
- ``cut`` (cut off): the head modifies a noun outside the mention as a
  compound, adjective or number ("sales" of "sales commissions", "future" of
  "future lease payments");
- ``circular``: the two arguments overlap, share their head's lemma, or are
  in one coreference cluster.

Mentions are found in the text by their words (first occurrence); a fact
is dropped when every occurrence of an argument breaks a rule. Dropping
removes facts only, so the reading needs no judge calls: the configs with
the rules applied are merged into the reference file (``m1v1c-n500``) and
read at v7.1's recall under GPT-6, DeepSeek and agreement.
"""
from __future__ import annotations

import argparse
import ast
import json
import pickle
from collections import Counter
from pathlib import Path

from .m1_precision import OUT

LAYERS = Path("v6/runs/k-checks/layers-m1p3a-n500.pkl")
NOT_THING = {"VERB", "ADJ", "ADV", "AUX", "ADP", "SCONJ", "CCONJ", "PART", "INTJ", "DET"}
MODIFIER = {"compound", "amod", "nummod", "compound:prt"}


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def layers() -> dict:
    out = {}
    for x in pickle.loads(LAYERS.read_bytes()):
        out[x["id"]] = {"dep": _lit(x["words"]), "pos": _lit(x["pos"]), "lemma": _lit(x["lemma"]),
                        "coref": _lit(x["coref"])}
    return out


def _spans(tokens, name):
    low = [w.lower() for w in tokens]
    mt = name.lower().split()
    return [(a, a + len(mt)) for a in range(len(low) - len(mt) + 1) if low[a:a + len(mt)] == mt]


def _head(span, dep):
    """The mention's syntactic head: its first token whose parent is outside it
    (parents in ``dep`` are 1-based word indices, 0 the root)."""
    a, b = span
    for i in range(a, b):
        if i >= len(dep):
            return None
        p = dep[i][0] - 1
        if not (a <= p < b):
            return i
    return a


def flags(tokens, lay, head_names, tail_names) -> set:
    dep, pos, lem = lay["dep"], lay["pos"], lay["lemma"]
    clusters = [{(a, b) for a, b in c} for c in lay["coref"]]

    def arg(names):
        sp = sorted({s for n in names for s in _spans(tokens, n)})
        out = []
        for s in sp:
            h = _head(s, dep)
            if h is None:
                continue
            f = set()
            if pos[h] in NOT_THING:
                f.add("verbal")
            p = dep[h][0] - 1
            if dep[h][1] in MODIFIER and not (s[0] <= p < s[1]) and 0 <= p < len(pos) and pos[p] in ("NOUN", "PROPN"):
                f.add("cut")
            out.append((s, h, f))
        return out
    H, T = arg(head_names), arg(tail_names)
    if not H or not T:
        return set()
    res = set()
    for rule in ("verbal", "cut"):
        if all(rule in f for _, _, f in H) or all(rule in f for _, _, f in T):
            res.add(rule)
    circ = True
    for sh, hh, _ in H:
        for st, ht, _ in T:
            same = (sh[0] < st[1] and st[0] < sh[1]) or lem[hh].lower() == lem[ht].lower() \
                or any(any(x[0] <= hh < x[1] for x in c) and any(x[0] <= ht < x[1] for x in c) for c in clusters)
            circ &= same
    if circ:
        res.add("circular")
    return res


def k2() -> int:
    L = layers()
    S = {r["n"]: r for r in json.loads(Path("v6/runs/k-checks/k2-sample.json").read_text())}
    rows = []
    for k in range(3):
        rows += json.loads(Path(f"v6/runs/k-checks/k2-sorted{k}.json").read_text())
    items = {it["id"]: it for it in pickle.loads((OUT / "preds-m1p3a-n500.pkl").read_bytes())["items"]}
    by = Counter()
    hit = Counter()
    for r in rows:
        s = S[r["n"]]
        it, lay = items.get(s["text_id"]), L.get(s["text_id"])
        if it is None or lay is None:
            continue
        hn = s["head"] if isinstance(s["head"], list) else [s["head"]]
        tn = s["tail"] if isinstance(s["tail"], list) else [s["tail"]]
        f = flags(it["tokens"], lay, hn, tn)
        by[r["category"]] += 1
        for x in f:
            hit[(r["category"], x)] += 1
        if f:
            hit[(r["category"], "any")] += 1
    print("| K2 cause | facts | any rule | verbal | cut | circular |\n|---|---|---|---|---|---|")
    for c, n in by.most_common():
        print(f"| {c} | {n} | " + " | ".join(f"{hit[(c, x)]} ({hit[(c, x)] / n:.0%})"
                                             for x in ("any", "verbal", "cut", "circular")) + " |")
    return 0


def m1(configs: list[str], ref: str = "m1v1c-n500") -> int:
    from .v2_read import merge
    L = layers()
    R = pickle.loads((OUT / f"preds-{ref}.pkl").read_bytes())
    for c in configs:
        kept, drop = [], Counter()
        for it, ms in zip(R["items"], R["margins"][c]):
            lay = L.get(it["id"])
            k2_ = {}
            for k, m in ms.items():
                f = flags(it["tokens"], lay, sorted(k[0]), sorted(k[2])) if lay else set()
                for x in f:
                    drop[x] += 1
                if f:
                    drop["any"] += 1
                else:
                    k2_[k] = m
            kept.append(k2_)
        n = sum(len(x) for x in R["margins"][c])
        print(f"{c}: {n} facts, {drop['any']} dropped ({dict(drop)})")
        S = {"items": R["items"], "preds": {"ver": [sorted(x, key=str) for x in kept]}, "margins": {"ver": kept}}
        p = OUT / f"preds-args-{c}.pkl"
        p.write_bytes(pickle.dumps(S))
        out = f"m1args-{c}-n500"
        merge(ref, str(p), f"{c}+args", out)
        ref = out
    print(f"merged into {ref}; read: m1_precision equal --run {ref} --version c3 --judge J --pool v7.1,e1b,d1,d12,d123")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["k2", "m1"])
    ap.add_argument("--configs", default="v1c,d123")
    a = ap.parse_args()
    return k2() if a.cmd == "k2" else m1(a.configs.split(","))


if __name__ == "__main__":
    raise SystemExit(main())
