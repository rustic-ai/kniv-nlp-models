"""Plan r4 stage 1 checks on the M1 pool (``m1p3a-n500``: v7.1, E1b, D1 on
the 500 short-dev texts, c3 labels). Outputs in ``v6/runs/k-checks/``.

    uv run python -m v6.eval.r4_checks k1 [--name m1p3a-n500]
    uv run python -m v6.eval.r4_checks k2-sample          # 300 of E1b's judged-false facts
    uv run python -m v6.eval.r4_checks k2-score           # with k2-sorted{0,1,2}.json
    uv run python -m v6.eval.r4_checks layers             # released xlarge-u parse/roles/coref
    uv run python -m v6.eval.r4_checks k3
    uv run python -m v6.eval.r4_checks k4-decode | k4-judge | k4-score

K1: how good a verifier over today's candidates must be for +10 judged
precision at v7.1's recall. K2: a sample of false facts for sorting by
cause. K3: are judged-true facts linked by the parse or roles more often
than false ones. K4: schema-free facts decoded from the linguistic layer by
rule, judged (c3).
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from collections import Counter, deque

import numpy as np

from ..config import RUNS_DIR
from .m1_precision import OUT, POSITIVE, _same, judge_prompt, labels_path, PROMPTS

K = RUNS_DIR / "k-checks"
XL = "dragonscale-ai/kniv-v7-xlarge-u-en"


def _load(name):
    P = pickle.loads((OUT / f"preds-{name}.pkl").read_bytes())
    L = json.loads(labels_path(name, "astra", "c3").read_text())
    return P, L


def _rows(P, L):
    """Per config: (margin, text index, key, judged fact or None)."""
    texts = L["texts"]
    keep = [i for i, t in enumerate(texts) if t["judged"]]
    rows = {c: [] for c in L["configs"]}
    for i in keep:
        for c in L["configs"]:
            for k, m in P["margins"][c][i].items():
                f = next((f for f in texts[i]["facts"] if c in f["src"]
                          and _same((set(f["key"][0]), f["key"][1], set(f["key"][2])), k)), None)
                rows[c].append((m, i, k, f))
    total = sum(sum(f["verdict"] in POSITIVE for f in texts[i]["facts"]) for i in keep)
    return rows, total


def _true(f):
    return f is not None and f["verdict"] in POSITIVE


# ── K1 ─────────────────────────────────────────────────────────────
def k1(name: str, gain: float = 0.10) -> None:
    P, L = _load(name)
    rows, total = _rows(P, L)
    cfgs = L["configs"]

    def auc(r):
        m = np.array([x[0] for x in r]); y = np.array([_true(x[3]) for x in r])
        o = m.argsort().argsort() + 1
        n1, n0 = y.sum(), (~y).sum()
        return (o[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

    b = rows[cfgs[0]]
    R0 = sum(_true(x[3]) for x in b) / total
    P0 = np.mean([_true(x[3]) for x in b])
    need, tgt = R0 * total, P0 + gain
    print(f"pool {total}; {cfgs[0]} P {P0:.3f} R {R0:.3f}; target P {tgt:.3f} at R {R0:.3f} "
          f"(keep {need:.0f} true, at most {need * (1 / tgt - 1):.0f} false)\n")
    print("| run | predicted | true | false | margin AUC | max R | TPR needed | FPR allowed |")
    print("|---|---|---|---|---|---|---|---|")
    for c in cfgs:
        T = sum(_true(x[3]) for x in rows[c]); F = len(rows[c]) - T
        t = need / T
        f = T * t * (1 / tgt - 1) / F if t <= 1 else float("nan")
        print(f"| {c} | {len(rows[c])} | {T} | {F} | {auc(rows[c]):.3f} | {T / total:.3f} | {t:.3f} | {f:.3f} |")


# ── K2 ─────────────────────────────────────────────────────────────
def k2_sample(name: str, run: str = "e1b", n: int = 300) -> None:
    from ..label_bank import load as load_bank
    P, L = _load(name)
    rows, total = _rows(P, L)
    R0 = sum(_true(x[3]) for x in rows[L["configs"][0]]) / total
    r = sorted(rows[run], key=lambda x: -x[0])
    ok = np.cumsum([_true(x[3]) for x in r])
    cut = r[int(np.searchsorted(ok / total, R0))][0]
    false = [x for x in r if x[0] >= cut and not _true(x[3])]
    print(f"{run}: cut {cut:.3f}, {len(false)} judged-false facts above it")
    random.seed(13)
    S = random.sample(false, n)
    bank = {x["name"]: x for x in load_bank()["relations"]}
    out = []
    for j, (m, i, k, f) in enumerate(S):
        it = P["items"][i]
        out.append({"n": j, "text_id": it["id"], "text": " ".join(it["tokens"]),
                    "speakers": it.get("speakers"), "participants": it.get("participants"),
                    "head": sorted(k[0]), "relation": k[1],
                    "description": bank.get(k[1], {}).get("description", ""),
                    "tail": sorted(k[2]), "judge_verdict": f["verdict"] if f else None,
                    "margin": round(float(m), 2)})
    K.mkdir(parents=True, exist_ok=True)
    (K / "k2-sample.json").write_text(json.dumps(out, indent=1))
    print("wrote", K / "k2-sample.json")


K2_DATA = RUNS_DIR.parent.parent / "data" / "v6-eval" / "r4_k2_false_facts.json"


def k2_score() -> None:
    """The sorted sample, kept with the evaluation data (``data/v6-eval``)."""
    lab = json.loads(K2_DATA.read_text())["items"]
    c = Counter(x["cause"] for x in lab)
    for k, v in c.most_common():
        print(f"{k:16s} {v:4d}  {100 * v / len(lab):.0f}%")


# ── linguistic layer ───────────────────────────────────────────────
def layers(name: str) -> None:
    import time
    from huggingface_hub import snapshot_download
    from kniv_v7 import Kniv
    k = Kniv(snapshot_download(XL, revision="v7.1.1"))
    P = pickle.loads((OUT / f"preds-{name}.pkl").read_bytes())
    out, t0 = [], time.time()
    for it in P["items"]:
        w = k.window({"tokens": it["tokens"], "sentence_spans": it["sentence_spans"]}, [], None, True, 5)
        out.append({"id": it["id"], "W": w["analysed_words"],
                    "words": [(x["head"], x["deprel"]) for x in w["words"]],
                    "lemma": [x["lemma"] for x in w["words"]], "pos": [x["pos"] for x in w["words"]],
                    "ner": w["ner"], "srl": w["srl"], "coref": w["coref"]})
    K.mkdir(parents=True, exist_ok=True)
    (K / f"layers-{name}.pkl").write_bytes(pickle.dumps(out))
    print(f"{len(out)} texts in {time.time() - t0:.0f}s")


def _children(words):
    ch = {i: [] for i in range(len(words))}
    for i, (h, _) in enumerate(words):
        if 0 < h <= len(words):
            ch[h - 1].append(i)
    return ch


# ── K3 ─────────────────────────────────────────────────────────────
def k3(name: str) -> None:
    P, L = _load(name)
    lay = pickle.loads((K / f"layers-{name}.pkl").read_bytes())
    cfgs = L["configs"]

    def occ(tokens, W, ms):
        low = [t.lower() for t in tokens[:W]]
        out = []
        for m in ms:
            mt = m.split()
            out += [(a, a + len(mt)) for a in range(len(low) - len(mt) + 1) if low[a:a + len(mt)] == mt]
        return out

    def dist(words, src):
        adj = {i: set() for i in range(len(words))}
        for i, (h, _) in enumerate(words):
            if 0 < h <= len(words):
                adj[i].add(h - 1); adj[h - 1].add(i)
        d = {s: 0 for s in src}; q = deque(src)
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in d:
                    d[y] = d[x] + 1; q.append(y)
        return d

    def link(it, ly, key):
        hs, ts = occ(it["tokens"], ly["W"], key[0]), occ(it["tokens"], ly["W"], key[2])
        if not hs or not ts:
            return None
        sp = it["sentence_spans"]
        sent = lambda i: next((j for j, (a, b) in enumerate(sp) if a <= i < b), -1)
        same = any(sent(a) == sent(c) for a, _ in hs for c, _ in ts)
        touch = lambda s, p: s[0] < p[1] and p[0] < s[1]
        frame = False
        for fr in ly["srl"]:
            parts = [(fr["predicate"], fr["predicate"] + 1)] + [(a[1], a[2]) for a in fr["args"]]
            if any(touch(h, p) and touch(t, q) for h in hs for t in ts
                   for pi, p in enumerate(parts) for qi, q in enumerate(parts) if pi != qi):
                frame = True; break
        hd = dist(ly["words"], [i for a, b in hs for i in range(a, b)])
        path = min((hd.get(i, 99) for a, b in ts for i in range(a, b)), default=99)
        return {"same": same, "frame": frame, "path": path}

    res = {c: {True: [], False: []} for c in cfgs + ["any"]}
    miss = Counter()
    for it, ly, t in zip(P["items"], lay, L["texts"]):
        if not t["judged"]:
            continue
        for f in t["facts"]:
            srcs = [c for c in cfgs if c in f["src"]]
            if not srcs:
                continue
            y = f["verdict"] in POSITIVE
            r = link(it, ly, f["key"])
            if r is None:
                miss[y] += 1; continue
            for c in srcs + ["any"]:
                res[c][y].append(r)
    print(f"facts with an argument not in the text: true {miss[True]}, false {miss[False]}\n")
    cols = [("same sentence", lambda r: r["same"]), ("one frame", lambda r: r["frame"]),
            ("path<=2", lambda r: r["path"] <= 2), ("path<=3", lambda r: r["path"] <= 3),
            ("frame or path<=3", lambda r: r["frame"] or r["path"] <= 3)]
    print("| facts | true / false | " + " | ".join(f"{n} T / F" for n, _ in cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for c in ["any"] + cfgs:
        T, F = res[c][True], res[c][False]
        print(f"| {c} | {len(T)} / {len(F)} | " + " | ".join(
            f"{100 * np.mean([fn(r) for r in T]):.0f} / {100 * np.mean([fn(r) for r in F]):.0f}"
            for _, fn in cols) + " |")


# ── K4 ─────────────────────────────────────────────────────────────
NP_KEEP = {"det", "amod", "compound", "nummod", "flat", "nmod:poss", "fixed", "flat:name"}
NOUNISH = {"NOUN", "PROPN", "PRON", "NUM", "X", "SYM"}


def _decode(it, ly):
    """Rules: each SRL frame's (ARG0, predicate, ARG1); with no ARG0, (ARG1,
    predicate, ARG2); prepositional ARG2-4 / ARGM-LOC / ARGM-DIR as
    (subject, predicate + preposition, object); copulas and appositions as
    (X, is, Y). Arguments cut to their noun phrase (head plus det, amod,
    compound, nummod, flat, possessive, fixed, and one "of"-style nmod),
    pronouns to their coreference cluster's longest name; clausal and
    nonsense arguments dropped. Negation kept on the fact."""
    toks, W = it["tokens"], ly["W"]
    words, pos, lem = ly["words"], ly["pos"], ly["lemma"]
    ch = _children(words)
    best = {}
    for cl in ly["coref"]:
        ms = [(a, b + 1) for a, b in cl]
        names = [" ".join(toks[a:b]) for a, b in ms
                 if not all(pos[i] in ("PRON", "DET") for i in range(a, b))]
        nm = max(names, key=len) if names else None
        for a, b in ms:
            for i in range(a, b):
                best.setdefault(i, (nm, [" ".join(toks[x:y]).lower() for x, y in ms]))

    def np_span(h, depth=0):
        keep = {h}
        for c in ch[h]:
            rel = words[c][1]
            if rel in NP_KEEP:
                keep |= np_span(c, depth + 1)
            elif rel == "nmod" and depth == 0 and any(words[g][1] == "case" and g < c for g in ch[c]):
                keep |= np_span(c, depth + 1) | {g for g in ch[c] if words[g][1] == "case"}
        return keep

    def head_of(a, b):
        hs = [i for i in range(a, b) if not (a <= words[i][0] - 1 < b)]
        return hs[0] if hs else None

    def arg(a, b):
        h = head_of(a, b)
        if h is None or pos[h] not in NOUNISH or words[h][1] == "case":
            return None
        idx = {i for i in np_span(h) if a <= i < b}
        x, y = min(idx), max(idx) + 1
        txt = " ".join(toks[x:y]); names = {txt.lower()}
        if pos[h] == "PRON":
            nm, ms = best.get(h, (None, []))
            if not nm:
                return None
            names, txt = {nm.lower()} | set(ms), nm
        elif h in best and best[h][0]:
            names |= set(best[h][1])
        if txt.lower() in {"it", "this", "that", "they", "which", "who"}:
            return None
        if not re.search(r"[A-Za-z]{3}", txt) or re.search(r"[_=/<>{}]|http|www", txt):
            return None
        return {"text": txt, "names": frozenset(names), "head": h}

    def prep_of(a, b):
        h = head_of(a, b)
        cs = [c for c in ch[h] if words[c][1] == "case" and c >= a] if h is not None else []
        return toks[cs[0]].lower() if cs else None

    facts = []

    def add(s, p, o, neg, src):
        if s and o and s["names"].isdisjoint(o["names"]) and s["head"] != o["head"]:
            facts.append({"s": s["text"], "p": p, "o": o["text"], "neg": neg, "src": src,
                          "names": [sorted(s["names"]), sorted(o["names"])]})

    for fr in ly["srl"]:
        p = fr["predicate"]
        if p >= W or (lem[p] in ("be", "have", "do") and pos[p] == "AUX"):
            continue
        roles = {}
        for lab, a, b in fr["args"]:
            roles.setdefault(lab, (a, b))
        neg = "ARGM-NEG" in roles or any(lem[c] in ("not", "never") for c in ch[p])
        pred = " ".join([toks[p]] + [toks[c] for c in ch[p] if words[c][1] == "compound:prt"])
        sl = "ARG0" if "ARG0" in roles else "ARG1"
        subj = arg(*roles[sl]) if sl in roles else None
        if subj is None:
            continue
        objs = ["ARG1", "ARG2"] if sl == "ARG0" else ["ARG2", "ARG3"]
        for lab in objs + ["ARG3", "ARG4", "ARGM-LOC", "ARGM-DIR"]:
            if lab not in roles or lab == sl:
                continue
            pr = prep_of(*roles[lab])
            if lab.startswith("ARGM") and not pr:
                continue
            add(subj, f"{pred} {pr}" if pr else pred, arg(*roles[lab]), neg, f"srl:{lab}")
    for i, (h, rel) in enumerate(words):
        if rel == "cop" and 0 < h <= W:
            pn = h - 1
            sb = [c for c in ch[pn] if words[c][1] in ("nsubj", "nsubj:pass")]
            if sb and pos[pn] in NOUNISH:
                si, oi = np_span(sb[0]), np_span(pn)
                add(arg(min(si), max(si) + 1), toks[i], arg(min(oi), max(oi) + 1),
                    any(lem[c] in ("not", "never") for c in ch[pn]), "cop")
        if rel == "appos" and 0 < h <= W:
            si, oi = np_span(h - 1), np_span(i)
            add(arg(min(si), max(si) + 1), "is", arg(min(oi), max(oi) + 1), False, "appos")
    seen, out = set(), []
    for f in facts:
        k = (f["s"].lower(), f["p"].lower(), f["o"].lower())
        if k not in seen:
            seen.add(k); out.append(f)
    return out


def k4_decode(name: str) -> None:
    P = pickle.loads((OUT / f"preds-{name}.pkl").read_bytes())
    lay = pickle.loads((K / f"layers-{name}.pkl").read_bytes())
    rows = [_decode(it, ly) for it, ly in zip(P["items"], lay)]
    (K / f"k4-open-{name}.json").write_text(json.dumps(rows))
    n = sum(map(len, rows))
    print(f"{n} open facts on {len(rows)} texts; by source "
          f"{Counter(f['src'].split(':')[0] for r in rows for f in r)}")


def k4_judge(name: str) -> None:
    import asyncio
    from ..annotate.base import CacheStore
    from ..annotate.kg import KGItem
    from ..config import load_annotators
    from ..kg_judge import CandJudge, _run
    from ..kg_teacher import CACHE
    P = pickle.loads((OUT / f"preds-{name}.pkl").read_bytes())
    rows = json.loads((K / f"k4-open-{name}.json").read_text())
    spec = load_annotators(["astra"])["astra"]
    items, keyed = [], []
    for it, facts in zip(P["items"], rows):
        if not facts:
            continue
        rels = [{"name": p, "description": f"the head {p} the tail, in the passage's own words"}
                for p in sorted({f["p"] for f in facts})]
        jf = [{"key": (frozenset({f["s"]}), f["p"], frozenset({f["o"]}))} for f in facts]
        q = KGItem(id=f"{it['id']}:k4open", tokens=it["tokens"], sentence_spans=it["sentence_spans"],
                   speakers=it.get("speakers"), participants=it["participants"], types=it["types"],
                   relations=rels, domain=it.get("domain") or "")
        q.layers = {"judge_prompt": judge_prompt({**it, "relations": rels}, jf)}
        items.append(q); keyed.append((it["id"], facts))
    print(f"{len(items)} calls, {sum(len(f) for _, f in keyed)} facts", flush=True)
    ann = CandJudge(spec, CacheStore(CACHE, "c3"))
    ann.prompt = PROMPTS["c3"]
    res = asyncio.run(_run(ann, items, "k4-open-astra-c3"))
    out = []
    for tid, facts in keyed:
        r = res.get(f"{tid}:k4open")
        v = r.payload if r is not None and r.ok else {}
        out.append({"id": tid, "judged": bool(v), "facts": [{**f, "verdict": v.get(n)} for n, f in enumerate(facts)]})
    (K / f"k4-labels-{name}.json").write_text(json.dumps(out))
    print(Counter(f["verdict"] for t in out for f in t["facts"]))


def k4_score(name: str) -> None:
    P, L = _load(name)
    lab = json.loads((K / f"k4-labels-{name}.json").read_text())
    nt = sum(t["judged"] for t in lab)
    F = [f for t in lab if t["judged"] for f in t["facts"]]
    ok = [f["verdict"] in POSITIVE for f in F]
    print(f"{len(F)} judged open facts on {nt} texts: judged precision {np.mean(ok):.3f}; "
          f"true facts per text {sum(ok) / max(nt, 1):.2f}")
    for s in sorted({f["src"] for f in F}):
        o = [f["verdict"] in POSITIVE for f in F if f["src"] == s]
        print(f"  {s:14s} {len(o):5d} facts, precision {np.mean(o):.3f}")
    byid = {t["id"]: t for t in lab}
    tot = cov = 0
    for it, t in zip(P["items"], L["texts"]):
        if not t["judged"] or not byid.get(it["id"], {}).get("judged"):
            continue
        opens = [(set(f["names"][0]), set(f["names"][1])) for f in byid[it["id"]]["facts"]
                 if f["verdict"] in POSITIVE]
        for f in t["facts"]:
            if f["verdict"] in POSITIVE:
                h, tl = set(f["key"][0]), set(f["key"][2])
                tot += 1
                cov += any((h & a and tl & b) or (h & b and tl & a) for a, b in opens)
    print(f"schema pool facts whose pair a true open fact covers: {cov}/{tot} = {cov / max(tot, 1):.3f}")


# ── type-fit filter and new-relation bar (M4, offline on saved rows) ──
def m4_filter(run: str, naming: str = "user") -> None:
    """Drop candidates whose argument types break the relation's constraint
    (``m4_business.CONSTRAINT``) before the bar; sweep the new-relation bar."""
    from . import m4_business as m4
    items = m4.load()

    def load_rows(mode):
        rows = json.loads((m4.OUT / f"rows-{run}-{naming}-{mode}.json").read_text())
        for r in rows:
            for c in r["cand"]:
                h, rel, t = c["key"]
                c["key"] = (h, rel, t) if mode == "gold" else (frozenset(h), rel, frozenset(t))
        return rows

    def types_of(it, c, mode):
        if mode == "gold":
            return it["entities"][c["key"][0]]["type"], it["entities"][c["key"][2]]["type"]
        return tuple(c["types"])

    for mode in ("gold", "e2e"):
        rows = load_rows(mode)
        frows = []
        for it, r in zip(items, rows):
            keep = []
            for c in r["cand"]:
                ch, ct = m4.CONSTRAINT[c["rel"]]
                th, tt = types_of(it, c, mode)
                if th in ch and tt in ct:
                    keep.append(c)
            frows.append({**r, "cand": keep})
        a, b = m4.score_rows(items, rows, mode), m4.score_rows(items, frows, mode)
        d = m4.bootstrap(b["per_item"], a["per_item"])
        print(f"\n## {run}, {mode}")
        for nm, sc in (("no filter", a), ("type filter", b)):
            p, r, f = sc["prf"]
            print(f"  {nm:11s}: P {p:.3f} R {r:.3f} F1 {f:.3f}  ({int(sc['counts'][1])} predicted; "
                  f"type violations {sc['type_violations']:.3f})")
        print(f"  F1 filter - none: {100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]")
        line = []
        for dl in np.arange(-4, 2.01, 0.5):
            fa = m4.score_rows(items, rows, mode, delta=dl)["prf"]
            fb = m4.score_rows(items, frows, mode, delta=dl)["prf"]
            line.append(f"{dl:+.1f}: {fa[2]:.3f}/{fb[2]:.3f} (P {fb[0]:.2f} R {fb[1]:.2f})")
        print("  new-relation bar sweep, F1 (none / filter):\n   " + "\n   ".join(line))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["k1", "k2-sample", "k2-score", "layers", "k3",
                                    "k4-decode", "k4-judge", "k4-score", "m4-filter"])
    ap.add_argument("--name", default="m1p3a-n500")
    ap.add_argument("--run", help="m4-filter: the run whose saved M4 rows to filter")
    a = ap.parse_args()
    {"k1": lambda: k1(a.name), "k2-sample": lambda: k2_sample(a.name), "k2-score": k2_score,
     "layers": lambda: layers(a.name), "k3": lambda: k3(a.name), "k4-decode": lambda: k4_decode(a.name),
     "k4-judge": lambda: k4_judge(a.name), "k4-score": lambda: k4_score(a.name),
     "m4-filter": lambda: m4_filter(a.run)}[a.cmd]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
