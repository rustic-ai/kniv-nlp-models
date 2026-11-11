"""P6 pilot (v8 plan r2): partial-name grouping on Re-DocRED end to end.

    uv run python -m v6.eval.partial_names_pilot --run ultrabert-380m-v7b8ev

Runs ``gold_eval.eval_redocred(oracle=True)`` on dev and test, which scores
every grouping mode of ``build_entities`` over the model's own mentions and
coreference at the same calibrated offset, and compares the run's deployed
rule (``calibration.json`` ``entity_coref``) with the same rule plus
``partial_names``: end-to-end F1 with a paired bootstrap over documents.

``build_entities`` is wrapped (at run time, not edited) to record both
groupings per document, so the pilot also reports, per mode, the share of
predicted entities that fuse two gold entities (merged) and of gold entities
split across predicted ones, and lists the merges the partial rule adds.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict

import numpy as np
import torch

from ..config import RUNS_DIR

REC: dict = {}


def wrap():
    import v6.entities as E
    orig = E.build_entities

    def rec(tokens, sentence_spans, ner, chains, coref="all", partial_names=False, **kw):
        out = orig(tokens, sentence_spans, ner, chains, coref=coref, partial_names=partial_names, **kw)
        if kw.get("link") is None and kw.get("link_merge") is None and ner != "clusters":
            REC[(tuple(tokens[:40]), len(tokens), coref, partial_names)] = out[1]
        return out
    E.build_entities = rec


def f1(c):
    tp, p, g = np.asarray(c).sum(0)
    pr, rc = tp / max(p, 1), tp / max(g, 1)
    return pr, rc, 2 * pr * rc / max(pr + rc, 1e-12)


def boot(a, b, n=2000, seed=13):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)
    d = [f1(a[i])[2] - f1(b[i])[2] for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return f1(a)[2] - f1(b)[2], float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def groups(vertex, sp):
    return [sorted({(sp[m["sent_id"]][0] + m["pos"][0], sp[m["sent_id"]][0] + m["pos"][1]) for m in v})
            for v in vertex or []]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="ultrabert-380m-v7b8ev")
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()
    wrap()
    from ..gold.redocred import load_rel_items
    from .. import infer
    from ..train import gold_eval as G
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ex = infer.Extractor(a.run, device=str(device))       # loads calibration onto the model
    model = ex.model
    model.entity_coref = ex.entity_coref
    rule = ex.entity_coref
    print(f"{a.run}: deployed entity_coref {rule}", flush=True)
    out = {"run": a.run, "rule": rule}
    for split in a.splits.split(","):
        REC.clear()
        t0 = time.time()
        res = G.eval_redocred(model, ex.tok, ex.spec, ex.vocab, device, a.limit, oracle=True, split=split)
        rows = res["items"]
        base_k = f"rel_e2e_{rule}"           # GRID key, same offset as its partial variant
        keys = {m: f"rel_e2e_{m}" for m in ("compatible", "none")}
        keys.update({f"{m}+partial": f"rel_e2e_{m}_partial" for m in ("all", "compatible", "none")})
        res_s = {}
        for name, k in keys.items():
            c = [r[k] for r in rows if k in r]
            res_s[name] = {"p": f1(c)[0], "r": f1(c)[1], "f1": f1(c)[2]}
        deployed = [r["rel_e2e_f1"] for r in rows]
        res_s["deployed (rel offsets)"] = dict(zip(("p", "r", "f1"), f1(deployed)))
        d = boot([r[f"rel_e2e_{rule}_partial"] for r in rows], [r[base_k] for r in rows])
        res_s["paired: partial - rule"] = d
        # merges and splits from the recorded groupings
        items = load_rel_items(split, limit=a.limit)
        stats = defaultdict(lambda: [0, 0, 0, 0])      # merged, predicted, split, gold
        added = []
        for it in items:
            sp = [tuple(x) for x in it.layers["sentence_spans"]]
            key = (tuple(it.tokens[:40]), len(it.tokens))
            ent_of = {(a_, e + 1): gi for gi, en in enumerate(it.entities) for a_, e in en["mentions"]}
            gv = {}
            for pn in (False, True):
                v = REC.get(key + (rule, pn))
                if v is None:
                    continue
                gs = groups(v, sp)
                gv[pn] = gs
                owners = [{ent_of[m] for m in g if m in ent_of} for g in gs]
                single = [next(iter(o)) for o in owners if len(o) == 1]
                s = stats[pn]
                s[0] += sum(len(o) > 1 for o in owners); s[1] += len(gs)
                s[2] += sum(single.count(x) - 1 for x in set(single)); s[3] += max(len(set(single)), 1)
            if False in gv and True in gv:
                before = [set(g) for g in gv[False]]
                for g in gv[True]:
                    parts = [b for b in before if b <= set(g)]
                    if len(parts) > 1:
                        names = [" ".join(it.tokens[x:y]) for x, y in sorted({p_ for b in parts
                                                                              for p_ in list(b)[:1]})]
                        own = [{ent_of.get(m) for m in b} - {None} for b in parts]
                        known = [o for o in own if o]
                        verdict = ("unmatched" if len(known) < 2 else
                                   "right" if len(set().union(*known)) == 1 else "wrong")
                        added.append({"names": names, "verdict": verdict})
        res_s["merged_rate"] = {str(k): v[0] / max(v[1], 1) for k, v in stats.items()}
        res_s["split_rate"] = {str(k): v[2] / max(v[3], 1) for k, v in stats.items()}
        res_s["merges_added"] = len(added)
        res_s["merges_added_by_verdict"] = {v: sum(x["verdict"] == v for x in added)
                                            for v in ("right", "wrong", "unmatched")}
        random.Random(0).shuffle(added)
        res_s["sample"] = added[:a.show]
        out[split] = res_s
        print(f"\n{split} ({len(rows)} docs, {time.time() - t0:.0f}s)")
        for k_, v in res_s.items():
            if isinstance(v, dict) and "f1" in v:
                print(f"  {k_:24s} F1 {v['f1']:.4f} P {v['p']:.4f} R {v['r']:.4f}")
        print(f"  paired {rule}+partial - {rule}: {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
        print(f"  merged rate {res_s['merged_rate']}, split rate {res_s['split_rate']}")
        print(f"  merges the partial rule adds: {len(added)} {res_s['merges_added_by_verdict']} "
              "(right: the matched parts are one gold entity; unmatched: under two parts match gold)")
        for x in res_s["sample"]:
            print(f"    {x['verdict']:9s}  {' | '.join(x['names'])}")
    p = RUNS_DIR / "partial-names-pilot" / f"{a.run}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"written {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
