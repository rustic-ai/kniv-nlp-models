"""Stage 3 (v8 plan r4): fewer verifier candidates, read on the judged M1 file.

    uv run python -m v6.eval.s3_cost fill  [--cfg CFG]       # verifier outputs, deployed settings
    uv run python -m v6.eval.s3_cost sweep --cfg CFG --floor F --ver-top K
    uv run python -m v6.eval.s3_cost read  --cfg CFG

The verifier reads the pair's best trained-relation candidates down to
``ver_floor`` under the bar, ``ver_top`` per pair. A lower floor or fewer per
pair only removes candidates, so every fact kept is one the stage-3 judge
pass already read (``m1s3b-n500``, cut at p 0.05). ``fill`` runs the
extractor once at the deployed settings and stores each candidate's verifier
output and the entity step's output per text; ``sweep`` re-runs the relation
step for one setting from that cache (the verifier is never called: a miss
is an error) and stores its predictions and candidates verified per text;
``read`` puts every swept setting beside v7.1 and the stage-3 reading in
``m1s3c-<cfg>`` and prints the M1 equal-recall table (frozen pool) under
GPT-6, DeepSeek and agreement.
"""
from __future__ import annotations

import argparse
import json
import pickle

from . import heldout_zs as hz
from .m1_precision import OUT, POSITIVE, _same, equal, labels_path

CFGS = {"nest": ("ner+np+nest+merge+ver:2", "s3nest"), "flat": ("ver", "s3")}
RUN, BASE = "ultrabert-380m-v8p3-d123", "m1s3b-n500"
# plan r4 V2: V1c's readings against the fixed reference label file
# (``m1v1c-n500``: v7.1, E1b, D1, D12, d123, V1b (s3) and V1c judged by both)
VERS = {"ver1": ("", "s3", "m1s3b-n500"), "ver1c": ("-v1c", "v1c", "m1v1c-n500"),
        # V2: its facts match any judged fact of the reference file (no config of its own in it)
        "ver2": ("-v2", None, "m1v1c-n500")}


def _ver():
    import os
    return os.environ.get("KNIV_VERIFIER", f"{RUN}-ver1").rsplit("-", 1)[-1]


def _lab(cfg):
    return CFGS[cfg][1] if _ver() == "ver1" else VERS[_ver()][1]


def _base():
    return VERS[_ver()][2]


def _cache(cfg):
    return OUT / f"s3cost-cache-{cfg}{VERS[_ver()][0]}.pkl"


def _extractor(cfg):
    import os
    os.environ.setdefault("KNIV_VER_THRESHOLD", "0.05")
    from ..infer import Extractor
    from .m1_precision import _configure
    ex = Extractor(RUN)
    boost0 = ex.span_boost
    ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
    ex.unseen_offset = ex.unseen_offset - hz.SHIFT
    ex.top = max(ex.top, 4)
    _configure(ex, CFGS[cfg][0], boost0)
    return ex


def _items():
    return pickle.loads((OUT / f"preds-{_base()}.pkl").read_bytes())["items"]


def fill(cfg):
    ex = _extractor(cfg)
    ents, ver = {}, {}
    orig_e, orig_v = ex._entities, ex._verify
    cur = {}

    def e(w, types):
        r = orig_e(w, types)
        ents[cur["id"]] = r
        return r

    def v(w, nodes, spans, parts, cands):
        out = orig_v(w, nodes, spans, parts, cands)
        for j, c in enumerate(cands):
            ver[(cur["id"], *c)] = (out[j], ex._last_verdicts.get(j))
        return out
    ex._entities, ex._verify = e, v
    items = _items()
    for n, it in enumerate(items):
        cur["id"] = it["id"]
        hz.predict(ex, [it], "e2e")
        if n % 50 == 0:
            print(f"{n}/{len(items)} texts, {len(ver)} candidates", flush=True)
    _cache(cfg).write_bytes(pickle.dumps({"ents": ents, "ver": ver}))
    print(f"filled: {len(ents)} texts, {len(ver)} candidates")
    return 0


def sweep(cfg, floor, ver_top, cap=0, rank="near"):
    C = pickle.loads(_cache(cfg).read_bytes())
    ex = _extractor(cfg)
    ex.ver_floor, ex.ver_top = float(floor), int(ver_top) or None
    ex.ver_cap, ex.ver_rank = (int(cap) or None), rank
    cur, n_read = {}, []
    ex._entities = lambda w, types: C["ents"][cur["id"]]

    def v(w, nodes, spans, parts, cands):
        n_read[-1] += len(cands)
        got = [C["ver"][(cur["id"], *c)] for c in cands]     # KeyError: not a subset of the fill
        ex._last_verdicts = {j: g[1] for j, g in enumerate(got) if g[1] is not None}
        return [g[0] for g in got]
    ex._verify = v
    preds, margins = [], []
    for it in _items():
        cur["id"] = it["id"]
        n_read.append(0)
        r = hz.predict(ex, [it], "e2e")[0]
        preds.append(sorted({c["key"] for c in r["cand"] if c["m"] > 0}, key=str))
        margins.append({c["key"]: max(c["m"], -1e9) for c in r["cand"] if c["m"] > 0})
    import os
    name = f"f{floor:g}k{ver_top}" + (f"c{cap}{rank}" if cap else "") + ("r" if os.environ.get("KNIV_UNSEEN_FROM") else "")
    p = OUT / f"s3cost-{cfg}{VERS[_ver()][0]}-{name}.pkl"
    p.write_bytes(pickle.dumps({"preds": preds, "margins": margins, "n_read": n_read}))
    print(f"{cfg} {name}: {sum(map(len, preds))} facts, {sum(n_read) / len(n_read):.1f} candidates verified per text")
    return 0


def read(cfg, drop_unjudged=False):
    """``drop_unjudged``: leave out swept facts no judged fact matches (span
    variants the merge kept once a sibling was dropped) instead of counting
    them false: the reading if they are as precise as the judged ones."""
    lab, base, tag = _lab(cfg), _base(), VERS[_ver()][0]
    P = pickle.loads((OUT / f"preds-{base}.pkl").read_bytes())
    sw = sorted(OUT.glob(f"s3cost-{cfg}{tag}-f*.pkl"))
    run = f"m1s3c-{cfg}{tag}" + ("-judged" if drop_unjudged else "")
    names = ["v7.1", "e1b", "d1", "d12", "d123"] + ([lab] if lab else ["s3", "v1c"])
    Q = {"run": run, "items": P["items"], "preds": {}, "margins": {}}
    for c in names:
        Q["preds"][c], Q["margins"][c] = P["preds"][c], P["margins"][c]
    cost = {}
    for f in sw:
        S = pickle.loads(f.read_bytes())
        c = f.stem.split("-")[-1]
        Q["preds"][c], Q["margins"][c] = S["preds"], S["margins"]
        if drop_unjudged:
            L0 = json.loads(labels_path(base, "astra", "c3").read_text())["texts"]
            keep = [{k: m for k, m in ms.items() if any(_same(k, (set(f["key"][0]), f["key"][1], set(f["key"][2])))
                                                         for f in t["facts"])} for t, ms in zip(L0, S["margins"])]
            Q["margins"][c] = keep
            Q["preds"][c] = [sorted(ms, key=str) for ms in keep]
        cost[c] = sum(S["n_read"]) / len(S["n_read"])
    (OUT / f"preds-{run}.pkl").write_bytes(pickle.dumps(Q))
    cfgs = names + [c for c in Q["preds"] if c not in names]
    for judge in ("astra", "deepseek", "both"):
        L = json.loads(labels_path(base, judge, "c3").read_text())
        un = 0
        for i, t in enumerate(L["texts"]):
            for c in cfgs[len(names):]:
                for k in Q["margins"][c][i]:
                    f = next((f for f in t["facts"] if (drop_unjudged or lab is None or lab in f["src"])
                              and _same((set(f["key"][0]), f["key"][1], set(f["key"][2])), k)), None)
                    if f is None:
                        un += 1
                    elif c not in f["src"]:
                        f["src"] = f["src"] + [c]
        L["configs"] = cfgs
        labels_path(run, judge, "c3").write_text(json.dumps(L))
        print(f"\n## {judge} (sweep facts not matched to a judged fact: {un})")
        equal(run, judge, "c3", pool=["v7.1", "e1b", "d1", "d12", "d123"])
    print("\ncandidates verified per text: " + ", ".join(f"{c} {v:.1f}" for c, v in cost.items()))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["fill", "sweep", "read"])
    ap.add_argument("--cfg", default="nest", choices=list(CFGS))
    ap.add_argument("--floor", type=float, default=3.0)
    ap.add_argument("--ver-top", type=int, default=4)
    ap.add_argument("--cap", type=int, default=0, help="sweep: at most CAP candidates verified per window")
    ap.add_argument("--rank", default="near", choices=["near", "top"])
    ap.add_argument("--drop-unjudged", action="store_true", help="read: leave unjudged swept facts out")
    a = ap.parse_args()
    if a.cmd == "fill":
        return fill(a.cfg)
    if a.cmd == "sweep":
        return sweep(a.cfg, a.floor, a.ver_top, a.cap, a.rank)
    return read(a.cfg, a.drop_unjudged)


if __name__ == "__main__":
    raise SystemExit(main())
