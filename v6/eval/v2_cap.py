"""A cap on verified candidates per window (plan r4, V2 step 1): what the
per-candidate verifier keeps of its gain when it reads only ``k`` of a
window's candidates, and what it costs.

    uv run python -m v6.eval.v2_cap s4 [--caps 8 16 32 64]

``s4``: stage 4's long windows (V1b, d123 flat, threshold 0.6). The pilot
logged every eligible candidate's margin and verified a random 40 of them,
and judged a random 30 of a text's margin / verifier disagreements (GPT-6
c3, and DeepSeek c3k on the same candidates). A cap verifies the window's
top ``k`` candidates under a ranking (``near``: closest to the bar;
``top``: highest margin; ``below``: highest under the bar, then above); the
rest are left to the margin. The verifier changes a decision only on a
disagreement, so each rule and cap is read as the change in true and false
facts per window against the margin alone, beside the uncapped verifier's:
each sampled candidate weighted by its window's candidates over those
sampled, each judged one by its window's sampled disagreements over those
judged. The ranking is chosen here (training-domain text), never on M1.

Cost per tier: the per-candidate verifier's measured CPU cost
(``verifier_onnx bench`` / ``student``, ONNX INT8) times ``min(n, k)``,
over the release's end-to-end time at that text length.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np

S4 = Path("v6/runs/s4-pilot")
THR = 0.6
POS = ("supported", "negated", "hedged")
# ms per candidate, ONNX INT8 CPU, at about 128 / 768 subwords (plan r4 review follow-up)
PER_CAND = {"tiny (17M)": (2, 20), "medium (68M)": (12, 115), "large (150M)": (24, 224), "xlarge-u (380M)": (66, 504)}
RELEASE = {"tiny (17M)": "ettin-17m", "medium (68M)": "ettin-68m", "large (150M)": "modernbert-base",
           "xlarge-u (380M)": "ultrabert-380m"}


def _rows():
    rows = []
    for p in sorted(S4.glob("pred-*.pkl")):
        rows += pickle.loads(p.read_bytes())
    return rows


def _labels():
    """(text id, head, tail, relation) -> {"gpt6": bool|None, "deepseek": bool|None}."""
    L = json.loads((S4 / "labels-astra-c3.json").read_text())
    D = json.loads(Path("v6/runs/s6-second/s6-labels-s4-deepseek-c3k.json").read_text())
    D = {t["id"]: t for t in D}
    out = {}
    for t in L:
        if not t["judged"]:
            continue
        d = D.get(t["id"])
        for n, c in enumerate(t["cands"]):
            g = None if c["judge"] is None else c["judge"] in POS
            s = None
            if d is not None and d["judged"] and n < len(d["facts"]) and d["facts"][n]["verdict"] is not None:
                s = d["facts"][n]["verdict"] in POS
            out[(t["id"], c["head"], c["tail"], c["relation"])] = {"gpt6": g, "deepseek": s}
    return out


def _rank(cands, rule):
    m = np.array([c["m"] for c in cands])
    if rule == "near":
        return np.argsort(np.abs(m), kind="stable")
    if rule == "top":
        return np.argsort(-m, kind="stable")
    if rule == "below":                     # highest under the bar first, then the lowest above it
        under = np.where(m <= 0, -m, 1e6 + m)
        return np.argsort(under, kind="stable")
    raise ValueError(rule)


def s4(caps) -> int:
    rows, lab = _rows(), _labels()
    judged_ids = {k[0] for k in lab}
    rules = ["near", "top", "below"]
    readings = ["gpt6", "deepseek", "both"]
    n_c = np.array([len(r["cands"]) for r in rows])
    print(f"{len(rows)} windows, {n_c.mean():.0f} eligible candidates per window (median {np.median(n_c):.0f}, "
          f"90th percentile {np.percentile(n_c, 90):.0f}); {len(judged_ids)} with judged disagreements")
    # per window: weights, and each judged disagreement's contribution
    acc = {(rule, k, rd): np.zeros(2) for rule in rules for k in caps + [None] for rd in readings}
    dis_kept = {(rule, k): 0.0 for rule in rules for k in caps}
    dis_all = 0.0
    n_win = 0
    for r in rows:
        cs = r["cands"]
        samp = [j for j, c in enumerate(cs) if c["p"] is not None]
        if not samp:
            continue
        n_win += 1
        w = len(cs) / len(samp)
        dis = [j for j in samp if (cs[j]["m"] > 0) != (cs[j]["p"] >= THR)]
        dis_all += w * len(dis)
        ranks = {rule: _rank(cs, rule) for rule in rules}
        inset = {(rule, k): set(ranks[rule][:k].tolist()) for rule in rules for k in caps}
        for key, s in inset.items():
            dis_kept[key] += w * sum(j in s for j in dis)
        if r["id"] not in judged_ids or not dis:
            continue
        J = [j for j in dis if (r["id"], cs[j]["head"], cs[j]["tail"], cs[j]["relation"]) in lab]
        if not J:
            continue
        w2 = w * len(dis) / len(J)
        for j in J:
            c = cs[j]
            L = lab[(r["id"], c["head"], c["tail"], c["relation"])]
            keep = c["p"] >= THR                      # the verifier keeps what the margin drops, or drops what it keeps
            sign = 1.0 if keep else -1.0
            for rd in readings:
                y = L["gpt6"] if rd == "gpt6" else L["deepseek"] if rd == "deepseek" else \
                    (None if L["gpt6"] is None or L["deepseek"] is None else L["gpt6"] and L["deepseek"])
                if y is None:
                    continue
                d = np.array([sign, 0.0]) if y else np.array([0.0, sign])     # (true facts, false facts)
                for rule in rules:
                    acc[(rule, None, rd)] += w2 * d
                    for k in caps:
                        if j in inset[(rule, k)]:
                            acc[(rule, k, rd)] += w2 * d
    print(f"\ndisagreements per window (where the verifier changes a decision): {dis_all / n_win:.1f}")
    print("\nshare of the verifier's decision changes inside the cap, by ranking:")
    print("| ranking | " + " | ".join(f"cap {k}" for k in caps) + " |\n|---|" + "---|" * len(caps))
    for rule in rules:
        print(f"| {rule} | " + " | ".join(f"{dis_kept[(rule, k)] / dis_all:.2f}" for k in caps) + " |")
    for rd in readings:
        print(f"\n## {rd}: change against the margin alone, per window (true facts, false facts; net = true − false)")
        full = acc[("near", None, rd)] / n_win
        print(f"uncapped verifier: true {full[0]:+.2f}, false {full[1]:+.2f}, net {full[0] - full[1]:+.2f}")
        print("| ranking | " + " | ".join(f"cap {k}: true / false (share of net)" for k in caps) + " |\n|---|"
              + "---|" * len(caps))
        for rule in rules:
            cells = []
            for k in caps:
                v = acc[(rule, k, rd)] / n_win
                cells.append(f"{v[0]:+.2f} / {v[1]:+.2f} ({(v[0] - v[1]) / (full[0] - full[1]):.2f})")
            print(f"| {rule} | " + " | ".join(cells) + " |")
    # cost
    print("\ncost of the capped per-candidate verifier, total time over the release's end to end (ONNX INT8, CPU)")
    long_n = np.array([len(r["cands"]) for r in rows if len(r["cands"])])
    print("| tier | release s, 40 / 500 words | " + " | ".join(f"cap {k}, short / 500 words" for k in caps)
          + " | uncapped |\n|---|---|" + "---|" * (len(caps) + 1))
    for tier, (ms_s, ms_l) in PER_CAND.items():
        rel = json.loads(Path(f"release/{RELEASE[tier]}/onnx/bench-int8.json").read_text())
        t40, t500 = rel["40 words"]["total_s"], rel["500 words"]["total_s"]
        # short: 22.4 verified candidates per M1 text (deployed floor 3); long: the pilot's windows
        cells = []
        for k in caps:
            s = 1 + min(22.4, k) * ms_s / 1000 / t40
            l_ = 1 + np.minimum(long_n, k).mean() * ms_l / 1000 / t500
            cells.append(f"{s:.1f}x / {l_:.1f}x")
        un = (1 + 22.4 * ms_s / 1000 / t40, 1 + long_n.mean() * ms_l / 1000 / t500)
        print(f"| {tier} | {t40:.3f} / {t500:.3f} | " + " | ".join(cells) + f" | {un[0]:.1f}x / {un[1]:.0f}x |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["s4"])
    ap.add_argument("--caps", type=int, nargs="+", default=[8, 16, 32, 64])
    a = ap.parse_args()
    return s4(a.caps)


if __name__ == "__main__":
    raise SystemExit(main())
