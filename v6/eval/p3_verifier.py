"""P3' (v8 plan r4): does one threshold on the verifier's probability serve
every text type?

    uv run python -m v6.eval.p3_verifier --run ultrabert-380m-v8p3-d123-ver1

The threshold is fitted on the verifier's held-out fifth (D1's judged
candidates on texts in ``heldout_base``'s fifth, gold entities; never
trained on), as the F1-best probability for "expressed" (supported, negated,
hedged). It is then applied unchanged to:

- M1, short dev, end to end: the verifier's re-ranking of d123's flat
  candidates and of v7.1's nested candidates (``verifier score-m1`` files),
  F1 from judged precision and pooled recall;
- CoNLL04 test, end to end: v7.1's flat and nested candidates, judged the
  same way (never in any training).

Pass (plan r4 P3'): F1 at the fitted threshold within 1 point of each set's
own best. p = 0.5 is reported too.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

GRID = np.linspace(0.02, 0.98, 49)


def _logit(p):
    return float(np.log(p + 1e-6) - np.log(1 - p + 1e-6))


def heldout_probs(run: str, device, limit: int) -> tuple[np.ndarray, np.ndarray]:
    from ..label_bank import load as load_bank
    from ..train.verifier import DATA, _corpus, collate, encode, load_verifier
    model, tok = load_verifier(run, device)
    bank = {r["name"]: r for r in load_bank()["relations"]}
    recs = json.loads((_corpus() / DATA).read_text())["records"]
    exs = []
    for r in recs:
        if r["split"] != "held":
            continue
        cache: dict = {}
        for c in r["cands"]:
            if c.get("src") != "d1":
                continue
            e = encode(r, c, tok, bank, None, None, cache=cache)
            if e is not None:
                exs.append(e)
        if len(exs) >= limit:
            break
    P, Y = [], []
    with torch.no_grad():
        for i in range(0, len(exs), 32):
            ch = exs[i:i + 32]
            P += (1 - torch.softmax(model(collate(ch, tok.pad_token_id, device)), -1)[:, 1]).tolist()
            Y += [e["y"] != 1 for e in ch]
    return np.array(P), np.array(Y)


def cls_f1(p, y, thr):
    k = p >= thr
    tp = (k & y).sum()
    return 2 * tp / max(k.sum() + y.sum(), 1)


def e2e_curve(name: str, cfg: str):
    """(logit, expressed) per prediction of ``cfg`` and the pool size, as
    ``m1_precision equal`` counts them."""
    from .m1_precision import OUT, POSITIVE, _same, labels_path
    P = pickle.loads((OUT / f"preds-{name}.pkl").read_bytes())
    L = json.loads(labels_path(name, "astra", "c3").read_text())
    rows, total = [], 0
    for i, t in enumerate(L["texts"]):
        if not t["judged"]:
            continue
        total += sum(f["verdict"] in POSITIVE for f in t["facts"])
        for k, m in P["margins"][cfg][i].items():
            f = next((f for f in t["facts"] if cfg in f["src"]
                      and _same((set(f["key"][0]), f["key"][1], set(f["key"][2])), k)), None)
            rows.append((m, f is not None and f["verdict"] in POSITIVE))
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), total


def e2e_f1(m, ok, total, thr):
    k = m >= _logit(thr)
    tp = ok[k].sum()
    p = tp / max(k.sum(), 1)
    r = tp / max(total, 1)
    return 2 * p * r / max(p + r, 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="ultrabert-380m-v8p3-d123-ver1")
    ap.add_argument("--tag", default="ver1b")
    ap.add_argument("--limit", type=int, default=8000)
    ap.add_argument("--device")
    ap.add_argument("--fit-set", help="fit on judged end-to-end candidates: NAME|CONFIG|CUT_THRESHOLD")
    ap.add_argument("--out", type=Path, default=Path("v6/runs/k-checks/p3-verifier.json"))
    a = ap.parse_args()
    if a.fit_set:     # judged end-to-end candidates outside M1 (``name|config|threshold it was cut at``)
        name, cfg, t0 = a.fit_set.split("|")
        m, ok, total = e2e_curve(name, cfg)
        m = m + _logit(float(t0))          # its margins are log-odds over t0: back to log-odds
        fs = [e2e_f1(m, ok, total, t) for t in GRID]
        thr = float(GRID[int(np.argmax(fs))])
        y = ok
        print(f"fitted on {name} ({cfg}; {len(m):,} judged candidates, pool {total}): threshold {thr:.2f} "
              f"(F1 {max(fs):.3f}; at 0.5 {e2e_f1(m, ok, total, 0.5):.3f})")
    else:
        device = torch.device(a.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        p, y = heldout_probs(a.run, device, a.limit)
        f = [cls_f1(p, y, t) for t in GRID]
        thr = float(GRID[int(np.argmax(f))])
        print(f"held-out fifth (D1 candidates, gold entities): {len(y):,}, {y.mean():.2f} expressed; "
              f"fitted threshold {thr:.2f} (F1 {max(f):.3f}; at 0.5 {cls_f1(p, y, 0.5):.3f})")
    sets = [("M1 short dev, d123 flat", f"m1p3b-n500-v-{a.tag}-d123", "d123+v"),
            ("M1 short dev, v7.1 nested", "v2-n500", "ner+np+nest+merge:2+v"),
            ("CoNLL04 test, v7.1 flat", f"ultrabert-380m-v7b8ev-conll04-v-{a.tag}-base", "base+v"),
            ("CoNLL04 test, v7.1 nested", f"ultrabert-380m-v7b8ev-conll04-v-{a.tag}-ner+np+nest+merge:2",
             "ner+np+nest+merge:2+v")]
    out = {"threshold": thr, "heldout_n": int(len(y)), "sets": {}}
    print("\n| set | F1 at fitted | F1 at 0.5 | best F1 (threshold) | fitted − best |\n|---|---|---|---|---|")
    ok_all = True
    for label, name, cfg in sets:
        m, ok, total = e2e_curve(name, cfg)
        fs = [e2e_f1(m, ok, total, t) for t in GRID]
        b = int(np.argmax(fs))
        ff, fh = e2e_f1(m, ok, total, thr), e2e_f1(m, ok, total, 0.5)
        ok_all &= (fs[b] - ff) <= 0.01
        out["sets"][label] = {"fitted": ff, "half": fh, "best": fs[b], "best_thr": float(GRID[b])}
        print(f"| {label} | {ff:.3f} | {fh:.3f} | {fs[b]:.3f} ({GRID[b]:.2f}) | {100 * (ff - fs[b]):+.1f} |")
    print("\nPASS" if ok_all else "\nFAIL: the fitted threshold is more than 1 F1 off some set's best")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.fit_set:
        a.out = a.out.with_name("p3-verifier-e2e.json")
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
