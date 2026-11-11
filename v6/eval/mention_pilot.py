"""P1 pilot: mention proposals from the linguistic layer, end to end.

    uv run python -m v6.eval.mention_pilot --run ultrabert-380m-v7b8ev --short 600

Each configuration (extra proposers, span boost) runs the ``heldout_zs`` e2e
mode on CoNLL04 test and a short-dev sample with the run's deployed
calibration, and reports end-to-end F1, entity mention F1 and the share of
gold facts whose two arguments were both found, with a paired bootstrap
against the deployed configuration (no extra proposers).

Pass (plan r2): both arguments found up 10+ points on short dev and CoNLL04,
with end-to-end F1 up.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from ..config import RUNS_DIR
from . import heldout_zs as hz


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="ultrabert-380m-v7b8ev")
    ap.add_argument("--short", type=int, default=600, help="short-dev texts")
    ap.add_argument("--configs", default="base,ner,np,ner+np",
                    help="comma-separated; each a +-joined set of proposers, 'base' = none; "
                         "':<boost>' overrides the span boost, e.g. ner+np:1.0")
    ap.add_argument("--device")
    a = ap.parse_args()
    from ..infer import Extractor
    data = {"conll04": hz.load_conll04(), "short_dev": hz.load_short_dev(a.short)}
    ex = Extractor(a.run, device=a.device)
    boost0 = ex.span_boost
    dep = ex.unseen_offset
    ex.rel_offsets = {k: v + hz.SHIFT for k, v in ex.rel_offsets.items()}
    ex.unseen_offset = dep - hz.SHIFT
    ex.top = max(ex.top, 4)
    print(f"{a.run}: deployed span_boost {boost0}, span_offset {ex.span_offset}, "
          f"unseen_offset {dep:+.2f}", flush=True)
    counts, summary = {}, {}
    for cfg in a.configs.split(","):
        name, _, b = cfg.partition(":")
        parts = name.split("+")
        ex.nested = "nest" in parts
        ex.proposers = tuple(p for p in parts if p not in ("base", "nest"))
        ex.span_boost = float(b) if b else (boost0 if boost0 is not None else 0.0)
        for s, items in data.items():
            t0 = time.time()
            rows = hz.predict(ex, items, "e2e")
            counts[(cfg, s)] = np.array([hz.score_item(r, "e2e") for r in rows])
            es = hz.entity_stats(rows)
            p, r, f = hz.f1(counts[(cfg, s)])
            summary[f"{cfg}|{s}"] = {"f1": f, "p": p, "r": r, "mention_f1": es["mention"][2],
                                     "mention_r": es["mention"][1], "reach": es["reach"],
                                     "boost": ex.span_boost}
            print(f"{cfg:14s} {s:9s} e2e F1 {f:.3f} (P {p:.3f} R {r:.3f}) | mentions F1 "
                  f"{es['mention'][2]:.3f} R {es['mention'][1]:.3f} | both arguments found "
                  f"{es['reach']:.3f} [{time.time() - t0:.0f}s]", flush=True)
    base = a.configs.split(",")[0]
    print(f"\npaired bootstrap of e2e F1 against {base} (texts resampled)")
    for cfg in a.configs.split(",")[1:]:
        for s in data:
            d, lo, hi = hz.bootstrap(counts[(cfg, s)], counts[(base, s)])
            summary[f"{cfg}-vs-{base}|{s}"] = [d, lo, hi]
            print(f"  {cfg:14s} {s:9s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out = RUNS_DIR / "mention-pilot" / f"{a.run}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
