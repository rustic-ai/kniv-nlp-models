"""Failure analysis: does the type's text decide zero-shot entity typing?
CrossNER, each domain's types described four ways, the kind a user would
write -- none, "a <name>" (the scorecard's), a short definition, and the
definition with a specificity hint ("use this rather than person") -- scored
as span F1 at the run's shipped span offset and as typing accuracy on gold
spans, with the head's most frequent confusions.

    uv run python -m v6.train.type_text_eval --runs ultrabert-380m-v7b8ev \\
        --descriptions data/v6-eval/type_descriptions.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from ..config import RUNS_DIR
from . import span_eval as SE
from .gold_eval import HF_REPO, load_run
from .probes import span_vectors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", required=True)
    ap.add_argument("--descriptions", type=Path, required=True)
    ap.add_argument("--device")
    args = ap.parse_args()
    d = json.loads(args.descriptions.read_text())
    defs, hints = d["definitions"], d["hints"]
    modes = {"none": lambda n: "", "name": lambda n: f"a {n}",
             "definition": lambda n: defs.get(n, f"a {n}"),
             "definition_hint": lambda n: defs.get(n, f"a {n}") + (f"; {hints[n]}" if n in hints else "")}
    device = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    out = {}
    for run in args.runs.split(","):
        model, tok, spec, vocab, _ = load_run(run, HF_REPO, device)
        model.eval()
        from huggingface_hub import hf_hub_download
        try:
            off = float(json.loads(Path(hf_hub_download(HF_REPO, f"{run}/calibration.json")).read_text())
                        .get("span_offset", 0.0))
        except Exception:
            off = 0.0
        res = {}
        for dom in SE.DOMAINS:
            items = SE.load_crossner(dom)
            labels = sorted({t for it in items for t, _, _ in it["spans"]})
            gold = [{(a, b, t) for t, a, b in it["spans"]} for it in items]
            r = {}
            for m, fn in modes.items():
                f1 = SE.f1(gold, SE.kniv_predict(model, tok, spec, vocab, device, items, labels, offset=off,
                                                 describe=fn))["f1"]
                _, _, H, G = span_vectors(model, tok, spec, vocab, device, items, labels, describe=fn)
                conf = Counter((labels[g], labels[h]) for h, g in zip(H, G) if h != g)
                r[m] = {"f1": f1, "typing_acc": float((H == G).mean()),
                        "confusions": [(a, b, n) for (a, b), n in conf.most_common(6)]}
            res[dom] = r
            print(f"TT {run} {dom} (offset {off:+.1f}): " + "  ".join(
                f"{m} F1 {v['f1']:.3f} typing {v['typing_acc']:.3f}" for m, v in r.items()), flush=True)
            print(f"TT {run} {dom} confusions (name): " + "; ".join(
                f"{a} -> {b} ({n})" for a, b, n in r["name"]["confusions"]), flush=True)
        out[run] = res
        (RUNS_DIR / "failure").mkdir(parents=True, exist_ok=True)
        (RUNS_DIR / "failure" / "type_text.json").write_text(json.dumps(out, indent=1))
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
