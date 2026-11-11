"""Entity spans by type name: kniv's span head against GLiNER, same sentences,
same type names, exact-match span F1 (micro over the set).

    uv run python -m v6.train.span_eval --runs deberta-v3-large-p2-zs-v62-ftr --gliner

Sets:
- ``fewnerd``: Few-NERD (supervised) test, a fixed 3,000-sentence sample, 66
  fine types. kniv trained on Few-NERD's training split; GLiNER did not.
- ``crossner_<domain>``: CrossNER (MIT) test, politics, science, music,
  literature and AI, each with its own type set -- zero-shot for kniv (none of
  these type sets were trained). GLiNER's training data may overlap them.

Type names are given as written in each dataset's tag set, made readable
(``programlang`` -> ``programming language``); kniv reads them as
``entity type: <name>``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..config import DATA_DIR, RUNS_DIR
from .gold_eval import HF_REPO, load_run

CROSSNER = DATA_DIR / "crossner"
DESCRIBE = False
DOMAINS = ("politics", "science", "music", "literature", "ai")
READABLE = {"programlang": "programming language", "misc": "miscellaneous",
            "organisation": "organisation", "politicalparty": "political party",
            "musicalartist": "musical artist", "musicalinstrument": "musical instrument",
            "musicgenre": "music genre", "literarygenre": "literary genre",
            "academicjournal": "academic journal", "chemicalcompound": "chemical compound",
            "chemicalelement": "chemical element", "astronomicalobject": "astronomical object",
            "researcher": "researcher", "university": "university", "task": "task",
            "metrics": "metric", "algorithm": "algorithm", "field": "field of study"}


def _readable(t: str) -> str:
    return READABLE.get(t, t)


def load_crossner(domain: str) -> list[dict]:
    items, toks, tags = [], [], []
    for line in (CROSSNER / f"{domain}-test.txt").read_text().splitlines() + [""]:
        if not line.strip():
            if toks:
                spans, cur = [], None
                for i, t in enumerate(tags + ["O"]):
                    if t.startswith("B-") or t == "O" or (t.startswith("I-") and (cur is None or cur[0] != t[2:])):
                        if cur is not None:
                            spans.append((cur[0], cur[1], i))
                            cur = None
                        if t.startswith(("B-", "I-")):
                            cur = (t[2:], i)
                items.append({"tokens": toks, "spans": [(_readable(a), s, e) for a, s, e in spans]})
            toks, tags = [], []
            continue
        w, t = line.split("\t") if "\t" in line else line.split()
        toks.append(w); tags.append(t)
    return items


def load_fewnerd(n: int = 3000) -> list[dict]:
    from ..gold import fewnerd
    rows = fewnerd.load("test")
    rng = np.random.default_rng(7)
    idx = sorted(rng.choice(len(rows), size=min(n, len(rows)), replace=False).tolist())
    fix = {"GPE": "geopolitical entity", "education": "educational institution",
           "government or governmentagency": "government or government agency"}
    name = lambda c: fix.get(fewnerd.type_text(c)[0], fewnerd.type_text(c)[0])
    return [{"tokens": rows[i]["tokens"],
             "spans": [(name(c), s, e) for c, s, e in rows[i]["spans"]]}
            for i in idx]


def f1(gold: list[set], pred: list[set]) -> dict:
    tp = sum(len(g & p) for g, p in zip(gold, pred))
    ng, np_ = sum(len(g) for g in gold), sum(len(p) for p in pred)
    pr, rc = tp / max(np_, 1), tp / max(ng, 1)
    return {"p": pr, "r": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-12), "gold": ng}


def _counts(gold: list[set], pred: list[set]) -> np.ndarray:
    return np.array([[len(g & p), len(p), len(g)] for g, p in zip(gold, pred)], dtype=float)


def _f1c(c: np.ndarray) -> float:
    tp, np_, ng = c[..., 0], c[..., 1], c[..., 2]
    return 2 * tp / np.maximum(np_ + ng, 1e-12)


def paired(gold: dict, pa: dict, pb: dict, singles: list[str], mean_of: list[str],
           n: int = 2000, seed: int = 0) -> dict:
    """Micro F1 difference per set (and for the mean over ``mean_of``),
    resampling sentences with replacement, the same draw for both runs."""
    rng = np.random.default_rng(seed)
    draws = {}
    for s in singles + mean_of:
        ca, cb = _counts(gold[s], pa[s]), _counts(gold[s], pb[s])
        idx = rng.integers(0, len(ca), size=(n, len(ca)))
        draws[s] = (100 * (_f1c(ca.sum(0)) - _f1c(cb.sum(0))),
                    100 * (_f1c(ca[idx].sum(1)) - _f1c(cb[idx].sum(1))))
    out = {s: (d, *np.percentile(bs, [2.5, 97.5])) for s, (d, bs) in draws.items()}
    pt = np.mean([draws[s][0] for s in mean_of])
    bs = np.mean([draws[s][1] for s in mean_of], axis=0)
    out[f"crossner mean of {len(mean_of)}"] = (pt, *np.percentile(bs, [2.5, 97.5]))
    return {k: tuple(float(x) for x in v) for k, v in out.items()}


@torch.no_grad()
def mention_proposals(o: dict, b: dict) -> list[set]:
    """Per item, the spans the entity-mention head (trained on Re-DocRED,
    untyped) finds: BIO Viterbi over each item's words."""
    from .data import ENT_TAGS
    from .gold_eval import bio_constraints, viterbi
    from ..entities import bio_spans
    allowed, start = bio_constraints(ENT_TAGS)
    lp = torch.log_softmax(o["ent"].float(), -1).cpu().numpy()
    n = b["word_mask"].sum(1).tolist()
    return [{(a, e) for _, a, e in bio_spans([ENT_TAGS[x] for x in viterbi(lp[i, :int(n[i])],
                                                                          allowed, start)])}
            for i in range(len(n))]


def kniv_predict(model, tok, spec, vocab, device, items: list[dict], labels: list[str],
                 batch: int = 16, offset: float = 0.0, describe=None,
                 boost: float | None = None, extra: list[set] | None = None,
                 nested: bool = False) -> list[set]:
    """``describe``: type name -> description (default: ``--describe``'s).
    ``boost``: also propose the entity-mention head's spans, their margin
    raised by ``boost`` (``kdata.decode_spans``); None: the span head alone.
    ``extra``: per item, more proposed ``(start, end)`` spans (``v6.mentions``),
    raised by the same boost."""
    from .data import blank_window, collate
    from .kdata import TypeCollator, decode_spans, span_targets
    # kniv's span head trained on "entity type: <name>. <description>"; a
    # description built from the name alone keeps that shape without giving
    # it anything GLiNER is not given (--describe).
    desc = describe or ((lambda n: f"a {n}") if DESCRIBE else (lambda n: ""))
    bank = {"entity_types": [{"id": i, "name": n, "description": desc(n)} for i, n in enumerate(labels)]}
    coll = TypeCollator(tok, bank, train=False)
    lid = {n: i for i, n in enumerate(labels)}
    feats = []
    for it in items:
        sp = [(0, len(it["tokens"]))]
        f = blank_window(it["tokens"], sp, tok, spec.max_subwords, vocab.size("cls"))
        f.update(coref=None, rel=None,
                 ents=span_targets([(lid[t], s, e) for t, s, e in it["spans"] if t in lid],
                                   f["_W"], f["word_ok"], list(range(len(labels))), False))
        feats.append(f)
    out: list[set] = [set() for _ in items]
    order = sorted(range(len(items)), key=lambda i: len(feats[i]["input_ids"]))
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        b = coll(collate([feats[i] for i in idx], tok.pad_token_id))
        b = {n: v.to(device) for n, v in b.items() if hasattr(v, "to")}
        o = model(b, relation_only=True)
        props = mention_proposals(o, b) if boost is not None and o.get("ent") is not None else None
        if extra is not None and boost is not None:
            props = [(props[j] if props else set()) | extra[i] for j, i in enumerate(idx)]
        for j, spans in enumerate(decode_spans(o["span"].float().cpu(), {n: v.cpu() for n, v in b.items()},
                                          offset=offset, proposals=props, boost=boost or 0.0,
                                          nested=nested)):
            out[idx[j]] = {(s, e, labels[t]) for s, e, t in spans}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", default="")
    ap.add_argument("--repo", default=HF_REPO)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--gliner", action="store_true", help="also score GLiNER on the same sets")
    ap.add_argument("--gliner-model", default="urchade/gliner_large-v2.1")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fewnerd", type=int, default=3000)
    ap.add_argument("--device")
    ap.add_argument("--calibrate-on", metavar="SET",
                    help="choose kniv's span offset on SET (e.g. crossner_politics) and report the "
                         "other sets at it; written into the run's calibration record")
    ap.add_argument("--describe", action="store_true",
                    help="give kniv 'a <name>' as each type's description (its trained format)")
    ap.add_argument("--paired", action="store_true",
                    help="paired bootstrap over sentences, first run minus last run")
    ap.add_argument("--cache", type=Path, metavar="DIR",
                    help="keep each run's predictions in DIR and reuse them when the run, "
                         "calibration set, Few-NERD size and --describe match")
    args = ap.parse_args()
    global DESCRIBE
    DESCRIBE = args.describe
    sets = {"fewnerd": load_fewnerd(args.fewnerd)}
    for d in DOMAINS:
        sets[f"crossner_{d}"] = load_crossner(d)
    labels = {n: sorted({t for it in s for t, _, _ in it["spans"]}) for n, s in sets.items()}
    gold = {n: [{(a, b, t) for t, a, b in it["spans"]} for it in s] for n, s in sets.items()}
    print(", ".join(f"{n}: {len(s)} sentences, {len(labels[n])} types" for n, s in sets.items()),
          flush=True)
    out = RUNS_DIR / "span-eval" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    preds: dict[str, dict] = {}
    if args.gliner:
        spec = {"model": args.gliner_model, "threshold": args.threshold,
                "sets": {n: {"labels": labels[n], "items": [{"tokens": it["tokens"]} for it in s]}
                         for n, s in sets.items()}}
        (out / "gliner_in.json").write_text(json.dumps(spec))
        here = Path(__file__).resolve().parents[1] / "gliner_baseline.py"
        subprocess.run(["uv", "run", "--no-project", "--with", "gliner", "python", str(here),
                        str(out / "gliner_in.json"), str(out / "gliner_out.json")], check=True)
        g = json.loads((out / "gliner_out.json").read_text())
        results[f"gliner:{args.gliner_model}"] = {
            n: f1(gold[n], [{tuple(x) for x in p} for p in g["preds"][n]]) for n in sets}
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    for run in [r for r in args.runs.split(",") if r]:
        key = (run, args.calibrate_on, args.fewnerd, DESCRIBE)
        cf = args.cache / f"{run}.pkl" if args.cache else None
        if cf and cf.exists():
            c_ = pickle.loads(cf.read_bytes())
            if c_["key"] == key:
                preds[run] = c_["preds"]
                results[run] = {n: f1(gold[n], preds[run][n]) for n in sets}
                if args.calibrate_on:
                    results[run]["_span_offset"] = c_["offset"]
                print(f"{run}: predictions reused from {cf}", flush=True)
                continue
        model, tok, spec_, vocab, _ = load_run(run, args.repo, device, args.local)
        off = 0.0
        if args.calibrate_on:
            c = args.calibrate_on
            sweep = {o_: f1(gold[c], kniv_predict(model, tok, spec_, vocab, device, sets[c], labels[c],
                                                  offset=o_))["f1"]
                     for o_ in np.round(np.arange(-5.0, 1.01, 0.5), 2)}
            off = float(max(sweep, key=sweep.get))
            print(f"{run}: span offset {off:+.2f} chosen on {c} "
                  f"({', '.join(f'{k:+.1f}:{v:.3f}' for k, v in sweep.items())})", flush=True)
            rec = RUNS_DIR / "calibration" / f"{run}.slim.json"
            if rec.exists():
                r = json.loads(rec.read_text()); r["span_offset"] = off
                r["span_offset_note"] = f"chosen on {c} (describe={DESCRIBE})"
                rec.write_text(json.dumps(r, indent=1))
        preds[run] = {n: kniv_predict(model, tok, spec_, vocab, device, s, labels[n], offset=off)
                      for n, s in sets.items()}
        results[run] = {n: f1(gold[n], preds[run][n]) for n in sets}
        if args.calibrate_on:
            results[run]["_span_offset"] = off
        if cf:
            cf.parent.mkdir(parents=True, exist_ok=True)
            cf.write_bytes(pickle.dumps({"key": key, "preds": preds[run], "offset": off}))
        del model
    for who, r in results.items():
        held = [d for d in DOMAINS if f"crossner_{d}" != args.calibrate_on]
        cn = np.mean([r[f"crossner_{d}"]["f1"] for d in held])
        print(f"{who}: Few-NERD {r['fewnerd']['f1']:.4f} | CrossNER " +
              " ".join(f"{d} {r[f'crossner_{d}']['f1']:.4f}" for d in DOMAINS)
              + f" | mean of {len(held)} {cn:.4f}",
              flush=True)
    if args.paired and len(preds) >= 2:
        a, b = list(preds)[0], list(preds)[-1]
        held = [f"crossner_{d}" for d in DOMAINS if f"crossner_{d}" != args.calibrate_on]
        results["_paired"] = paired(gold, preds[a], preds[b], ["fewnerd"], held)
        print(f"paired bootstrap, {a} - {b} (F1 points, 95% interval):")
        for k, (d, lo, hi) in results["_paired"].items():
            print(f"  {k}: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    (out / "results.json").write_text(json.dumps(results, indent=1))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
