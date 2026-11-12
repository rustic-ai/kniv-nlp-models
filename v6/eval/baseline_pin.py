"""M5: one pinned baseline -- the six v7.1.1 release models through the
released ONNX runtime, one harness, one calibration protocol.

    uv run python -m v6.eval.baseline_pin predict --model xlarge-u
    uv run python -m v6.eval.baseline_pin score

Protocol (fixed; every later v8 gate compares against these numbers):

- **Model:** ``dragonscale-ai/kniv-v7-<size>-en`` at tag ``v7.1.1``, the fp32
  ONNX graphs (the README's reference precision), run by the ``kniv_v7``
  runtime shipped in that snapshot, on CPU.
- **Calibration:** the shipped ``calibration.json`` as is: per-relation
  offsets, the global offset for the other calibrated relations,
  ``unseen_offset`` for every relation outside them, ``rel_top``, span and
  coreference settings. No refit on any set. A model directory without
  ``calibration.json`` is refused.
- **Input:** the gold tokens and gold sentences of each text, as one runtime
  window (``Kniv.window``); conversation texts carry their per-sentence
  speakers. Words past the model's subword limit are not seen, and gold facts
  on them count as misses.
- **Modes:** ``gold``: the gold entities are given (``Kniv.relations``), only
  the relation step is scored. ``e2e``: the model finds the entities: the
  set's entity types through the span head (CoNLL04, short dev), the default
  entity head on Re-DocRED (no types, as the model card scores it).
- **Scoring:** ``entity`` (primary): a predicted entity stands for the gold
  entity sharing the most exact mention spans, which must be unique (a
  speaker stands for the participant of that name); a fact is right when its
  mapped head, relation and mapped tail are a gold fact; predicted facts are
  counted at the level of predicted entities, so duplicates and unmatched
  arguments are false positives. This is ``gold_eval``'s relaxed end-to-end
  rule, the one behind the published Re-DocRED numbers. ``mention``
  (continuity with ``heldout_zs``): any predicted mention text equal
  (case-insensitive) to a gold mention text, each gold fact matched once.
- **Candidates are kept with margins** (every offset shifted by ``SHIFT`` so
  the runtime returns its top-k per pair), so a later threshold study reads
  the same predictions; the pinned score is at the deployed bar (margin > 0).

Diagnostics on CoNLL04 only (``--diag``), to explain the two older numbers:
``e2e_default`` (default entity head, no types) and ``e2e_text`` (raw text
through the runtime's own windows and tokenizer, then the default head).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from ..config import RUNS_DIR
from .heldout_zs import (SHIFT, bootstrap, entity_stats, f1, load_conll04,
                         load_short_dev, score_item)

SIZES = ["tiny", "small", "medium", "large", "xlarge-d", "xlarge-u"]
REVISION = "v7.1.1"
OUT = RUNS_DIR / "m5-baseline"
PATTERNS = ["calibration.json", "config.json", "label_bank.json", "tokenizer.json",
            "tokenizer_config.json", "kniv_v7/*.py", "onnx/onnx_meta.json", "onnx/int8_eval.json",
            "onnx/plain.onnx", "onnx/marker.onnx", "onnx/label.onnx", "onnx/attr.onnx",
            "onnx/span.onnx", "onnx/coref_spans.onnx"]


# ── the model ────────────────────────────────────────────────────────
def model_dir(size: str, precision: str = "fp32") -> Path:
    from huggingface_hub import snapshot_download
    pat = PATTERNS + (["onnx/*-int8.onnx", "onnx/*-fp16.onnx"] if precision == "int8" else [])
    return Path(snapshot_download(f"dragonscale-ai/kniv-v7-{size}-en", revision=REVISION,
                                  allow_patterns=pat))


def runtime(path: Path):
    """The ``kniv_v7`` package shipped in the snapshot (not the repo copy)."""
    name = f"kniv_v7_pin_{hashlib.sha1(str(path).encode()).hexdigest()[:8]}"
    spec = importlib.util.spec_from_file_location(name, path / "kniv_v7" / "__init__.py",
                                                  submodule_search_locations=[str(path / "kniv_v7")])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return sys.modules[f"{name}.runtime"]


def load_model(size: str, threads: int, precision: str = "fp32"):
    path = model_dir(size, precision)
    if not (path / "calibration.json").exists():
        raise SystemExit(f"{size}: no calibration.json in {path}; refusing to score an "
                         "uncalibrated model")
    rt = runtime(path)

    class Backend(rt.OnnxBackend):                  # the runtime's graph choice, fixed threads
        def _session(self, name):
            if name not in self._sess:
                f = self.dir / f"{name}.onnx"
                for v in (self.variant or "").split(","):   # as OnnxBackend: int8, then fp16
                    if v and (self.dir / f"{name}-{v}.onnx").exists():
                        f = self.dir / f"{name}-{v}.onnx"
                        break
                so = self._ort.SessionOptions()
                so.intra_op_num_threads = threads
                so.inter_op_num_threads = 1
                self._sess[name] = self._ort.InferenceSession(
                    str(f), so, providers=["CPUExecutionProvider"])
            return self._sess[name]

    k = rt.Kniv(path, backend=Backend(path / "onnx", int8=precision == "int8"))
    deployed = {"rel_offsets": dict(k.rel_offsets), "unseen_offset": k.unseen_offset,
                "top": k.top}
    # Shift every bar by SHIFT: the runtime then returns its top-k per pair
    # with margin + SHIFT; ``m`` below is the margin over the deployed bar.
    k.rel_offsets = {n: v + SHIFT for n, v in k.rel_offsets.items()}
    k.unseen_offset -= SHIFT
    sha = {f: hashlib.sha256((path / f).read_bytes()).hexdigest()[:16]
           for f in ("calibration.json", "label_bank.json", "kniv_v7/runtime.py",
                     "kniv_v7/entities.py")}
    info = {"size": size, "revision": REVISION, "commit": path.name, "precision": precision,
            "calibration": json.loads((path / "calibration.json").read_text()), "sha256": sha,
            "deployed": deployed}
    return k, rt, info


# ── sets ─────────────────────────────────────────────────────────────
def load_redocred_dev(limit=None) -> list[dict]:
    from ..gold.redocred import load_rel_items
    items = []
    for it in load_rel_items("dev", limit=limit):
        items.append({"id": it.id, "tokens": it.tokens, "participants": [],
                      "sentence_spans": it.layers["sentence_spans"],
                      "entities": [{"type": e["type"], "spans": [[a, b + 1] for a, b in e["mentions"]]}
                                   for e in it.entities],
                      "facts": [{"head": h, "tail": t, "relation": r} for h, t, r in it.layers["rel"]],
                      "relations": None, "types": None})
    return items


SETS = {"conll04": load_conll04, "short_dev": load_short_dev, "redocred_dev": load_redocred_dev}


def schema(it: dict, bank: dict) -> tuple[list[dict], list[dict] | None]:
    """Relations and entity types for one text, descriptions from the
    release's own label bank."""
    if it["relations"] is None:                     # Re-DocRED: the 96 calibrated relations
        return ([{"name": r["name"], "description": r.get("description", "")}
                 for r in bank["relations"] if r.get("calibrated")], None)
    rel = {r["name"]: r for r in bank["relations"]}
    typ = {t["name"]: t for t in bank.get("entity_types", [])}
    rels = [{"name": r["name"], "description": rel.get(r["name"], {}).get("description")
             or r.get("description", "")} for r in it["relations"]]
    types = [{"name": t["name"], "description": typ.get(t["name"], {}).get("description")
              or t.get("description", "")} for t in it["types"]]
    return rels, types


# ── prediction ───────────────────────────────────────────────────────
def text_window(k, rt, it):
    """Raw text through the runtime's windows (``e2e_text``): the gold tokens
    joined by spaces, re-tokenized; returns the window and a map from its
    word index to the gold word index (None where they disagree)."""
    text = " ".join(it["tokens"])
    ws = rt.windows(text)
    w = ws[0] if ws else {"tokens": [], "sentence_spans": []}
    gold_off, c = [], 0
    for t in it["tokens"]:
        gold_off.append((c, c + len(t)))
        c += len(t) + 1
    start_of = {a: i for i, (a, _) in enumerate(gold_off)}
    end_of = {b: i for i, (_, b) in enumerate(gold_off)}
    pos, wmap_s, wmap_e = 0, [], []
    for t in w["tokens"]:
        j = text.find(t, pos)
        pos = j + len(t)
        wmap_s.append(start_of.get(j))
        wmap_e.append(end_of.get(pos))
    return w, wmap_s, wmap_e, len(ws)


def predict_item(k, rt, it, bank, mode):
    rels, types = schema(it, bank)
    w = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"]}
    if it.get("speakers"):
        w["sentence_speakers"] = it["speakers"]
    smap = None
    if mode == "gold":
        sp = [tuple(x) for x in it["sentence_spans"]]
        ents = [{"type": e["type"], "mentions": sorted(tuple(s) for s in e["spans"])}
                for e in it["entities"]]
        nodes, facts = k.relations(w, it["tokens"], sp, ents, rels, k.top)
    elif mode == "e2e_text":
        w, ms, me, nwin = text_window(k, rt, it)
        out = k.window(w, rels, None, False, k.top) if w["tokens"] else {"entities": [], "facts": []}
        nodes, facts = out["entities"], out["facts"]
        smap = (ms, me, nwin, w["tokens"] == it["tokens"])
    else:
        out = k.window(w, rels, None if mode == "e2e_default" else types, False, k.top)
        nodes, facts = out["entities"], out["facts"]
    known = set(k.rel_offsets)
    if smap is not None:                            # spans back to gold words
        ms, me = smap[0], smap[1]
        for n in nodes:
            if n.get("spans"):
                sp2 = []
                for a, b in n["spans"]:
                    if ms[a] is not None and me[b - 1] is not None:
                        sp2.append([ms[a], me[b - 1] + 1])
                n["spans"] = sp2 or [[-1, -1]]
    cand = [{"h": f["head"], "t": f["tail"], "r": f["relation"], "m": f["margin"] - SHIFT,
             "unseen": f["relation"] not in known, "pol": f.get("polarity"),
             "mod": f.get("modality")} for f in facts]
    row = {"id": it["id"], "nodes": [{"speaker": bool(n.get("speaker")), "text": n["text"],
                                      "mentions": n.get("mentions", [n["text"]]),
                                      "spans": [tuple(s) for s in n.get("spans", [])]}
                                     for n in nodes],
           "cand": cand}
    if smap is not None:
        row["same_tokens"], row["windows"] = smap[3], smap[2]
    return row


def cmd_predict(args) -> int:
    global OUT
    if args.limit:                                  # smoke runs never mix with the pin
        OUT = OUT / "smoke"
    k, rt, info = load_model(args.model, args.threads, args.precision)
    bank = json.loads((model_dir(args.model) / "label_bank.json").read_text())
    tag = args.model + ("" if args.precision == "fp32" else f"_{args.precision}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"model-{tag}.json").write_text(json.dumps(info, indent=1))
    for s in args.sets.split(","):
        items = SETS[s](args.limit)
        modes = ["gold", "e2e"] + (["e2e_default", "e2e_text"] if args.diag and s == "conll04" else [])
        for mode in modes:
            dest = OUT / f"preds-{tag}-{s}-{mode}.pkl"
            if dest.exists() and not args.force:
                print(f"{tag} {s} {mode}: exists", flush=True)
                continue
            t0, rows = time.time(), []
            for i, it in enumerate(items):
                rows.append(predict_item(k, rt, it, bank, mode))
                if (i + 1) % 200 == 0:
                    print(f"{tag} {s} {mode}: {i + 1}/{len(items)} "
                          f"({(time.time() - t0) / (i + 1):.2f} s/text)", flush=True)
            dest.write_bytes(pickle.dumps({"rows": rows, "n": len(items), "limit": args.limit,
                                           "seconds": time.time() - t0}))
            print(f"{tag} {s} {mode}: done, {len(items)} texts, "
                  f"{time.time() - t0:.0f}s", flush=True)
    print("DONE", flush=True)
    return 0


# ── scoring ──────────────────────────────────────────────────────────
def gold_index(it: dict):
    P = len(it.get("participants") or [])
    span_of = {}
    for i, e in enumerate(it["entities"]):
        for a, b in e["spans"]:
            span_of[(a, b)] = P + i
    return P, span_of


def map_nodes(row: dict, it: dict, mode: str) -> list:
    """Each predicted node to a gold argument index (None: no unique match)."""
    P, span_of = gold_index(it)
    parts = [p.lower() for p in it.get("participants") or []]
    gold_m = [{tuple(s) for s in e["spans"]} for e in it["entities"]]
    out = []
    for n in row["nodes"]:
        if n["speaker"]:
            out.append(parts.index(n["text"].lower()) if n["text"].lower() in parts else None)
        elif mode == "gold":
            out.append(span_of.get(tuple(n["spans"][0])))
        else:
            ov = [len(set(n["spans"]) & g) for g in gold_m]
            m = max(ov) if ov else 0
            out.append(P + ov.index(m) if m and ov.count(m) == 1 else None)
    return out


def score_entity(row: dict, it: dict, mode: str, delta: float = 0.0) -> tuple[int, int, int]:
    gold = {(f["head"], f["relation"], f["tail"]) for f in it["facts"]}
    mp = map_nodes(row, it, mode)
    pred = {(c["h"], c["r"], c["t"]) for c in row["cand"]
            if c["m"] > (delta if c["unseen"] else 0.0)}
    mapped = {(mp[h], r, mp[t]) for h, r, t in pred
              if mp[h] is not None and mp[t] is not None and mp[h] != mp[t]}
    return len(mapped & gold), len(pred), len(gold)


def as_heldout_row(row: dict, it: dict, mode: str) -> dict:
    """The row in ``heldout_zs``'s format, for its mention rule and entity stats."""
    mp = map_nodes(row, it, "gold") if mode == "gold" else None
    keys = []
    for i, n in enumerate(row["nodes"]):
        if mode == "gold":
            keys.append(mp[i])
        else:
            keys.append(frozenset(m.lower() for m in n["mentions"]))
    cand = [{"key": (keys[c["h"]], c["r"], keys[c["t"]]), "m": c["m"], "unseen": c["unseen"]}
            for c in row["cand"]]
    ents = [(None, [tuple(s) for s in n["spans"]]) for n in row["nodes"] if not n["speaker"]]
    return {"cand": cand, "gold": it["facts"], "item": it, "ents": ents}


def cmd_score(args) -> int:
    global OUT
    if args.limit:
        OUT = OUT / "smoke"
    data = {s: SETS[s](args.limit) for s in args.sets.split(",")}
    res, counts = {}, {}
    for size in args.models.split(","):
        for s, items in data.items():
            byid = {it["id"]: it for it in items}
            for p in sorted(OUT.glob(f"preds-{size}-{s}-*.pkl")):
                mode = p.stem.split(f"-{s}-", 1)[1]
                blob = pickle.loads(p.read_bytes())
                rows = [r for r in blob["rows"] if r["id"] in byid]
                if len(rows) != len(items):
                    print(f"{size} {s} {mode}: {len(rows)} of {len(items)} texts, skipped")
                    continue
                gmode = "gold" if mode == "gold" else "e2e"
                ce = np.array([score_entity(r, byid[r["id"]], gmode) for r in rows])
                hz = [as_heldout_row(r, byid[r["id"]], gmode) for r in rows]
                cm = np.array([score_item(h, gmode) for h in hz])
                key = f"{size}|{s}|{mode}"
                counts[key] = ce
                rng = np.random.default_rng(13)
                bs = [f1(ce[rng.integers(0, len(ce), len(ce))])[2] for _ in range(1000)]
                pe, re_, fe = f1(ce)
                res[key] = {"entity": {"p": pe, "r": re_, "f1": fe,
                                       "ci": [float(np.percentile(bs, 2.5)),
                                              float(np.percentile(bs, 97.5))]},
                            "mention": dict(zip("prf", f1(cm))), "texts": len(rows),
                            "seconds": blob.get("seconds")}
                if gmode == "e2e":
                    es = entity_stats(hz)
                    res[key]["entities"] = {"mention_f1": es["mention"][2], "reach": es["reach"]}
                if mode == "e2e_text":
                    res[key]["same_tokens"] = float(np.mean([r["same_tokens"] for r in rows]))
                    res[key]["multi_window"] = int(sum(r["windows"] > 1 for r in rows))
                print(f"{key}: entity F1 {fe:.3f} [{res[key]['entity']['ci'][0]:.3f}, "
                      f"{res[key]['entity']['ci'][1]:.3f}] P {pe:.3f} R {re_:.3f} | "
                      f"mention F1 {res[key]['mention']['f']:.3f}"
                      + (f" | ent mention F1 {res[key]['entities']['mention_f1']:.3f} reach "
                         f"{res[key]['entities']['reach']:.3f}" if gmode == "e2e" else ""),
                      flush=True)
    (OUT / "scores.json").write_text(json.dumps(res, indent=1))
    if args.pair:                                   # paired bootstrap, entity rule
        a, b = args.pair.split(",")
        for key in sorted(counts):
            size, s, mode = key.split("|")
            if size == a and f"{b}|{s}|{mode}" in counts:
                d, lo, hi = bootstrap(counts[key], counts[f"{b}|{s}|{mode}"])
                print(f"{a} - {b} {s} {mode}: {d:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True, choices=SIZES)
    p.add_argument("--sets", default="conll04,short_dev,redocred_dev")
    p.add_argument("--limit", type=int)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--diag", action="store_true")
    p.add_argument("--precision", choices=["fp32", "int8"], default="fp32",
                   help="int8: the runtime's int8=True graph choice (diagnostic; the pin is fp32)")
    p.add_argument("--force", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--models", default=",".join(SIZES))
    s.add_argument("--sets", default="conll04,short_dev,redocred_dev")
    s.add_argument("--limit", type=int)
    s.add_argument("--pair", help="a,b: paired bootstrap a - b per set and mode")
    args = ap.parse_args()
    return cmd_predict(args) if args.cmd == "predict" else cmd_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
