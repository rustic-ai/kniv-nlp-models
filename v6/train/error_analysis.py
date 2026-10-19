"""Where coreference and relation extraction go wrong.

    uv run python -m v6.train.error_analysis --local data/v6-eval/checkpoints --run R

**Coreference** (LitBank, the gold_eval windows; spans inclusive): mention
detection by mention kind (pronoun, name, nominal); for every gold mention
the model found, whether it is linked to its own entity (**correct**), only
to other entities (**wrong link**) or to nothing (**unlinked**); gold
entities split over several predicted ones and predicted entities fusing
several gold ones; and how far back a wrong link reaches.

**Relations** (Re-DocRED test, gold entities, the dev-calibrated threshold
as deployed): every missed gold fact is the right relation ranked first but
under the threshold (**threshold**), another relation predicted for the
pair (**confused**), or nothing predicted for the pair (**pair missed**);
every false fact is a confusion on a related pair, the gold relation's
direction reversed, or a pair with no gold relation (**spurious**). Broken
down by how many sentences separate the entities, by entity types, and by
relation; the most frequent confusions are listed.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .gold_eval import (HF_REPO, _pairs, _sentences, calibrate_offset, load_run, phase2_window,
                        run_phase2)

PRONOUNS = {"i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "he", "him",
            "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "we", "us",
            "our", "ours", "ourselves", "they", "them", "their", "theirs", "themselves", "thee",
            "thou", "thy", "thine", "ye", "this", "that", "these", "those", "who", "whom", "whose",
            "which", "one"}


def kind(tokens, span) -> str:
    s, e = span
    words = tokens[s:e + 1]
    if len(words) == 1 and words[0].lower() in PRONOUNS:
        return "pronoun"
    if any(w[:1].isupper() for w in words) and words[0].lower() not in ("the", "a", "an"):
        return "name"
    return "nominal"


def coref(model, tok, spec, vocab, device) -> dict:
    from ..gold.litbank import load_coref_items
    items = load_coref_items()
    feats = [phase2_window(it.tokens, _sentences(it.tokens), tok, spec, vocab) for it in items]
    outs = run_phase2(model, feats, tok, device)
    c = Counter()
    dist_wrong, dist_right = [], []
    for it, f, o in zip(items, feats, outs):
        W = f["_W"]
        gold = [[tuple(m) for m in cl if m[1] < W] for cl in it.layers["coref"]]
        gold = [g for g in gold if len(g) >= 2]
        pred = [[tuple(m) for m in cl] for cl in o["coref"]]
        g_of = {m: gi for gi, cl in enumerate(gold) for m in cl}
        p_of = {m: pi for pi, cl in enumerate(pred) for m in cl}
        for m, gi in g_of.items():
            k = kind(it.tokens, m)
            c[f"gold_{k}"] += 1
            if m not in p_of:
                c[f"missed_{k}"] += 1
                # A boundary disagreement, not an absent mention: some
                # predicted mention overlaps it.
                if any(a <= m[1] and m[0] <= b for a, b in p_of):
                    c[f"missed_overlap_{k}"] += 1
                continue
            mates = [x for x in pred[p_of[m]] if x != m]
            own = [x for x in mates if g_of.get(x) == gi]
            if own:
                c[f"correct_{k}"] += 1
            elif mates:
                c[f"wrong_{k}"] += 1
                prev = [x for x in mates if x[0] < m[0]]
                if prev:
                    dist_wrong.append(m[0] - max(prev)[0])
            else:
                c[f"unlinked_{k}"] += 1
            prev_own = [x for x in own if x[0] < m[0]]
            if prev_own:
                dist_right.append(m[0] - max(prev_own)[0])
        for m in p_of:
            k = kind(it.tokens, m)
            c[f"pred_{k}"] += 1
            if m not in g_of:
                c[f"spurious_{k}"] += 1
                if any(a <= m[1] and m[0] <= b for a, b in g_of):
                    c[f"spurious_overlap_{k}"] += 1
        for gl in gold:
            owners = {p_of[m] for m in gl if m in p_of}
            c["gold_entities"] += 1
            c["split_entities"] += int(len(owners) > 1)
        for pl in pred:
            owners = {g_of[m] for m in pl if m in g_of}
            c["pred_entities"] += 1
            c["fused_entities"] += int(len(owners) > 1)
    return {"counts": dict(c), "dist_wrong": dist_wrong, "dist_right": dist_right,
            "windows": len(items)}


def relations(model, tok, spec, vocab, device) -> dict:
    from ..gold.redocred import load_rel_items
    items = load_rel_items("test")
    names = vocab.labels["relations"]
    offset = calibrate_offset(model, tok, spec, vocab, device)
    feats = [phase2_window(it.tokens, [tuple(x) for x in it.layers["sentence_spans"]], tok, spec,
                           vocab, [[(a, e + 1) for a, e in en["mentions"]] for en in it.entities],
                           markers=model.rel_markers) for it in items]
    outs = run_phase2(model, feats, tok, device, batch=1)
    c = Counter()
    by = defaultdict(Counter)
    confusions, per_rel = Counter(), defaultdict(Counter)
    for it, f, o in zip(items, feats, outs):
        keep = f["_ent_keep"]
        pairs = _pairs(len(keep))
        spans = [tuple(x) for x in it.layers["sentence_spans"]]

        def sent(pos):
            return max(i for i, (s, _) in enumerate(spans) if s <= pos)
        ent_sents = [{sent(a) for a, _ in en["mentions"]} for en in it.entities]
        gold = defaultdict(set)
        for h, t, r in it.layers["rel"]:
            gold[(h, t)].add(r)
        logits = {(keep[h], keep[t]): lg for (h, t), lg in zip(pairs, o["relation"])}
        for (h, t), lg in logits.items():
            above = [k for k in np.argsort(-lg[1:])[:4] if lg[k + 1] > lg[0] - offset]
            pred = {names[k] for k in above}
            top = names[int(np.argmax(lg[1:]))]
            g = gold.get((h, t), set())
            d = min(abs(a - b) for a in ent_sents[h] for b in ent_sents[t])
            dk = "same" if d == 0 else "adjacent" if d == 1 else "2+ apart"
            tk = f"{it.entities[h]['type']}->{it.entities[t]['type']}"
            for r in g:
                for key in ("all", dk, tk):
                    by[key]["gold"] += 1
                if r in pred:
                    c["tp"] += 1; per_rel[r]["tp"] += 1
                    for key in ("all", dk, tk):
                        by[key]["tp"] += 1
                    continue
                per_rel[r]["fn"] += 1
                if top == r:
                    c["fn_threshold"] += 1
                elif pred:
                    c["fn_confused"] += 1
                    for p in pred - g:
                        confusions[(r, p)] += 1
                else:
                    c["fn_pair_missed"] += 1
            for p in pred - g:
                per_rel[p]["fp"] += 1
                for key in ("all", dk, tk):
                    by[key]["fp"] += 1
                if g:
                    c["fp_confused"] += 1
                elif p in gold.get((t, h), set()):
                    c["fp_reversed"] += 1
                else:
                    c["fp_spurious"] += 1
            for key in ("all", dk, tk):
                by[key]["pred"] += len(pred)
    return {"counts": dict(c), "by": {k: dict(v) for k, v in by.items()},
            "confusions": [(a, b, n) for (a, b), n in confusions.most_common(25)],
            "per_relation": {r: dict(v) for r, v in per_rel.items()}, "offset": offset}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--part", choices=["coref", "relations", "both"], default="both")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device")
    args = ap.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok, spec, vocab, _ = load_run(args.run, HF_REPO, device, args.local)
    res = {}
    if args.part in ("coref", "both"):
        res["coref"] = coref(model, tok, spec, vocab, device)
    if args.part in ("relations", "both"):
        res["relations"] = relations(model, tok, spec, vocab, device)
    out = args.out or Path(f"error-analysis-{args.run}.json")
    out.write_text(json.dumps(res, indent=1, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
