"""P5 pilot (v8 plan r2): where do negated and hedged facts die?

    uv run python -m v6.eval.negation_pilot --run ultrabert-380m-v7b8ev --per-kind 300

For a statement whose fact is negated or hedged, extraction can fail in two
places: the relation step does not return the fact (its margin falls under
the bar), or it does and the fact-attribute head gives it the wrong polarity
or modality. On gold entities this reports, per rewrite kind:

- the expected fact's margin above the deployed bar (median; share returned,
  share ranked first in its pair);
- the attribute head's polarity and modality for that fact, scored whether
  or not it passed the bar ("forced"), against the expected ones.

Sets:
- ``pairs``: the short-text minimal pairs (``kg_short.json`` "pairs"; teacher
  rewrites of an asserted base fact as negated, reported, planned, desired,
  questioned, hypothetical), only rewrites whose teacher label agrees with the
  rewrite kind. The asserted control is the base fact's evidence sentence from
  its training text: a different sentence, so margin differences against it
  are confounded by wording. v7.1 never trained on either.
- ``hand``: ``data/v6-eval/p5_negation.json``, 8 business facts each in an
  asserted form and in the forms an outside evaluation failed on ("did not",
  "no longer", "denied", "plans to", "might", "rumoured"): exact minimal pairs.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from ..config import DATA_DIR, RUNS_DIR
from ..schemas import MODALITIES, POLARITIES
from .heldout_zs import _lit

HAND = DATA_DIR / "v6-eval" / "p5_negation.json"


def heldout_base(text_id: str) -> bool:
    """One fifth of the short training texts, with every minimal-pair rewrite
    of them, held out of the attribute-head retrain (``attr_finetune
    --sources``) so ``--heldout`` scores texts it never saw."""
    import hashlib
    base = text_id.split(":var:")[0].split("#")[0]
    return int(hashlib.sha1(base.encode()).hexdigest(), 16) % 5 == 0


def heldout_form(text_id: str) -> bool:
    """One fifth of D3's bases (all outside ``heldout_base``), held out of
    training with ``--hold-out-m3`` so M3 can read D3's forms on texts the
    model never saw (plan r4 C4). A second hash, independent of the first."""
    import hashlib
    base = text_id.split(":var:")[0].split("#")[0]
    return int(hashlib.sha1(f"{base}:d3".encode()).hexdigest(), 16) % 5 == 0
KIND_EXPECT = {"negated": ("negative", "asserted"), "reported": ("positive", "reported"),
               "planned": ("positive", "planned"), "desired": ("positive", "desired"),
               "questioned": ("positive", "questioned"), "hypothetical": ("positive", "hypothetical")}


@torch.no_grad()
def score_fact(ex, tokens, sentence_spans, ents, relations, target):
    """The target fact ``(head entity, relation name, tail entity)`` on gold
    entities: its margin above the deployed bar, its rank among the pair's
    relations, the best margin of any relation on the pair, and the attribute
    head's polarity / modality probabilities for it. ``None`` when an
    argument has no usable mention."""
    from ..train.data import blank_window, collate, marker_features
    from ..train.kdata import label_text
    from ..infer import UNSEEN_ID
    m = ex.model
    sp = [tuple(x) for x in sentence_spans]
    f = blank_window(tokens, sp, ex.tok, ex.spec.max_subwords, ex.vocab.size("cls"))
    W = f["_W"]
    node, spans = {}, []
    for i, e in enumerate(ents):
        ms = [(a, b) for a, b in e["spans"] if b <= W and f["word_ok"][a]]
        if ms:
            node[i] = len(spans)
            spans.append(np.asarray(ms, dtype=np.int32))
    h, rel, t = target
    if h not in node or t not in node or node[h] == node[t] or len(spans) < 2:
        return None
    f.update(coref=None, _n_rel=1, rel={"spans": spans, "types": np.zeros(len(spans), np.int32),
                                        "triples": []})
    f["rm"] = marker_features(tokens, sp, spans, ex.tok, turns=[], participants=[])
    b = collate([f], ex.tok.pad_token_id)
    enc = ex.tok([label_text(r) for r in relations], padding=True, truncation=True, max_length=48,
                 return_tensors="pt")
    cols = [ex.bank_ids.get(r["name"], UNSEEN_ID + i) if ex.frozen_path else UNSEEN_ID + i
            for i, r in enumerate(relations)]
    b = {k: v.to(ex.device) for k, v in b.items() if hasattr(v, "to")}
    b["rel_label_emb"] = m.label_embeddings(enc["input_ids"].to(ex.device),
                                            enc["attention_mask"].to(ex.device))
    b["rel_label_cols"] = torch.tensor(cols, device=ex.device)
    o = m(b, relation_only=True)
    lg = o["relation"].float()
    k = [r["name"] for r in relations].index(rel)
    pairs = list(zip(b["pair_h"].tolist(), b["pair_t"].tolist()))
    p = pairs.index((node[h], node[t]))
    off = torch.tensor([ex.rel_offsets.get(r["name"], -ex.unseen_offset) for r in relations],
                       device=lg.device)
    marg = lg[p, 1:] - lg[p, 0] + off
    rank = int((marg > marg[k]).sum()) + 1
    a = m.attr(m.relation._pair, m.relation._labels, m.relation._ent, b,
               torch.tensor([p], device=ex.device), torch.tensor([k], device=ex.device))
    return {"margin": float(marg[k]), "rank": rank, "best_other": float(marg.max()),
            "polarity": torch.softmax(a["polarity"][0], -1).cpu().numpy().tolist(),
            "modality": torch.softmax(a["modality"][0], -1).cpu().numpy().tolist()}


def load_pairs(per_kind: int, seed: int = 13, heldout: bool = False) -> list[dict]:
    """Rewrites (teacher label = the kind's expected one) and, per base fact,
    its evidence sentence as the asserted control."""
    from ..label_bank import load as load_bank
    desc = {r["name"]: r.get("description", "") for r in load_bank()["relations"]}
    d = json.loads((DATA_DIR / "v6-corpus" / "kg_short.json").read_text())
    base = {x["id"]: x for x in d["train"]}
    by_kind = defaultdict(list)
    for x in d["pairs"]:
        b, rest = x["id"].split(":var:")
        if heldout and not heldout_base(b):
            continue
        key, kind = rest.rsplit(":", 1)
        if kind not in KIND_EXPECT:
            continue
        hh, tt, rel = key.split("-", 2)
        p = _lit(x["payload"])
        fs = [g for g in p["facts"] if g["relation"] == rel]
        if not fs or (fs[0]["polarity"], fs[0]["modality"]) != KIND_EXPECT[kind]:
            continue
        by_kind[kind].append((x, b, int(hh), int(tt), rel, fs[0]))
    rng = random.Random(seed)
    out, controls = [], {}
    for kind, xs in sorted(by_kind.items()):
        rng.shuffle(xs)
        for x, b, hh, tt, rel, fct in xs[:per_kind]:
            rels = [{"name": n, "description": desc.get(n, "")} for n in _lit(x["relations"])]
            p = _lit(x["payload"])
            k = len(p.get("participants") or [])
            if fct["head"] < k or fct["tail"] < k:
                continue
            out.append({"set": "pairs", "kind": kind, "id": x["id"], "tokens": _lit(x["tokens"]),
                        "sentence_spans": _lit(x["sentence_spans"]), "entities": p["entities"],
                        "fact": (fct["head"] - k, rel, fct["tail"] - k), "relations": rels,
                        "expect": KIND_EXPECT[kind]})
            # asserted control: the base fact's first evidence sentence
            if (b, hh, tt, rel) in controls:
                continue
            B = base[b]
            bp = _lit(B["payload"])
            bk = len(bp.get("participants") or [])
            bf = [g for g in bp["facts"] if g["head"] == hh and g["tail"] == tt and g["relation"] == rel]
            if not bf or hh < bk or tt < bk or bf[0]["polarity"] != "positive" \
                    or bf[0]["modality"] != "asserted" or not bf[0]["evidence"]:
                continue
            toks, ssp = _lit(B["tokens"]), _lit(B["sentence_spans"])
            s0, s1 = ssp[bf[0]["evidence"][0]]
            ents, idx = [], {}
            for i, e in enumerate(bp["entities"]):
                ms = [[a - s0, z - s0] for a, z in e["spans"] if s0 <= a and z <= s1]
                if ms:
                    idx[i] = len(ents)
                    ents.append({"type": e["type"], "spans": ms})
            if hh - bk not in idx or tt - bk not in idx:
                continue
            controls[(b, hh, tt, rel)] = {
                "set": "pairs", "kind": "asserted (base sentence)", "id": b, "tokens": toks[s0:s1],
                "sentence_spans": [[0, s1 - s0]], "entities": ents,
                "fact": (idx[hh - bk], rel, idx[tt - bk]),
                "relations": [{"name": n, "description": desc.get(n, "")} for n in _lit(B["relations"])],
                "expect": ("positive", "asserted")}
    return list(controls.values()) + out


def load_hand() -> list[dict]:
    out = []
    for x in json.loads(HAND.read_text()):
        f = x["fact"]
        out.append({"set": "hand", "kind": x["kind"], "id": x["id"], "base": x["base"],
                    "tokens": x["tokens"], "sentence_spans": x["sentence_spans"],
                    "entities": x["entities"], "fact": (f["head"], f["relation"], f["tail"]),
                    "relations": x["relations"], "expect": (f["polarity"], f["modality"])})
    return out


def summarise(rows: list[dict]) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["set"], r["kind"])].append(r)
    for (s, kind), rs in sorted(groups.items()):
        mg = np.array([r["margin"] for r in rs])
        pol = Counter(POLARITIES[int(np.argmax(r["polarity"]))] for r in rs)
        mod = Counter(MODALITIES[int(np.argmax(r["modality"]))] for r in rs)
        ep, em = rs[0]["expect"]
        ret = [r for r in rs if r["margin"] > 0]
        out[f"{s}|{kind}"] = {
            "n": len(rs), "median_margin": float(np.median(mg)),
            "returned": float((mg > 0).mean()), "rank1": float(np.mean([r["rank"] == 1 for r in rs])),
            "pol_acc": pol[ep] / len(rs), "mod_acc": mod[em] / len(rs),
            "p_negative_mean": float(np.mean([r["polarity"][1] for r in rs])),
            "right_when_returned": (sum(POLARITIES[int(np.argmax(r["polarity"]))] == ep
                                        and MODALITIES[int(np.argmax(r["modality"]))] == em
                                        for r in ret) / max(len(ret), 1)),
            "end_to_end_right": sum(r["margin"] > 0 and POLARITIES[int(np.argmax(r["polarity"]))] == ep
                                    and MODALITIES[int(np.argmax(r["modality"]))] == em
                                    for r in rs) / len(rs),
            "polarity_pred": dict(pol), "modality_pred": dict(mod), "expect": [ep, em]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="ultrabert-380m-v7b8ev")
    ap.add_argument("--per-kind", type=int, default=300)
    ap.add_argument("--local", action="store_true",
                    help="checkpoint from data/v6-eval/checkpoints (not uploaded)")
    ap.add_argument("--heldout", action="store_true",
                    help="only pairs whose base text the attribute retrain held out")
    ap.add_argument("--device")
    a = ap.parse_args()
    from ..infer import Extractor
    from ..train.gold_eval import CKPT_DIR
    ex = Extractor(a.run, device=a.device, local=CKPT_DIR if a.local else None)
    if getattr(ex.model, "attr", None) is None:
        raise SystemExit(f"{a.run} has no fact-attribute head")
    items = load_hand() + load_pairs(a.per_kind, heldout=a.heldout)
    print(f"{a.run}: {len(items)} statements; unseen_offset {ex.unseen_offset:+.2f}", flush=True)
    rows, skipped, t0 = [], 0, time.time()
    for n, it in enumerate(items):
        r = score_fact(ex, it["tokens"], it["sentence_spans"], it["entities"], it["relations"], it["fact"])
        if r is None:
            skipped += 1
            continue
        rows.append({**{k: it[k] for k in ("set", "kind", "id", "expect")},
                     **({"base": it["base"]} if "base" in it else {}), **r})
        if n % 300 == 0:
            print(f"  {n}/{len(items)} [{time.time() - t0:.0f}s]", flush=True)
    summ = summarise(rows)
    print(f"\n{skipped} skipped (an argument past the window)\n")
    print("| set | kind | n | median margin | returned | ranked 1st | polarity right (forced) | "
          "modality right (forced) | mean P(negative) | both right when returned | returned and right |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in summ.items():
        s, kind = k.split("|")
        print(f"| {s} | {kind} | {v['n']} | {v['median_margin']:+.2f} | {v['returned']:.2f} | "
              f"{v['rank1']:.2f} | {v['pol_acc']:.2f} | {v['mod_acc']:.2f} | {v['p_negative_mean']:.2f} | "
              f"{v['right_when_returned']:.2f} | {v['end_to_end_right']:.2f} |")
    print("\npredicted modality per kind:")
    for k, v in summ.items():
        print(f"  {k}: polarity {v['polarity_pred']}, modality {v['modality_pred']}")
    hand = [r for r in rows if r["set"] == "hand"]
    if hand:
        print("\nhand set, per statement (margin, P(negative), modality):")
        for r in hand:
            print(f"  {r['id']:28s} {r['margin']:+6.2f}  {r['polarity'][1]:.2f}  "
                  f"{MODALITIES[int(np.argmax(r['modality']))]}")
    out = RUNS_DIR / "negation-pilot" / f"{a.run}-{time.strftime('%Y%m%d-%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.json").write_text(json.dumps(rows))
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
