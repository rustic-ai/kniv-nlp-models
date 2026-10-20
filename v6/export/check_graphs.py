"""Does each export graph compute what the model's own forward computes?

    uv run python -m v6.export.check_graphs --run ultrabert-380m-v7b8ev --local data/v6-eval/checkpoints

On real windows (Re-DocRED dev documents with their entities, a
conversation), every graph's output is compared with the trained model's
forward on the same input: max absolute difference per output, which must be
at most ``TOL`` (fp32, CPU).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ..train.data import collate, word_ends
from ..train.gold_eval import load_run, phase2_window, window
from ..train.model import NEG
from .graphs import COREF_K, GRAPHS

TOL = 2e-3


def windows(n: int = 4) -> list[dict]:
    """Re-DocRED dev documents (entities given) and a conversation window."""
    from ..gold.redocred import load_rel_items
    out = []
    for it in load_rel_items("dev")[:n]:
        out.append({"tokens": it.tokens, "spans": [tuple(x) for x in it.layers["sentence_spans"]],
                    "entities": [[(a, e + 1) for a, e in en["mentions"]] for en in it.entities]})
    toks = "I moved to Lisbon last year . My sister Ana still lives in Porto . Do you like Lisbon ?".split()
    out.append({"tokens": toks, "spans": [(0, 6), (6, 13), (13, 17)],
                "entities": [[(3, 4), (15, 16)], [(7, 9)], [(11, 12)]]})
    return out


def diff(a, b) -> float:
    a, b = torch.as_tensor(a).float(), torch.as_tensor(b).float()
    if a.shape != b.shape:
        return float("inf")
    if a.numel() == 0:
        return 0.0
    big = (a.abs() > 1e8) | (b.abs() > 1e8)          # NEG-masked entries: both masked is equal
    if bool(((a.abs() > 1e8) != (b.abs() > 1e8)).any()):
        return float("inf")
    return float((a - b).abs()[~big].max()) if bool((~big).any()) else 0.0


def plain_inputs(b: dict) -> tuple:
    S = b["sent_whole"].size(1)
    return (b["input_ids"], b["attention_mask"], b["first_subword"], b["word_end"],
            b["word_mask"], b["sent_of_word"], torch.arange(S))


@torch.no_grad()
def check(model, tok, spec, vocab) -> dict:
    from ..label_bank import load as load_bank
    from ..train.kdata import TypeCollator, label_text
    g = {k: v(model).eval() for k, v in GRAPHS.items()}
    worst: dict[str, float] = {}

    def note(k, v):
        worst[k] = max(worst.get(k, 0.0), v)
    rels = [r for r in load_bank()["relations"] if r["source"] == "redocred"][:24] + \
        [{"name": "lives in", "description": "the head resides in the tail place"}]
    enc = tok([label_text(r) for r in rels], padding=True, truncation=True, max_length=48,
              return_tensors="pt")
    types = ["person", "location", "organization", "date"]
    for w in windows():
        tokens, spans = w["tokens"], w["spans"]
        # Plain pass, with two predicates for the SRL frames.
        f = window(tokens, spans, tok, spec, vocab, predicates=[1, min(5, len(tokens) - 1)])
        f.update(coref=None, rel=None, _n_rel=vocab.size("relations"))
        b = collate([f], tok.pad_token_id)
        o = model(b, predict=False, predict_heads=True)
        (pos, morph, lemma, ner, arc, rel_dep, rel_head, srl_pred, s, cls, sent, cw, scores, cand,
         ent, span_words) = g["plain"](*plain_inputs(b))
        for k, a, r in (("pos", pos, o["pos"]), ("morph", morph, o["morph"]), ("lemma", lemma, o["lemma"]),
                        ("ner", ner, o["ner"]), ("arc", arc, o["arc"]), ("srl_pred", srl_pred, o["srl_pred"]),
                        ("cls", cls, o["cls"]), ("sentiment", sent, o["sentiment"]), ("ent", ent, o["ent"])):
            note(k, diff(a, r))
        W = b["first_subword"].size(1)
        k = min(COREF_K, W)
        note("coref_scores", diff(scores[..., :k + 1], o["coref_scores"]))
        valid = o["coref_scores"][..., 1:] > NEG / 2       # ties among masked candidates are arbitrary
        note("coref_cand", float((cand[..., :k] != o["coref_cand"])[valid].any()) * 1e9)
        note("dep_rel", diff(g["dep_rel"](rel_dep, rel_head, o["rel_head"]), o["rel"]))
        # SRL roles, laid out per frame as the forward lays them out.
        fb, fp = o["frame_batch"], o["frame_pred"]
        same = model._same_sentence(b["sent_of_word"], b["word_mask"])[fb, fp]
        start = same.float().argmax(1)
        end = same.size(1) - 1 - same.flip(1).float().argmax(1)
        n = end - start + 1
        L = int(n.max())
        m = torch.arange(L).unsqueeze(0) < n.unsqueeze(1)
        idx = start.unsqueeze(1) + torch.arange(L)
        words = s[fb.unsqueeze(1), idx.clamp(max=W - 1)]
        lg = g["srl_roles"](words, fp - start, m)
        ref = torch.stack([o["srl_role"][j, idx[j][m[j]]] for j in range(len(fb))])
        note("srl_roles", diff(torch.stack([lg[j][m[j]] for j in range(len(fb))]), ref))
        heads = torch.arange(0, W, max(1, W // 7))
        a, pos_ = g["coref_spans"](cw, heads, b["sent_of_word"])
        ra, rp = model.coref.spans(o["coref_a"], torch.zeros_like(heads), heads, b["sent_of_word"])
        note("coref_spans", max(diff(a, ra), float((pos_ != rp).any()) * 1e9))

        # Label pass and the span head.
        temb = g["label"](enc["input_ids"], enc["attention_mask"])
        note("label", diff(temb, model.label_embeddings(enc["input_ids"], enc["attention_mask"])))
        bank = {"entity_types": [{"id": i, "name": t, "description": ""} for i, t in enumerate(types)]}
        fs = window(tokens, spans, tok, spec, vocab)
        from ..train.kdata import span_targets
        fs.update(coref=None, rel=None, ents=span_targets([], fs["_W"], fs["word_ok"],
                                                          list(range(len(types))), False))
        bs = TypeCollator(tok, bank, train=False)(collate([fs], tok.pad_token_id))
        os_ = model(bs, relation_only=True)
        sp = g["span"](span_words, bs["span_start"], bs["span_end"],
                       g["label"](bs["type_label_ids"], bs["type_label_mask"]))
        note("span", diff(sp, os_["span"]))

        # Marker pass and attributes.
        fm = phase2_window(tokens, spans, tok, spec, vocab, w["entities"], markers=True)
        fm["coref"] = None
        bm = collate([fm], tok.pad_token_id)
        bm["rel_label_emb"] = temb
        bm["rel_label_cols"] = torch.arange(len(rels)) + 10 ** 6
        om = model(bm, relation_only=True)
        E = len(bm["ent_batch"])
        logits, pr, q, entv = g["marker"](bm["rm_input_ids"], bm["rm_attention_mask"],
                                          bm["rm_men_pos"], bm["men_ent"], torch.arange(E),
                                          bm["pair_h"], bm["pair_t"], temb)
        note("marker", diff(logits, om["relation"]))
        note("marker_pair", diff(pr, model.relation._pair))
        note("marker_labels", diff(q, model.relation._labels[0]))
        note("marker_ent", diff(entv, model.relation._ent))
        A = min(6, len(pr))
        cols = torch.arange(A) % len(rels)
        po, mo, ho, to = g["attr"](pr[:A], q[cols], entv)
        ref = model.attr.score(model.relation._pair[:A], model.relation._labels[0][cols],
                               model.relation._ent, torch.ones(A, E, dtype=torch.bool))
        note("attr", max(diff(po, ref["polarity"]), diff(mo, ref["modality"]),
                         diff(ho, ref["holder"]), diff(to, ref["time"])))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--local", type=Path)
    args = ap.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    model, tok, spec, vocab, _ = load_run(args.run, None if args.local else __import__(
        "v6.train.gold_eval", fromlist=["HF_REPO"]).HF_REPO, torch.device("cpu"), args.local)
    worst = check(model, tok, spec, vocab)
    bad = {k: v for k, v in worst.items() if not v <= TOL}
    for k, v in worst.items():
        print(f"{'FAIL' if k in bad else 'ok  '} {k:14s} max |diff| {v:.2e}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
