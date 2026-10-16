"""Local checks for the schema-conditioned knowledge layer, before any GPU run.

    uv run python -m v6.train.smoke_knowledge

On deberta-v3-xsmall (the production encoder's family; Ettin cannot return
attention maps, which the marker head pools). Each check prints PASS or
raises. They exist because the bugs they guard against are silent: a
transplant that is merely close, a mask that leaks false negatives, a bf16
index-put, gradients recomputed through the wrong adapter state.
"""
from __future__ import annotations

from collections import Counter

import torch

from ..label_bank import load as load_bank
from .data import collate, redocred_features
from .encoders import get
from .kdata import LabelCollator
from .model import V6Model, atlop_loss, losses
from .vocab import Vocab


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        raise SystemExit(f"smoke check failed: {name}")


def no_dropout(m):
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0


def main() -> int:
    torch.manual_seed(0)
    spec = get("deberta-v3-xsmall")
    vocab = Vocab.load()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    sizes = {k: vocab.size(k) for k in vocab.labels}
    bank = load_bank()
    feats = redocred_features("dev", tok, spec.max_subwords, vocab, Counter(), markers=True)[:4]
    R = vocab.size("relations")

    # 1. masked loss
    logits = torch.randn(30, R + 1, requires_grad=True)
    labels = torch.zeros(30, R + 1)
    labels[torch.arange(30), torch.randint(0, R + 1, (30,))] = 1.0
    a, b = atlop_loss(logits, labels), atlop_loss(logits, labels, torch.ones_like(labels))
    check("masked loss = ATLOP loss when nothing is masked", torch.allclose(a, b), f"{a.item():.6f}")
    mask = torch.ones_like(labels)
    mask[:, 50:] = 0
    atlop_loss(logits, labels, mask).backward()
    check("masked columns get no gradient", float(logits.grad[:, 50:].abs().max()) == 0.0)

    # 2. collator round trip
    coll = LabelCollator(tok, bank, train=True, seed=1)
    batch = collate(feats, tok.pad_token_id)
    old = batch["pair_labels"].clone()
    out = coll(dict(batch))
    cols = out["rel_label_cols"].tolist()
    same = torch.equal(out["pair_labels"][:, 1:], old[:, [c + 1 for c in cols]])
    none_ok = torch.equal(out["pair_labels"][:, 0], (old[:, [c + 1 for c in cols]].sum(1) == 0).float())
    check("collator keeps the sampled columns' targets", same, f"K={len(cols)}")
    check("collator marks 'no listed relation' exactly when no sampled positive", none_ok)
    check("label texts tokenised for every column",
          out["rel_label_ids"].size(0) == len(cols) and out["rel_label_mask"].sum() > 0)

    # 3. transplant reproduces the ATLOP head
    A = V6Model(spec.hf_id, sizes, phase2=True, rel_adapter=4, rel_markers=True).eval()
    no_dropout(A)
    with torch.no_grad():
        for p in A.rel_lora.B.values():
            p.normal_(0, 0.02)
        for p in A.relation.parameters():
            p.normal_(0, 0.02)
    B_ = V6Model(spec.hf_id, sizes, phase2=True, rel_adapter=4, rel_markers=True,
                 rel_head="label").eval()
    no_dropout(B_)
    state = {k: v for k, v in A.state_dict().items() if not k.startswith("relation.bilinear")}
    missing, unexpected = B_.load_state_dict(state, strict=False)
    check("label model loads the ATLOP model except its bilinear",
          not unexpected and all(m.startswith(("relation.", "label_enc.")) for m in missing))
    full = LabelCollator(tok, bank, train=False)
    from .loop import _transplant
    _transplant(B_, A.state_dict(), full, torch.device("cpu"))
    eb = full(collate(feats, tok.pad_token_id))
    with torch.no_grad():
        la = A(collate(feats, tok.pad_token_id))["relation"]
        lb = B_(eb)["relation"]
    diff = float((la - lb).abs().max())
    check("transplanted label head reproduces the ATLOP logits", diff < 1e-3, f"max |diff| {diff:.2e}")
    pa = (la[:, 1:] > la[:, :1])
    pb = (lb[:, 1:] > lb[:, :1])
    check("and the same predicted triples", torch.equal(pa, pb), f"{int(pa.sum())} predicted")

    # 4. bf16 autocast, relation-only training pass
    B_.train()
    no_dropout(B_)
    for n, p in B_.named_parameters():
        p.requires_grad = n.startswith(("rel_lora.", "relation.", "ent.", "mix.ent.", "label_enc."))
    B_.rel_lora.sticky = True
    tb = coll(collate(feats, tok.pad_token_id))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        L = losses(B_(tb, relation_only=True), tb)
    sum(l for l, _ in L.values()).backward()
    got = sum(p.grad is not None for n, p in B_.named_parameters() if n.startswith("label_enc."))
    check("bf16 autocast forward and backward", all(torch.isfinite(l) for l, _ in L.values()),
          ", ".join(f"{k} {l.item():.3f}" for k, (l, _) in L.items()))
    check("gradients reach the label encoder", got > 0)

    # 5. gradient checkpointing with the sticky adapter
    def grads(ckpt: bool):
        B_.zero_grad(set_to_none=True)
        if ckpt:
            B_.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            B_.encoder.gradient_checkpointing_disable()
        B_.rel_lora.sticky = True
        L = losses(B_(tb, relation_only=True), tb)
        sum(l for l, _ in L.values()).backward()
        B_.rel_lora.on = False
        return torch.cat([p.grad.flatten() for p in B_.rel_lora.A.values()])
    g0, g0b, g1 = grads(False), grads(False), grads(True)
    rel = lambda x, y: float((x - y).norm() / y.norm())
    check("adapter gradients identical with gradient checkpointing",
          rel(g1, g0) <= max(1e-5, 2 * rel(g0b, g0)), f"{rel(g1, g0):.1e} vs run-to-run {rel(g0b, g0):.1e}")

    # 6. the frozen-tensor guard
    from .loop import _frozen_digest
    d0 = _frozen_digest(B_)
    with torch.no_grad():
        B_.pos.weight[0, 0] += 1e-3
    check("--assert-frozen notices a changed frozen tensor", _frozen_digest(B_) != d0)
    # 7. teacher windows: speaker tags, schema masks, partial labels
    from ..annotate.kg import KGItem
    from ..windows import tokenize
    from .kdata import teacher_features
    sents = [tokenize(x) for x in ("I moved to Lisbon last year .", "Nice , Portugal suits you .",
                                   "My sister Ana lives in Porto .")]
    toks = sum(sents, [])
    spans, o_ = [], 0
    for x in sents:
        spans.append([o_, o_ + len(x)]); o_ += len(x)
    rel = {r["name"]: r for r in bank["relations"] if r["source"] == "redocred"}
    names = ["residence", "country", "sibling", "located in the administrative territorial entity"]
    item = KGItem(id="t", tokens=toks, sentence_spans=spans, speakers=["user", "assistant", "user"],
                  participants=["user", "assistant"], types=[],
                  relations=[{"name": n, "description": ""} for n in names])
    payload = {"participants": ["user", "assistant"],
               "entities": [{"type": "LOC", "spans": [[3, 4]]}, {"type": "LOC", "spans": [[9, 10]]},
                            {"type": "PER", "spans": [[15, 16]]}, {"type": "LOC", "spans": [[18, 19]]}],
               "facts": [{"head": 0, "tail": 2, "relation": "residence", "polarity": "positive",
                          "modality": "asserted", "holder": 0, "time": -1, "evidence": [0]},
                         {"head": 4, "tail": 5, "relation": "residence", "polarity": "positive",
                          "modality": "asserted", "holder": 0, "time": -1, "evidence": [2]}]}
    tf = teacher_features(item, payload, tok, spec.max_subwords, bank, vocab.size("cls"))
    rm = tf["rm"]
    star = tok.convert_tokens_to_ids("▁*")
    part_ok = all(int(rm["ids"][p]) == star for ms in rm["men_pos"][:2] for p in ms) \
        and [len(x) for x in rm["men_pos"][:2]] == [2, 1]
    check("speaker tags are marker mentions of the participants", part_ok,
          " ".join(tok.convert_ids_to_tokens(rm["ids"][:8].tolist())))
    check("teacher facts become bank-id triples", tf["rel"]["triples"] ==
          [(0, 2, rel["residence"]["id"]), (4, 5, rel["residence"]["id"])])
    tcoll = LabelCollator(tok, bank, train=True, seed=3, unlabelled_keep=1.0, min_labels=200)
    mixed = collate(feats[:2] + [tf], tok.pad_token_id)
    old = mixed["pair_labels"].clone()
    mo = tcoll(dict(mixed))
    cols = mo["rel_label_cols"].tolist()
    tsel = mo["pair_batch"] == 2
    sch = {rel[n]["id"] for n in names}
    want = torch.tensor([1.0] + [float(c in sch) for c in cols])
    check("teacher pairs see only their schema's columns",
          bool((mo["pair_label_mask"][tsel] == want).all()))
    rd = {r["id"] for r in bank["relations"] if r["source"] == "redocred"}
    want_rd = torch.tensor([1.0] + [float(c in rd) for c in cols])
    check("Re-DocRED pairs see only Re-DocRED columns",
          bool((mo["pair_label_mask"][~tsel] == want_rd).all()))
    check("teacher positives survive collation", int(mo["pair_labels"][tsel, 1:].sum()) == 2)
    # a schema sample without 'residence': its positive pairs are masked whole
    tcoll2 = LabelCollator(tok, bank, train=True, seed=3, unlabelled_keep=1.0)
    tcoll2.columns = lambda pool, keep=(): torch.as_tensor(sorted(sch - {rel["residence"]["id"]})).numpy()
    m2 = tcoll2(dict(collate([tf], tok.pad_token_id)))
    pos_rows = old[mixed["pair_batch"] == 2][:, 1:].sum(1) > 0
    check("a pair whose teacher positive fell outside the columns is masked whole",
          bool((m2["pair_label_mask"][pos_rows] == 0).all())
          and bool((m2["pair_label_mask"][~pos_rows] != 0).any()))
    tcoll3 = LabelCollator(tok, bank, train=True, seed=3, unlabelled_keep=0.0)
    m3 = tcoll3(dict(collate([tf], tok.pad_token_id)))
    check("unlabelled_keep 0 masks every unlabelled teacher pair",
          int((m3["pair_label_mask"].sum(1) > 0).sum()) == 2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        L = losses(B_(mo, relation_only=True), mo)
    check("mixed Re-DocRED + teacher batch trains", all(torch.isfinite(l) for l, _ in L.values()),
          ", ".join(f"{k} {l.item():.3f}" for k, (l, _) in L.items()))
    # 8. the hybrid: a frozen ATLOP path for its relations, the label head for the rest
    import tempfile
    from .kdata import TypeCollator, knowledge_features
    from .loop import Chain, _load_legacy
    from .model import SpanTypeHead  # noqa: F401
    H_ = V6Model(spec.hf_id, sizes, phase2=True, rel_adapter=4, rel_markers=True,
                 rel_head="label", legacy_relation=True, span_head=True).eval()
    no_dropout(H_)
    H_.load_state_dict({k: v for k, v in B_.state_dict().items()
                        if not k.startswith(("rel_lora.", "relation.", "ent.", "mix.ent."))
                        or True}, strict=False)
    with tempfile.TemporaryDirectory() as td:
        torch.save({"model": A.state_dict()}, f"{td}/best.pt")
        _load_legacy(H_, f"{td}/best.pt")
    rd_cols = torch.tensor([r["id"] for r in bank["relations"] if r["source"] == "redocred"])
    hb = full(collate(feats, tok.pad_token_id))
    hb["rel_label_cols"] = rd_cols
    with torch.no_grad():
        ho = H_(hb)
        la = A(collate(feats, tok.pad_token_id))
    want = la["relation"][:, 1:] - la["relation"][:, :1]
    got = ho["relation"][:, 1:]
    check("hybrid: Re-DocRED columns are the ATLOP head's margins",
          float((want - got).abs().max()) < 1e-4, f"max |diff| {float((want - got).abs().max()):.1e}")
    check("hybrid: entity head is the legacy run's", torch.allclose(ho["ent"], la["ent"], atol=1e-5))
    mixed_cols = torch.cat([rd_cols[:5], torch.tensor([200, 300, 400])])
    mb_ = full(collate(feats, tok.pad_token_id))
    mb_["rel_label_cols"] = mixed_cols
    keep_ids = [0] + [1 + c for c in mixed_cols[:5].tolist()]
    mb_["rel_label_ids"] = mb_["rel_label_ids"][:len(mixed_cols)]
    mb_["rel_label_mask"] = mb_["rel_label_mask"][:len(mixed_cols)]
    with torch.no_grad():
        mo_ = H_(mb_)
    check("hybrid: a mixed schema routes old relations to ATLOP and new ones to the label head",
          torch.allclose(mo_["relation"][:, 1:6], want[:, mixed_cols[:5]], atol=1e-4)
          and torch.allclose(mo_["relation"][:, 6:], (mo_["relation_label"][:, 6:]
                                                      - mo_["relation_label"][:, :1]), atol=1e-5))

    # 9. training the hybrid leaves the legacy path untouched
    H_.train(); no_dropout(H_)
    for n, prm in H_.named_parameters():
        prm.requires_grad = n.startswith(("rel_lora.", "relation.", "label_enc.", "span.",
                                          "mix.ent_span."))
    H_.rel_lora.sticky = True
    before = {k: v.clone() for k, v in H_.state_dict().items() if k.startswith(("legacy_", "ent."))}
    d_before = _frozen_digest(H_)
    xs_tok = tok
    fr = knowledge_features("fewrel", "train", xs_tok, spec.max_subwords, vocab.size("cls"), bank, limit=4)
    fn = knowledge_features("fewnerd", "train", xs_tok, spec.max_subwords, vocab.size("cls"), bank, limit=6)
    dd = knowledge_features("docred_distant", "train", xs_tok, spec.max_subwords, vocab.size("cls"), bank, limit=2)
    tc = Chain(TypeCollator(tok, bank, train=True, seed=5),
               LabelCollator(tok, bank, train=True, seed=5, unlabelled_keep=1.0))
    mix_items = feats[:1] + [tf] + fr[:2] + fn[:3] + dd[:1]
    tb2 = tc(collate(mix_items, tok.pad_token_id))
    opt = torch.optim.SGD([p for p in H_.parameters() if p.requires_grad], lr=0.1)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        o2 = H_(tb2, relation_only=True)
        L2 = losses(o2, tb2)
    o2["span"].retain_grad()
    sum(l for l, _ in L2.values()).backward()
    check("mixed batch of every source trains (bf16)",
          all(torch.isfinite(l) for l, _ in L2.values()) and L2["ent_span"][1] > 0 and L2["relation"][1] > 0,
          ", ".join(f"{k} {l.item():.3f}/{n}" for k, (l, n) in L2.items()))
    sm = tb2["span_label_mask"]
    check("span head: masked type columns get no gradient",
          float(o2["span"].grad[sm == 0].abs().max()) == 0.0 if (sm == 0).any() else True)
    opt.step()
    after = {k: v for k, v in H_.state_dict().items() if k.startswith(("legacy_", "ent."))}
    check("a training step leaves the legacy path and entity head untouched",
          all(torch.equal(before[k], after[k]) for k in before) and _frozen_digest(H_) == d_before)

    # 10. FewRel: only the labelled pair supervises
    fb = LabelCollator(tok, bank, train=True, seed=5)(collate(fr[:1], tok.pad_token_id))
    rows = fb["pair_label_mask"].sum(1)
    check("FewRel: the labelled pair is supervised, its reverse is masked",
          len(rows) == 2 and rows[0] > 1 and rows[1] == 0, f"mask row sums {rows.tolist()}")
    # 11. v6.2: the zero-shot head, the label adapter, a grown relation adapter
    Z = V6Model(spec.hf_id, sizes, phase2=True, rel_adapter=8, rel_markers=True, rel_head="zs",
                legacy_relation=True, span_head=True, label_adapter=4, legacy_rank=4).eval()
    no_dropout(Z)
    src = B_.state_dict()
    n_grown = Z.rel_lora.load_grown(src, "rel_lora.")
    with torch.no_grad():
        B_.eval(); B_.rel_lora.sticky = False
        sa, _ = B_._marker_pass(eb)
        sz, _ = Z._marker_pass(eb)
    check("a relation adapter grown from rank 4 to 8 gives the same marker pass",
          n_grown > 0 and float((sa - sz).abs().max()) < 1e-4, f"max |diff| {float((sa - sz).abs().max()):.1e}")
    zb = full(collate(feats, tok.pad_token_id))
    with torch.no_grad():
        Z.relation.xattn_off = False
        on = Z(zb, relation_only=True)["relation"]
        Z.relation.xattn_off = True
        off = Z(zb, relation_only=True)["relation"]
        Z.relation.xattn_off = False
    check("cross-attention starts as the identity (gates at zero)", torch.allclose(on, off, atol=1e-5))
    Z.train(); no_dropout(Z)
    for n, prm in Z.named_parameters():
        prm.requires_grad = n.startswith(("rel_lora.", "relation.", "label_enc.", "span.",
                                          "mix.ent_span.", "label_lora."))
    Z.rel_lora.sticky = True
    zt = tc(collate(mix_items, tok.pad_token_id))
    before_z = {k: v.clone() for k, v in Z.state_dict().items() if k.startswith(("legacy_", "ent."))}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        Lz = losses(Z(zt, relation_only=True), zt)
    sum(l for l, _ in Lz.values()).backward()
    got = {pre: any(p.grad is not None and float(p.grad.abs().sum()) > 0
                    for n, p in Z.named_parameters() if n.startswith(pre))
           for pre in ("label_lora.", "relation.xattn.", "relation.pair.", "label_enc.score", "span.")}
    check("v6.2 head trains in bf16 on a mixed batch", all(torch.isfinite(l) for l, _ in Lz.values()),
          ", ".join(f"{k} {l.item():.3f}" for k, (l, _) in Lz.items()))
    check("gradients reach the label adapter, cross-attention, pair MLP and attention pooling",
          all(got.values()), str(got))

    def label_grads(ckpt: bool):
        Z.zero_grad(set_to_none=True)
        if ckpt:
            Z.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            Z.encoder.gradient_checkpointing_disable()
        Z.rel_lora.sticky = True
        Z.rel_lora.on = True                     # as in relation-only training
        e = Z.label_embeddings(zt["rel_label_ids"], zt["rel_label_mask"])
        e.pow(2).mean().backward()
        Z.rel_lora.on = False
        return torch.cat([p.grad.flatten() for p in list(Z.label_lora.A.values())
                          + list(Z.label_lora.B.values())])
    with torch.no_grad():                       # a fresh B is zero: A would get no gradient
        for prm in Z.label_lora.B.values():
            prm.normal_(0, 0.02)
    g0, g1 = label_grads(False), label_grads(True)
    check("label adapter gradients identical with gradient checkpointing on (and the relation adapter left on)",
          float((g1 - g0).norm() / g0.norm()) < 1e-5, f"{float((g1 - g0).norm() / g0.norm()):.1e}")
    Z.encoder.gradient_checkpointing_disable()
    check("the label pass leaves the relation adapter as it found it", Z.rel_lora.on is False)
    after_z = {k: v for k, v in Z.state_dict().items() if k.startswith(("legacy_", "ent."))}
    check("v6.2 training leaves the legacy path untouched",
          all(torch.equal(before_z[k], after_z[k]) for k in before_z))
    Z.eval()
    hz = full(collate(feats, tok.pad_token_id)); hz["rel_label_cols"] = rd_cols
    with torch.no_grad():
        Z.legacy_lora.load_state_dict({k[len("legacy_lora."):]: v for k, v in H_.state_dict().items()
                                       if k.startswith("legacy_lora.")})
        Z.legacy_relation.load_state_dict({k[len("legacy_relation."):]: v for k, v in H_.state_dict().items()
                                           if k.startswith("legacy_relation.")})
        zo_ = Z(hz)
    check("v6.2 hybrid: Re-DocRED columns are still the ATLOP head's margins",
          float((zo_["relation"][:, 1:] - want).abs().max()) < 1e-4)
    # v7: T-REx documents, the label-ranking terms, word pooling.
    Z.train(); no_dropout(Z)
    Z.rel_lora.on = Z.rel_lora.sticky = False
    for prm in Z.parameters():
        prm.grad = None
    trex = knowledge_features("trex", "train", tok, spec.max_subwords, vocab.size("cls"), bank, limit=6)
    never = {r["id"] for r in bank["relations"] if not r["train"]}
    check("T-REx documents carry typed spans and facts, none on a never-trained relation",
          all(len(f["ents"]["spans"]) for f in trex) and sum(len(f["rel"]["triples"]) for f in trex if f["rel"])
          and not any(r in never for f in trex if f["rel"] for *_, r in f["rel"]["triples"]))
    tb = tc(collate(trex, tok.pad_token_id))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        Lt = losses(Z(tb, relation_only=True), tb)
    (Lt["relation_rank"][0] + Lt["ent_span_rank"][0]).backward()
    gr = any(p.grad is not None and float(p.grad.abs().sum()) > 0
             for n, p in Z.named_parameters() if n.startswith(("relation.", "span.")))
    check("label-ranking terms are positive and reach the relation and span heads",
          float(Lt["relation_rank"][0]) > 0 and float(Lt["ent_span_rank"][0]) > 0 and gr,
          f"relation_rank {Lt['relation_rank'][0].item():.3f}, ent_span_rank {Lt['ent_span_rank'][0].item():.3f}")
    Z.eval()
    with torch.no_grad():
        Z.word_pool = "first"
        a = Z(tb, relation_only=True)["span"]
        Z.word_pool = "mean"
        b = Z(tb, relation_only=True)["span"]
        Z.word_pool = "first"
    multi = bool(((tb["word_end"] - tb["first_subword"]) > 1)[tb["word_mask"]].any())
    check("mean word pooling changes span scores only where words have several subwords",
          multi and not torch.allclose(a, b), f"max |diff| {float((a - b).abs().max()):.2e}")
    # v7: fact attributes on teacher windows.
    from .model import FactAttrHead
    tw = knowledge_features("teacher", "train", tok, spec.max_subwords, vocab.size("cls"), bank, limit=12)
    tw = [f for f in tw if f["rel"] and f["rel"].get("attrs")][:4]
    ab = tc(collate(tw, tok.pad_token_id))
    n_attr = len(ab.get("attr_pair", []))
    same_doc = all(int(ab["ent_batch"][e]) == int(ab["pair_batch"][p])
                   for p, h, t in zip(ab["attr_pair"].tolist(), ab["attr_holder"].tolist(),
                                      ab["attr_time"].tolist()) for e in (h, t) if e >= 0)
    check("teacher facts carry attribute targets, holder and time pointers inside their window",
          n_attr > 0 and same_doc and bool((ab["attr_col"] >= 0).all()), f"{n_attr} facts")
    A = V6Model(spec.hf_id, sizes, phase2=True, rel_adapter=8, rel_markers=True, rel_head="zs",
                span_head=True, label_adapter=4, attr_head=True)
    A.train(); no_dropout(A)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        oa = A(ab, relation_only=True)
        La = losses(oa, ab)
    La["fact_attr"][0].backward()
    ga = any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in A.named_parameters()
             if n.startswith("attr."))
    check("fact attribute loss is finite and reaches the attribute head (bf16)",
          isinstance(A.attr, FactAttrHead) and torch.isfinite(La["fact_attr"][0]) and ga
          and La["fact_attr"][1] == n_attr, f"fact_attr {La['fact_attr'][0].item():.3f} over {La['fact_attr'][1]}")
    print("all knowledge-layer smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
