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
    tcoll2.columns = lambda pool: torch.as_tensor(sorted(sch - {rel["residence"]["id"]})).numpy()
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
    print("all knowledge-layer smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
