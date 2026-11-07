"""Does a negation cue explain the attribute head's polarity errors? On the
teacher dev facts (gold entities and relations), per fact: the head's polarity,
a negation word in the fact's evidence sentences, and the model's own SRL
ARGM-NEG there.

    uv run python -m v6.eval.negation_cues RUN
"""
import sys, collections, torch
from v6.label_bank import load as load_bank
from v6.train.kdata import knowledge_features, LabelCollator, TypeCollator
from v6.train.gold_eval import load_run
from v6.train.data import collate
from v6.train.loop import Chain

RUN = sys.argv[1]
CUES = {"not", "n't", "never", "no", "none", "neither", "nor", "without", "cannot", "nothing", "nobody", "nowhere"}
dev_ = torch.device("mps")
bank = load_bank()
model, tok, spec, vocab, _ = load_run(RUN, "dragonscale-ai/kniv-v6-runs", dev_)
model.eval()
NEG = vocab.id("srl", "B-ARGM-NEG") if vocab.id("srl", "B-ARGM-NEG") is not None else vocab.id("srl", "ARGM-NEG")
print("ARGM-NEG id", NEG, [l for l in vocab.labels["srl"] if "NEG" in l])
feats = knowledge_features("teacher", "dev", tok, spec.max_subwords, vocab.size("cls"), bank)
coll = Chain(TypeCollator(tok, bank, train=False), LabelCollator(tok, bank, train=False))
rows = []
for n, f in enumerate(feats):
    r = f["rel"]
    if not r or not r.get("attrs"):
        continue
    if n % 20 == 0:
        model.__dict__.get("_label_cache", {}).clear(); torch.mps.empty_cache()
        print(n, len(rows), flush=True)
    b = coll(collate([f], tok.pad_token_id))
    if "attr_pair" not in b:
        continue
    b = {k: v.to(dev_) if hasattr(v, "to") else v for k, v in b.items()}
    with torch.no_grad():
        o = model({k: v for k, v in b.items() if k != "coref_ok"}, predict=True)
        model(b, relation_only=True)
        rel = model.relation
        ok = (b["attr_col"] >= 0)
        same = b["ent_batch"].unsqueeze(0) == b["pair_batch"][b["attr_pair"]].unsqueeze(1)
        fa = model.attr.score(rel._pair[b["attr_pair"]].float(), rel._labels[0, b["attr_col"].clamp(min=0)].float(), rel._ent.float(), same)
    pol = fa["polarity"].float().softmax(-1)[:, 1].cpu()
    # words
    ids = f["input_ids"]; fs = list(f["first_subword"]); W = len(fs)
    words = []
    for i in range(W):
        e = fs[i + 1] if i + 1 < W else len(ids)
        words.append(tok.decode(ids[fs[i]:e]).strip().lower())
    sow = list(f["sent_of_word"])
    # ARGM-NEG words from predicted frames
    negw = set()
    if o["srl_role"].numel():
        lab = o["srl_role"].argmax(-1).cpu()  # [F,W]
        for fr in range(lab.size(0)):
            for w in (lab[fr] == NEG).nonzero().flatten().tolist():
                negw.add(w)
    ev = {(h, t): s for h, t, s in r.get("evidence") or []}
    E = len(r["spans"])
    attrs = [a for a in r["attrs"] if a["h"] != a["t"] and a["h"] < E and a["t"] < E]
    assert len(attrs) == len(pol), (len(attrs), len(pol))
    for a, p, k in zip(attrs, pol.tolist(), ok.cpu().tolist()):
        if not k:
            continue
        sents = ev.get((a["h"], a["t"]))
        if not sents:
            sents = {sow[s] for s, _ in r["spans"][a["h"]]} & {sow[s] for s, _ in r["spans"][a["t"]]}
        sents = set(sents)
        inside = [i for i in range(W) if sow[i] in sents]
        lex = any(words[i] in CUES for i in inside)
        srl = any(i in negw for i in inside)
        rows.append((a["polarity"] == "negative", p > 0.5, lex, srl))
print(len(rows), "facts")
def f1(gold, pred):
    tp = sum(g and q for g, q in zip(gold, pred)); return 2 * tp / max(sum(gold) + sum(pred), 1)
g = [x[0] for x in rows]
print("model F1", round(f1(g, [x[1] for x in rows]), 3))
print("lexical-cue rule F1", round(f1(g, [x[2] for x in rows]), 3), " ARGM-NEG rule F1", round(f1(g, [x[3] for x in rows]), 3))
c = collections.Counter((x[0], x[1], x[2], x[3]) for x in rows)
print("gold_neg pred_neg lex srl : count")
for k, v in sorted(c.items(), key=lambda kv: -kv[1]): print(k, v)
