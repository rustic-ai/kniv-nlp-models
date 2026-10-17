"""Fine-tune the linguistic heads with the encoder frozen: POS, morphology,
lemma and dependencies on UD English EWT gold, NER on the v5 training set
(``kniv-corpus-en``: gold-filtered corpus sentences).

    uv run python -m v6.train.head_finetune --run ultrabert-380m-v7b3-srlft

The v5 recipe and data for the linguistic layer, as
``srl_finetune`` is for SRL: the encoder, every scalar mix and every other head
stay exactly as they were, so the knowledge layer and coreference are
bit-identical afterwards. In a shared multi-task run the linguistic heads fit
the corpus's silver labels but lose to the knowledge objectives on gold; a
head trained alone on cached encoder outputs recovers part of that.

Each head's scalar-mix output is computed once per sentence by the frozen
model and cached; training runs over the cache in three steps, each choosing
its epoch on its own dev set:

1. POS, morphology, lemma -- UD EWT train, UD dev accuracy;
2. NER -- ``prepared/deberta-v3-large-nlp-en/ner_train.json`` from
   ``dragonscale-ai/kniv-corpus-en`` (45k sentences, 18 OntoNotes-style types,
   spaCy trf + GPT validation, kept only if Qwen3-8B confirmed every entity),
   reading the new POS head's probabilities, span F1 on a sample of its dev;
3. DEP arcs and labels -- UD EWT train, reading the new POS and NER heads'
   probabilities, UD dev LAS.

Licence: UD EWT is CC BY-SA 4.0 (training on it is the v5 recipe; the UD
score then means "fine-tuned on UD", as v5's did). OntoNotes (LDC) is not
trained on; kniv-corpus-en is CC BY-SA 4.0.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import IGNORE, collate, encode_row
from .gold_eval import CKPT_DIR, HF_REPO, load_run

MIXES = ("pos", "morph", "lemma", "ner", "dep")
TAGGERS = ("pos", "morph", "lemma")
DEP_MODULES = ("root", "arc_dep", "arc_head", "arc", "rel_head", "rel_dep", "rel")
LABELS = ("pos", "morph", "lemma", "ner", "dep_head", "dep_rel")


def ud_features(split: str, tok, spec, vocab, limit: int | None = None) -> list[dict]:
    """UD EWT sentences as corpus-style rows, through the corpus encoder, so
    labels (lemma edit scripts, morph strings, relations) use the run's vocab."""
    from ..gold.ud_ewt import load_ud_items
    stats = Counter()
    out = []
    for it in load_ud_items(split, limit):
        n = len(it.tokens)
        heads = it.layers["dep"]["heads"]
        row = {"window_id": it.id, "domain": "ud", "tokens": it.tokens, "loss_mask": None,
               "sentence_spans": [[0, n]], "pos": it.layers["pos"], "morph": it.layers["morph"],
               "lemma": it.layers["lemma"], "ner": None, "dep": it.layers["dep"]["rels"],
               "dep_heads": [h - 1 for h in heads],      # 0-based, root -1
               "srl_frames": None, "cls": None, "sentiment": None}
        out.append(encode_row(row, tok, spec.max_subwords, vocab, stats))
    return out


V5_DATA = "dragonscale-ai/kniv-corpus-en"


def v5_ner_features(split: str, limit: int, tok, spec, vocab, rng) -> list[dict]:
    """v5's NER sentences (one sentence per item) as corpus-style rows."""
    from huggingface_hub import hf_hub_download
    rows = json.load(open(hf_hub_download(
        V5_DATA, f"prepared/deberta-v3-large-nlp-en/ner_{split}.json", repo_type="dataset")))
    if limit and limit < len(rows):
        rows = [rows[i] for i in sorted(rng.choice(len(rows), limit, replace=False))]
    stats = Counter()
    out = []
    for k, r in enumerate(rows):
        n = len(r["words"])
        tags = []
        for t in r["ner_tags"]:                 # ~0.4% bare types from validation fixes
            if t.split("-", 1)[-1] in ("None", "NONE", "null", "_MISSING"):
                t = "O"
            elif t != "O" and t[:2] not in ("B-", "I-"):
                t = ("I-" if tags and tags[-1] != "O" and tags[-1][2:] == t else "B-") + t
            tags.append(t)
        row = {"window_id": f"v5-ner-{split}-{k}", "domain": "v5", "tokens": r["words"],
               "loss_mask": None, "sentence_spans": [[0, n]], "pos": None, "morph": None,
               "lemma": None, "ner": tags, "dep": None, "dep_heads": None,
               "srl_frames": None, "cls": None, "sentiment": None}
        out.append(encode_row(row, tok, spec.max_subwords, vocab, stats))
    if stats["ner_out_of_vocab"]:
        print(f"  v5 ner {split}: {stats['ner_out_of_vocab']} tags outside the run's NER vocab")
    return out


@torch.no_grad()
def cache(model, feats: list[dict], tok, device, mixes, batch: int = 32) -> list[dict]:
    """Per item: each mix's word vectors ``[W,H]`` (fp16, CPU) and its labels.
    Length-sorted batches keep the number of distinct input shapes small."""
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    out: list[dict | None] = [None] * len(feats)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        b = collate([feats[i] for i in idx], tok.pad_token_id)
        b = {n: b[n].to(device) for n in ("input_ids", "attention_mask", "first_subword", "word_end")}
        o = model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                          output_hidden_states=True)
        layers = tuple(F.layer_norm(h, h.shape[-1:]) for h in o.hidden_states)
        model._word_span = (b["first_subword"], b["word_end"])
        reps = {m: model._words(layers, m, b["first_subword"]) for m in mixes}
        for j, i in enumerate(idx):
            f = feats[i]
            W = len(f["word_ok"])
            out[i] = {**{m + "_w": reps[m][j, :W].to(torch.float16).cpu() for m in mixes},
                      **{k_: torch.from_numpy(np.asarray(f[k_][:W]).astype(np.int64))
                         for k_ in LABELS if k_ in f},
                      "word_ok": torch.from_numpy(np.asarray(f["word_ok"][:W])),
                      "sent_of_word": torch.from_numpy(np.asarray(f["sent_of_word"][:W]).astype(np.int64))}
        if (k // batch) % 50 == 0:
            if device.type == "mps":
                torch.mps.empty_cache()
            print(f"  cached {min(k + batch, len(order)):,}/{len(order):,}", flush=True)
    return out


def pad(items: list[dict], device) -> dict:
    W = max(len(x["word_ok"]) for x in items)
    B = len(items)
    b = {}
    for k, v in items[0].items():
        if v.dim() == 2:                                   # word vectors
            t = torch.zeros(B, W, v.size(1), dtype=torch.float32)
            for i, x in enumerate(items):
                t[i, :len(x[k])] = x[k].float()
        elif k == "word_ok":
            t = torch.zeros(B, W, dtype=torch.bool)
            for i, x in enumerate(items):
                t[i, :len(x[k])] = x[k]
        else:
            t = torch.full((B, W), -1 if k == "sent_of_word" else IGNORE, dtype=torch.long)
            for i, x in enumerate(items):
                t[i, :len(x[k])] = x[k]
        b[k] = t.to(device)
    return b


def dep_logits(model, b: dict, predict: bool):
    pos_p = torch.softmax(model.pos(b["pos_w"]).float(), -1)
    ner_p = torch.softmax(model.ner(torch.cat([b["ner_w"], pos_p], -1)).float(), -1)
    d = torch.cat([b["dep_w"], pos_p, ner_p], -1)
    wm = b["word_ok"]
    B = d.size(0)
    cand = torch.cat([model.root.expand(B, 1, -1), d], 1)
    arc = model.arc.pairwise(model.arc_dep(d), model.arc_head(cand)).squeeze(-1)
    allowed = torch.cat([wm.unsqueeze(-1), model._same_sentence(b["sent_of_word"], wm)], -1)
    arc = arc.masked_fill(~allowed, -1e4)
    head = arc.argmax(-1) if predict else b["dep_head"].clamp(min=0)
    hrep = torch.gather(model.rel_head(cand), 1, head.unsqueeze(-1).expand(-1, -1, model.rel.U.size(1)))
    return arc, model.rel.paired(model.rel_dep(d), hrep), head


def epochs(name, params, train, dev, step_loss, score, device, args, rng) -> float:
    for p in params:
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    best = score(dev)
    print(f"{name}: dev before {json.dumps({k: round(v, 4) for k, v in best.items()})}", flush=True)
    keep = [p.detach().clone() for p in params]
    stale = 0
    for ep in range(1, args.epochs + 1):
        order = rng.permutation(len(train))
        tot, n, skipped = 0.0, 0, 0
        for k in range(0, len(order), args.batch):
            b = pad([train[i] for i in order[k:k + args.batch]], device)
            loss = step_loss(b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not (torch.isfinite(loss) and torch.isfinite(gn)):
                skipped += 1                       # one inf grad would turn every weight NaN
                continue
            opt.step()
            tot += loss.item(); n += 1
        r = score(dev)
        mark = ""
        if r["main"] > best["main"]:
            best, stale, mark = r, 0, "  * best"
            keep = [p.detach().clone() for p in params]
        else:
            stale += 1
        print(f"{name} epoch {ep}: loss {tot / max(n, 1):.4f}  " + (f"skipped {skipped}  " if skipped else "") +
              "  ".join(f"{k} {v:.4f}" for k, v in r.items()) + mark, flush=True)
        if stale >= args.patience:
            break
    with torch.no_grad():
        for p, v in zip(params, keep):
            p.copy_(v)
    for p in params:
        p.requires_grad = False
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="base checkpoint (e.g. an SRL fine-tune)")
    ap.add_argument("--out-run", help="default: <run> with -srlft replaced by -ft, else <run>-ft")
    ap.add_argument("--local", type=Path, help="read <dir>/<run>/best.pt instead of the hub")
    ap.add_argument("--ud-limit", type=int, help="UD sentences per split (testing)")
    ap.add_argument("--ner-train", type=int, default=45000, help="v5 NER sentences (all 45k)")
    ap.add_argument("--ner-dev", type=int, default=3000, help="v5 NER dev sample for selection")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    out_run = args.out_run or (args.run.replace("-srlft", "-ft") if args.run.endswith("-srlft")
                               else f"{args.run}-ft")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok, spec, vocab, best_metric = load_run(args.run, HF_REPO, device, args.local)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    t0 = time.time()
    ud_tr = cache(model, ud_features("train", tok, spec, vocab, args.ud_limit), tok, device, MIXES)
    ud_dv = cache(model, ud_features("dev", tok, spec, vocab, args.ud_limit), tok, device, MIXES)
    ctr = v5_ner_features("train", args.ner_train, tok, spec, vocab, rng)
    cdv = v5_ner_features("dev", args.ner_dev, tok, spec, vocab, rng)
    c_tr = cache(model, ctr, tok, device, ("pos", "ner"))
    c_dv = cache(model, cdv, tok, device, ("pos", "ner"))
    print(f"cached UD {len(ud_tr):,}/{len(ud_dv):,} sentences, v5 NER {len(c_tr):,}/{len(c_dv):,} "
          f"sentences in {time.time() - t0:.0f}s", flush=True)
    for x in c_tr + c_dv:                      # NER items carry no dep vectors
        for k in ("dep_head", "dep_rel", "pos", "morph", "lemma"):
            x.pop(k, None)

    def ce(logits, target):
        m = target != IGNORE
        return F.cross_entropy(logits[m].float(), target[m]) if bool(m.any()) else logits.sum() * 0.0

    @torch.no_grad()
    def score_tag(items):
        c = Counter()
        for k in range(0, len(items), 128):
            b = pad(items[k:k + 128], device)
            for h in TAGGERS:
                m = b[h] != IGNORE
                c[h] += int((getattr(model, h)(b[h + "_w"]).argmax(-1)[m] == b[h][m]).sum())
                c[h + "_n"] += int(m.sum())
        r = {h: c[h] / max(c[h + "_n"], 1) for h in TAGGERS}
        r["main"] = sum(r.values()) / 3
        return r

    # Heads see only labelled word rows in training: MPS LayerNorm backward on
    # the all-zero padding rows gives wrong (and in larger batches NaN) gradients.
    def loss_tag(b):
        out = 0.0
        for h in TAGGERS:
            m = b[h] != IGNORE
            out = out + ce(getattr(model, h)(b[h + "_w"][m]), b[h][m]) / 3
        return out

    tag_params = [p for h in TAGGERS for p in getattr(model, h).parameters()]
    r_tag = epochs("pos/morph/lemma", tag_params, ud_tr, ud_dv, loss_tag, score_tag, device, args, rng)

    from .loop import _spans
    L = vocab.labels["ner"]

    @torch.no_grad()
    def score_ner(items):
        tp = p_ = g_ = 0
        for k in range(0, len(items), 16):
            b = pad(items[k:k + 16], device)
            pos_p = torch.softmax(model.pos(b["pos_w"]).float(), -1)
            pred = model.ner(torch.cat([b["ner_w"], pos_p], -1)).argmax(-1).tolist()
            for pp, gg in zip(pred, b["ner"].tolist()):
                g = [L[x] if x != IGNORE else "O" for x in gg]
                q = [L[x] if y != IGNORE else "O" for x, y in zip(pp, gg)]
                gs, ps = _spans(g), _spans(q)
                tp += len(gs & ps); p_ += len(ps); g_ += len(gs)
        f1 = 2 * tp / max(p_ + g_, 1)
        return {"ner_f1": f1, "main": f1}

    def loss_ner(b):
        m = b["ner"] != IGNORE
        pos_p = torch.softmax(model.pos(b["pos_w"][m]).float(), -1).detach()
        return ce(model.ner(torch.cat([b["ner_w"][m], pos_p], -1)), b["ner"][m])

    r_ner = epochs("ner", list(model.ner.parameters()), c_tr, c_dv, loss_ner, score_ner,
                   device, args, rng)

    @torch.no_grad()
    def score_dep(items):
        ua = la = n = 0
        for k in range(0, len(items), 64):
            b = pad(items[k:k + 64], device)
            arc, rel, head = dep_logits(model, b, predict=True)
            m = b["dep_head"] != IGNORE
            ok = (head == b["dep_head"]) & m
            ua += int(ok.sum()); la += int((ok & (rel.argmax(-1) == b["dep_rel"])).sum()); n += int(m.sum())
        return {"uas": ua / max(n, 1), "las": la / max(n, 1), "main": la / max(n, 1)}

    def loss_dep(b):
        arc, rel, _ = dep_logits(model, b, predict=False)
        t = b["dep_rel"].masked_fill(b["dep_head"] == IGNORE, IGNORE)
        return ce(arc, b["dep_head"]) + ce(rel, t)

    dep_params = [p for name in DEP_MODULES
                  for p in (getattr(model, name).parameters()
                            if isinstance(getattr(model, name), torch.nn.Module)
                            else [getattr(model, name)])]
    r_dep = epochs("dep", dep_params, ud_tr, ud_dv, loss_dep, score_dep, device, args, rng)

    src = CKPT_DIR / args.run / "best.pt" if args.local is None else args.local / args.run / "best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    changed = 0
    for k, v in model.state_dict().items():
        if k.split(".")[0] in TAGGERS + ("ner",) + DEP_MODULES:
            assert k in ck["model"] and ck["model"][k].shape == v.shape, k
            ck["model"][k] = v.detach().cpu().to(ck["model"][k].dtype)
            changed += 1
    ck["head_finetune"] = {"base": args.run, "tag": r_tag, "ner": r_ner, "dep": r_dep, "args": vars(args)}
    out = CKPT_DIR / out_run
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "best.pt")
    (out / "best.metric").write_text(json.dumps({
        "value": best_metric["value"], "step": best_metric["step"],
        "metric": "dev composite of the base run; linguistic heads fine-tuned (UD EWT, v5 NER)",
        "ud_dev": {**r_tag, **r_dep}, "v5_dev_ner": r_ner}))
    print(f"{changed} head tensors replaced; wrote {out / 'best.pt'}", flush=True)
    from .srl_finetune import upload
    upload(out, out_run, not args.no_upload, "linguistic head fine-tune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
