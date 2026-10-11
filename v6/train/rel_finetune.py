"""Fine-tune only the relation head on Re-DocRED, everything else frozen.

    uv run python -m v6.train.rel_finetune --run deberta-v3-large-p2-joint-ev2

The SRL head fine-tune's recipe (``srl_finetune``) applied to relations. The
encoder, every scalar mix and every other head stay as trained, so phase 1,
coreference and the entity head are bit-identical afterwards; only the
relation head (its context layer, projections and bilinear) learns, from
Re-DocRED's human-annotated train documents alone.

It tests one hypothesis: that the corpus's teacher labels hold the head back.
In phase-2 training about 84% of relation-supervised windows carry ATLOP's
labels on non-Wikipedia text -- under 5 triples a window against Re-DocRED's
28 a document -- and the head learns a mixture of the two densities.

Because nothing below the head moves, each document's relation word vectors
(the ``relation`` scalar mix over the frozen encoder) are computed once and
cached. Epochs are chosen on Re-DocRED dev; test is scored once, by
``gold_eval``. The corpus dev windows' relation F1 (agreement with the
teacher on the corpus's own domains) is reported before and after, so a head
that wins on Wikipedia by forgetting everything else shows it.
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

from .data import collate, featurise, load_phase2, redocred_features
from .gold_eval import CKPT_DIR, HF_REPO, load_run
from .model import atlop_loss

REL_PARAMS = ("relation.",)
EVIDENCE_WEIGHT = 0.1           # DREEAM's, as in phase-2 training


@torch.no_grad()
def cache_words(model, feats: list[dict], tok, device, batch: int = 8) -> list[torch.Tensor]:
    """Each window's relation word vectors ``[W, H]`` (fp16, CPU): the frozen
    encoder through the frozen ``relation`` mix, without dropout."""
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    out: list[torch.Tensor | None] = [None] * len(feats)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        b = {n: v.to(device) for n, v in collate([feats[i] for i in idx],
                                                 tok.pad_token_id).items()
             if n in ("input_ids", "attention_mask", "first_subword")}
        # The relation pass: through the relation adapter when there is one.
        layers = model._relation_layers(b)
        w = model._words(layers, "relation", b["first_subword"])
        for j, i in enumerate(idx):
            out[i] = w[j, :len(feats[i]["first_subword"])].to(torch.float16).cpu()
        if (k // batch) % 100 == 0:
            print(f"  cached {min(k + batch, len(order))}/{len(order)}", flush=True)
    return out


def head_batch(feats: list[dict], words: list[torch.Tensor], pad_id: int, device) -> tuple:
    """The relation head's inputs for these windows, from the cache."""
    b = collate(feats, pad_id)
    B, W = b["word_mask"].shape
    x = torch.zeros(B, W, words[0].size(-1))
    for j, w in enumerate(words):
        x[j, :len(w)] = w.float()
    keep = ("word_mask", "sent_of_word", "ent_batch", "men_ent", "men_start", "men_end", "pair_batch",
            "pair_h", "pair_t", "pair_labels", "pair_evidence", "pair_ev_ok")
    return x.to(device), {k: b[k].to(device) for k in keep if k in b}


def step_loss(model, x, b) -> torch.Tensor:
    logits, sent_att = model.relation(x, b)
    loss = atlop_loss(logits, b["pair_labels"])
    if sent_att is not None and b["pair_ev_ok"].any():
        ok = b["pair_ev_ok"]
        t = b["pair_evidence"][ok]
        kl = (t * (torch.log(t.clamp(min=1e-12)) - torch.log(sent_att[ok].clamp(min=1e-12)))).sum(-1)
        loss = loss + EVIDENCE_WEIGHT * kl.mean()
    return loss


@torch.no_grad()
def triple_f1(model, feats, words, pad_id, device, batch: int = 2) -> float:
    """Micro F1 over (pair, relation) with gold entities, threshold at the
    learned class -- the training-time relation metric."""
    model.eval()
    tp = p = g = 0
    for k in range(0, len(feats), batch):
        x, b = head_batch(feats[k:k + batch], words[k:k + batch], pad_id, device)
        if "pair_h" not in b or not len(b["pair_h"]):
            continue
        lg, _ = model.relation(x, b)
        pred, gold = lg[:, 1:] > lg[:, :1], b["pair_labels"][:, 1:] > 0.5
        tp += int((pred & gold).sum()); p += int(pred.sum()); g += int(gold.sum())
    model.train()
    return 2 * tp / max(p + g, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="base checkpoint (hub run name)")
    ap.add_argument("--out-run", help="default: <run>-relft")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--corpus-dev", type=int, default=600,
                    help="corpus dev windows for the generalisation check (0 = skip)")
    ap.add_argument("--pool", choices=["first", "mean"], default="first",
                    help="mention representation: first word, or mean over the span")
    ap.add_argument("--reinit", action="store_true",
                    help="train a freshly initialised relation head")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--local", type=Path, help="read <dir>/<run>/best.pt instead of the hub, "
                                               "and write the result there")
    args = ap.parse_args()
    args.out_run = args.out_run or f"{args.run}-relft"
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model, tok, spec, vocab, best = load_run(args.run, HF_REPO, device, args.local)
    if not model.phase2:
        raise SystemExit(f"{args.run} has no relation head")
    model.relation.pool = args.pool
    if args.reinit:
        # A fresh head: compares mention poolings on equal footing, instead
        # of one inheriting a head trained for the other.
        from .model import RelationHead
        old = model.relation
        model.relation = RelationHead(old.proj[0].in_features, old.bilinear.out_features - 1,
                                      context=len(old.context.layers) if old.context else 0,
                                      pool=args.pool).to(device)
    model.eval()
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(REL_PARAMS)
    trained = sorted({n.split(".")[0] + "." + n.split(".")[1] for n, p in model.named_parameters()
                      if p.requires_grad})
    print(f"{args.run}: training {sum(p.numel() for p in model.parameters() if p.requires_grad):,} "
          f"parameters ({', '.join(trained)}) on {device}", flush=True)

    st = Counter()
    train = redocred_features("train", tok, spec.max_subwords, vocab, st)
    dev = redocred_features("dev", tok, spec.max_subwords, vocab, st)
    corpus = []
    if args.corpus_dev:
        corpus = [f for f in featurise(spec, "dev", vocab, tok, phase2=True) if f.get("rel")]
        corpus = [corpus[i] for i in sorted(rng.choice(len(corpus), min(args.corpus_dev, len(corpus)),
                                                       replace=False))]
    print(f"Re-DocRED train {len(train)} / dev {len(dev)}; corpus dev windows {len(corpus)}",
          flush=True)
    t0 = time.time()
    w_train = cache_words(model, train, tok, device)
    w_dev = cache_words(model, dev, tok, device)
    w_corpus = cache_words(model, corpus, tok, device) if corpus else []
    print(f"cached in {time.time() - t0:.0f}s", flush=True)
    # Only the relation head runs from here. On MPS it moves to the CPU: the
    # pair count changes every batch, so every batch is a new shape, and the
    # MPS graph cache grows with each one until it exhausts memory.
    model.encoder.to("cpu")
    if device.type == "mps":
        model.to("cpu")
        torch.mps.empty_cache()
        device = torch.device("cpu")
        torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))
        print("training the relation head on the CPU (MPS shape cache)", flush=True)
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    pad = tok.pad_token_id
    before = {"redocred_dev": triple_f1(model, dev, w_dev, pad, device),
              "corpus_dev": triple_f1(model, corpus, w_corpus, pad, device) if corpus else None}
    print(f"before: {before}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    warm = max(1, int(0.06 * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (steps - s) / max(steps - warm, 1))))
    model.relation.train()
    best_f1, best_state, bad, history = before["redocred_dev"], None, 0, []
    for ep in range(args.epochs):
        order = rng.permutation(len(train))
        tot, n = 0.0, 0
        for k in range(0, len(order), args.batch):
            idx = order[k:k + args.batch]
            x, b = head_batch([train[i] for i in idx], [w_train[i] for i in idx], pad, device)
            loss = step_loss(model, x, b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            tot += loss.item(); n += 1
        f1 = triple_f1(model, dev, w_dev, pad, device)
        history.append({"epoch": ep + 1, "loss": round(tot / n, 4), "redocred_dev": round(f1, 5)})
        mark = ""
        if f1 > best_f1:
            best_f1, bad, mark = f1, 0, "  * best"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()
                          if k.startswith(REL_PARAMS)}
        else:
            bad += 1
        print(f"epoch {ep + 1}: loss {tot / n:.4f}  Re-DocRED dev F1 {f1:.4f}{mark}", flush=True)
        if bad >= args.patience:
            break
    if best_state is None:
        print("no epoch beat the base head; nothing written")
        return 1
    model.load_state_dict(best_state, strict=False)
    after = {"redocred_dev": triple_f1(model, dev, w_dev, pad, device),
             "corpus_dev": triple_f1(model, corpus, w_corpus, pad, device) if corpus else None}
    print(f"after:  {after}", flush=True)

    root = args.local or CKPT_DIR
    src = root / args.run / "best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for k, v in best_state.items():
        assert k in ck["model"] and ck["model"][k].shape == v.shape, k
        ck["model"][k] = v.detach().cpu().to(ck["model"][k].dtype)
    ck["rel_finetune"] = {"base": args.run, "history": history, "before": before,
                          "after": after, "args": vars(args), "data": "Re-DocRED train"}
    out = root / args.out_run
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "best.pt")
    (out / "best.metric").write_text(json.dumps({
        "value": best["value"], "step": best["step"],
        "metric": "dev composite of the base run; relation head fine-tuned on Re-DocRED train",
        "redocred_dev": after["redocred_dev"], "corpus_dev_relation": after["corpus_dev"]}))
    print(f"{len(best_state)} relation tensors replaced; wrote {out / 'best.pt'}")
    print(f"score it: uv run python -m v6.train.gold_eval --local {root} "
          f"--runs {args.run},{args.out_run} --benchmarks redocred")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
