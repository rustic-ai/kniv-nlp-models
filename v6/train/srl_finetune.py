"""Fine-tune only the SRL role head on PropBank gold, everything else frozen.

    uv run python -m v6.train.srl_finetune                 # DeBERTa full run's best.pt

Option A of the SRL follow-up (EVALUATION.md): the encoder, every scalar mix
and every other head stay exactly as trained, so POS, morph, lemma, DEP, NER,
CLS and sentiment are bit-identical afterwards; only the SRL role head
(``srl_p``, ``srl_a``, ``srl_role``) learns, from PropBank gold instead of v5's
silver. It measures how much of the gap to v5 is the head's calibration --
what a thin head can relearn without the encoder.

Because nothing below the head moves, each sentence's SRL word
representation is computed once and cached; training then runs over the
cache in seconds per epoch.

**The predicate head is not trained.** The prepared PropBank files hold about
one predicate per sentence (40,872 frames over 39,949 train sentences), so a
predicate left unlabelled would be taught as "not a predicate". Role scoring
is trained on gold frames of every part of speech, including the adjectival
and nominal predicates the corpus never marks; *finding* those at inference
would need fully annotated sentences.

Data: ``srl_train.json`` and ``srl_dev.json`` from ``dragonscale-ai/kniv-corpus-en``.
**``srl_train.json`` is not pure gold**: v5's ``prepare_srl.py`` merged PropBank
EWT gold with MASC (converted from dependency SRL), QA-SRL (crowdsourced,
converted) and model-generated silver. Fine-tuned on the mix, dev F1 fell
from 0.799 to 0.778 in one epoch. So train is restricted to its PropBank EWT
gold portion -- frames whose sentence is a UD EWT train sentence: 8,690 of
40,872 -- unless ``--all-sources``. Dev is entirely PropBank EWT. Sentences
that also occur in the PropBank test split are dropped from both, so the test
stays unseen. Epochs are chosen on dev; test is scored once, by ``gold_eval``.

This trains on public gold, which the v6 corpus otherwise does not: an SRL
score from a checkpoint made here measures "fine-tuned on PropBank", like v5's.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..config import PREPARED_DIR
from ..score.metrics import bio_spans
from .data import IGNORE, collate
from .gold_eval import CKPT_DIR, HF_REPO, bio_constraints, load_run, viterbi, window

SRL_PARAMS = ("srl_p.", "srl_a.", "srl_role.")


def ewt_sentences(split: str) -> set[tuple]:
    import conllu
    from ..config import UD_DIR
    text = (UD_DIR / f"en_ewt-ud-{split}.conllu").read_text(encoding="utf-8")
    return {tuple(t["form"] for t in s if isinstance(t["id"], int))
            for s in conllu.parse(text)}


def load_frames(name: str, exclude: set, only: set | None = None) -> dict[tuple, list]:
    """``{sentence tokens: [(predicate_idx, tags), ...]}`` minus ``exclude``,
    restricted to sentences in ``only`` when given."""
    path = PREPARED_DIR / f"srl_{name}.json"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        hf_hub_download("dragonscale-ai/kniv-corpus-en",
                        f"prepared/kniv-deberta-cascade/srl_{name}.json",
                        repo_type="dataset", local_dir=str(PREPARED_DIR.parents[1]))
    out: dict[tuple, list] = defaultdict(list)
    for e in json.loads(path.read_text()):
        w, t, p = e.get("words"), e.get("srl_tags"), e.get("predicate_idx")
        if not w or len(w) != len(t) or p is None or not 0 <= p < len(w) or len(w) > 128:
            continue
        if tuple(w) in exclude or (only is not None and tuple(w) not in only):
            continue
        out[tuple(w)].append((p, t))
    return out


@torch.no_grad()
def cache_reps(model, sentences: list[tuple], tok, spec, vocab, device,
               batch: int = 64) -> list[torch.Tensor]:
    """Each sentence's SRL word representations, from the frozen encoder and
    the frozen SRL scalar mix: ``[n_words, H]`` fp16 on the CPU."""
    feats = [window(list(s), [(0, len(s))], tok, spec, vocab) for s in sentences]
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    reps: list[torch.Tensor | None] = [None] * len(feats)
    for k in range(0, len(order), batch):
        idx = order[k:k + batch]
        b = {n: v.to(device) for n, v in collate([feats[i] for i in idx],
                                                 tok.pad_token_id).items()}
        out = model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                            output_hidden_states=True)
        layers = tuple(F.layer_norm(h, h.shape[-1:]) for h in out.hidden_states)
        s = model._words(layers, "srl", b["first_subword"])          # eval: no dropout
        for j, i in enumerate(idx):
            reps[i] = s[j, :feats[i]["_W"]].to(torch.float16).cpu()
        if (k // batch) % 50 == 0:
            print(f"  cached {min(k + batch, len(order))}/{len(order)}", flush=True)
    return reps


def frames_of(groups: dict, sentences: list[tuple], vocab) -> list[tuple[int, int, np.ndarray]]:
    """``(sentence index, predicate, tag ids)`` for every gold frame."""
    ix = vocab.index["srl"]
    out = []
    for si, s in enumerate(sentences):
        for p, tags in groups[s]:
            out.append((si, p, np.array([ix[t] for t in tags], dtype=np.int64)))
    return out


def role_logits(model, reps: list[torch.Tensor], frames: list, device) -> tuple:
    """Role logits ``[F, W, C]`` for a batch of frames, padded to the longest."""
    W = max(len(f[2]) for f in frames)
    H = reps[0].size(1)
    words = torch.zeros(len(frames), W, H, device=device)
    for k, (si, _, tags) in enumerate(frames):
        words[k, :len(tags)] = reps[si].to(device, torch.float32)
    pred = words[torch.arange(len(frames), device=device),
                 torch.tensor([f[1] for f in frames], device=device)]
    logits = model.srl_role.against(model.srl_p(pred), model.srl_a(words))
    target = torch.full((len(frames), W), IGNORE, dtype=torch.long, device=device)
    for k, (_, _, tags) in enumerate(frames):
        target[k, :len(tags)] = torch.from_numpy(tags).to(device)
    return logits, target


@torch.no_grad()
def score(model, reps, frames, tags, device, batch: int = 256) -> float:
    """Span F1 on ``frames``, Viterbi-decoded with the predicate given and V
    excluded -- the same protocol as ``gold_eval``."""
    model.eval()
    allowed, start = bio_constraints(tags)
    v = tags.index("V")
    tp = n_pred = n_gold = 0
    for k in range(0, len(frames), batch):
        chunk = frames[k:k + batch]
        logits, _ = role_logits(model, reps, chunk, device)
        lp = torch.log_softmax(logits.float(), -1).cpu().numpy()
        for (si, p, gold), l in zip(chunk, lp):
            l = l[:len(gold)]
            keep = np.ones_like(l, dtype=bool)
            keep[:, v] = False
            keep[p, :] = False
            keep[p, v] = True
            pr = bio_spans([tags[x] for x in viterbi(l, allowed, start, keep)])
            gs = bio_spans([tags[x] for x in gold])
            tp += len(pr & gs); n_pred += len(pr); n_gold += len(gs)
    P, R = tp / max(n_pred, 1), tp / max(n_gold, 1)
    return 2 * P * R / max(P + R, 1e-12)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="deberta-v3-large", help="base checkpoint (hub)")
    ap.add_argument("--out-run", default="deberta-v3-large-srlft")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--all-sources", action="store_true",
                    help="train on all of srl_train.json (PropBank + MASC + QA-SRL + silver)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))

    model, tok, spec, vocab, best = load_run(args.run, HF_REPO, device)
    assert model.srl_mode == "joint", "fine-tunes the joint SRL head"
    tags = vocab.labels["srl"]
    test = {tuple(e["words"]) for e in json.loads((PREPARED_DIR / "srl_test.json").read_text())
            if e.get("words")}
    train_g = load_frames("train", test, None if args.all_sources else ewt_sentences("train"))
    dev_g = load_frames("dev", test)
    train_s, dev_s = list(train_g), list(dev_g)
    print(f"{args.run} (dev {best['value']}): "
          f"{'all sources' if args.all_sources else 'PropBank EWT gold'} train {len(train_s)} sentences / "
          f"{sum(map(len, train_g.values()))} frames, dev {len(dev_s)} / "
          f"{sum(map(len, dev_g.values()))}, test sentences excluded from both", flush=True)

    t0 = time.time()
    train_r = cache_reps(model, train_s, tok, spec, vocab, device)
    dev_r = cache_reps(model, dev_s, tok, spec, vocab, device)
    print(f"cached representations in {time.time() - t0:.0f}s", flush=True)
    train_f, dev_f = frames_of(train_g, train_s, vocab), frames_of(dev_g, dev_s, vocab)

    # Only the SRL role head learns. Everything else is frozen and in eval
    # mode, and the cache means it is not even run.
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(SRL_PARAMS)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in params):,} parameters "
          f"({', '.join(SRL_PARAMS)})", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    base = score(model, dev_r, dev_f, tags, device)
    print(f"dev SRL F1 before fine-tuning: {base:.4f}", flush=True)
    best_f1, best_state, stale = base, {n: p.detach().clone() for n, p in
                                        model.named_parameters() if p.requires_grad}, 0
    history = [{"epoch": 0, "dev_f1": base}]
    for epoch in range(1, args.epochs + 1):
        for m in (model.srl_p, model.srl_a, model.srl_role):
            m.train()
        order = rng.permutation(len(train_f))
        total, n = 0.0, 0
        for k in range(0, len(order), args.batch):
            chunk = [train_f[i] for i in order[k:k + args.batch]]
            logits, target = role_logits(model, train_r, chunk, device)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   target.reshape(-1), ignore_index=IGNORE)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item(); n += 1
        f1 = score(model, dev_r, dev_f, tags, device)
        history.append({"epoch": epoch, "train_loss": total / n, "dev_f1": f1})
        mark = ""
        if f1 > best_f1:
            best_f1, stale, mark = f1, 0, "  * best"
            best_state = {n_: p.detach().clone() for n_, p in model.named_parameters()
                          if p.requires_grad}
        else:
            stale += 1
        print(f"epoch {epoch}: train loss {total / n:.4f}  dev SRL F1 {f1:.4f}{mark}", flush=True)
        if stale >= args.patience:
            print(f"no dev improvement for {args.patience} epochs; stopping", flush=True)
            break

    # The fine-tuned checkpoint: the base state dict with only the SRL role
    # head replaced, beside the base in the evaluation directory.
    src = CKPT_DIR / args.run / "best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    changed = 0
    for n_, p in best_state.items():
        assert n_ in ck["model"] and ck["model"][n_].shape == p.shape
        ck["model"][n_] = p.detach().cpu().to(ck["model"][n_].dtype)
        changed += 1
    ck["srl_finetune"] = {"base": args.run, "base_dev": best["value"],
                          "data": ("srl_train.json all sources" if args.all_sources else
                                   "PropBank EWT gold portion of srl_train.json") +
                                  " (kniv-corpus-en); test sentences excluded",
                          "trained": list(SRL_PARAMS), "history": history,
                          "dev_f1_before": base, "dev_f1_after": best_f1, "args": vars(args)}
    out = CKPT_DIR / args.out_run
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "best.pt")
    (out / "best.metric").write_text(json.dumps({
        "value": best["value"], "step": best["step"],
        "metric": "dev composite of the base run; SRL role head fine-tuned on PropBank gold",
        "srl_dev_f1": best_f1}))
    print(f"dev SRL F1 {base:.4f} -> {best_f1:.4f}; {changed} SRL tensors replaced; "
          f"wrote {out / 'best.pt'}", flush=True)
    print(f"score it: uv run python -m v6.train.gold_eval --local {CKPT_DIR} "
          f"--runs {args.run},{args.out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
