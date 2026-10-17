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
PRED_PARAMS = ("srl_pred.",)


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
        model._word_span = (b["first_subword"], b["word_end"])       # the model's word pooling
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


# ── --data full: every PropBank predicate, predicate head trained too ──
def load_full(split: str, exclude: set) -> list[dict]:
    from ..config import DATA_DIR
    path = DATA_DIR / "prepared" / "propbank-full" / f"{split}.json"
    if not path.exists():
        raise SystemExit(f"{path} missing; run python -m v6.gold.propbank_full")
    return [r for r in json.loads(path.read_text())
            if tuple(r["words"]) not in exclude and len(r["words"]) <= 128 and r["frames"]]


def _decode(lp, p, tags, allowed, start, v):
    keep = np.ones_like(lp, dtype=bool)
    keep[:, v] = False
    keep[p, :] = False
    keep[p, v] = True
    return bio_spans([tags[x] for x in viterbi(lp, allowed, start, keep)])


@torch.no_grad()
def score_full(model, reps, rows, frames, tags, device) -> dict:
    """Predicate-ID F1, roles F1 on gold predicates, and end-to-end F1 (the
    model's own predicates; predicate, role and span must all match)."""
    model.eval()
    allowed, start = bio_constraints(tags)
    v = tags.index("V")
    pid = [0, 0, 0]; role = [0, 0, 0]; e2e = [0, 0, 0]
    found = []
    for si, r in enumerate(rows):
        logits = model.srl_pred(reps[si].to(device, torch.float32))
        fp = torch.nonzero(logits.argmax(-1) == 1).flatten().tolist()
        found.append(fp)
        g = {f["predicate_idx"] for f in r["frames"]}
        pid[0] += len(g & set(fp)); pid[1] += len(fp); pid[2] += len(g)
    by_sent = defaultdict(list)
    for fr in frames:
        by_sent[fr[0]].append(fr)
    gold_args = defaultdict(set)
    for k in range(0, len(frames), 256):
        chunk = frames[k:k + 256]
        lg, _ = role_logits(model, reps, chunk, device)
        lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
        for (si, p, gold), l in zip(chunk, lp):
            gs = bio_spans([tags[x] for x in gold])
            ps = _decode(l[:len(gold)], p, tags, allowed, start, v)
            role[0] += len(gs & ps); role[1] += len(ps); role[2] += len(gs)
            gold_args[si] |= {(p,) + s for s in gs}
    pred_frames = [(si, p, np.zeros(len(rows[si]["words"]), dtype=np.int64))
                   for si, fp in enumerate(found) for p in fp]
    pred_args = defaultdict(set)
    for k in range(0, len(pred_frames), 256):
        chunk = pred_frames[k:k + 256]
        lg, _ = role_logits(model, reps, chunk, device)
        lp = torch.log_softmax(lg.float(), -1).cpu().numpy()
        for (si, p, z), l in zip(chunk, lp):
            pred_args[si] |= {(p,) + s for s in _decode(l[:len(z)], p, tags, allowed, start, v)}
    for si in range(len(rows)):
        e2e[0] += len(gold_args[si] & pred_args[si])
        e2e[1] += len(pred_args[si]); e2e[2] += len(gold_args[si])

    def f1(c):
        P, R = c[0] / max(c[1], 1), c[0] / max(c[2], 1)
        return 2 * P * R / max(P + R, 1e-12)
    return {"pred_id_f1": f1(pid), "role_f1": f1(role), "e2e_f1": f1(e2e)}


def upload(out, run: str, enabled: bool, what: str = "SRL head fine-tune") -> None:
    """Mirror a fine-tuned checkpoint to the runs repo beside the training
    runs' (``<run>/best.pt``, ``best.metric``), so a local fine-tune is not
    the only copy."""
    if not enabled:
        return
    from huggingface_hub import HfApi
    HfApi().upload_folder(repo_id=HF_REPO, folder_path=str(out), path_in_repo=run,
                          allow_patterns=["best.pt", "best.metric"],
                          commit_message=f"Add {run} ({what})")
    print(f"uploaded to {HF_REPO}/{run}", flush=True)


def main_full(args, model, tok, spec, vocab, best, device, rng) -> int:
    tags = vocab.labels["srl"]
    test = {tuple(r["words"]) for r in load_full("test", set())} | {
        tuple(e["words"]) for e in json.loads((PREPARED_DIR / "srl_test.json").read_text())
        if e.get("words")}
    train, dev = load_full("train", test), load_full("dev", test)
    print(f"{args.run} (dev {best['value']}): full PropBank EWT, train {len(train)} sentences / "
          f"{sum(len(r['frames']) for r in train)} frames, dev {len(dev)} / "
          f"{sum(len(r['frames']) for r in dev)}; test sentences excluded", flush=True)
    t0 = time.time()
    train_r = cache_reps(model, [tuple(r["words"]) for r in train], tok, spec, vocab, device)
    dev_r = cache_reps(model, [tuple(r["words"]) for r in dev], tok, spec, vocab, device)
    print(f"cached representations in {time.time() - t0:.0f}s", flush=True)
    ix = vocab.index["srl"]

    def frames(rows):
        return [(si, f["predicate_idx"], np.array([ix[x] for x in f["srl_tags"]], dtype=np.int64))
                for si, r in enumerate(rows) for f in r["frames"]]
    train_f, dev_f = frames(train), frames(dev)
    pred_target = [torch.tensor([int(i in {f["predicate_idx"] for f in r["frames"]})
                                 for i in range(len(r["words"]))]) for r in train]
    for n, p in model.named_parameters():
        p.requires_grad = n.startswith(SRL_PARAMS + PRED_PARAMS)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable: {sum(p.numel() for p in params):,} parameters "
          f"({', '.join(SRL_PARAMS + PRED_PARAMS)})", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    base = score_full(model, dev_r, dev, dev_f, tags, device)
    print("dev before: " + "  ".join(f"{k} {v:.4f}" for k, v in base.items()), flush=True)
    frames_of_sent = defaultdict(list)
    for fr in train_f:
        frames_of_sent[fr[0]].append(fr)
    best_e2e, stale, history = base["e2e_f1"], 0, [{"epoch": 0, **base}]
    best_state = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    sent_batch = max(8, args.batch // 4)          # ~4 frames per sentence
    for epoch in range(1, args.epochs + 1):
        for m in (model.srl_p, model.srl_a, model.srl_role, model.srl_pred):
            m.train()
        order = rng.permutation(len(train))
        tot = n = 0
        for k in range(0, len(order), sent_batch):
            sents = order[k:k + sent_batch].tolist()
            # predicate head: every word of these fully annotated sentences
            reps = [train_r[s].to(device, torch.float32) for s in sents]
            plog = model.srl_pred(torch.cat(reps))
            ploss = F.cross_entropy(plog.float(), torch.cat([pred_target[s] for s in sents]).to(device))
            # role head: every frame of these sentences
            chunk = [fr for s in sents for fr in frames_of_sent[s]]
            lg, tgt = role_logits(model, train_r, chunk, device)
            rloss = F.cross_entropy(lg.reshape(-1, lg.size(-1)).float(), tgt.reshape(-1),
                                    ignore_index=IGNORE)
            loss = ploss + rloss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tot += loss.item(); n += 1
        r = score_full(model, dev_r, dev, dev_f, tags, device)
        history.append({"epoch": epoch, "train_loss": tot / n, **r})
        mark = ""
        if r["e2e_f1"] > best_e2e:
            best_e2e, stale, mark = r["e2e_f1"], 0, "  * best"
            best_state = {n_: p.detach().clone() for n_, p in model.named_parameters()
                          if p.requires_grad}
        else:
            stale += 1
        print(f"epoch {epoch}: train loss {tot / n:.4f}  dev " +
              "  ".join(f"{k} {v:.4f}" for k, v in r.items()) + mark, flush=True)
        if stale >= args.patience:
            print(f"no dev end-to-end improvement for {args.patience} epochs; stopping", flush=True)
            break
    return _write(args, best_state, best, history, base, {"e2e_f1": best_e2e},
                  "full PropBank EWT (every predicate; gold.propbank_full)",
                  SRL_PARAMS + PRED_PARAMS)


def _write(args, best_state, best, history, before, after, data, trained) -> int:
    src = CKPT_DIR / args.run / "best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for n_, p in best_state.items():
        assert n_ in ck["model"] and ck["model"][n_].shape == p.shape
        ck["model"][n_] = p.detach().cpu().to(ck["model"][n_].dtype)
    ck["srl_finetune"] = {"base": args.run, "base_dev": best["value"], "data": data,
                          "trained": list(trained), "history": history,
                          "dev_before": before, "dev_after": after, "args": vars(args)}
    out = CKPT_DIR / args.out_run
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "best.pt")
    (out / "best.metric").write_text(json.dumps({
        "value": best["value"], "step": best["step"],
        "metric": f"dev composite of the base run; {', '.join(trained)} fine-tuned on {data}",
        **{f"srl_dev_{k}": v for k, v in after.items()}}))
    print(f"{len(best_state)} SRL tensors replaced; wrote {out / 'best.pt'}", flush=True)
    upload(out, args.out_run, not args.no_upload)
    print(f"score it: uv run python -m v6.train.gold_eval --local {CKPT_DIR} "
          f"--runs {args.run},{args.out_run} --benchmarks srl,srl_full")
    return 0


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
    ap.add_argument("--data", choices=["full", "root"], default="full",
                    help="full: every PropBank predicate (gold.propbank_full), predicate head "
                         "trained too; root: v5's one-root-predicate files, role head only")
    ap.add_argument("--local", type=Path, help="read <dir>/<run>/best.pt instead of the hub")
    ap.add_argument("--word-pool", choices=["first", "mean"],
                    help="override the run's word pooling (a probe; the checkpoint keeps its own)")
    ap.add_argument("--no-upload", action="store_true",
                    help="keep the fine-tuned checkpoint local (default: also upload it to the runs repo)")
    ap.add_argument("--all-sources", action="store_true",
                    help="train on all of srl_train.json (PropBank + MASC + QA-SRL + silver)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))

    model, tok, spec, vocab, best = load_run(args.run, HF_REPO, device, args.local)
    assert model.srl_mode == "joint", "fine-tunes the joint SRL head"
    if args.word_pool:                     # a probe: the same encoder, other word vectors
        model.word_pool = args.word_pool
    if args.data == "full":
        if args.out_run == "deberta-v3-large-srlft":
            args.out_run = "deberta-v3-large-srlft-full"
        return main_full(args, model, tok, spec, vocab, best, device, rng)
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
    upload(out, args.out_run, not args.no_upload)
    print(f"score it: uv run python -m v6.train.gold_eval --local {CKPT_DIR} "
          f"--runs {args.run},{args.out_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
