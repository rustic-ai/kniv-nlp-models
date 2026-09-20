"""Train ATLOP on **Re-DocRED** train, for use as a v6 relation annotator.

Why retrain at all: the released ATLOP checkpoint was trained on the
ORIGINAL DocRED, whose systematic false negatives taught it to under-
propose. Measured against Re-DocRED's corrected gold it reaches 0.952
precision at only 0.302 recall. Published work reports ~+13 F1 for models
trained and evaluated on Re-DocRED rather than DocRED, so the ceiling here
is roughly 0.75-0.80 F1 — a different regime from the 0.459 the released
checkpoint gives us, and far beyond anything prompting achieved (best LLM
combination: 0.586).

For the corpus this matters as *precision at usable recall*. Relation
labels are training data: a missing triple costs supervision, a wrong one
teaches an error. The intended corpus rule is supervised-model AND LLM for
positives, with everything uncertain masked rather than labelled
``no_relation`` — the same per-token masking already used for morph.

LICENCE NOTE: Re-DocRED and DocRED are MIT. The ATLOP repository declares
**no licence**, so weights trained with its code are fine for internal
evaluation but need a decision before shipping. Re-implementing the
architecture from the paper (it is small: entity logsumexp pooling,
attention-based context pooling, grouped bilinear, adaptive thresholding)
removes that constraint if we ever publish.

SETUP: see ``v6/experiments/atlop_runner.py`` for the checkout and the
``meta/rel2id.json`` provenance. Then:

    ATLOP_DIR=<scratch> uv run python -m v6.experiments.atlop_train \\
        --epochs 30 --train-batch-size 4 --lr 3e-5

On a single A100 this is a few hours. The loop is plain PyTorch AMP — apex
and wandb are not required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .atlop_runstate import RunState

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "re-docred"


def _setup_atlop():
    base = os.environ.get("ATLOP_DIR")
    if not base:
        raise SystemExit("set ATLOP_DIR (see module docstring)")
    from .atlop_compat import patch_long_seq
    patch_long_seq(Path(base) / "ATLOP")
    sys.path.insert(0, str(Path(base) / "ATLOP"))
    os.chdir(Path(base) / "ATLOP")
    return base


def featurise_cached(read_docred, src: str, tok, max_len: int, cache_dir: Path):
    """Featurise once, reuse forever.

    Preprocessing the 3,053-document train split takes minutes, and on a
    preemptible box that cost is paid on every restart — which is enough to
    make resume not worth using. Keyed on the file's content hash plus the
    tokenizer and length, so a changed split invalidates itself.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(
        Path(src).read_bytes() + f"{tok.name_or_path}:{max_len}".encode()
    ).hexdigest()[:16]
    cached = cache_dir / f"feat-{Path(src).stem}-{digest}.pkl"
    if cached.exists():
        print(f"features: reusing {cached.name}", flush=True)
        with cached.open("rb") as fh:
            return pickle.load(fh)
    feats = read_docred(src, tok, max_seq_length=max_len)
    tmp = cached.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(feats, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cached)
    print(f"features: cached {cached.name} ({len(feats)} docs)", flush=True)
    return feats


def evaluate(model, features, device, batch_size, collate_fn, id2rel, gold):
    """Micro-F1 over (doc, h, t, relation) against Re-DocRED gold."""
    model.eval()
    loader = DataLoader(features, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn)
    preds = []
    with torch.no_grad():
        for batch in loader:
            out = model(input_ids=batch[0].to(device),
                        attention_mask=batch[1].to(device),
                        entity_pos=batch[3], hts=batch[4])
            logits = out[0] if isinstance(out, (tuple, list)) else out
            preds.append(logits.float().cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    tp = n_pred = n_gold = 0
    k = 0
    for f in features:
        got = set()
        for (h, t) in f["hts"]:
            row = preds[k]; k += 1
            for r in np.nonzero(row)[0]:
                if r != 0:
                    got.add((int(h), int(t), id2rel[int(r)]))
        g = gold.get(f["title"], set())
        tp += len(got & g); n_pred += len(got); n_gold += len(g)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--epochs", type=float, default=30)
    ap.add_argument("--train-batch-size", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--classifier-lr", type=float, default=1e-4)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--model", default="roberta-large")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--num-labels", type=int, default=4)
    ap.add_argument("--seed", type=int, default=66)
    ap.add_argument("--limit-train", type=int, default=0,
                    help="truncate the train split (smoke tests only)")
    ap.add_argument("--out-dir", default=str(REPO / "data" / "re-docred" / "atlop-run"),
                    help="checkpoints, status.json, events.jsonl. On Colab "
                         "point this at mounted Drive so it survives the VM.")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--save-every", type=int, default=200,
                    help="steps between full-state checkpoints; this is the "
                         "most work a preemption can cost")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore an existing latest.pt and start over")
    args = ap.parse_args()

    _setup_atlop()
    from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
    from model import DocREModel
    from prepro import read_docred
    from utils import collate_fn
    from .atlop_compat import patch_tokenizer

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rs = RunState(Path(args.out_dir), run_id, save_every=args.save_every)
    rs.p.meta.update({"argv": sys.argv[1:], "seed": args.seed,
                      "model": args.model})
    rs.status(state="featurising", device=device)
    rs.event("start", device=device, args=vars(args))
    print(f"device={device} | run {run_id} | out {args.out_dir}", flush=True)

    tok = patch_tokenizer(AutoTokenizer.from_pretrained(args.model))
    config = AutoConfig.from_pretrained(args.model, num_labels=97)
    config.cls_token_id, config.sep_token_id = tok.cls_token_id, tok.sep_token_id
    config.transformer_type = "roberta" if "roberta" in args.model else "bert"
    # eager: ATLOP reads attention weights, which SDPA does not materialise
    enc = AutoModel.from_pretrained(args.model, config=config,
                                    attn_implementation="eager")

    train_src = str(DATA / "train_revised.json")
    if args.limit_train:
        docs = json.load(open(train_src))[:args.limit_train]
        train_src = "/tmp/_redocred_train_slice.json"
        json.dump(docs, open(train_src, "w"))
    fcache = Path(args.out_dir) / "features"
    train_f = featurise_cached(read_docred, train_src, tok,
                               args.max_seq_length, fcache)
    dev_f = featurise_cached(read_docred, str(DATA / "dev_revised.json"), tok,
                             args.max_seq_length, fcache)
    print(f"train {len(train_f)} docs | dev {len(dev_f)} docs", flush=True)
    rs.event("featurised", train=len(train_f), dev=len(dev_f))

    rel2id = json.load(open("meta/rel2id.json"))
    id2rel = {v: k for k, v in rel2id.items()}
    gold = {d["title"]: {(l["h"], l["t"], l["r"]) for l in d["labels"]}
            for d in json.load(open(DATA / "dev_revised.json"))}

    model = DocREModel(config, enc, num_labels=args.num_labels).to(device)
    new = ["extractor", "bilinear", "classifier"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(k in n for k in new)], "lr": args.lr},
        {"params": [p for n, p in model.named_parameters()
                    if any(k in n for k in new)], "lr": args.classifier_lr},
    ]
    opt = torch.optim.AdamW(grouped, lr=args.lr, eps=1e-6)
    loader = DataLoader(train_f, batch_size=args.train_batch_size, shuffle=True,
                        collate_fn=collate_fn, drop_last=True)
    total = int(len(loader) * args.epochs // args.grad_accum)
    sched = get_linear_schedule_with_warmup(
        opt, int(total * args.warmup_ratio), total)
    scaler = torch.amp.GradScaler(device) if device == "cuda" else None
    print(f"total steps {total}", flush=True)

    start_epoch, step = (0, 0) if args.no_resume else rs.try_resume(
        model, opt, sched, scaler)
    best = rs.p.best_f1
    rs.status(state="training", total_steps=total, epoch=start_epoch, step=step)

    for epoch in range(start_epoch, int(args.epochs)):
        model.zero_grad()
        for i, batch in enumerate(loader):
            model.train()
            kw = {"input_ids": batch[0].to(device),
                  "attention_mask": batch[1].to(device),
                  "labels": batch[2], "entity_pos": batch[3], "hts": batch[4]}
            if scaler:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(**kw)[0] / args.grad_accum
                scaler.scale(loss).backward()
            else:
                loss = model(**kw)[0] / args.grad_accum
                loss.backward()
            if (i + 1) % args.grad_accum == 0:
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if scaler:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                sched.step(); model.zero_grad(); step += 1
                cur = loss.item() * args.grad_accum
                if step % args.log_every == 0:
                    rs.status(step=step, epoch=epoch, loss=round(cur, 4))
                    print(f"  epoch {epoch} step {step}/{total} loss {cur:.4f} "
                          f"eta {rs.p.eta_minutes}m", flush=True)
                if step % args.save_every == 0:
                    rs.p.step, rs.p.epoch = step, epoch
                    rs.save_latest(model, opt, sched, scaler)
                if rs.stop_requested:
                    # Preemption or Ctrl-C: checkpoint at a safe boundary and
                    # exit non-zero so a supervisor knows to restart it.
                    rs.p.step, rs.p.epoch = step, epoch
                    rs.save_latest(model, opt, sched, scaler)
                    rs.status(state="interrupted")
                    rs.event("interrupted", step=step, epoch=epoch)
                    print(f"\ninterrupted at step {step}; resume by re-running "
                          f"the same command", flush=True)
                    return 130
        rs.p.step, rs.p.epoch = step, epoch
        rs.status(state="evaluating")
        f1, p, r = evaluate(model, dev_f, device, args.eval_batch_size,
                            collate_fn, id2rel, gold)
        rs.event("eval", epoch=epoch, step=step, f1=f1, precision=p, recall=r)
        print(f"epoch {epoch}: dev F1={f1:.4f} P={p:.4f} R={r:.4f}"
              f"{'  <- best' if f1 > best else ''}", flush=True)
        if f1 > best:
            best = f1
            rs.save_best(model, f1, epoch)
        rs.status(state="training",
                  last_eval={"epoch": epoch, "f1": round(f1, 4),
                             "precision": round(p, 4), "recall": round(r, 4)})
        rs.save_latest(model, opt, sched, scaler)     # epoch boundary
    rs.status(state="done")
    rs.event("done", best_f1=best, best_epoch=rs.p.best_epoch)
    print(f"\nbest dev F1 {best:.4f} -> {rs.best}", flush=True)
    return 0


def guarded() -> int:
    """Run main, recording a crash in the event log before it is lost.

    On a disconnecting VM the traceback in the terminal is the first thing
    that disappears. events.jsonl is on disk (and on Drive, if --out-dir
    points there), so a post-mortem is possible.
    """
    try:
        return main()
    except SystemExit:
        raise
    except BaseException as exc:                     # noqa: BLE001
        try:
            out = Path([a.split("=", 1)[1] for a in sys.argv
                        if a.startswith("--out-dir=")][0]
                       if any(a.startswith("--out-dir=") for a in sys.argv)
                       else (sys.argv[sys.argv.index("--out-dir") + 1]
                             if "--out-dir" in sys.argv
                             else REPO / "data" / "re-docred" / "atlop-run"))
            out.mkdir(parents=True, exist_ok=True)
            with (out / "events.jsonl").open("a") as fh:
                fh.write(json.dumps({
                    "ts": time.time(), "kind": "crash",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-4000:],
                }) + "\n")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(guarded())
