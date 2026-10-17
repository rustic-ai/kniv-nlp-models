"""Fine-tune only the fact attribute head (polarity, modality, holder, time)
on the teacher corpus, everything else frozen.

    uv run python -m v6.train.attr_finetune --run deberta-v3-large-v7b

Like ``srl_finetune``: nothing below the head moves, so every other output is
bit-identical afterwards. The head's inputs -- each fact's pair vector, its
relation's label vector and its window's entity vectors -- are computed once
per window by the frozen model and cached; training then runs over the cache.

Why: in a shared run the polarity output can settle into "always positive"
(the teacher's facts are 1.7% negative) and stay there; a head trained alone
on cached inputs, with every window that holds a negation, gets out of it.

Train: every teacher training window with a negated fact plus a sample of
the rest (``--windows``). Dev: the teacher dev windows. The epoch is chosen on
the mean of negation F1 and modality, holder and time accuracy.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from .gold_eval import CKPT_DIR, HF_REPO, load_run
from .model import fact_attr_loss
from .srl_finetune import upload


@torch.no_grad()
def cache(model, feats, tok, bank, device) -> list[dict]:
    from .data import collate
    from .kdata import LabelCollator, TypeCollator
    from .loop import Chain
    coll = Chain(TypeCollator(tok, bank, train=False), LabelCollator(tok, bank, train=False))
    out = []
    for f in feats:
        b = coll(collate([f], tok.pad_token_id))
        if "attr_pair" not in b:
            continue
        b = {k: v.to(device) if hasattr(v, "to") else v for k, v in b.items()}
        model(b, relation_only=True)
        rel = model.relation
        ok = b["attr_col"] >= 0
        if not bool(ok.any()):
            continue
        p, c = b["attr_pair"][ok], b["attr_col"][ok]
        out.append({"pr": rel._pair[p].half().cpu(), "q": rel._labels[0, c].half().cpu(),
                    "ent": rel._ent.half().cpu(),
                    **{k: b[f"attr_{k}"][ok].cpu() for k in ("polarity", "modality", "holder", "time")}})
    return out


def batches(items: list[dict], idx, device):
    """Several windows' facts at once: entities concatenated, each fact allowed
    to point only into its own window's."""
    pr, q, ent, owner, eown, tgt = [], [], [], [], [], {k: [] for k in ("polarity", "modality",
                                                                        "holder", "time")}
    base = 0
    for w, i in enumerate(idx):
        x = items[i]
        n, E = len(x["pr"]), len(x["ent"])
        pr.append(x["pr"]); q.append(x["q"]); ent.append(x["ent"])
        owner += [w] * n; eown += [w] * E
        for k in ("polarity", "modality"):
            tgt[k].append(x[k])
        for k in ("holder", "time"):
            tgt[k].append(torch.where(x[k] >= 0, x[k] + base, x[k]))
        base += E
    owner, eown = torch.tensor(owner), torch.tensor(eown)
    same = (owner.unsqueeze(1) == eown.unsqueeze(0)).to(device)
    t = {f"attr_{k}_ok": torch.cat(v).to(device) for k, v in tgt.items()}
    return (torch.cat(pr).float().to(device), torch.cat(q).float().to(device),
            torch.cat(ent).float().to(device), same, t)


@torch.no_grad()
def score(head, items, device, chunk: int = 64) -> dict:
    head.eval()
    c = dict(neg_tp=0, neg_p=0, neg_g=0, mod=0, hold=0, time=0, n=0)
    for k in range(0, len(items), chunk):
        pr, q, ent, same, t = batches(items, range(k, min(k + chunk, len(items))), device)
        o = head.score(pr, q, ent, same)
        pp, gp = o["polarity"].argmax(-1), t["attr_polarity_ok"]
        c["neg_tp"] += int(((pp == 1) & (gp == 1)).sum()); c["neg_p"] += int((pp == 1).sum())
        c["neg_g"] += int((gp == 1).sum())
        c["mod"] += int((o["modality"].argmax(-1) == t["attr_modality_ok"]).sum())
        c["hold"] += int((o["holder"].argmax(-1) - 1 == t["attr_holder_ok"]).sum())
        c["time"] += int((o["time"].argmax(-1) - 1 == t["attr_time_ok"]).sum())
        c["n"] += len(gp)
    r = {"negation_f1": 2 * c["neg_tp"] / max(c["neg_p"] + c["neg_g"], 1),
         "modality_acc": c["mod"] / max(c["n"], 1), "holder_acc": c["hold"] / max(c["n"], 1),
         "time_acc": c["time"] / max(c["n"], 1)}
    r["mean"] = sum(r.values()) / 4
    return r


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True, help="base checkpoint (hub) with an attribute head")
    ap.add_argument("--out-run", help="default: <run>-attrft")
    ap.add_argument("--windows", type=int, default=3000,
                    help="teacher training windows without a negation to add to those with one")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=32, help="windows per step")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--train-limit", type=int, help="first N teacher training windows (a quick check)")
    args = ap.parse_args()
    out_run = args.out_run or f"{args.run}-attrft"
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok, spec, vocab, best = load_run(args.run, HF_REPO, device)
    if model.attr is None:
        raise SystemExit(f"{args.run} has no attribute head")
    model.eval()
    from ..label_bank import load as load_bank
    from .kdata import knowledge_features
    bank = load_bank()
    n_cls = vocab.size("cls")
    train = knowledge_features("teacher", "train", tok, spec.max_subwords, n_cls, bank)
    dev = knowledge_features("teacher", "dev", tok, spec.max_subwords, n_cls, bank)
    if args.train_limit:
        train, dev = train[:args.train_limit], dev[:max(args.train_limit // 4, 20)]

    def has_neg(f):
        return any(a["polarity"] == "negative" for a in (f["rel"] or {}).get("attrs") or [])
    neg = [f for f in train if f["rel"] and has_neg(f)]
    rest = [f for f in train if f["rel"] and f["rel"].get("attrs") and not has_neg(f)]
    pick = [rest[i] for i in rng.choice(len(rest), size=min(args.windows, len(rest)), replace=False)]
    t0 = time.time()
    tr, dv = cache(model, neg + pick, tok, bank, device), cache(model, dev, tok, bank, device)
    n_neg = sum(int((x["polarity"] == 1).sum()) for x in tr)
    print(f"{args.run}: cached {len(tr)} training windows ({sum(len(x['pr']) for x in tr):,} facts, "
          f"{n_neg:,} negated) and {len(dv)} dev windows in {time.time() - t0:.0f}s", flush=True)

    head = model.attr
    for p in model.parameters():
        p.requires_grad = False
    params = list(head.parameters())
    for p in params:
        p.requires_grad = True
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    before = score(head, dv, device)
    print(f"dev before: {json.dumps({k: round(v, 4) for k, v in before.items()})}", flush=True)
    best_r, best_state, stale = before, {k: v.clone() for k, v in head.state_dict().items()}, 0
    history = [{"epoch": 0, **before}]
    for epoch in range(1, args.epochs + 1):
        head.train()
        order = rng.permutation(len(tr))
        total, n = 0.0, 0
        for k in range(0, len(order), args.batch):
            pr, q, ent, same, t = batches(tr, order[k:k + args.batch], device)
            loss = fact_attr_loss(head.score(pr, q, ent, same), t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += loss.item(); n += 1
        r = score(head, dv, device)
        history.append({"epoch": epoch, "train_loss": total / n, **r})
        mark = ""
        if r["mean"] > best_r["mean"]:
            best_r, stale, mark = r, 0, "  * best"
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        print(f"epoch {epoch}: loss {total / n:.4f}  " +
              "  ".join(f"{k} {v:.4f}" for k, v in r.items()) + mark, flush=True)
        if stale >= args.patience:
            break

    src = CKPT_DIR / args.run / "best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    for k, v in best_state.items():
        name = f"attr.{k}"
        assert name in ck["model"] and ck["model"][name].shape == v.shape
        ck["model"][name] = v.detach().cpu().to(ck["model"][name].dtype)
    ck["attr_finetune"] = {"base": args.run, "history": history, "dev_before": before,
                           "dev_after": best_r, "args": vars(args)}
    out = CKPT_DIR / out_run
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ck, out / "best.pt")
    (out / "best.metric").write_text(json.dumps({
        "value": best["value"], "step": best["step"],
        "metric": "dev composite of the base run; fact attribute head fine-tuned on the teacher corpus",
        **{f"attr_dev_{k}": v for k, v in best_r.items()}}))
    print(f"dev {before['mean']:.4f} -> {best_r['mean']:.4f}; wrote {out / 'best.pt'}", flush=True)
    upload(out, out_run, not args.no_upload, "fact attribute head fine-tune")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
