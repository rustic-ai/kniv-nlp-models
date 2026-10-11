"""Train one encoder on the v6 corpus.

    uv run python -m v6.train.loop --encoder ettin-1b --hf-repo dragonscale-ai/kniv-v6-runs
    uv run python -m v6.train.loop --encoder ettin-1b --probe 20     # A100 fit check

Every head trains jointly on every window (TRAINING.md §1, §6). Per-head
losses start uniform; after ``--calib-steps`` each is divided by its own
early mean so a head with more classes or more tokens does not dominate.
Per-head losses are logged every interval: a head flat from step zero is
invisible in the total.

Dev evaluation scores agreement with the corpus annotators and is used only
to select ``best.pt``. Headline numbers come from public gold (TRAINING.md §8).

Operations follow TRAINING.md §7 and OPERATIONS.md: run in the kernel, never
detached; resume reads ``latest.pt`` (pulled from the hub if this VM is new)
and replays the same sample order; SIGTERM checkpoints and exits.

**Control files** in the output directory are checked every step, so a run
can be steered without a kernel interrupt (``colab.py stop`` / ``checkpoint``
upload them):

``STOP``        checkpoint and exit cleanly; its text is recorded as the reason
``CHECKPOINT``  write ``latest.pt`` now and carry on

**Non-finite steps are skipped, never applied.** A NaN or inf loss or
gradient norm skips the update (the data is still consumed, so the sample
order is unchanged) and is recorded. ``--max-nonfinite`` in a row stops the
run *without* writing ``latest.pt``, so the last good resume point survives.
Resuming replays the same data, so a run stopped this way needs a change --
usually a lower learning rate, under a new run name -- not a plain relaunch.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

from ..config import RUNS_DIR
from . import encoders
from .data import IGNORE, batch_order, collate, featurise
from .model import HEAD_WEIGHT, HEADS, PHASE2_HEADS, RELATION_HEADS, V6Model, losses
from .runstate import Hub, RunState, memory
from .vocab import Vocab


# ── evaluation ──────────────────────────────────────────────────────
def _spans(tags: list[str]) -> set[tuple]:
    out, cur = set(), None
    for i, t in enumerate(tags + ["O"]):
        if cur and not (t.startswith("I-") and t[2:] == cur[0]):
            out.add((cur[0], cur[1], i))
            cur = None
        if t.startswith("B-") or (t.startswith("I-") and cur is None):
            cur = (t[2:], i)
    return out


def _f1(tp: float, n_pred: float, n_gold: float) -> float:
    return 2 * tp / max(n_pred + n_gold, 1)


@torch.no_grad()
def evaluate(model, feats, vocab: Vocab, pad_id: int, device, amp,
             batch_size: int, rel_gold: list | None = None) -> dict:
    """Dev scores and their composite. ``rel_gold``: human-annotated relation
    documents (Re-DocRED dev), scored as ``relation_gold_f1`` and counted in
    the composite."""
    on_gold = None
    sticky = model.rel_lora is not None and model.rel_lora.sticky
    if sticky:                       # the plain pass must be plain while evaluating
        model.rel_lora.sticky = model.rel_lora.on = False
    if rel_gold:
        on_gold = evaluate(model, rel_gold, vocab, pad_id, device, amp, max(1, batch_size // 4))
    model.eval()
    L = vocab.labels
    c = defaultdict(float)
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]))
    for k in range(0, len(order), batch_size):
        b = _to(collate([feats[i] for i in order[k:k + batch_size]], pad_id), device)
        with amp():
            o = model(b, predict_heads=True)
        for h in ("pos", "morph", "lemma"):
            t, m = b[h], b[h] != IGNORE
            c[f"{h}_ok"] += (o[h].argmax(-1)[m] == t[m]).sum().item()
            c[f"{h}_n"] += m.sum().item()
        t, m = b["dep_head"], b["dep_head"] != IGNORE
        ua = (o["rel_head"] == t) & m
        c["uas_ok"] += ua.sum().item()
        c["las_ok"] += (ua & (o["rel"].argmax(-1) == b["dep_rel"])).sum().item()
        c["dep_n"] += m.sum().item()
        t, m = b["srl_pred"], b["srl_pred"] != IGNORE
        p = o["srl_pred"].argmax(-1)
        c["pred_tp"] += ((p == 1) & (t == 1) & m).sum().item()
        c["pred_p"] += ((p == 1) & m).sum().item()
        c["pred_g"] += ((t == 1) & m).sum().item()
        ner_p, ner_g = o["ner"].argmax(-1).tolist(), b["ner"].tolist()
        for pp, gg in zip(ner_p, ner_g):
            g = [L["ner"][x] if x != IGNORE else "O" for x in gg]
            q = [L["ner"][x] if y != IGNORE else "O" for x, y in zip(pp, gg)]
            gs, ps = _spans(g), _spans(q)
            c["ner_tp"] += len(gs & ps); c["ner_p"] += len(ps); c["ner_g"] += len(gs)
        if len(b["frame_batch"]):
            rp, rg = o["srl_role"].argmax(-1).tolist(), b["frame_roles"].tolist()
            for pp, gg in zip(rp, rg):
                g = [L["srl"][x] if x != IGNORE else "O" for x in gg]
                q = [L["srl"][x] if y != IGNORE else "O" for x, y in zip(pp, gg)]
                gs = {s for s in _spans(g) if s[0] != "V"}
                ps = {s for s in _spans(q) if s[0] != "V"}
                c["role_tp"] += len(gs & ps); c["role_p"] += len(ps); c["role_g"] += len(gs)
        ok = b["cls_ok"]
        pc, gc = (o["cls"][ok] > 0), b["cls"][ok] > 0.5
        c["cls_tp"] += (pc & gc).sum().item()
        c["cls_p"] += pc.sum().item(); c["cls_g"] += gc.sum().item()
        t, m = b["sentiment"], b["sentiment"] != IGNORE
        c["sent_ok"] += (o["sentiment"].argmax(-1)[m] == t[m]).sum().item()
        c["sent_n"] += m.sum().item()
        if "coref_scores" in o:
            _coref_counts(o, b, c)
        if "ent" in o and "ent" in b:
            from .data import ENT_TAGS
            for pp, gg in zip(o["ent"].argmax(-1).tolist(), b["ent"].tolist()):
                g = [ENT_TAGS[x] if x != IGNORE else "O" for x in gg]
                q = [ENT_TAGS[x] if y != IGNORE else "O" for x, y in zip(pp, gg)]
                gs, ps = _spans(g), _spans(q)
                c["ent_tp"] += len(gs & ps); c["ent_p"] += len(ps); c["ent_g"] += len(gs)
        if "relation" in o and len(o["relation"]):
            lg, gold = o["relation"], b["pair_labels"][:, 1:] > 0.5
            pred = lg[:, 1:] > lg[:, :1]
            c["rel_tp"] += (pred & gold).sum().item()
            c["rel_p"] += pred.sum().item(); c["rel_g"] += gold.sum().item()
    model.train()
    if sticky:
        model.rel_lora.sticky = True
    r = {h: c[f"{h}_ok"] / max(c[f"{h}_n"], 1) for h in ("pos", "morph", "lemma")}
    r.update(uas=c["uas_ok"] / max(c["dep_n"], 1), las=c["las_ok"] / max(c["dep_n"], 1),
             ner_f1=_f1(c["ner_tp"], c["ner_p"], c["ner_g"]),
             srl_pred_f1=_f1(c["pred_tp"], c["pred_p"], c["pred_g"]),
             srl_role_f1=_f1(c["role_tp"], c["role_p"], c["role_g"]),
             cls_f1=_f1(c["cls_tp"], c["cls_p"], c["cls_g"]),
             sentiment=c["sent_ok"] / max(c["sent_n"], 1))
    keys = ["pos", "morph", "lemma", "ner_f1", "las", "srl_pred_f1", "srl_role_f1",
            "cls_f1", "sentiment"]
    if c["link_g"] or c["rel_g"]:
        r.update(coref_link_f1=_f1(c["link_tp_p"], c["link_p"], c["link_g"])
                 if c["link_g"] else 0.0,
                 relation_f1=_f1(c["rel_tp"], c["rel_p"], c["rel_g"]))
        keys += ["coref_link_f1", "relation_f1"]
    if c["ent_g"]:
        r["ent_f1"] = _f1(c["ent_tp"], c["ent_p"], c["ent_g"])
    if on_gold is not None:
        r["relation_gold_f1"] = on_gold["relation_f1"]
        keys.append("relation_gold_f1")
        if "ent_f1" in on_gold:
            r["ent_gold_f1"] = on_gold["ent_f1"]
            keys.append("ent_gold_f1")
    r = {k: round(v, 5) for k, v in r.items()}
    # One number to select best.pt on: every head counts equally.
    r["composite"] = round(sum(r[k] for k in keys) / len(keys), 5)
    return r


def _coref_counts(o: dict, b: dict, c: dict) -> None:
    """Antecedent-link F1 on gold mentions: a predicted link (word -> earlier
    word) is right when both are gold mentions of one cluster; recall is over
    gold mentions that have an earlier mention in their cluster."""
    s, cand = o["coref_scores"], o["coref_cand"]
    B, W, _ = cand.shape
    cl = torch.full((B, W), -1, dtype=torch.long, device=cand.device)
    cl[b["coref_batch"], b["coref_head"]] = b["coref_cluster"]
    ok = b["coref_ok"].unsqueeze(1) & b["word_mask"]
    best = s.argmax(-1)                                                # 0 = dummy
    ante = torch.gather(cand, 2, (best - 1).clamp(min=0).unsqueeze(-1)).squeeze(-1)
    linked = (best > 0) & ok
    right = linked & (cl >= 0) & (torch.gather(cl, 1, ante) == cl)
    # gold mentions with an earlier mention of their cluster
    has_prev = torch.zeros(B, W, dtype=torch.bool, device=cand.device)
    order = torch.argsort(b["coref_head"] + b["coref_batch"] * (W + 1))
    seen = set()
    for i in order.tolist():
        key = int(b["coref_cluster"][i])
        if key in seen:
            has_prev[b["coref_batch"][i], b["coref_head"][i]] = True
        seen.add(key)
    c["link_tp_p"] += right.sum().item()
    c["link_p"] += linked.sum().item()
    c["link_g"] += (has_prev & ok).sum().item()


def _load_init(model, init: str, vocab: Vocab) -> None:
    """Weights from a phase-1 checkpoint; the phase-2 heads start fresh. A
    vocabulary that disagrees on any shared inventory refuses to load."""
    path = Path(init)
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download("dragonscale-ai/kniv-v6-runs", f"{init}/best.pt"))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    for k, labels in ck["vocab"]["labels"].items():
        if vocab.labels.get(k) != labels:
            raise SystemExit(f"--init vocabulary differs from this one on {k!r}")
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    fresh = [m for m in missing if not m.startswith(("coref.", "relation.",
                                                      "mix.coref.", "mix.relation.", "ent.",
                                                      "mix.ent.", "rel_lora."))]
    if fresh or unexpected:
        raise SystemExit(f"--init does not fit this model: missing {fresh[:5]}, "
                         f"unexpected {unexpected[:5]}")
    print(f"initialised from {init}: {len(ck['model'])} tensors loaded, "
          f"{len(missing)} phase-2 tensors fresh", flush=True)


def _to(batch: dict, device) -> dict:
    # non_blocking only on CUDA: on MPS a non-blocking copy from unpinned
    # memory races with the source being freed and delivers garbage indices.
    nb = device.type == "cuda"
    return {k: v.to(device, non_blocking=nb) for k, v in batch.items()}


# ── training ────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--encoder", required=True, choices=list(encoders.ENCODERS))
    ap.add_argument("--run-name", help="default: the encoder name")
    ap.add_argument("--out", type=Path, help="default: v6/runs/train/<run-name>")
    ap.add_argument("--hf-repo", help="private HF repo to mirror checkpoints to")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, help="stop after this many steps")
    ap.add_argument("--encoder-lr", type=float)
    ap.add_argument("--head-lr", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--grad-accum", type=int)
    ap.add_argument("--phase2", action="store_true",
                    help="add the coreference and relation heads (TRAINING.md §4)")
    ap.add_argument("--init", help="start from a checkpoint: a best.pt path, or a run "
                                   "name in the hub repo (e.g. deberta-v3-large-srlft-full)")
    ap.add_argument("--rel-adapter", type=int, default=0, metavar="R",
                    help="rank of a LoRA adapter for a second, relation-only encoder pass "
                         "feeding the relation and entity heads (0 = none)")
    ap.add_argument("--train-scope", choices=["all", "phase2", "relation"], default="all",
                    help="phase2: train only the phase-2 heads and their layer mixes; "
                         "everything else stays exactly as --init left it")
    ap.add_argument("--relation-data", choices=["all", "redocred"], default="all",
                    help="redocred: train on Re-DocRED documents alone, without the "
                         "corpus windows and their teacher relation labels "
                         "(with --train-scope relation, where nothing else trains)")
    ap.add_argument("--redocred-train", type=int, default=0, metavar="N",
                    help="add Re-DocRED train, N times per epoch, as relation-only "
                         "supervision; its dev then joins best.pt selection (0 = off)")
    ap.add_argument("--head-weight", action="append", default=[], metavar="HEAD=W",
                    help="fixed weight on a head's scaled loss, e.g. relation=3 to give "
                         "it a larger share of the encoder update (repeatable)")
    ap.add_argument("--p2-context", type=int, default=0, metavar="N",
                    help="transformer layers over words inside each phase-2 head "
                         "(for a frozen encoder; 0 = none)")
    ap.add_argument("--srl-mode", choices=["joint", "conditioned"], default="joint",
                    help="joint: every predicate in one pass (v6); conditioned: one "
                         "pass per predicate over its sentence (v5), for MODEL_CHANGES §1")
    ap.add_argument("--grad-checkpointing", choices=["on", "off"],
                    help="override the encoder's default")
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--calib-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-minutes", type=float, default=30.0)
    ap.add_argument("--push-minutes", type=float, default=5.0,
                    help="mirror status, events and train.log to the hub this often")
    ap.add_argument("--max-nonfinite", type=int, default=5,
                    help="stop after this many consecutive non-finite steps")
    ap.add_argument("--train-limit", type=int)
    ap.add_argument("--dev-limit", type=int)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device")
    ap.add_argument("--probe", type=int, metavar="STEPS",
                    help="memory/speed probe on the longest batches; saves nothing")
    args = ap.parse_args()

    spec = encoders.get(args.encoder)
    over = {k: v for k, v in (("encoder_lr", args.encoder_lr), ("head_lr", args.head_lr),
                              ("batch_size", args.batch_size),
                              ("grad_accum", args.grad_accum),
                              ("grad_checkpointing", None if args.grad_checkpointing is None
                               else args.grad_checkpointing == "on")) if v is not None}
    spec = dataclasses.replace(spec, **over)
    run_name = args.run_name or spec.name
    out = args.out or RUNS_DIR / "train" / run_name
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available()
                                          else "cpu"))
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()

    def amp():
        return (torch.autocast("cuda", dtype=torch.bfloat16) if bf16
                else torch.autocast(device.type, enabled=False))

    torch.manual_seed(args.seed)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    vocab = Vocab.load()
    cond = args.srl_mode == "conditioned"
    train = featurise(spec, "train", vocab, tok, limit=args.train_limit, srl_sentences=cond,
                      phase2=args.phase2)
    dev = (featurise(spec, "dev", vocab, tok, limit=args.dev_limit, srl_sentences=cond,
                     phase2=args.phase2) if not args.probe else [])
    heads = HEADS + (PHASE2_HEADS if args.phase2 else ())
    weight = dict(HEAD_WEIGHT)
    for hw in args.head_weight:
        h, _, w = hw.partition("=")
        if h not in heads or not w:
            raise SystemExit(f"--head-weight {hw!r}: expected HEAD=W with HEAD one of {heads}")
        weight[h] = float(w)
    rd_dev = []
    if args.redocred_train:
        if not args.phase2:
            raise SystemExit("--redocred-train needs --phase2")
        from .data import redocred_features
        st = Counter()
        rd = redocred_features("train", tok, spec.max_subwords, vocab, st)
        if args.relation_data == "redocred":
            if args.train_scope != "relation":
                raise SystemExit("--relation-data redocred needs --train-scope relation: "
                                 "the corpus windows supervise every other head")
            train = rd * args.redocred_train
        else:
            train = train + rd * args.redocred_train
        rd_dev = (redocred_features("dev", tok, spec.max_subwords, vocab, st)
                  if not args.probe else [])
        print("Re-DocRED: " + ", ".join(f"{k} {v:,}" for k, v in sorted(st.items()))
              + f"; train x{args.redocred_train}", flush=True)

    model = V6Model(spec.hf_id, {k: vocab.size(k) for k in vocab.labels},
                    attn_implementation=spec.attn_implementation,
                    srl_mode=args.srl_mode, phase2=args.phase2,
                    p2_context=args.p2_context, rel_adapter=args.rel_adapter).to(device)
    if args.init:
        _load_init(model, args.init, vocab)
    if args.train_scope == "phase2":
        if not args.phase2:
            raise SystemExit("--train-scope phase2 needs --phase2")
        for n, prm in model.named_parameters():
            prm.requires_grad = n.startswith(("coref.", "relation.", "mix.coref.",
                                              "mix.relation.", "ent.", "mix.ent."))
        model.freeze_encoder = True
        model.train()
    elif args.train_scope == "relation":
        # The relation adapter, relation and entity heads alone, on the adapted
        # pass only: encoder weights, phase 1 and coref stay bit-identical.
        if not args.rel_adapter:
            raise SystemExit("--train-scope relation needs --rel-adapter")
        for n, prm in model.named_parameters():
            prm.requires_grad = n.startswith(("rel_lora.", "relation.", "mix.relation.",
                                              "ent.", "mix.ent."))
        heads = RELATION_HEADS
        model.rel_lora.sticky = True
    wrong = {str(p.dtype) for p in model.parameters()} - {"torch.float32"}
    if wrong:
        raise SystemExit(f"parameters must be fp32 master weights; found {wrong}")
    if spec.grad_checkpointing:
        model.encoder.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    enc_decay, enc_plain = [], []
    for n, p in model.encoder.named_parameters():
        if p.requires_grad:
            (enc_plain if p.ndim < 2 or "norm" in n.lower() else enc_decay).append(p)
    groups = [{"params": enc_decay, "lr": spec.encoder_lr, "weight_decay": args.weight_decay},
              {"params": enc_plain, "lr": spec.encoder_lr, "weight_decay": 0.0},
              {"params": [p for p in model.head_parameters() if p.requires_grad],
               "lr": spec.head_lr, "weight_decay": 0.0}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.98),
                            eps=1e-6, fused=device.type == "cuda")

    micro = spec.batch_size
    if args.probe:
        # Worst case: the longest windows, every step.
        longest = sorted(range(len(train)), key=lambda i: -len(train[i]["input_ids"]))
        order = [longest[i:i + micro] for i in range(0, micro * spec.grad_accum, micro)]
        if args.phase2:
            # The relation head grows with entity pairs, not length: the last
            # micro-batch is the windows with the most pairs instead.
            def n_pairs(i):
                r = train[i].get("rel")
                return len(r["spans"]) ** 2 if r else 0
            order[-1] = sorted(range(len(train)), key=lambda i: -n_pairs(i))[:micro]
        return _probe(model, opt, train, order, args.probe, spec, tok, device, amp)

    per_epoch = len(batch_order(train, micro, args.seed, 0)) // spec.grad_accum
    total = args.max_steps or per_epoch * args.epochs
    warm = max(1, int(total * args.warmup))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total - s) / max(total - warm, 1))))

    hub = Hub(args.hf_repo, run_name) if args.hf_repo else None
    rs = RunState(out, run_id=f"{run_name}-{int(time.time())}", hub=hub,
                  meta={"encoder": spec.asdict(), "args": vars(args) | {"out": str(out)},
                        "train_windows": len(train), "dev_windows": len(dev),
                        "steps_per_epoch": per_epoch})
    rs.push_minutes = args.push_minutes
    scale, calib = {h: 1.0 for h in heads}, defaultdict(list)
    ck = rs.try_resume(model, opt, sched)
    if ck:
        scale, calib = ck["loss_scale"], defaultdict(list, ck["calib"])
    rs.status(state="training", total_steps=total, push=True)
    print(f"{spec.name} on {device}{' bf16' if bf16 else ''}: {len(train)} train / "
          f"{len(dev)} dev windows, {per_epoch} steps/epoch, {total} steps, "
          f"batch {micro}x{spec.grad_accum}", flush=True)

    def extra():
        return {"loss_scale": scale, "calib": dict(calib), "vocab": vocab.to_json(),
                "encoder": spec.asdict(),
                "model_config": {"srl_mode": args.srl_mode, "phase2": args.phase2,
                                 "p2_context": args.p2_context, "ent_head": True,
                                 "rel_adapter": args.rel_adapter,
                                 "redocred_train": args.redocred_train,
                                 "relation_data": args.relation_data,
                                 "init": args.init, "train_scope": args.train_scope}}

    def stop(state, reason=""):
        # Checkpoint upload first, status last: a mirrored "stopped" or "done"
        # then means latest.pt is already on the hub, so the VM may be released.
        rs.save_latest(model, opt, sched, extra())
        if hub:
            hub.wait()
        rs.status(state=state, reason=reason, push=True)

    def control() -> str | None:
        """Act on a control file; return a stop reason if the run must end."""
        ckf, stopf = out / "CHECKPOINT", out / "STOP"
        if ckf.exists():
            ckf.unlink()
            rs.event("control", action="checkpoint")
            rs.save_latest(model, opt, sched, extra())
            rs.status(push=True)
            print(f"  checkpoint requested: saved at step {rs.p.step}", flush=True)
        if stopf.exists():
            why = stopf.read_text().strip() or "STOP file"
            stopf.unlink()                 # or a relaunch would stop at once
            rs.event("control", action="stop", reason=why)
            return why
        return None

    model.train()
    last_save = time.time()
    log, n_log = defaultdict(float), 0
    try:
        while rs.p.step < total:
            epoch = rs.p.epoch
            batches = batch_order(train, micro, args.seed, epoch)
            start = (rs.p.step - epoch * per_epoch) * spec.grad_accum
            for m0 in range(start, per_epoch * spec.grad_accum, spec.grad_accum):
                this, seen, bad = defaultdict(float), set(), []
                for m in range(m0, m0 + spec.grad_accum):
                    b = _to(collate([train[i] for i in batches[m]], tok.pad_token_id), device)
                    with amp():
                        o = model(b, relation_only=args.train_scope == "relation")
                    L = losses(o, b)
                    vals = {h: l.item() for h, (l, n) in L.items() if n}
                    bad = [h for h, v in vals.items() if not math.isfinite(v)]
                    if bad:
                        break                  # never backpropagate a non-finite loss
                    loss = sum(scale.get(h, 1.0) * weight.get(h, 1.0) * l
                               for h, (l, _) in L.items()) / spec.grad_accum
                    loss.backward()
                    for h, v in vals.items():
                        this[h] += v / spec.grad_accum
                        seen.add(h)
                    this["total"] += loss.item()
                gn = None
                if not bad:
                    gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    if not math.isfinite(gn):
                        bad = ["grad_norm"]
                if bad:
                    opt.zero_grad(set_to_none=True)
                    sched.step()               # the step is consumed, not applied
                else:
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                rs.p.step += 1
                step = rs.p.step
                rs.p.lr = sched.get_last_lr()[0]
                if bad:
                    rs.p.nonfinite_steps += 1
                    rs.p.nonfinite_streak += 1
                    rs.event("nonfinite", step=step, heads=bad,
                             streak=rs.p.nonfinite_streak)
                    print(f"  NON-FINITE at step {step} ({', '.join(bad)}): update "
                          f"skipped, {rs.p.nonfinite_streak} in a row", flush=True)
                    if rs.p.nonfinite_streak >= args.max_nonfinite:
                        why = (f"{rs.p.nonfinite_streak} consecutive non-finite steps "
                               f"at step {step}; latest.pt left at the last good state")
                        rs.event("abort", reason=why)
                        rs.status(state="failed", reason=why, push=True)
                        if hub:
                            hub.wait()
                        print(f"STOPPED: {why}", flush=True)
                        return 2
                    rs.status()
                else:
                    rs.p.nonfinite_streak = 0
                    rs.p.grad_norm = round(gn, 4)
                    for k, v in this.items():
                        log[k] += v
                    n_log += 1

                # Loss scale: each head divided by its own mean over the first
                # calib_steps steps, so the heads start on an equal footing.
                # A skipped step's partial losses are not evidence.
                if step <= args.calib_steps and not bad:
                    for h in seen:
                        calib[h].append(this[h])
                if step == args.calib_steps:
                    scale = {h: 1.0 / max(sum(calib[h]) / len(calib[h]), 1e-3)
                             if calib[h] else 1.0 for h in heads}
                    rs.event("loss_scale", scale=scale)
                    print("  loss scales: " + " ".join(
                        f"{h} {s:.3f}" for h, s in scale.items()), flush=True)

                if (step % args.log_every == 0 or step == total) and n_log:
                    hl = {h: round(log[h] / n_log, 4) for h in heads}
                    rs.status(loss=round(log["total"] / n_log, 4), head_loss=hl)
                    p = rs.p
                    print(f"step {step}/{total} ep {epoch} loss {p.loss:.4f} | "
                          + " ".join(f"{h} {v:.3f}" for h, v in hl.items())
                          + f" | gn {p.grad_norm} lr {p.lr:.2e} | {p.steps_per_sec or 0:.2f} "
                          f"st/s eta {p.eta_minutes}m | mem {p.memory} disk {p.disk_free_gb}G",
                          flush=True)
                    log, n_log = defaultdict(float), 0

                if step % args.eval_every == 0 or step == total:
                    rs.status(state="evaluating")
                    r = evaluate(model, dev, vocab, tok.pad_token_id, device, amp, micro * 2,
                                 rel_gold=rd_dev)
                    rs.event("eval", step=step, **r)
                    better = rs.maybe_save_best(model, r["composite"], extra())
                    rs.status(state="training", last_eval=r, push=True)
                    print(f"  eval step {step}: " + " ".join(f"{k} {v:.4f}" for k, v in r.items())
                          + ("  * best" if better else ""), flush=True)

                why = control()
                if why:
                    stop("stopped", why)
                    print(f"stopped at step {step} ({why}); checkpointed", flush=True)
                    return 0
                if rs.stop_requested:
                    stop("interrupted", "signal")
                    print(f"interrupted at step {step}; checkpointed", flush=True)
                    return 0
                if time.time() - last_save > args.save_minutes * 60:
                    rs.save_latest(model, opt, sched, extra())
                    last_save = time.time()
                if step >= total:
                    break
            else:
                rs.p.epoch = epoch + 1
                rs.save_latest(model, opt, sched, extra())
                last_save = time.time()
                continue
            break
    except Exception as exc:
        rs.event("error", error=f"{type(exc).__name__}: {exc}", step=rs.p.step,
                 memory=memory())
        rs.status(state="failed", reason=f"{type(exc).__name__}: {exc}"[:300], push=True)
        if hub:
            hub.wait()
        raise
    stop("done")
    print(f"done: {rs.p.step} steps, best composite {rs.p.best:.4f} "
          f"at step {rs.p.best_step}", flush=True)
    return 0


def _probe(model, opt, train, order, steps, spec, tok, device, amp) -> int:
    """Run ``steps`` optimiser steps on the longest windows and report peak
    memory and speed. This is the A100 fit check: batch size, gradient
    checkpointing and bf16 are measured here, not assumed."""
    import json
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for s in range(steps):
        t0 = time.time()
        for idx in order:
            b = _to(collate([train[i] for i in idx], tok.pad_token_id), device)
            with amp():
                L = losses(model(b), b)
            (sum(l for l, _ in L.values()) / len(order)).backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.time() - t0)
        print(f"  probe step {s + 1}/{steps}: {times[-1]:.2f}s {memory()}", flush=True)
    n_words = sum(len(train[i]["first_subword"]) for idx in order for i in idx)
    rep = {"encoder": spec.name, "batch_size": spec.batch_size,
           "grad_accum": spec.grad_accum, "grad_checkpointing": spec.grad_checkpointing,
           "longest_subwords": max(len(train[i]["input_ids"]) for idx in order for i in idx),
           "sec_per_step": round(sum(times[1:]) / max(len(times) - 1, 1), 3),
           "words_per_sec": round(n_words / (sum(times[1:]) / max(len(times) - 1, 1)), 1),
           **memory()}
    print("PROBE " + json.dumps(rep), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
