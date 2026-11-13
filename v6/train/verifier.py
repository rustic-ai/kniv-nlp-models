"""Fact verifier (plan r4 5.2, pilot V1): reads the text with one candidate
fact written after it and returns supported / not supported / negated /
hedged.

    uv run python -m v6.train.verifier build                     # -> data/v6-corpus/verifier_v1.json
    uv run python -m v6.train.verifier train --run ultrabert-380m-v8e-w1half --out-run ultrabert-380m-v8v1
    uv run python -m v6.train.verifier score-m1 --run ultrabert-380m-v8v1 --name m1p3b-n500 --config e1b

Model. The run's encoder and heads stay frozen. The input is the marker
pass's (every entity mention between ``*``) followed by ``[SEP] head |
relation | tail | description [SEP]``, encoded through a new LoRA adapter
started as a copy of the run's relation adapter, so the pass starts as
today's marker pass while text and candidate attend to each other. The
pair's head, tail and context vectors come from the run's own relation head
(``pair_features``) over that pass; with the mean of the candidate's tokens
they feed a small MLP to four verdicts. Only the adapter and the MLP train.

Data (all on disk; ``build``). Per short training text, every judged
candidate of D1 (gold entities), and as free negatives: the same pair with
another relation of the text's schema, the pair reversed (asymmetric
relations), and an argument cut or widened by one word. D3's forms and the
minimal pairs give negated and hedged statements, D2's concept passages
business and legal facts. D1e's judged end-to-end candidates are added when
their labels exist. Texts in ``negation_pilot.heldout_base``'s fifth, and
D3 bases in ``heldout_form``'s, are the held-out split. D2's generated names
replace a fifth of the relation names in training.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

VERDICTS = ["supported", "not_supported", "negated", "hedged"]
POSITIVE = {"supported", "negated", "hedged"}
SYMMETRIC_HINT = ("sibling", "spouse", "partner", "shares", "adjacent", "neighbo", "twinned",
                  "related to", "associated with", "married", "friend", "same as", "different from",
                  "competes with", "collaborat", "interacts with")


def _corpus() -> Path:
    from ..config import DATA_DIR
    return DATA_DIR / "v6-corpus"


DATA = "verifier_v1.json"


# ── build ──────────────────────────────────────────────────────────
def _entities(row: dict) -> tuple[list[list[tuple[int, int]]], list[str]]:
    """Participants (speakers, one mention per turn) then the payload's
    entities, as word spans; the names used in candidate texts. Index order is
    the payload's: a fact's head ``k`` is participant ``k`` below their count."""
    from .data import speaker_turns
    p = row["payload"]
    parts = list(p.get("participants") or row.get("participants") or [])
    spans = [tuple(x) for x in row["sentence_spans"]]
    turns = speaker_turns(spans, row.get("speakers"), parts) if parts else []
    ents, names = [], []
    for k, who in enumerate(parts):
        ents.append([(s, s + 1) for s, q in turns if q == k])
        names.append(who)
    for e in p["entities"]:
        ms = [tuple(x) for x in e["spans"]]
        ents.append(ms)
        names.append(" ".join(row["tokens"][ms[0][0]:ms[0][1]]) if ms else "")
    return ents, names


def _symmetric(rel: str) -> bool:
    return any(h in rel for h in SYMMETRIC_HINT)


def _record(row: dict, cands: list[dict], split: str, src: str, rng: random.Random,
            neg: bool = True) -> dict | None:
    """One text with its candidates (``h``, ``t``, ``rel``, ``y``); free
    negatives added when ``neg``: another relation of the schema, the pair
    reversed, an argument cut or widened by a word."""
    ents, names = _entities(row)
    if len(ents) < 2:
        return None
    toks = row["tokens"]
    schema = list(row["relations"])
    judged = {(c["h"], c["rel"], c["t"]) for c in cands}
    pos = {(c["h"], c["rel"], c["t"]) for c in cands if VERDICTS[c["y"]] in POSITIVE}
    out = [dict(c, src=src) for c in cands]
    if neg:
        for c in cands:
            if VERDICTS[c["y"]] not in POSITIVE:
                continue
            h, r, t = c["h"], c["rel"], c["t"]
            if rng.random() < 0.5:
                alts = [x for x in schema if x != r and (h, x, t) not in judged]
                if alts:
                    x = rng.choice(alts)
                    out.append({"h": h, "t": t, "rel": x, "y": 1, "src": "neg:relation"})
            if rng.random() < 0.3 and not _symmetric(r) and (t, r, h) not in pos:
                out.append({"h": t, "t": h, "rel": r, "y": 1, "src": "neg:reversed"})
            if rng.random() < 0.3:
                k = rng.choice([h, t])
                ms = ents[k]
                if ms and k >= len(ents) - len(row["payload"]["entities"]):
                    a, b = ms[0]
                    sent = next(((s, e) for s, e in row["sentence_spans"] if s <= a < e), (0, len(toks)))
                    opts = []
                    if b - a >= 2:
                        opts += [(a + 1, b), (a, b - 1)]
                    if b < sent[1] and toks[b] not in (".", ",", ";", ":"):
                        opts.append((a, b + 1))
                    if a > sent[0] and toks[a - 1] not in (".", ",", ";", ":"):
                        opts.append((a - 1, b))
                    if opts:
                        na, nb = rng.choice(opts)
                        # the cut or widened span stands in for argument k in this candidate only
                        out.append({"h": h, "t": t, "rel": r, "y": 1, "src": "neg:argument",
                                    "swap": [k, [na, nb]]})
    return {"id": row["id"], "split": split, "tokens": toks, "sentence_spans": row["sentence_spans"],
            "speakers": row.get("speakers"), "participants": list(row["payload"].get("participants") or []),
            "entities": [[list(m) for m in ms] for ms in ents], "names": names, "cands": out}


def _from_payload(row: dict) -> list[dict]:
    """A labelled row's facts as candidates, the verdict from polarity and
    modality (D3 forms, minimal pairs, D2 concepts)."""
    out = []
    for f in row["payload"]["facts"]:
        y = 2 if f.get("polarity") == "negative" else 3 if f.get("modality", "asserted") != "asserted" else 0
        out.append({"h": f["head"], "t": f["tail"], "rel": f["relation"], "y": y})
    return out


def build(seed: int = 13, d1e: Path | None = None) -> int:
    from ..config import RUNS_DIR
    from ..eval.negation_pilot import heldout_base, heldout_form
    rng = random.Random(seed)
    c = Counter()
    recs = []
    short = json.loads((_corpus() / "kg_short.json").read_text())
    labels = {r["id"]: r for r in json.loads(
        (RUNS_DIR / "kg-judge" / "labels-d1-full-ultrabert-380m-v7b8ev.json").read_text())}
    for row in short["train"]:
        lab = labels.get(row["id"])
        if not lab or not lab.get("judged"):
            continue
        cands = [{"h": f["head"], "t": f["tail"], "rel": f["relation"], "y": VERDICTS.index(f["verdict"])}
                 for f in lab["facts"] if f.get("verdict") in VERDICTS and f["head"] != f["tail"]]
        split = "held" if heldout_base(row["id"]) else "train"
        r = _record(row, cands, split, "d1", rng, neg=split == "train")
        if r:
            recs.append(r); c[f"d1 texts {split}"] += 1
    for row in short["pairs"]:
        split = "held" if heldout_base(row["id"]) else "train"
        r = _record(row, _from_payload(row), split, "pairs", rng, neg=False)
        if r:
            recs.append(r); c[f"pairs {split}"] += 1
    for row in json.loads((_corpus() / "kg_d3_forms.json").read_text())["train"]:
        split = "held" if heldout_form(row["id"]) else "train"
        r = _record(row, _from_payload(row), split, "forms", rng, neg=False)
        if r:
            recs.append(r); c[f"forms {split}"] += 1
    for row in json.loads((_corpus() / "kg_d2_concepts.json").read_text())["train"]:
        r = _record(row, _from_payload(row), "train", "concepts", rng, neg=True)
        if r:
            recs.append(r); c["concepts train"] += 1
    if d1e is not None and d1e.exists():
        n = _add_d1e(recs, d1e, rng)
        c["d1e texts"] += n
    for r in recs:
        for x in r["cands"]:
            c[f"cand {r['split']} {x['src']} {VERDICTS[x['y']]}"] += 1
    (_corpus() / DATA).write_text(json.dumps({"records": recs}))
    for k in sorted(c):
        print(f"  {k}: {c[k]:,}")
    print("wrote", _corpus() / DATA)
    return 0


def _add_d1e(recs: list[dict], labels_file: Path, rng: random.Random) -> int:
    """D1e (plan r4 5.2): judged end-to-end candidates, keyed by mention text
    (M1's format). Each argument is every occurrence of its mention texts."""
    from ..eval.m1_precision import OUT
    import pickle
    L = json.loads(labels_file.read_text())
    name = labels_file.name.split("-astra")[0].removeprefix("labels-")
    P = pickle.loads((OUT / f"preds-{name}.pkl").read_bytes())
    n = 0
    for it, t in zip(P["items"], L["texts"]):
        if not t["judged"] or not t["facts"]:
            continue
        low = [w.lower() for w in it["tokens"]]
        idx, ents, names, cands = {}, [], [], []

        def ent(ms):
            key = tuple(sorted(ms))
            if key not in idx:
                spans = []
                for m in ms:
                    mt = m.split()
                    spans += [(a, a + len(mt)) for a in range(len(low) - len(mt) + 1) if low[a:a + len(mt)] == mt]
                if not spans:
                    return None
                idx[key] = len(ents)
                ents.append(sorted(set(spans)))
                names.append(max(ms, key=len))
            return idx[key]
        for f in t["facts"]:
            if f["verdict"] not in VERDICTS or "teacher" in f["src"] and len(f["src"]) == 1:
                continue
            h, tl = ent(f["key"][0]), ent(f["key"][2])
            if h is None or tl is None or h == tl:
                continue
            cands.append({"h": h, "t": tl, "rel": f["key"][1], "y": VERDICTS.index(f["verdict"]), "src": "d1e"})
        if cands:
            recs.append({"id": it["id"], "split": "train", "tokens": it["tokens"],
                         "sentence_spans": it["sentence_spans"], "speakers": it.get("speakers"),
                         "participants": it.get("participants") or [],
                         "entities": [[list(m) for m in ms] for ms in ents], "names": names, "cands": cands})
            n += 1
    return n


# ── model ──────────────────────────────────────────────────────────
class Verifier(nn.Module):
    def __init__(self, base, r: int = 64, hidden: int = 512):
        super().__init__()
        from .model import RelationAdapter
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        self.ver = RelationAdapter(base.encoder, r)
        if getattr(base, "rel_lora", None) is not None:      # start as the relation pass
            st = {f"A.{k}": v for k, v in base.rel_lora.A.items()}
            st.update({f"B.{k}": v for k, v in base.rel_lora.B.items()})
            self.ver.load_grown(st, "")
        H = base.encoder.config.hidden_size
        self.head = nn.Sequential(nn.LayerNorm(4 * H), nn.Linear(4 * H, hidden), nn.GELU(),
                                  nn.Dropout(0.1), nn.Linear(hidden, len(VERDICTS)))

    def trainable(self):
        return list(self.ver.parameters()) + list(self.head.parameters())

    def forward(self, b: dict) -> torch.Tensor:
        base = self.base
        if getattr(base, "rel_lora", None) is not None:
            base.rel_lora.on = False
        self.ver.on = True
        try:
            seq, att = base._encode_with_attn(b["ids"], b["mask"])
        finally:
            self.ver.on = False
        base.relation.pair_features(seq, att, b)
        hs, ts, rs = base.relation._pair_in
        cm = b["cand_mask"].unsqueeze(-1).to(seq.dtype)
        cand = (seq * cm).sum(1) / cm.sum(1).clamp(min=1)
        x = torch.cat([hs.float(), ts.float(), rs.float(), cand.float()], -1)
        return self.head(x).float()


def _markers(rec: dict, tok, swap=None) -> dict:
    """The marker pass's input for a text; ``swap`` = ``[k, [a, b]]`` marks
    span ``a:b`` in place of entity ``k``'s mentions (a broken argument)."""
    from .data import marker_features, speaker_turns
    spans = [tuple(x) for x in rec["sentence_spans"]]
    parts = rec.get("participants") or []
    ents = [[tuple(m) for m in ms] for ms in rec["entities"]]
    if swap is not None:
        ents[swap[0]] = [tuple(swap[1])]
    n_part = len(parts)
    turns = speaker_turns(spans, rec.get("speakers"), parts) if parts else []
    return marker_features(rec["tokens"], spans, [np.asarray(ms) for ms in ents[n_part:]], tok,
                           turns=[(s, p) for s, p in turns], participants=parts)


def encode(rec: dict, c: dict, tok, bank: dict, para: dict | None, rng: random.Random | None,
           max_subwords: int = 1024, cache: dict | None = None) -> dict | None:
    """One (text, candidate) example: marker ids with the candidate after
    them, each mention's marker position, the candidate's token mask."""
    swap = c.get("swap")
    key = (id(rec), None if swap is None else (swap[0], tuple(swap[1])))
    rm = cache.get(key) if cache is not None else None
    if rm is None:
        rm = _markers(rec, tok, swap)
        if cache is not None:
            cache[key] = rm
    names = list(rec["names"])
    if swap is not None:
        names[swap[0]] = " ".join(rec["tokens"][swap[1][0]:swap[1][1]])
    men = rm["men_pos"]                       # participants' turns first, then entities
    if any(len(men[k]) == 0 or men[k].max() < 0 for k in (c["h"], c["t"])):
        return None
    men = [m if len(m) else np.asarray([-1], dtype=np.int64) for m in men]
    rel = bank.get(c["rel"], {"name": c["rel"], "description": ""})
    name, desc = rel["name"], rel.get("description", "")
    if para and rng is not None and rng.random() < 0.2 and rel.get("id") in para:
        pn, pd = rng.choice(para[rel["id"]])
        name, desc = pn, pd or desc
    # names first, so the 96-token cap only ever cuts the description
    text = f"{names[c['h']]} | {name} | {names[c['t']]} | {desc}"
    cand = tok(text, add_special_tokens=False)["input_ids"][:96]
    ids = [int(x) for x in rm["ids"]] + cand + [tok.sep_token_id]
    if len(ids) > max_subwords:
        return None
    return {"ids": ids, "n_text": len(rm["ids"]), "men": men, "h": c["h"], "t": c["t"], "y": c["y"]}


def collate(exs: list[dict], pad: int, device) -> dict:
    T = max(len(e["ids"]) for e in exs)
    ids = torch.full((len(exs), T), pad, dtype=torch.long)
    mask = torch.zeros(len(exs), T, dtype=torch.long)
    cmask = torch.zeros(len(exs), T, dtype=torch.bool)
    ent_batch, men_ent, men_pos, ph, pt = [], [], [], [], []
    for b, e in enumerate(exs):
        n = len(e["ids"])
        ids[b, :n] = torch.tensor(e["ids"]); mask[b, :n] = 1
        cmask[b, e["n_text"]:n - 1] = True
        base = len(ent_batch)
        for k, pos in enumerate(e["men"]):
            ent_batch.append(b)
            for p in pos.tolist():
                men_ent.append(base + k); men_pos.append(p)
        ph.append(base + e["h"]); pt.append(base + e["t"])
    out = {"ids": ids, "mask": mask, "cand_mask": cmask,
           "ent_batch": torch.tensor(ent_batch), "men_ent": torch.tensor(men_ent),
           "rm_men_pos": torch.tensor(men_pos), "pair_h": torch.tensor(ph), "pair_t": torch.tensor(pt),
           "pair_batch": torch.arange(len(exs)), "y": torch.tensor([e["y"] for e in exs])}
    return {k: v.to(device) for k, v in out.items()}


def examples(recs, tok, bank, para, rng, split: str, limit: int | None = None) -> list[dict]:
    out = []
    for r in recs:
        if r["split"] != split:
            continue
        cache: dict = {}
        for c in r["cands"]:
            e = encode(r, c, tok, bank, para if split == "train" else None, rng, cache=cache)
            if e is not None:
                e["src"] = c.get("src", "")
                out.append(e)
        if limit and len(out) >= limit:
            break
    return out


@torch.no_grad()
def evaluate(model, exs, pad, device, bs: int = 32) -> dict:
    model.eval()
    P, Y, S = [], [], []
    for i in range(0, len(exs), bs):
        chunk = exs[i:i + bs]
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(collate(chunk, pad, device))
        P.append(torch.softmax(lg, -1).cpu().numpy()); Y += [e["y"] for e in chunk]; S += [e["src"] for e in chunk]
    model.train()
    P, Y = np.concatenate(P), np.array(Y)
    expressed = 1 - P[:, 1]
    yb = Y != 1
    o = expressed.argsort().argsort() + 1
    n1, n0 = yb.sum(), (~yb).sum()
    auc = (o[yb].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    pred = P.argmax(1)
    res = {"n": len(Y), "acc": float((pred == Y).mean()), "auc_expressed": float(auc)}
    for k, v in enumerate(VERDICTS):
        sel = Y == k
        if sel.any():
            res[f"recall_{v}"] = float((pred[sel] == k).mean())
    for s in sorted(set(S)):
        sel = np.array([x == s for x in S])
        res[f"acc_{s}"] = float((pred[sel] == Y[sel]).mean())
    return res


def train(args) -> int:
    from .gold_eval import load_run
    from ..label_bank import load as load_bank
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    base, tok, spec, vocab, _ = load_run(args.run, args.repo, device, args.local)
    base.eval()
    bank_l = load_bank()
    bank = {r["name"]: r for r in bank_l["relations"]}
    para_raw = json.loads((_corpus() / "label_para.json").read_text()) if (_corpus() / "label_para.json").exists() else {}
    para = {int(k): v for k, v in para_raw.items()}
    recs = json.loads((_corpus() / DATA).read_text())["records"]
    rng = random.Random(args.seed)
    tr = examples(recs, tok, bank, para, rng, "train", args.limit)
    held = examples(recs, tok, bank, None, rng, "held", args.dev_limit)
    print(f"{len(tr):,} training examples, {len(held):,} held out; {Counter(e['y'] for e in tr)}", flush=True)
    model = Verifier(base, r=args.rank).to(device)
    opt = torch.optim.AdamW([{"params": list(model.ver.parameters()), "lr": args.lr},
                             {"params": list(model.head.parameters()), "lr": args.head_lr}], weight_decay=0.01)
    steps = args.steps or (len(tr) * args.epochs) // args.batch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=steps,
                                                pct_start=0.05, anneal_strategy="linear")
    w = torch.tensor([1.0, 1.0, args.neg_weight, args.hedge_weight], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w)
    out = Path(args.out_dir or f"data/v6-eval/checkpoints/{args.out_run}")
    out.mkdir(parents=True, exist_ok=True)
    best, t0, order = -1.0, time.time(), []
    model.train()
    for step in range(1, steps + 1):
        if not order:
            order = list(range(len(tr))); rng.shuffle(order)
        idx = [order.pop() for _ in range(min(args.batch, len(order)))]
        b = collate([tr[i] for i in idx], tok.pad_token_id, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(b)
        loss = loss_fn(lg, b["y"])
        if not torch.isfinite(loss):
            print(f"step {step}: non-finite loss, skipped", flush=True)
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.trainable(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} [{time.time() - t0:.0f}s]", flush=True)
        if step % args.eval_every == 0 or step == steps:
            res = evaluate(model, held, tok.pad_token_id, device)
            print(f"eval {step}: {json.dumps({k: round(v, 4) for k, v in res.items()})}", flush=True)
            if res["auc_expressed"] > best:
                best = res["auc_expressed"]
                torch.save({"ver": model.ver.state_dict(), "head": model.head.state_dict(),
                            "base_run": args.run, "rank": args.rank, "step": step, "eval": res,
                            "data": DATA}, out / "verifier.pt")
                (out / "best.metric").write_text(json.dumps({"value": best, "step": step,
                                                             "metric": "held-out AUC, expressed"}))
            (out / "status.json").write_text(json.dumps({"state": "running", "step": step, "of": steps,
                                                         "best": best, "time": time.time()}))
    (out / "status.json").write_text(json.dumps({"state": "done", "step": steps, "best": best,
                                                 "time": time.time()}))
    if args.upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=args.repo, folder_path=str(out), path_in_repo=args.out_run,
                              allow_patterns=["verifier.pt", "best.metric", "status.json", "train.log"],
                              commit_message=f"Add {args.out_run} (fact verifier)")
        print(f"uploaded to {args.repo}/{args.out_run}", flush=True)
    print(f"done: best held-out AUC {best:.4f}")
    return 0


def load_verifier(run: str, device, repo: str | None = None, local: Path | None = None):
    from .gold_eval import CKPT_DIR, HF_REPO, load_run
    path = (local or CKPT_DIR) / run / "verifier.pt"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo or HF_REPO, f"{run}/verifier.pt", local_dir=str(CKPT_DIR)))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    base, tok, spec, vocab, _ = load_run(ck["base_run"], repo or HF_REPO, device)
    base.eval()
    m = Verifier(base, r=ck["rank"]).to(device)
    m.ver.load_state_dict(ck["ver"]); m.head.load_state_dict(ck["head"])
    m.eval()
    return m, tok


@torch.no_grad()
def score_m1(args) -> int:
    """Re-rank one config's judged M1 predictions by the verifier's
    probability that the fact is expressed, and read precision at v7.1's
    recall against the margin (``m1_precision equal`` on a copy of the
    predictions file whose margins are the verifier's)."""
    import pickle
    from ..eval.m1_precision import OUT
    from ..label_bank import load as load_bank
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok = load_verifier(args.run, device)
    bank = {r["name"]: r for r in load_bank()["relations"]}
    P = pickle.loads((OUT / f"preds-{args.name}.pkl").read_bytes())
    new = {}
    for i, it in enumerate(P["items"]):
        keys = list(P["margins"][args.config][i])
        if not keys:
            new[i] = {}
            continue
        low = [w.lower() for w in it["tokens"]]
        idx, ents, names = {}, [], []

        def ent(ms):
            key = tuple(sorted(ms))
            if key not in idx:
                spans = []
                for m in ms:
                    mt = m.split()
                    spans += [(a, a + len(mt)) for a in range(len(low) - len(mt) + 1) if low[a:a + len(mt)] == mt]
                idx[key] = len(ents) if spans else -1
                if spans:
                    ents.append(sorted(set(spans))); names.append(max(ms, key=len))
            return idx[key]
        cands = []
        for k in keys:
            h, t = ent(k[0]), ent(k[2])
            cands.append((k, h, t))
        rec = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"], "speakers": it.get("speakers"),
               "participants": it.get("participants") or [], "entities": [[list(m) for m in ms] for ms in ents],
               "names": names}
        exs, ks = [], []
        for k, h, t in cands:
            if h < 0 or t < 0 or h == t:
                continue
            e = encode(rec, {"h": h, "t": t, "rel": k[1], "y": 0}, tok, bank, None, None)
            if e is not None:
                exs.append(e); ks.append(k)
        out = {k: -1e3 for k in keys}           # facts the verifier cannot read go last
        for j in range(0, len(exs), 32):
            lg = model(collate(exs[j:j + 32], tok.pad_token_id, device))
            pr = 1 - torch.softmax(lg, -1)[:, 1]
            for k, p in zip(ks[j:j + 32], pr.tolist()):
                out[k] = float(np.log(p + 1e-6) - np.log(1 - p + 1e-6))
        new[i] = out
        if i % 100 == 0:
            print(f"  {i}/{len(P['items'])}", flush=True)
    P2 = {**P, "margins": {**P["margins"], f"{args.config}+v": [new[i] for i in range(len(P['items']))]},
          "preds": {**P["preds"], f"{args.config}+v": P["preds"][args.config]}}
    name2 = f"{args.name}-v-{args.run.split('-')[-1]}"
    (OUT / f"preds-{name2}.pkl").write_bytes(pickle.dumps(P2))
    # the labels are the same facts: copy, adding the new config as a source of each fact
    from ..eval.m1_precision import labels_path
    L = json.loads(labels_path(args.name, "astra", "c3").read_text())
    for t in L["texts"]:
        for f in t["facts"]:
            if args.config in f["src"]:
                f["src"].append(f"{args.config}+v")
    L["configs"] = L["configs"] + [f"{args.config}+v"]
    labels_path(name2, "astra", "c3").write_text(json.dumps(L))
    print(f"wrote preds-{name2}.pkl; read it with: m1_precision equal --run {name2} --version c3")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["build", "train", "score-m1"])
    ap.add_argument("--run", default="ultrabert-380m-v8e-w1half")
    ap.add_argument("--out-run", default="ultrabert-380m-v8v1")
    ap.add_argument("--out-dir")
    ap.add_argument("--repo", default="dragonscale-ai/kniv-v6-runs")
    ap.add_argument("--local", type=Path)
    ap.add_argument("--device")
    ap.add_argument("--d1e", type=Path, help="build: D1e's labels file (M1 format)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--limit", type=int, help="train: first N training examples (pipeline test)")
    ap.add_argument("--dev-limit", type=int, default=4000)
    ap.add_argument("--neg-weight", type=float, default=2.0)
    ap.add_argument("--hedge-weight", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--name", default="m1p3b-n500", help="score-m1: the M1 predictions file")
    ap.add_argument("--config", default="e1b", help="score-m1: the config whose predictions to re-rank")
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a.seed, a.d1e)
    if a.cmd == "train":
        return train(a)
    return score_m1(a)


if __name__ == "__main__":
    raise SystemExit(main())
