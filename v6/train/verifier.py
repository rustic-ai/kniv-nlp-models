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
# V1c (plan r4 review step 3): hedged split into its kinds (schemas.MODALITIES
# after "asserted", same order); label HEDGED_ANY = hedged, kind unknown (the
# loss is on the kinds' summed probability).
VERDICTS_K = ["supported", "not_supported", "negated", "hypothetical", "planned", "desired", "questioned", "reported"]
HEDGED_ANY = -1
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


def build(seed: int = 13, d1e: list[Path] | None = None) -> int:
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
    for f in d1e or []:
        n = _add_d1e(recs, f, rng)
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


# ── V1c: both judges, hedge kinds, long windows ───────────────────
DATA_K = "verifier_v1c.json"


def _held_long(text_id: str) -> bool:
    """A fifth of the stage-4 pilot's texts, by hash: the long-text reading."""
    import hashlib
    return int(hashlib.sha1(text_id.encode()).hexdigest(), 16) % 5 == 0


def _second(name: str) -> dict:
    """DeepSeek c3k labels (``v6.eval.s6_second``) by text id: per fact
    (GPT-6 verdict, DeepSeek verdict, kind), in the GPT-6 file's order."""
    from ..eval.s6_second import _path
    p = _path(name)
    if not p.exists():
        return {}
    return {t["id"]: [(f["gpt6"], f["verdict"], f["kind"]) for f in t["facts"]]
            for t in json.loads(p.read_text()) if t["judged"]}


def _k_label(gpt6: str, second: tuple | None, c: Counter) -> int | None:
    """The V1c label of one judged fact: None (dropped) when the judges
    disagree on the verdict (hedged against hedged agrees whatever the kinds);
    DeepSeek's kind for a hedged fact; HEDGED_ANY when the kind is unknown."""
    if gpt6 not in VERDICTS:
        return None
    if second is None or second[1] not in VERDICTS:
        c["single judge"] += 1
        return HEDGED_ANY if gpt6 == "hedged" else VERDICTS_K.index(gpt6)
    if second[1] != gpt6:
        c["dropped: judges disagree"] += 1
        return None
    c["both judges agree"] += 1
    if gpt6 == "hedged":
        return VERDICTS_K.index(second[2]) if second[2] in VERDICTS_K else HEDGED_ANY
    return VERDICTS_K.index(gpt6)


def _mention_entities(tokens: list[str]):
    """Entities found by mention text (every occurrence), as in ``_add_d1e``."""
    low = [w.lower() for w in tokens]
    idx, ents, names = {}, [], []

    def ent(ms):
        key = tuple(sorted(ms))
        if key not in idx:
            spans = []
            for m in ms:
                mt = m.lower().split()
                spans += [(a, a + len(mt)) for a in range(len(low) - len(mt) + 1) if low[a:a + len(mt)] == mt]
            if not spans:
                return None
            idx[key] = len(ents)
            ents.append(sorted(set(spans)))
            names.append(max(ms, key=len))
        return idx[key]
    return ent, ents, names


def build_k(seed: int = 13, news: bool = False, out: str = DATA_K) -> int:
    """V1b's set rebuilt for V1c: hedge kinds (from the payload's modality
    for generated rows, from DeepSeek where it judged), the agreement filter
    wherever both judges labelled a text (D1's sample, D1e, the stage-4
    pilot), and the pilot's long windows (a fifth held out by text)."""
    from ..config import RUNS_DIR
    from ..eval.m1_precision import OUT as M1, labels_path
    from ..eval.negation_pilot import heldout_base, heldout_form
    from ..schemas import MODALITIES
    rng = random.Random(seed)
    c = Counter()
    recs = []
    sec = {k: _second(k) for k in ("d1", "d1e", "s4", "news-train")}
    short = json.loads((_corpus() / "kg_short.json").read_text())
    labels = {r["id"]: r for r in json.loads(
        (RUNS_DIR / "kg-judge" / "labels-d1-full-ultrabert-380m-v7b8ev.json").read_text())}
    for row in short["train"]:
        lab = labels.get(row["id"])
        if not lab or not lab.get("judged"):
            continue
        two = sec["d1"].get(row["id"])
        cands = []
        for j, f in enumerate(lab["facts"]):
            if f["head"] == f["tail"]:
                continue
            y = _k_label(f.get("verdict"), two[j] if two else None, c)
            if y is not None:
                cands.append({"h": f["head"], "t": f["tail"], "rel": f["relation"], "y": y})
        split = "held" if heldout_base(row["id"]) else "train"
        r = _record_k(row, cands, split, "d1", rng, neg=split == "train")
        if r:
            recs.append(r); c[f"d1 texts {split}"] += 1

    def payload(row):
        out = []
        for f in row["payload"]["facts"]:
            mod = f.get("modality", "asserted")
            y = (2 if f.get("polarity") == "negative" else 0 if mod == "asserted" else
                 VERDICTS_K.index(mod) if mod in VERDICTS_K else HEDGED_ANY)
            out.append({"h": f["head"], "t": f["tail"], "rel": f["relation"], "y": y})
        return out
    for row in short["pairs"]:
        split = "held" if heldout_base(row["id"]) else "train"
        r = _record_k(row, payload(row), split, "pairs", rng, neg=False)
        if r:
            recs.append(r); c[f"pairs {split}"] += 1
    for row in json.loads((_corpus() / "kg_d3_forms.json").read_text())["train"]:
        split = "held" if heldout_form(row["id"]) else "train"
        r = _record_k(row, payload(row), split, "forms", rng, neg=False)
        if r:
            recs.append(r); c[f"forms {split}"] += 1
    for row in json.loads((_corpus() / "kg_d2_concepts.json").read_text())["train"]:
        r = _record_k(row, payload(row), "train", "concepts", rng, neg=True)
        if r:
            recs.append(r); c["concepts train"] += 1
    import pickle
    from ..eval.s6_second import D1E
    for run in D1E:
        L = json.loads(labels_path(run, "astra", "c3").read_text())
        P = pickle.loads((M1 / f"preds-{run}.pkl").read_bytes())
        for it, t in zip(P["items"], L["texts"]):
            if not t["judged"] or not t["facts"]:
                continue
            two = sec["d1e"].get(it["id"])
            ent, ents, names = _mention_entities(it["tokens"])
            cands = []
            for j, f in enumerate(t["facts"]):
                if "teacher" in f["src"] and len(f["src"]) == 1:
                    continue
                y = _k_label(f["verdict"], two[j] if two else None, c)
                h, tl = ent(f["key"][0]), ent(f["key"][2])
                if y is None or h is None or tl is None or h == tl:
                    continue
                cands.append({"h": h, "t": tl, "rel": f["key"][1], "y": y, "src": "d1e"})
            if cands:
                recs.append({"id": it["id"], "split": "train", "tokens": it["tokens"],
                             "sentence_spans": it["sentence_spans"], "speakers": it.get("speakers"),
                             "participants": it.get("participants") or [],
                             "entities": [[list(m) for m in ms] for ms in ents], "names": names, "cands": cands})
                c["d1e texts"] += 1
    # plan r4 readiness C: the judged news training texts (the held-out news set stays out), as D1e
    for run in (["ultrabert-380m-v8p3-d123-news-train"] if news else []):
        L = json.loads(labels_path(run, "astra", "c3").read_text())
        P = pickle.loads((M1 / f"preds-{run}.pkl").read_bytes())
        for it, t in zip(P["items"], L["texts"]):
            if not t["judged"] or not t["facts"]:
                continue
            two = sec["news-train"].get(it["id"])
            ent, ents, names = _mention_entities(it["tokens"])
            cands = []
            for j, f in enumerate(t["facts"]):
                if "teacher" in f["src"] and len(f["src"]) == 1:
                    continue
                y = _k_label(f["verdict"], two[j] if two else None, c)
                h, tl = ent(f["key"][0]), ent(f["key"][2])
                if y is None or h is None or tl is None or h == tl:
                    continue
                cands.append({"h": h, "t": tl, "rel": f["key"][1], "y": y, "src": "news"})
            if cands:
                recs.append({"id": it["id"], "split": "train", "tokens": it["tokens"],
                             "sentence_spans": it["sentence_spans"], "speakers": it.get("speakers"),
                             "participants": it.get("participants") or [],
                             "entities": [[list(m) for m in ms] for ms in ents], "names": names, "cands": cands})
                c["news texts"] += 1
    S = RUNS_DIR / "s4-pilot"
    items = {it["id"]: it for it in json.loads((S / "items.json").read_text())}
    for t in json.loads((S / "labels-astra-c3.json").read_text()):
        if not t["judged"]:
            continue
        it = items[t["id"]]
        two = sec["s4"].get(t["id"])
        ent, ents, names = _mention_entities(it["tokens"])
        cands = []
        for j, x in enumerate(t["cands"]):
            y = _k_label(x["judge"], two[j] if two else None, c)
            h, tl = ent(x["head_m"]), ent(x["tail_m"])
            if y is None or h is None or tl is None or h == tl:
                continue
            cands.append({"h": h, "t": tl, "rel": x["relation"], "y": y, "src": "s4"})
        if cands:
            split = "held_long" if _held_long(t["id"]) else "train"
            recs.append({"id": t["id"], "split": split, "tokens": it["tokens"],
                         "sentence_spans": it["sentence_spans"], "speakers": it.get("speakers"),
                         "participants": it.get("participants") or [],
                         "entities": [[list(m) for m in ms] for ms in ents], "names": names, "cands": cands})
            c[f"s4 texts {split}"] += 1
    for r in recs:
        for x in r["cands"]:
            c[f"cand {r['split']} {x['src']} {'hedged?' if x['y'] == HEDGED_ANY else VERDICTS_K[x['y']]}"] += 1
    (_corpus() / out).write_text(json.dumps({"records": recs, "verdicts": VERDICTS_K}))
    for k in sorted(c):
        print(f"  {k}: {c[k]:,}")
    print("wrote", _corpus() / out)
    return 0


def _record_k(row, cands, split, src, rng, neg=True):
    """``_record`` with labels in VERDICTS_K (free negatives: not supported)."""
    expressed = [dict(c, y=0 if c["y"] != 1 else 1) for c in cands]       # _record reads y for POSITIVE only
    r = _record(row, expressed, split, src, rng, neg=neg)
    if r is None:
        return None
    for x, c in zip(r["cands"], cands):                                      # the judged ones come first, in order
        x["y"] = c["y"]
    return r


# ── model ──────────────────────────────────────────────────────────
class Verifier(nn.Module):
    def __init__(self, base, r: int = 64, hidden: int = 512, n_out: int = len(VERDICTS)):
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
                                  nn.Dropout(0.1), nn.Linear(hidden, n_out))

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


def coarse(P: np.ndarray) -> np.ndarray:
    """Verdict probabilities in VERDICTS order (the kinds summed into hedged)."""
    return P if P.shape[1] == len(VERDICTS) else np.concatenate([P[:, :3], P[:, 3:].sum(1, keepdims=True)], 1)


def k_loss(lg: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Weighted NLL over VERDICTS_K; HEDGED_ANY rows on log sum of the kinds."""
    lp = torch.log_softmax(lg.float(), -1)
    known = y >= 0
    yk = y.clamp(min=0)
    nll = -lp.gather(1, yk.unsqueeze(1)).squeeze(1)
    any_h = -torch.logsumexp(lp[:, 3:], -1)
    loss = torch.where(known, nll, any_h)
    wt = torch.where(known, w[yk], w[3])
    return (loss * wt).sum() / wt.sum()


@torch.no_grad()
def evaluate(model, exs, pad, device, bs: int = 32) -> dict:
    model.eval()
    P, Y, S = [], [], []
    i = 0
    while i < len(exs):
        k = 1                               # as many as fit bs x 384 squared (long windows: a few)
        while i + k < len(exs) and k < bs and (k + 1) * max(len(e["ids"]) for e in exs[i:i + k + 1]) ** 2 <= bs * 384 ** 2:
            k += 1
        chunk = exs[i:i + k]
        i += k
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            lg = model(collate(chunk, pad, device))
        P.append(torch.softmax(lg, -1).cpu().numpy()); Y += [e["y"] for e in chunk]; S += [e["src"] for e in chunk]
    model.train()
    P, Y = np.concatenate(P), np.array(Y)
    res_k = {}
    if P.shape[1] == len(VERDICTS_K):        # kinds: their accuracy on rows with a known kind, then coarse
        kk = Y >= 3
        if kk.any():
            res_k["acc_kind"] = float((P[kk, 3:].argmax(1) + 3 == Y[kk]).mean())
            res_k["n_kind"] = int(kk.sum())
        P = coarse(P)
        Y = np.where((Y >= 3) | (Y == HEDGED_ANY), 3, Y)
    expressed = 1 - P[:, 1]
    yb = Y != 1
    o = expressed.argsort().argsort() + 1
    n1, n0 = yb.sum(), (~yb).sum()
    auc = (o[yb].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1)
    pred = P.argmax(1)
    res = {"n": len(Y), "acc": float((pred == Y).mean()), "auc_expressed": float(auc), **res_k}
    for k, v in enumerate(VERDICTS):
        sel = Y == k
        if sel.any():
            res[f"recall_{v}"] = float((pred[sel] == k).mean())
    for s in sorted(set(S)):
        sel = np.array([x == s for x in S])
        res[f"acc_{s}"] = float((pred[sel] == Y[sel]).mean())
    return res


def next_batch(tr: list, order: list, longq: list, batch: int, max_tokens: int) -> list:
    """Up to ``batch`` examples within a memory budget of ``batch`` x
    ``max_tokens``² (every layer's attention map is kept for backward, so
    memory grows with the square of the length). Examples
    too long for a full batch wait in ``longq`` (mutated) and go out in a
    batch of their own when one more would pass the budget, or when
    ``order`` runs out; short ones drawn meanwhile go back to ``order``."""
    idx = []
    while order and len(idx) < batch:
        j = order.pop()
        n = len(tr[j]["ids"])
        if n <= max_tokens:
            idx.append(j)
            continue
        if longq and (len(longq) + 1) * max(n, *(len(tr[x]["ids"]) for x in longq)) ** 2 > batch * max_tokens ** 2:
            out = list(longq)
            longq[:] = [j]
            order.extend(idx)
            return out
        longq.append(j)
    if not order and longq:
        order.extend(idx)
        out = list(longq)
        longq[:] = []
        return out
    return idx


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
    D = json.loads((_corpus() / args.data).read_text())
    recs, verdicts = D["records"], D.get("verdicts", VERDICTS)
    rng = random.Random(args.seed)
    tr = examples(recs, tok, bank, para, rng, "train", args.limit)
    held = examples(recs, tok, bank, None, rng, "held", args.dev_limit)
    held_long = examples(recs, tok, bank, None, rng, "held_long", args.dev_limit)
    print(f"{len(tr):,} training examples, {len(held):,} held out, {len(held_long):,} long held out; "
          f"{Counter(e['y'] for e in tr)}", flush=True)
    model = Verifier(base, r=args.rank, n_out=len(verdicts)).to(device)
    opt = torch.optim.AdamW([{"params": list(model.ver.parameters()), "lr": args.lr},
                             {"params": list(model.head.parameters()), "lr": args.head_lr}], weight_decay=0.01)
    steps = int(args.steps or (len(tr) * args.epochs) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=steps,
                                                pct_start=0.05, anneal_strategy="linear")
    w = torch.tensor([1.0, 1.0, args.neg_weight] + [args.hedge_weight] * (len(verdicts) - 3), device=device)
    loss_fn = nn.CrossEntropyLoss(weight=w) if len(verdicts) == len(VERDICTS) else (lambda lg, y: k_loss(lg, y, w))
    out = Path(args.out_dir or f"data/v6-eval/checkpoints/{args.out_run}")
    out.mkdir(parents=True, exist_ok=True)
    best, t0, order, longq = -1.0, time.time(), [], []
    model.train()
    for step in range(1, steps + 1):
        if not order:
            order = list(range(len(tr))); rng.shuffle(order)
        idx = next_batch(tr, order, longq, args.batch, args.max_tokens)
        if not idx:
            order = list(range(len(tr))); rng.shuffle(order)
            idx = next_batch(tr, order, longq, args.batch, args.max_tokens)
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
            if held_long:
                res.update({f"long_{k}": v for k, v in evaluate(model, held_long, tok.pad_token_id, device).items()
                            if k in ("n", "auc_expressed", "acc", "acc_kind")})
            print(f"eval {step}: {json.dumps({k: round(v, 4) for k, v in res.items()})}", flush=True)
            if res["auc_expressed"] > best:
                best = res["auc_expressed"]
                torch.save({"ver": model.ver.state_dict(), "head": model.head.state_dict(),
                            "base_run": args.run, "rank": args.rank, "step": step, "eval": res,
                            "data": args.data, "verdicts": verdicts}, out / "verifier.pt")
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


def trained_relations(data: str = DATA) -> set[str]:
    """Relation names in the verifier set's training split (cached beside it)."""
    cache = _corpus() / data.replace(".json", "-relations.json")
    if cache.exists():
        return set(json.loads(cache.read_text()))
    recs = json.loads((_corpus() / data).read_text())["records"]
    names = sorted({c["rel"] for r in recs if r["split"] == "train" for c in r["cands"]})
    cache.write_text(json.dumps(names))
    return set(names)


def load_verifier(run: str, device, repo: str | None = None, local: Path | None = None):
    from .gold_eval import CKPT_DIR, HF_REPO, load_run
    path = (local or CKPT_DIR) / run / "verifier.pt"
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(repo or HF_REPO, f"{run}/verifier.pt", local_dir=str(CKPT_DIR)))
    ck = torch.load(path, map_location="cpu", weights_only=False)
    base, tok, spec, vocab, _ = load_run(ck["base_run"], repo or HF_REPO, device)
    base.eval()
    m = Verifier(base, r=ck["rank"], n_out=len(ck.get("verdicts", VERDICTS))).to(device)
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
    from .gold_eval import CKPT_DIR, HF_REPO, load_run
    vp = CKPT_DIR / args.run / "verifier.pt"
    if not vp.exists():
        from huggingface_hub import hf_hub_download
        vp = Path(hf_hub_download(HF_REPO, f"{args.run}/verifier.pt", local_dir=str(CKPT_DIR)))
    v3 = torch.load(vp, map_location="cpu", weights_only=False).get("form") == "v3"
    if v3:                                  # the list verifier: each text's candidates in shared passes, fixed order
        from . import verifier3 as V3
        base3, tok, _, _, _ = load_run(V3.RUN, HF_REPO, device)
        base3.eval()
        model = V3.load_v3(vp, base3, device)
    else:
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
        out = {k: -1e3 for k in keys}           # facts the verifier cannot read go last
        if v3:
            ok = [(k, h, t) for k, h, t in cands if h >= 0 and t >= 0 and h != t]
            prs = V3.score_window(model, rec, [(h, t, k[1]) for k, h, t in ok], tok, bank, device)
            for (k, _, _), pr in zip(ok, prs):
                if pr is not None:
                    p = 1 - float(coarse(np.array([pr]))[0, 1])
                    out[k] = float(np.log(p + 1e-6) - np.log(1 - p + 1e-6))
            new[i] = out
            if i % 100 == 0:
                print(f"  {i}/{len(P['items'])}", flush=True)
            continue
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
    name2 = f"{args.name}-v-{args.tag or args.run.split('-')[-1]}-{args.config}"
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


@torch.no_grad()
def score_m3(args) -> int:
    """The verifier's verdict on each expected fact of the M3 hand set, with
    gold entities and the fact given (the relation step's recall is read
    apart): right = supported for asserted, negated for negated, hedged for a
    hedged statement (supported also right where the set accepts asserted)."""
    from ..eval import m3_negation as m3
    from ..label_bank import load as load_bank
    device = torch.device(args.device or "cpu")
    model, tok = load_verifier(args.run, device)
    bank = {r["name"]: r for r in load_bank()["relations"]}
    rows = []
    for it in m3.load_hand():
        parts = list(dict.fromkeys(s for s in (it.get("sentence_speakers") or []) if s))
        ents = [[] for _ in parts] + [[tuple(x) for x in e["spans"]] for e in it["entities"]]
        names = parts + [e["name"] for e in it["entities"]]
        rec = {"tokens": it["tokens"], "sentence_spans": it["sentence_spans"],
               "speakers": it.get("sentence_speakers"), "participants": parts,
               "entities": [[list(m) for m in ms] for ms in ents], "names": names}

        def ix(a):
            return parts.index(a[1:]) if isinstance(a, str) else len(parts) + a
        for f in it["facts"]:
            if f.get("optional"):
                continue
            e = encode(rec, {"h": ix(f["head"]), "t": ix(f["tail"]), "rel": f["relation"], "y": 0},
                       tok, bank, None, None)
            if e is None:
                continue
            p = torch.softmax(model(collate([e], tok.pad_token_id, device)), -1)[0].tolist()
            mods = set(f["modality"])
            if f["polarity"] == "negative":
                ok = {2}; fam = "negated"
            elif "asserted" in mods and len(mods) == 1:
                ok = {0}; fam = "asserted"
            else:
                ok = {3} | ({0} if "asserted" in mods else set()); fam = "hedged"
            rows.append((fam, int(np.argmax(p)) in ok, p, it["id"]))
    for fam in ("negated", "hedged", "asserted"):
        r = [x for x in rows if x[0] == fam]
        v = Counter(VERDICTS[int(np.argmax(x[2][:4]))] for x in r)
        print(f"{fam:9s} ({len(r)}): verdict right {np.mean([x[1] for x in r]):.3f}; verdicts {dict(v)}")
    return 0


@torch.no_grad()
def score_m2(args) -> int:
    """M2 (never-trained relations): the verifier over a run's top 4
    relations per pair (by its margin), the pair's score the best verdict
    logit; AUC positives against negatives, top-1 and F1 at p > 0.5 against
    the run's own margins on the same items."""
    import pickle
    from ..eval import m2_unseen as m2
    from ..eval.heldout_zs import _lit
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else
                                          "mps" if torch.backends.mps.is_available() else "cpu"))
    model, tok = load_verifier(args.run, device)
    data = json.loads(Path(args.m2_set).read_text())
    rels = data["relations"]
    bank = {r["code"]: {"name": r["name"], "description": r.get("description", "")} for r in rels}
    names = [r["code"] for r in rels]
    preds = m2.load_preds(f"pt:{args.base}", args.m2_set)
    tb = m2.item_table(data, preds)
    wins = {w["id"]: w for w in data["windows"]}
    vbest, vtop = np.full(len(data["items"]), -1e3), np.full(len(data["items"]), -1)
    outdir = Path(args.out_dir or "v6/runs/m2-unseen")
    si, sn = map(int, args.slice.split("/")) if args.slice else (0, 1)
    if args.combine:            # merge the slices' files, then score
        for f in sorted(outdir.glob(f"verifier-{args.run}-*of{sn}.pkl")):
            x = pickle.loads(f.read_bytes())
            sel = x["done"]
            vbest[sel], vtop[sel] = x["vbest"][sel], x["vtop"][sel]
    t0 = time.time()
    done = np.zeros(len(data["items"]), dtype=bool)
    for n, (it, M) in enumerate(zip(data["items"], tb["M"])):
        if args.combine or (args.limit and n >= args.limit):
            break
        if n % sn != si:        # MPS memory grows over a long run: slice in fresh processes
            continue
        done[n] = True
        if (M <= -1e2).all():
            continue
        w = wins[it["window"]]
        ents = [[tuple(m) for m in e["mentions"]] for e in _lit(w["entities"])]
        toks = _lit(w["tokens"])
        rec = {"tokens": toks, "sentence_spans": _lit(w["sentence_spans"]), "speakers": None,
               "participants": [], "entities": [[list(m) for m in ms] for ms in ents],
               "names": [" ".join(toks[ms[0][0]:ms[0][1]]) if ms else "" for ms in ents]}
        top = np.argsort(-M)[:4]
        exs, ks = [], []
        cache = {}
        for k in top:
            e = encode(rec, {"h": it["head"], "t": it["tail"], "rel": names[k], "y": 0}, tok, bank, None,
                       None, cache=cache)
            if e is not None:
                exs.append(e); ks.append(k)
        if not exs:
            continue
        lg = model(collate(exs, tok.pad_token_id, device))
        pr = 1 - torch.softmax(lg, -1)[:, 1]
        logit = torch.log(pr + 1e-6) - torch.log(1 - pr + 1e-6)
        j = int(logit.argmax())
        vbest[n], vtop[n] = float(logit[j]), ks[j]
        if (n + 1) % 500 == 0:
            print(f"  {n + 1}/{len(data['items'])} [{time.time() - t0:.0f}s]", flush=True)
    if args.slice and not args.combine:
        f = outdir / f"verifier-{args.run}-{si}of{sn}.pkl"
        f.write_bytes(pickle.dumps({"vbest": vbest, "vtop": vtop, "done": done}))
        print(f"wrote {f}")
        return 0
    if args.limit:          # pipeline test: the first items only
        keep = np.arange(min(args.limit, len(tb["pos"])))
        tb = {k: v[keep] for k, v in tb.items()}
        vbest, vtop = vbest[keep], vtop[keep]
    pos, gold = tb["pos"], tb["gold"]

    def auc(sc):
        sp, sn = sc[pos], sc[~pos]
        allv = np.concatenate([sp, sn]); r = allv.argsort().argsort() + 1.0
        return (r[:len(sp)].sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn))
    base = m2.metrics(tb)
    pred = vbest > 0
    tp = int((pos & pred & (vtop == gold)).sum())
    P, R = tp / max(pred.sum(), 1), tp / max(pos.sum(), 1)
    print(f"{args.base} margins: AUC {base['auc']:.3f}, top-1 {base['top1']:.3f}, F1 {base['f1']:.3f}")
    print(f"{args.run} over its top 4: AUC {auc(vbest):.3f}, top-1 {float((vtop[pos] == gold[pos]).mean()):.3f}, "
          f"F1 {2 * P * R / max(P + R, 1e-12):.3f} (P {P:.3f} R {R:.3f}) at p > 0.5")
    out = outdir / f"verifier-{args.run}.pkl"
    out.write_bytes(pickle.dumps({"vbest": vbest, "vtop": vtop}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--news", action="store_true", help="build-k: add the judged news training texts")
    ap.add_argument("--out-data", default=DATA_K, help="build-k: the data file written")
    ap.add_argument("cmd", choices=["build", "build-k", "train", "score-m1", "score-m3", "score-m2"])
    ap.add_argument("--run", default="ultrabert-380m-v8e-w1half")
    ap.add_argument("--out-run", default="ultrabert-380m-v8v1")
    ap.add_argument("--out-dir")
    ap.add_argument("--repo", default="dragonscale-ai/kniv-v6-runs")
    ap.add_argument("--local", type=Path)
    ap.add_argument("--device")
    ap.add_argument("--d1e", type=Path, nargs="*", help="build: D1e's labels files (M1 format)")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=384,
                    help="train: longer examples go in batches of their own, within batch x max-tokens squared")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--limit", type=int, help="train: first N training examples (pipeline test)")
    ap.add_argument("--dev-limit", type=int, default=4000)
    ap.add_argument("--neg-weight", type=float, default=2.0)
    ap.add_argument("--hedge-weight", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--data", default=DATA, help="train: the verifier set (data/v6-corpus/...)")
    ap.add_argument("--name", default="m1p3b-n500", help="score-m1: the M1 predictions file")
    ap.add_argument("--tag", help="score-m1: the verifier's tag in the output name (default the run's last word)")
    ap.add_argument("--config", default="e1b", help="score-m1: the config whose predictions to re-rank")
    ap.add_argument("--base", default="ultrabert-380m-v8e-w1half", help="score-m2: the run whose top 4 to verify")
    ap.add_argument("--m2-set", default="v6/runs/m2-unseen/set.json")
    ap.add_argument("--slice", help="score-m2: i/n, items with index %% n == i (fresh process each)")
    ap.add_argument("--combine", action="store_true", help="score-m2: merge the --slice files and score")
    a = ap.parse_args()
    if a.cmd == "build":
        return build(a.seed, a.d1e)
    if a.cmd == "build-k":
        return build_k(a.seed, a.news, a.out_data)
    if a.cmd == "train":
        return train(a)
    if a.cmd == "score-m3":
        return score_m3(a)
    if a.cmd == "score-m2":
        return score_m2(a)
    return score_m1(a)


if __name__ == "__main__":
    raise SystemExit(main())
