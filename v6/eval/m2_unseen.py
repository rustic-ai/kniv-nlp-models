"""v8 M2: a zero-shot relation test on Wikidata relations no training source has.

    uv run python -m v6.eval.m2_unseen build [--docs data/trex/docs.jsonl] [--min-facts 5] [--allow-seen]
    uv run python -m v6.eval.m2_unseen predict --model pt:ultrabert-380m-v7b8ev
    uv run python -m v6.eval.m2_unseen predict --model onnx:dragonscale-ai/kniv-v7-xlarge-u-en
    uv run python -m v6.eval.m2_unseen score --models pt:ultrabert-380m-v7b8ev,onnx:...

``build`` takes T-REx documents in ``v6.gold.trex convert`` form and keeps the
Wikidata properties that pass every leak filter:

1. the property's code is not in the label bank (any relation, trained or held
   out: FewRel validation and the zero-shot holdout are in the bank);
2. neither its name nor any alias equals a bank name or alias (normalised), nor
   a CoNLL04 or CrossRE relation name (evaluation-only sets);
3. it relates two entities (Wikidata item or time); coordinates, quantities,
   strings and identifiers are values kniv does not normalise;
4. no near-paraphrase of a bank relation: every candidate's nearest bank labels
   under the model's label encoder are listed (``neighbours.json``) and the
   ones that mean the same are dropped by hand (``PARAPHRASE``);
5. at least ``--min-facts`` aligned facts after the text filters.

Text filters: a document used for training (``kg_trex.json``) or sharing a
10-word sequence with any training corpus is dropped. ``--allow-seen`` keeps
the ``kg_trex.json`` documents (their facts on these properties were never
labelled; the pairs were trained as "no listed relation") and marks the set
provisional.

Items are entity pairs with gold entities given, scored against the full M2
schema (every kept relation, by name and Wikidata description):

- positive: the pair has a fact on an M2 relation (the label);
- negative: the pair has an aligned Wikidata fact on a bank relation only, so
  a relation holds but none in the schema does (the pair decision).

Metrics (``score``): fact F1 at each model's deployed new-relation bar
(``unseen_offset``); top-1 accuracy on positives over the whole schema and in
5-way draws (threshold-free); AUC of the best margin, positives against
negatives. Intervals by bootstrap over documents; the detectable difference
is (1.96 + 0.84) x the standard error of the paired difference between two
models (80% power at the 5% level).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..config import DATA_DIR, RUNS_DIR

OUT = RUNS_DIR / "m2-unseen"
TREX_DOCS = DATA_DIR / "trex" / "docs.jsonl"
PROPS = DATA_DIR / "wikidata" / "properties.json"
WINDOW = 400            # words per window, whole sentences
NGRAM = 10
DATATYPES = ("WikibaseItem", "Time")
CONLL04 = ["works for", "killed", "based in", "lives in", "located in", "organization based in",
           "live in", "work for", "kill"]
CROSSRE = ["part-of", "physical", "usage", "role", "social", "general-affiliation", "named",
           "temporal", "artifact", "origin", "topic", "win-defeat", "cause-effect", "compare",
           "opposite", "type-of", "related-to"]
# Candidates whose meaning a bank relation already carries (read from
# neighbours.json): dropped as near-paraphrases. Code -> the bank relation.
PARAPHRASE: dict[str, str] = {
    "P1018": "regulated by",                 # language regulatory body
    "P114": "member of",                     # airline alliance: the airline is a member of it
    "P517": "physically interacts with",     # interaction (fundamental force)
    # the full archive (docs-rest.jsonl), read from neighbours.json on 2026-11-12
    "P1537": "has contributing factor",      # contributing factor of: its inverse
    "P2828": "officer of",                   # corporate officer: its inverse
    "P703": "natural product of taxon",      # found in taxon
    "P2852": "has phone number",             # emergency phone number
    "P1916": "genetic association",          # gene substitution association with
    "P943": "developed by",                  # programmer
}
NEG_PER_POS = 1.0
TRAIN_CORPORA = ["kg_trex.json", "kg_pretrain.json", "docred_distant_rel.json", "redocred_rel.json",
                 "fewrel_rel.json", "fewnerd_spans.json", "kg_short.json", "kg_teacher.json"]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _words(tokens) -> list[str]:
    return [w for w in (re.sub(r"[^a-z0-9]", "", t.lower()) for t in tokens) if w]


def _grams(tokens) -> set[int]:
    w = _words(tokens)
    return {hash(" ".join(w[i:i + NGRAM])) for i in range(len(w) - NGRAM + 1)}


def _token_lists(x):
    """Every list of word strings under ``x`` (any corpus layout)."""
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "tokens":
                if isinstance(v, str) and v.startswith("["):
                    try:
                        import ast
                        v = ast.literal_eval(v)
                    except Exception:
                        continue
                if isinstance(v, list) and v and isinstance(v[0], str):
                    yield v
                    continue
            yield from _token_lists(v)
    elif isinstance(x, list):
        for v in x:
            if isinstance(v, (dict, list)):
                yield from _token_lists(v)


def training_overlap(doc_grams: dict[str, set[int]], skip: tuple[str, ...] = ()) -> dict[str, set]:
    """Corpus -> ids of the candidate documents sharing a 10-gram with it.
    Streams each training corpus against the (small) candidate gram set."""
    index: dict[int, set[str]] = defaultdict(set)
    for d, gs in doc_grams.items():
        for g in gs:
            index[g].add(d)
    hits: dict[str, set] = {}
    for name in TRAIN_CORPORA:
        if name in skip:
            continue
        p = DATA_DIR / "v6-corpus" / name
        if not p.exists():
            print(f"  {name}: missing, not checked", flush=True)
            continue
        t0 = time.time()
        found: set[str] = set()
        for toks in _token_lists(json.loads(p.read_text())):
            for g in _grams(toks):
                if g in index:
                    found |= index[g]
        hits[name] = found
        print(f"  {name}: {len(found)} candidate documents overlap ({time.time() - t0:.0f}s)",
              flush=True)
    return hits


def windows_of(doc: dict) -> list[tuple[int, int, list[list[int]]]]:
    """Whole-sentence windows of at most WINDOW words: (start, end, sentence spans)."""
    out, cur = [], []
    for a, b in doc["sentence_spans"]:
        if cur and b - cur[0][0] > WINDOW:
            out.append((cur[0][0], cur[-1][1], cur))
            cur = []
        if b - a > WINDOW:          # one over-long sentence: cut it
            b = a + WINDOW
        cur.append([a, b])
    if cur:
        out.append((cur[0][0], cur[-1][1], cur))
    return out


def build(args) -> dict:
    from ..label_bank import load as load_bank
    bank = load_bank()
    codes = {r["code"] for r in bank["relations"] if r.get("code")}
    bank_names = set()
    for r in bank["relations"]:
        bank_names.add(norm(r["name"]))
        bank_names.update(norm(a) for a in r.get("aliases") or [])
    eval_names = {norm(n) for n in CONLL04 + CROSSRE}
    props = {p["code"]: p for p in json.loads(PROPS.read_text())}
    docs = [json.loads(line) for line in Path(args.docs).open()]
    report: dict = {"docs_in": len(docs), "relations": {}, "items": {}}

    # 1-3: relation filters on every property with a fact
    facts_by_p = Counter(p for d in docs for *_, p in d["facts"])
    why: dict[str, str] = {}
    for p in facts_by_p:
        info = props.get(p)
        if p in codes:
            why[p] = "code in bank"
        elif not info or not info.get("name"):
            why[p] = "no Wikidata label"
        elif norm(info["name"]) in bank_names or any(norm(a) in bank_names
                                                       for a in info.get("aliases") or []):
            why[p] = "name or alias in bank"
        elif norm(info["name"]) in eval_names:
            why[p] = "CoNLL04/CrossRE name"
        elif info.get("type") not in DATATYPES:
            why[p] = f"value datatype ({info.get('type')})"
        elif p in PARAPHRASE:
            why[p] = f"near-paraphrase of bank '{PARAPHRASE[p]}'"
    cand = [p for p in facts_by_p if p not in why]
    r = report["relations"]
    r["properties_with_facts"] = len(facts_by_p)
    for reason in sorted(set(v.split(" (")[0].split(" '")[0] for v in why.values())):
        ps = [p for p, v in why.items() if v.startswith(reason)]
        r[f"removed: {reason}"] = {"relations": len(ps), "facts": sum(facts_by_p[p] for p in ps)}
    r["after relation filters"] = {"relations": len(cand), "facts": sum(facts_by_p[p] for p in cand)}

    # text filters, on documents with a candidate fact
    cset = set(cand)
    m2_docs = {d["id"]: d for d in docs if any(p in cset for *_, p in d["facts"])}
    trex_ids = {x["id"] for x in json.loads((DATA_DIR / "v6-corpus" / "kg_trex.json").read_text())["docs"]}
    seen = {i for i in m2_docs if i in trex_ids}
    t = report["items"]
    t["documents with a candidate fact"] = len(m2_docs)
    t["in kg_trex.json (trained)"] = len(seen)
    print("10-gram overlap with the training corpora:", flush=True)
    hits = training_overlap({i: _grams(d["tokens"]) for i, d in m2_docs.items()},
                            skip=("kg_trex.json",))
    for name, ids in hits.items():
        t[f"share a 10-gram with {name}"] = len(ids)
    drop = set().union(*hits.values()) if hits else set()
    if not args.allow_seen:
        drop |= seen
    keep_docs = [d for i, d in m2_docs.items() if i not in drop]
    t["documents kept"] = len(keep_docs)
    t["provisional (trained texts kept)"] = bool(args.allow_seen)

    # items per window
    items, windows = [], []
    for d in keep_docs:
        facts = defaultdict(set)
        for h, tl, p in d["facts"]:
            facts[(h, tl)].add(p)
        for wi, (a, b, sps) in enumerate(windows_of(d)):
            ents, local = [], {}
            for k, e in enumerate(d["entities"]):
                ms = [[x - a, y - a] for x, y in e["mentions"] if x >= a and y <= b]
                if ms:
                    local[k] = len(ents)
                    ents.append({"type": "date" if e["kind"] == "date" else "entity",
                                 "mentions": ms, "key": e["key"]})
            if len(ents) < 2:
                continue
            pos, neg = [], []
            for (h, tl), ps in facts.items():
                if h not in local or tl not in local:
                    continue
                m2 = sorted(ps & cset)
                if m2:
                    pos.extend((local[h], local[tl], p) for p in m2)
                elif ps & codes:        # a bank relation holds, no M2 one
                    neg.append((local[h], local[tl], None))
            if not pos:
                continue
            wid = f"{d['id']}#{wi}"
            windows.append({"id": wid, "doc": d["id"], "tokens": d["tokens"][a:b],
                            "sentence_spans": [[x - a, y - a] for x, y in sps], "entities": ents})
            rng = np.random.default_rng(int(hashlib.sha1(wid.encode()).hexdigest()[:8], 16))
            n_neg = min(len(neg), max(1, int(round(NEG_PER_POS * len(pos)))))
            neg = [neg[i] for i in sorted(rng.choice(len(neg), n_neg, replace=False))] if neg else []
            for h, tl, p in pos + neg:
                items.append({"window": wid, "doc": d["id"], "head": h, "tail": tl, "relation": p})
    support = Counter(it["relation"] for it in items if it["relation"])
    rels = sorted(p for p, n in support.items() if n >= args.min_facts)
    low = [p for p in support if p not in rels]
    r[f"removed: under {args.min_facts} facts in kept texts"] = {
        "relations": len(low) + len(cset - set(support)), "facts": sum(support[p] for p in low)}
    keep = set(rels)
    items = [it for it in items if it["relation"] is None or it["relation"] in keep]
    used = {it["window"] for it in items if it["relation"]}
    items = [it for it in items if it["window"] in used]
    windows = [w for w in windows if w["id"] in used]
    relations = [{"code": p, "name": props[p]["name"], "description": props[p].get("description") or "",
                  "datatype": props[p].get("type"), "facts": support[p]} for p in rels]
    r["kept"] = {"relations": len(rels), "facts": sum(support[p] for p in rels)}
    t["windows"] = len(windows)
    t["documents"] = len({w["doc"] for w in windows})
    t["positive items"] = sum(1 for it in items if it["relation"])
    t["negative items"] = sum(1 for it in items if not it["relation"])
    out = {"provisional": bool(args.allow_seen), "min_facts": args.min_facts,
           "relations": relations, "windows": windows, "items": items, "report": report,
           "removed": why}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / ("set-provisional.json" if args.allow_seen else "set.json")
    path.write_text(json.dumps(out))
    print(json.dumps(report, indent=1))
    print(f"wrote {path}: {len(rels)} relations, {len(windows)} windows, {len(items)} items")
    return out


# ── models ──────────────────────────────────────────────────────────
def label_rel(r: dict) -> dict:
    return {"name": r["name"], "description": r["description"]}


class PtModel:
    """A training run through the PyTorch extractor, CPU."""

    def __init__(self, run: str):
        from ..infer import Extractor
        self.ex = Extractor(run, device="cpu")
        self.bar = self.ex.unseen_offset

    def margins(self, w: dict, schema: list[dict]):
        """[E, E, R] raw margins over the schema (relation logit minus the
        no-relation logit; the deployed zs offset applied), every ordered pair."""
        import torch
        from ..train.data import blank_window, collate, marker_features
        from ..train.kdata import label_text
        from ..infer import UNSEEN_ID
        ex = self.ex
        tokens, sp = w["tokens"], [tuple(x) for x in w["sentence_spans"]]
        f = blank_window(tokens, sp, ex.tok, ex.spec.max_subwords, ex.vocab.size("cls"))
        W = f["_W"]
        spans, idx = [], []
        for k, e in enumerate(w["entities"]):
            ms = [(a, b) for a, b in e["mentions"] if b <= W and f["word_ok"][a]]
            if ms:
                spans.append(np.asarray(ms, dtype=np.int32))
                idx.append(k)
        E = len(w["entities"])
        out = np.full((E, E, len(schema)), np.nan, dtype=np.float32)
        if len(spans) < 2:
            return out
        f.update(coref=None, _n_rel=1, rel={"spans": spans, "types": np.zeros(len(spans), np.int32),
                                            "triples": []})
        f["rm"] = marker_features(tokens, sp, spans, ex.tok, turns=[], participants=[])
        b = collate([f], ex.tok.pad_token_id)
        enc = ex.tok([label_text(r) for r in schema], padding=True, truncation=True, max_length=48,
                     return_tensors="pt")
        b = {k: v.to(ex.device) for k, v in b.items() if hasattr(v, "to")}
        with torch.no_grad():
            b["rel_label_emb"] = ex.model.label_embeddings(enc["input_ids"], enc["attention_mask"])
            b["rel_label_cols"] = torch.tensor([UNSEEN_ID + i for i in range(len(schema))])
            o = ex.model(b, relation_only=True)
        lg = o["relation"].float().cpu().numpy()
        for h, t, row in zip(b["pair_h"].tolist(), b["pair_t"].tolist(), lg):
            out[idx[h], idx[t]] = row[1:] - row[0]
        return out


class OnnxModel:
    """A release repo through its own runtime (fp32 ONNX, CPU)."""

    def __init__(self, repo: str):
        import importlib.util
        import sys
        from huggingface_hub import snapshot_download
        path = Path(snapshot_download(repo, allow_patterns=[
            "onnx/marker.onnx", "onnx/label.onnx", "onnx/onnx_meta.json", "tokenizer.json",
            "calibration.json", "label_bank.json", "kniv_v7/*"]))
        spec = importlib.util.spec_from_file_location("kniv_v7.runtime", path / "kniv_v7" / "runtime.py")
        if "kniv_v7" not in sys.modules:
            import types
            pkg = types.ModuleType("kniv_v7")
            pkg.__path__ = [str(path / "kniv_v7")]
            sys.modules["kniv_v7"] = pkg
        self.rt = importlib.util.module_from_spec(spec)
        sys.modules["kniv_v7.runtime"] = self.rt
        spec.loader.exec_module(self.rt)
        self.k = self.rt.Kniv(path, providers=["CPUExecutionProvider"])
        self.bar = self.k.unseen_offset

    def margins(self, w: dict, schema: list[dict]):
        rt, k = self.rt, self.k
        tokens, sp = w["tokens"], [tuple(x) for x in w["sentence_spans"]]
        _, first, W = rt.align(tokens, k.tok, k.meta["max_subwords"])
        ok = [x >= 0 for x in first]
        spans, idx = [], []
        for i, e in enumerate(w["entities"]):
            ms = [(a, b) for a, b in e["mentions"] if b <= W and ok[a]]
            if ms:
                spans.append(ms)
                idx.append(i)
        E = len(w["entities"])
        out = np.full((E, E, len(schema)), np.nan, dtype=np.float32)
        if len(spans) < 2:
            return out
        ids, men_pos = rt.marker_input(tokens, sp, spans, k.tok, turns=[], participants=[])
        men_ent = [j for j, ms in enumerate(spans) for _ in ms]
        pos = [x for j in range(len(spans)) for x in men_pos[j]]
        n = len(spans)
        pairs = [(h, t) for h in range(n) for t in range(n) if h != t]
        temb = k.label_emb([rt.label_text(r) for r in schema])
        logits = k.run("marker", rm_input_ids=np.asarray([ids], dtype=np.int64),
                       rm_attention_mask=np.ones((1, len(ids)), dtype=np.int64),
                       men_pos=np.asarray(pos, dtype=np.int64),
                       men_ent=np.asarray(men_ent, dtype=np.int64),
                       ent_ids=np.arange(n, dtype=np.int64),
                       pair_h=np.asarray([h for h, _ in pairs], dtype=np.int64),
                       pair_t=np.asarray([t for _, t in pairs], dtype=np.int64),
                       label_emb=temb)[0]
        for (h, t), row in zip(pairs, logits):
            out[idx[h], idx[t]] = row[1:] - (row[0] + k.zs_offset)
        return out


def load_model(spec: str):
    kind, name = spec.split(":", 1)
    return PtModel(name) if kind == "pt" else OnnxModel(name)


def tag(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", spec)


def predict(args) -> None:
    data = json.loads(Path(args.set).read_text())
    schema = [label_rel(r) for r in data["relations"]]
    want = defaultdict(list)
    for it in data["items"]:
        want[it["window"]].append((it["head"], it["tail"]))
    wins = data["windows"]
    if args.slice:
        i, n = map(int, args.slice.split("/"))
        wins = wins[i::n]
    model = load_model(args.model)
    res, t0 = {}, time.time()
    for j, w in enumerate(wins):
        m = model.margins(w, schema)
        res[w["id"]] = {(h, t): m[h, t] for h, t in want[w["id"]]}
        if (j + 1) % 25 == 0:
            print(f"{args.model}: {j + 1}/{len(wins)} windows ({time.time() - t0:.0f}s)", flush=True)
    dest = OUT / (f"preds-{tag(args.model)}{set_tag(args.set)}"
                  f"{'-' + args.slice.replace('/', 'of') if args.slice else ''}.pkl")
    dest.write_bytes(pickle.dumps({"model": args.model, "bar": model.bar, "set": str(args.set),
                                   "margins": res}))
    print(f"wrote {dest} ({len(res)} windows, {time.time() - t0:.0f}s, bar {model.bar:+.2f})")


def set_tag(path) -> str:
    """'' for the provisional set (its prediction files predate the tag), else
    '-<set file stem>'."""
    stem = Path(path).stem
    return "" if stem == "set-provisional" else f"-{stem}"


def load_preds(spec: str, set_path=None) -> dict:
    paths = sorted(OUT.glob(f"preds-{tag(spec)}*.pkl"))
    out = None
    for p in paths:
        x = pickle.loads(p.read_bytes())
        if set_path is not None and Path(x["set"]).stem != Path(set_path).stem:
            continue
        if out is None:
            out = x
        else:
            out["margins"].update(x["margins"])
    if out is None:
        raise SystemExit(f"no predictions for {spec}")
    return out


def item_table(data: dict, preds: dict, bar: float | None = None) -> dict:
    """Per item arrays: positive?, gold index, margins over the schema (minus
    the bar), the 5-way draw's correctness."""
    codes = [r["code"] for r in data["relations"]]
    ci = {c: i for i, c in enumerate(codes)}
    R = len(codes)
    bar = preds["bar"] if bar is None else bar
    pos, gold, M, five, docs = [], [], [], [], []
    for it in data["items"]:
        m = preds["margins"][it["window"]][(it["head"], it["tail"])]
        if np.isnan(m).all():
            m = np.full(R, -1e3, dtype=np.float32)       # entity cut off by the subword limit
        M.append(m - bar)
        p = it["relation"] is not None
        pos.append(p)
        g = ci[it["relation"]] if p else -1
        gold.append(g)
        if p:
            rng = np.random.default_rng(int(hashlib.sha1(
                f"{it['window']}|{it['head']}|{it['tail']}|{g}".encode()).hexdigest()[:8], 16))
            others = rng.choice([k for k in range(R) if k != g], 4, replace=False)
            five.append(bool(m[g] > m[others].max()))
        else:
            five.append(False)
        docs.append(it["doc"])
    return {"pos": np.array(pos), "gold": np.array(gold), "M": np.array(M), "five": np.array(five),
            "docs": np.array(docs)}


def metrics(tb: dict, sel: np.ndarray | None = None) -> dict:
    """F1 at the bar (top-1 relation per pair), top-1 and 5-way accuracy on
    positives, AUC of the best margin."""
    if sel is None:
        sel = np.arange(len(tb["pos"]))
    pos, gold, M, five = tb["pos"][sel], tb["gold"][sel], tb["M"][sel], tb["five"][sel]
    top = M.argmax(1)
    best = M.max(1)
    pred = best > 0
    tp = int((pos & pred & (top == gold)).sum())
    np_, ng = int(pred.sum()), int(pos.sum())
    P, Rc = tp / max(np_, 1), tp / max(ng, 1)
    s_pos, s_neg = best[pos], best[~pos]
    auc = float("nan")
    if len(s_pos) and len(s_neg):
        allv = np.concatenate([s_pos, s_neg])
        ranks = allv.argsort().argsort() + 1.0
        auc = (ranks[:len(s_pos)].sum() - len(s_pos) * (len(s_pos) + 1) / 2) / (len(s_pos) * len(s_neg))
    return {"f1": 2 * P * Rc / max(P + Rc, 1e-12), "p": P, "r": Rc,
            "top1": float((top[pos] == gold[pos]).mean()) if ng else float("nan"),
            "five_way": float(five[pos].mean()) if ng else float("nan"),
            "auc": float(auc), "neg_kept_out": float((~pred[~pos]).mean()) if (~pos).any() else float("nan")}


def doc_boot(tb: dict, n: int = 2000, seed: int = 13):
    """Index samples: documents resampled with replacement (items of a document move together)."""
    rng = np.random.default_rng(seed)
    by = defaultdict(list)
    for i, d in enumerate(tb["docs"]):
        by[d].append(i)
    groups = [np.array(v) for v in by.values()]
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        yield np.concatenate([groups[g] for g in pick])


def score(args) -> None:
    data = json.loads(Path(args.set).read_text())
    specs = args.models.split(",")
    tabs = {s: item_table(data, load_preds(s, args.set)) for s in specs}
    keys = ("f1", "p", "r", "top1", "five_way", "auc", "neg_kept_out")
    summary = {"set": str(args.set), "provisional": data["provisional"],
               "relations": len(data["relations"]), "items": len(data["items"]),
               "positives": int(tabs[specs[0]]["pos"].sum()),
               "documents": len(set(tabs[specs[0]]["docs"])), "models": {}, "paired": {}}
    samples = list(doc_boot(tabs[specs[0]], args.boot))
    for s in specs:
        tb = tabs[s]
        point = metrics(tb)
        boots = [metrics(tb, i) for i in samples]
        ci = {k: [float(np.nanpercentile([b[k] for b in boots], q)) for q in (2.5, 97.5)] for k in keys}
        se = {k: float(np.nanstd([b[k] for b in boots])) for k in keys}
        summary["models"][s] = {"bar": float(load_preds(s, args.set)["bar"]), **{k: point[k] for k in keys},
                                "ci": ci, "se": se}
        print(f"{s} (bar {summary['models'][s]['bar']:+.2f}): " + ", ".join(
            f"{k} {point[k]:.3f} [{ci[k][0]:.3f}, {ci[k][1]:.3f}]" for k in keys))
    if len(specs) > 1:
        base = specs[-1]
        for s in specs[:-1]:
            rec = {}
            for k in ("f1", "top1", "five_way", "auc"):
                d = [metrics(tabs[s], i)[k] - metrics(tabs[base], i)[k] for i in samples]
                diff = metrics(tabs[s])[k] - metrics(tabs[base])[k]
                se = float(np.nanstd(d))
                rec[k] = {"diff": diff, "ci": [float(np.nanpercentile(d, 2.5)),
                                               float(np.nanpercentile(d, 97.5))],
                          "se": se, "detectable": 2.80 * se}
            summary["paired"][f"{s} vs {base}"] = rec
            print(f"{s} vs {base}: " + "; ".join(
                f"{k} {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] (detectable {v['detectable']:.3f})"
                for k, v in rec.items()))
    # per relation, first model
    tb = tabs[specs[0]]
    per = []
    for i, r in enumerate(data["relations"]):
        sel = np.where(tb["gold"] == i)[0]
        per.append({"code": r["code"], "name": r["name"], "n": len(sel),
                    "top1": float((tb["M"][sel].argmax(1) == i).mean()),
                    "found": float(((tb["M"][sel].argmax(1) == i) & (tb["M"][sel].max(1) > 0)).mean())})
    summary["per_relation"] = {specs[0]: per}
    dest = OUT / f"score-{Path(args.set).stem}.json"
    dest.write_text(json.dumps(summary, indent=1))
    print(f"wrote {dest}")


def neighbours(args) -> None:
    """Each candidate's nearest bank relations under a model's label encoder
    (cosine of 'relation: name. description'), for the paraphrase review."""
    import torch
    from ..infer import Extractor
    from ..label_bank import load as load_bank
    from ..train.kdata import label_text
    data = json.loads(Path(args.set).read_text())
    props = {p["code"]: p for p in json.loads(PROPS.read_text())}
    cands = [r["code"] for r in data["relations"]]
    if args.all_candidates:
        allc = Counter(p for line in Path(args.docs).open() for *_, p in json.loads(line)["facts"])
        cands = [p for p in allc if p not in data["removed"]]
    bank = [r for r in load_bank()["relations"]]
    ex = Extractor(args.run, device="cpu")

    def emb(rels):
        out = []
        for i in range(0, len(rels), 64):
            enc = ex.tok([label_text(r) for r in rels[i:i + 64]], padding=True, truncation=True,
                         max_length=48, return_tensors="pt")
            with torch.no_grad():
                e = ex.model.label_embeddings(enc["input_ids"], enc["attention_mask"]).float()
            out.append(torch.nn.functional.normalize(e.reshape(e.shape[0], -1), dim=-1))
        return torch.cat(out)
    B = emb([{"name": r["name"], "description": r.get("description") or ""} for r in bank])
    crels = [{"name": props[c]["name"], "description": props[c].get("description") or ""} for c in cands]
    C = emb(crels)
    S = (C @ B.T).numpy()
    rows = []
    for i, c in enumerate(cands):
        top = np.argsort(-S[i])[:5]
        rows.append({"code": c, "name": crels[i]["name"], "description": crels[i]["description"],
                     "nearest": [{"name": bank[j]["name"], "sim": round(float(S[i, j]), 3),
                                  "description": bank[j].get("description") or ""} for j in top]})
    rows.sort(key=lambda r: -r["nearest"][0]["sim"])
    dest = OUT / "neighbours.json"
    dest.write_text(json.dumps(rows, indent=1))
    for r in rows:
        n = r["nearest"][0]
        print(f"{r['code']:7s} {r['name'][:32]:32s} -> {n['name'][:32]:32s} {n['sim']:.3f}")
    print(f"wrote {dest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--docs", default=str(TREX_DOCS))
    b.add_argument("--min-facts", type=int, default=5)
    b.add_argument("--allow-seen", action="store_true")
    n = sub.add_parser("neighbours")
    n.add_argument("--set", default=str(OUT / "set-provisional.json"))
    n.add_argument("--docs", default=str(TREX_DOCS))
    n.add_argument("--run", default="ultrabert-380m-v7b8ev")
    n.add_argument("--all-candidates", action="store_true")
    p = sub.add_parser("predict")
    p.add_argument("--model", required=True, help="pt:<run> or onnx:<repo>")
    p.add_argument("--set", default=str(OUT / "set-provisional.json"))
    p.add_argument("--slice", help="i/n: every n-th window from i (fresh-process slices)")
    s = sub.add_parser("score")
    s.add_argument("--models", required=True, help="comma-separated; the last is the baseline")
    s.add_argument("--set", default=str(OUT / "set-provisional.json"))
    s.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    {"build": build, "neighbours": neighbours, "predict": predict, "score": score}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
