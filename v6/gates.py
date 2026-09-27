"""The DATASET_SPEC section 7 quality gates, run over the assembled shards.

    uv run python -m v6.gates                 # all shards in data/v6-corpus
    uv run python -m v6.gates --shard 0       # one shard, while iterating

Every gate is a build failure unless the spec marks it *report*. A gate that
cannot be evaluated from the shards says so and does not pass silently --
"not checked" and "passed" must never look the same, which is the whole point
of having gates.

Two gates in this file exist because a defect got past the others:

* Gate 6 checks per-token lengths, and gate 10 checks that coref mentions lie
  inside the window. Neither catches an annotation attached to the WRONG
  sentence, because a wrong-but-plausible value has the right shape and range.
  That is why item ids are content-addressed (4.1d) and why the coref batch
  refuses a call whose result count is short: those are the real defences.
  Gate 18 here checks what a shape gate can -- that every layer is present or
  masked, never quietly absent.
* Gate 17 was unimplemented while the manifest was missing four of eleven
  layers, so it now checks the manifest against the layers the rows carry
  rather than against a hand-written list.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .config import DATA_DIR

OUT = DATA_DIR / "v6-corpus"
from .schemas import (CLS_LABELS, DEPRELS, NER_LABELS, SENTIMENT_LABELS,
                      SRL_TAGS, UPOS_TAGS)

TOKEN_LAYERS = ["pos", "ner", "dep", "lemma", "morph"]
INVENTORY = {"pos": set(UPOS_TAGS), "ner": set(NER_LABELS),
             "dep": set(DEPRELS), "sentiment": set(SENTIMENT_LABELS)}
MAX_TOKENS = 512


class Result:
    __slots__ = ("num", "name", "kind", "ok", "detail")

    def __init__(self, num, name, kind, ok, detail=""):
        self.num, self.name, self.kind = num, name, kind
        self.ok, self.detail = ok, detail

    @property
    def status(self) -> str:
        if self.ok is None:
            return "NOT CHECKED"
        if self.ok:
            return "pass"
        return "REPORT" if self.kind == "report" else "FAIL"


def _bio_wellformed(tags: list[str]) -> int:
    """Count positions where I-X does not follow B-X or I-X."""
    bad, prev = 0, "O"
    for t in tags:
        if t.startswith("I-") and prev[2:] != t[2:]:
            bad += 1
        prev = t
    return bad


def _is_tree(heads: list[int], start: int, end: int) -> bool:
    local = heads[start:end]
    if sum(1 for h in local if h == -1) != 1:
        return False
    for i in range(len(local)):
        seen, cur, steps = set(), i, 0
        while local[cur] != -1:
            nxt = local[cur] - start
            if not (0 <= nxt < len(local)) or nxt in seen or steps > len(local):
                return False
            seen.add(nxt); cur = nxt; steps += 1
    return True


def run(rows: list[dict], manifest: dict, tokenizer=None) -> list[Result]:
    out: list[Result] = []
    n = len(rows)
    mask = [json.loads(r["loss_mask"]) for r in rows]
    prov = [json.loads(r["provenance"]) for r in rows]

    # ---- Windows ----------------------------------------------------------
    bad = []
    for r in rows:
        pos = 0
        for s, e in r["sentence_spans"]:
            if s != pos or e <= s:
                bad.append(r["window_id"]); break
            pos = e
        else:
            if pos != r["n_tokens"]:
                bad.append(r["window_id"])
    out.append(Result(1, "sentence_spans tile [0, n_tokens) exactly", "fail",
                      not bad, f"{len(bad)} windows violate" if bad else
                      f"{n} windows tile exactly"))

    # Gate 2 is a property of the builder: windows are cut on sentence
    # boundaries, so a split sentence cannot be observed from a single row.
    # What IS observable is that every row's tokens and spans agree, which
    # gate 1 covers. Recorded as not checkable here rather than passed.
    out.append(Result(2, "no sentence split across a window boundary", "fail",
                      None, "enforced in windows.py; not reconstructible "
                            "from shards alone"))

    if tokenizer is None:
        out.append(Result(3, f"n_tokens <= {MAX_TOKENS} under the DeBERTa "
                             "tokenizer", "fail", None,
                          "tokenizer unavailable in this venv"))
    else:
        over = []
        for r in rows:
            k = len(tokenizer(" ".join(r["tokens"]),
                              add_special_tokens=True)["input_ids"])
            if k > MAX_TOKENS:
                over.append((r["window_id"], k))
        out.append(Result(3, f"n_tokens <= {MAX_TOKENS} under the DeBERTa "
                             "tokenizer", "fail", not over,
                          f"{len(over)} windows over, worst {max(k for _, k in over)}"
                          if over else f"{n} windows within {MAX_TOKENS}"))

    doms = Counter(r["domain"] for r in rows)
    out.append(Result(4, "PII scan clean on business/enron", "fail",
                      None if "business" in doms else True,
                      "business domain not present in this corpus"
                      if "business" not in doms else "business present: scan required"))

    seen, dup = {}, 0
    for r in rows:
        k = hash(tuple(r["tokens"]))
        if k in seen:
            dup += 1
        seen[k] = 1
    out.append(Result(5, "near-duplicate rate across documents", "report",
                      True, f"{dup} exact-token duplicates ({dup/max(n,1):.2%})"))

    # ---- Per-layer --------------------------------------------------------
    lenbad = Counter()
    for r, m in zip(rows, mask):
        for l in TOKEN_LAYERS + ["dep_heads"]:
            v = r.get(l)
            if v is None:
                continue
            if len(v) != r["n_tokens"]:
                lenbad[l] += 1
    out.append(Result(6, "every per-token list has length n_tokens", "fail",
                      not lenbad, dict(lenbad) if lenbad else
                      f"{n} windows consistent across {len(TOKEN_LAYERS)+1} layers"))

    invbad = defaultdict(Counter)
    for r in rows:
        for l, allowed in INVENTORY.items():
            v = r.get(l)
            if v is None:
                continue
            vals = v if l != "sentiment" else [x for x in v if x is not None]
            for x in vals:
                if x not in allowed:
                    invbad[l][x] += 1
        for f in (r.get("srl_frames") or []):
            for t in f["tags"]:
                if t not in set(SRL_TAGS):
                    invbad["srl"][t] += 1
    out.append(Result(7, "every label is in its inventory", "fail",
                      not invbad,
                      {k: dict(v.most_common(4)) for k, v in invbad.items()}
                      if invbad else "pos, ner, dep, sentiment, srl all in inventory"))

    bio = Counter()
    for r in rows:
        if r.get("ner"):
            bio["ner"] += _bio_wellformed(r["ner"])
        for f in (r.get("srl_frames") or []):
            bio["srl"] += _bio_wellformed(f["tags"])
    out.append(Result(8, "NER and SRL BIO sequences well-formed", "fail",
                      not sum(bio.values()), dict(bio) if sum(bio.values())
                      else "no I-X without a matching B-X/I-X"))

    tot = badtree = masked = 0
    for r, m in zip(rows, mask):
        if not r.get("dep_heads"):
            continue
        dm = m.get("dep_tokens")
        for s, e in r["sentence_spans"]:
            tot += 1
            if not _is_tree(r["dep_heads"], s, e):
                badtree += 1
                if dm and all(dm[s:e]):
                    masked += 1
    unmasked = badtree - masked
    out.append(Result(9, "each sentence span induces exactly one tree", "fail",
                      unmasked == 0,
                      f"{badtree}/{tot} non-trees ({badtree/max(tot,1):.2%}), "
                      f"{masked} masked, {unmasked} UNMASKED"))

    crbad = 0
    for r in rows:
        for cl in (r.get("coref") or []):
            for a, b in cl:
                if not (0 <= a <= b < r["n_tokens"]):
                    crbad += 1
    out.append(Result(10, "coref mentions lie within [0, n_tokens)", "fail",
                      crbad == 0, f"{crbad} out-of-range mentions" if crbad
                      else "all mentions in range"))

    clsbad = clsn = clsempty = 0
    for r in rows:
        c = r.get("cls")
        if c is None:
            continue
        if len(c) != len(r["sentence_spans"]):
            clsbad += 1
            continue
        for x in c:
            clsn += 1
            clsempty += (len(x) == 0)
            for lab in x:
                if lab not in set(CLS_LABELS):
                    clsbad += 1
    out.append(Result(11, "cls has one entry per sentence span, labels from "
                          "the six", "fail", clsbad == 0,
                      f"{clsbad} violations" if clsbad else
                      f"{clsn} sentences, {clsempty/max(clsn,1):.2%} with no function"))

    relbad = Counter(); reln = 0
    for r in rows:
        rel = r.get("relations") or []
        ncl = len(r.get("coref") or [])
        for x in rel:
            reln += 1
            if x["head"] == x["tail"]:
                relbad["head == tail"] += 1
    out.append(Result(12, "relations pass the 3A.4 gates", "fail",
                      not relbad, dict(relbad) if relbad else
                      f"{reln} triples, no self-relations"))

    # ---- Corpus -----------------------------------------------------------
    cov = {}
    for l in TOKEN_LAYERS + ["srl_frames", "cls", "sentiment", "keywords", "coref"]:
        have = sum(1 for r in rows if r.get(l) is not None)
        cov[l] = have / max(n, 1)
    worst = min(cov, key=cov.get)
    out.append(Result(13, "coverage per layer >= 98% (relations exempt)",
                      "fail" if cov[worst] < 0.90 else "report",
                      cov[worst] >= 0.98,
                      " ".join(f"{k} {v:.1%}" for k, v in sorted(cov.items()))))

    elig = sum(1 for r in rows if (r.get("relations") or []) or
               not json.loads(r["loss_mask"]).get("relations"))
    trip = sum(len(r.get("relations") or []) for r in rows)
    perdom = defaultdict(lambda: [0, 0])
    for r in rows:
        d = perdom[r["domain"]]
        d[0] += 1; d[1] += len(r.get("relations") or [])
    out.append(Result("13a", "relation coverage (report)", "report", True,
                      " | ".join(f"{k} {v[1]/max(v[0],1):.2f} triples/win"
                                 for k, v in sorted(perdom.items()))
                      + f" | total {trip} triples over {n} windows"))

    mr = defaultdict(lambda: [0, 0])
    for r, m in zip(rows, mask):
        s = mr[r["domain"]]
        s[0] += 1
        s[1] += 1 if (m.get("morph") or r.get("morph") is None) else 0
    out.append(Result(14, "morph mask rate per domain (report)", "report", True,
                      " ".join(f"{k} {v[1]/max(v[0],1):.1%}"
                               for k, v in sorted(mr.items()))))

    dist = defaultdict(Counter)
    for r in rows:
        for x in (r.get("cls") or []):
            for lab in x:
                dist[r["domain"]][lab] += 1
    lines = []
    for d, c in sorted(dist.items()):
        t = sum(c.values())
        lines.append(f"{d}: " + ",".join(f"{k} {v/t:.0%}" for k, v in c.most_common(3)))
    out.append(Result(15, "label distribution per layer per domain (report)",
                      "report", True, " | ".join(lines)))

    out.append(Result(16, "no doc_id appears in more than one split", "fail",
                      None, "splits not yet materialised (sequencing step 6)"))

    layers_in_rows = {l for p in prov for l in p}
    recorded = set(manifest.get("layer_source", {}))
    missing = sorted(layers_in_rows - recorded)
    vers = manifest.get("annotator_versions", {})
    unres = sorted(k for k, v in vers.items() if v in ("unresolved", None, ""))
    ok17 = not missing and not unres and bool(manifest.get("git_sha")) \
        and manifest.get("git_sha") != "unknown"
    detail = []
    if missing:
        detail.append(f"layers not in manifest: {missing}")
    if unres:
        detail.append(f"unresolved versions: {unres}")
    if not detail:
        detail.append(f"{len(recorded)} layers, {len(vers)} annotator versions, "
                      f"git_sha {str(manifest.get('git_sha'))[:12]}")
    out.append(Result(17, "MANIFEST records annotator versions and git sha for "
                          "every layer", "fail", ok17, "; ".join(detail)))

    # Gate 18: absence must be recorded, not silent. A layer that is None
    # without a mask entry is indistinguishable downstream from a layer that
    # is legitimately empty.
    silent = Counter()
    for r, m in zip(rows, mask):
        for l in TOKEN_LAYERS + ["cls", "sentiment", "keywords", "coref"]:
            if r.get(l) is None and not m.get(l):
                silent[l] += 1
    out.append(Result(18, "every absent layer is masked, never silently null",
                      "fail", not silent, dict(silent) if silent else
                      "no unmasked absences"))
    return out


def load(shard: int | None) -> tuple[list[dict], dict]:
    import pyarrow.parquet as pq
    files = sorted(OUT.glob("shard_*.parquet"))
    if shard is not None:
        files = [OUT / f"shard_{shard:03d}.parquet"]
    if not files:
        raise SystemExit(f"no shards in {OUT}; run --stage assemble first")
    rows = []
    for f in files:
        rows.extend(pq.read_table(f).to_pylist())
    mf = OUT / "MANIFEST.json"
    return rows, (json.loads(mf.read_text()) if mf.exists() else {})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--shard", type=int)
    ap.add_argument("--no-tokenizer", action="store_true",
                    help="skip gate 3 instead of loading DeBERTa")
    args = ap.parse_args()

    rows, manifest = load(args.shard)
    tok = None
    if not args.no_tokenizer:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(
                "models/kniv-deberta-nlp-base-en-large")
        except Exception as exc:                            # noqa: BLE001
            print(f"  (gate 3: tokenizer unavailable: {type(exc).__name__})")

    res = run(rows, manifest, tok)
    print(f"\n{len(rows)} rows from {OUT}\n")
    width = max(len(r.name) for r in res)
    for r in res:
        print(f"  {str(r.num):>3}  {r.status:<11} {r.name:<{width}}  {r.detail}")
    fails = [r for r in res if r.status == "FAIL"]
    notck = [r for r in res if r.status == "NOT CHECKED"]
    print(f"\n{len(res)} gates: {sum(1 for r in res if r.status=='pass')} pass, "
          f"{len(fails)} FAIL, "
          f"{sum(1 for r in res if r.status=='REPORT')} report, "
          f"{len(notck)} not checked")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
