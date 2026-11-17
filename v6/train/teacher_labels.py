"""v8 teacher labels for the students (plan r4 stage 6, decisions 16 and 17).

    python -m v6.train.teacher_labels dump --run R --ver R-ver3f --route RT --set student --shard i/n [--limit N] [--upload]
    (on an A100: colab post --jobs tl, one VM at a time)

Per window, what ``student_targets`` dumps for the teacher except the
linguistic heads frozen since v7 (``LINGUISTIC``: v7's targets serve), so
entities, coreference, spans, relation margins above ``FLOOR`` and
attributes, and
from the v8 knowledge layer as the teacher is deployed (its V3-form verifier
at 0.6, never-trained relations through the route, the article trim):

| key | content |
|---|---|
| ``ver`` | the verifier's candidates: head, tail, relation index, margin over the bar, P(expressed), four verdicts (supported, not, negated, hedged); None where the window does not fit |
| ``kept`` | the teacher's facts: head, tail, relation index |
| ``rel_route`` | the route's relation margins above ``FLOOR`` for the routed relations (indices into the window's schema) |
| ``routed`` | indices of the routed relations |
| ``open`` | schema-free facts decoded from the teacher's linguistic layer (``kniv_v7.open_facts``) with the verifier's probability and verdicts (the students learn the filter; the teacher keeps those at 0.2) |
| ``pair_vec`` | the teacher's pair vectors for the verifier's pairs and pairs with a fact above -1 (the 48 best) |
| ``span_vec`` | span words at kept spans' boundaries (at most 32 words) |

Entities: the window's own (student windows); for the stage 4 windows and
the news training texts, the teacher's entity step over the text's types
(end to end), stored under ``entities``. Relation and type indices refer to
the window's schema lists, stored under ``schema`` for sets other than the
student windows. Heads and tails are entity indices; speakers are ``-1-p``.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np

from ..config import DATA_DIR
from .student_targets import OUT, WINDOWS, Dumper

ROUTE = {"ultrabert-380m-v8p3-d123": "ultrabert-380m-v8e-w1half", "deberta-v3-large-v8p3-d123": "deberta-v3-large-v8b"}
S4 = Path("v6/runs/s4-pilot/items.json")
NEWS = Path("v6/runs/m1-precision/preds-ultrabert-380m-v8p3-d123-news-train.pkl")
EXTRA = DATA_DIR / "v6-corpus" / "teacher_label_extra.json"     # the s4 and news windows, for the VM (``extra``)
THRESHOLD, OPEN_KEEP = 0.6, 0.2


def _lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def windows(name: str) -> list[dict]:
    """The labelling set ``name`` as dumper windows (``tokens``, ``sentence_spans``,
    ``speakers``, ``relations``, ``types`` and, for the student windows, ``entities``)."""
    if name == "student":
        return [json.loads(x) for x in WINDOWS.read_text().splitlines()]
    if EXTRA.exists() and not (S4.exists() and NEWS.exists()):
        return json.loads(EXTRA.read_text())[name]
    if name == "s4":
        items = json.loads(S4.read_text())
    elif name == "news":
        items = pickle.loads(NEWS.read_bytes())["items"]
    else:
        raise SystemExit(f"unknown set {name}")
    out = []
    for it in items:
        sp = _lit(it.get("speakers"))
        out.append({"id": it["id"], "tokens": _lit(it["tokens"]), "sentence_spans": _lit(it["sentence_spans"]),
                    "speakers": sp or [], "relations": _lit(it["relations"]), "types": _lit(it["types"]),
                    "domain": it.get("domain")})
    return out


def _name(r):
    return r["name"] if isinstance(r, dict) else r


class Labeller:
    def __init__(self, run: str, ver: str, route: str | None, device: str, release: Path | None = None,
                 local: Path | None = None):
        import sys
        import torch
        if route:
            os.environ["KNIV_UNSEEN_FROM"] = route
        else:
            os.environ.pop("KNIV_UNSEEN_FROM", None)
        os.environ.pop("KNIV_VERIFIER", None)
        from ..infer import Extractor
        self.ex = ex = Extractor(run, device=device, local=local)
        ex.use_verifier(ver, threshold=THRESHOLD)
        ex.trim_det = True
        rel_dir = release or Path("release") / run
        if not (rel_dir / "onnx" / "onnx_meta.json").exists():
            from ..export.release_dir import minimal
            minimal(run, rel_dir)
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from kniv_v7 import Kniv
        from ..export.torch_backend import TorchBackend
        torch.set_grad_enabled(False)
        self.k = Kniv(rel_dir, backend=TorchBackend.load(run, local=local, device=device))
        self.d = Dumper(self.k, vectors=True)
        self.cap: dict = {}
        if ex.unseen_ex is not None:               # the route's relation margins, as its relation step computes them
            m0 = ex.unseen_ex.model
            orig = m0.forward

            def fwd(b, **kw):
                o = orig(b, **kw)
                if kw.get("relation_only"):
                    self.cap["route"] = (o["relation"].float().cpu().numpy(), b["pair_h"].tolist(), b["pair_t"].tolist())
                return o
            m0.forward = fwd

    def window(self, w: dict) -> dict:
        ex = self.ex
        win = {"tokens": w["tokens"], "sentence_spans": w["sentence_spans"]}
        if w.get("speakers") and any(w["speakers"]):
            win["sentence_speakers"] = w["speakers"]
        extra = {}
        if "entities" not in w:                    # end to end: the teacher's entities over the text's types
            ents = ex._entities(win, w.get("types") or None)
            w = dict(w, entities=[{"type": e["type"], "spans": [list(m) for m in e["mentions"]]} for e in ents])
            extra["entities"] = w["entities"]
            from ..infer import trim_articles           # the trim: what the teacher reports as the mentions
            tr = trim_articles(w["tokens"], [tuple(x) for x in w["sentence_spans"]],
                               [{"type": e["type"], "text": "", "spans": [tuple(m) for m in e["spans"]]} for e in w["entities"]])
            extra["entities_trimmed"] = [[list(m) for m in n["spans"]] for n in tr]
            extra["schema"] = {"relations": [_name(r) for r in w.get("relations") or []],
                               "types": [_name(t) for t in w.get("types") or []]}
        rels = [r if isinstance(r, dict) else {"name": r} for r in w.get("relations") or []]
        ix = {r["name"]: i for i, r in enumerate(rels)}
        ver, kept, routed, vec = [], [], [], set()
        if rels and len(w.get("entities") or []) + len(set(s for s in w.get("speakers") or [] if s)) >= 2:
            ents = [{"type": e["type"], "mentions": [tuple(m) for m in e["spans"]]} for e in w["entities"]]
            ex.ver_log = []
            self.cap.clear()
            ex.trim_det = False                         # the trim changes only the reported mentions: untrimmed nodes map back
            try:
                nodes, facts = ex._relations(win, ents, rels, ex.top)
            finally:
                ex.trim_det = True
            parts = list(dict.fromkeys(s for s in win.get("sentence_speakers") or [] if s))
            # nodes: speakers, then the entities with a mention in the window, in order
            ids, ei = [], 0
            for nd in nodes:
                if nd.get("speaker"):
                    ids.append(-1 - parts.index(nd["text"]))
                    continue
                want = {tuple(m) for m in nd["spans"]}
                while ei < len(ents) and not want <= set(ents[ei]["mentions"]):
                    ei += 1
                ids.append(ei if ei < len(ents) else -999)
                ei += 1
            assert -999 not in ids, "node not matched to an entity"
            nid = ids.__getitem__
            for v in ex.ver_log:
                h, t = nid(v["head"]), nid(v["tail"])
                ver.append((h, t, ix[v["relation"]], v["m"], v["p"], v.get("verdict")))
                vec.add((h, t))
            for f in facts:
                kept.append((nid(f["head"]), nid(f["tail"]), ix[f["relation"]]))
            routed = [ix[r["name"]] for r in rels if ex.unseen_ex is not None and r["name"] not in ex.ver_trained
                      and r["name"] not in ex.rel_offsets]
            if "route" in self.cap and routed:
                lg, ph, pt = self.cap["route"]
                names = [r["name"] for r in rels if ix[r["name"]] in routed]      # the routed list, in order
                m = lg[:, 1:] - lg[:, :1]
                q, r = np.nonzero(m > -6.0)
                extra["rel_route"] = (np.asarray([nid(ph[a]) for a in q], dtype=np.int32),
                                      np.asarray([nid(pt[a]) for a in q], dtype=np.int32),
                                      np.asarray([ix[names[b]] for b in r], dtype=np.int16),
                                      m[q, r].astype(np.float16))
        self.d.vec_pairs = vec
        t = self.d.window(w)
        t.update(extra)
        t["ver"], t["kept"], t["routed"] = ver, kept, routed
        t["open"] = self.open_facts(win)
        return t

    def open_facts(self, win: dict) -> list:
        from kniv_v7.open_facts import decode
        L = self.d.last
        o, sent, wm, W, tokens, sp = (L[k] for k in ("o", "sent", "wm", "W", "tokens", "sp"))
        a = {"tokens": tokens, "sentences": [{"start": s, "end": e} for s, e in sp]}
        if win.get("sentence_speakers"):
            for d, who in zip(a["sentences"], win["sentence_speakers"]):
                if who:
                    d["speaker"] = who
        a.update(self.k.linguistic(o, tokens, sp, sent, wm, W))
        a["coref"] = [[list(m) for m in c] for c in self.k.coref(o, sent, wm, W)]
        fs = decode(a)
        if not fs:
            return []
        fs = self.ex.verify_open(win, fs)
        return [{"s": tuple(f["subject"]["span"]), "p": f["predicate"]["text"], "o": tuple(f["object"]["span"]),
                 "src": f["source"], "polarity": f["polarity"], "modality": f["modality"],
                 "verifier": f["verifier"], "verdicts": f.get("verdicts"),
                 "keep": f["verifier"] is not None and f["verifier"] >= OPEN_KEEP} for f in fs]


def extra(a) -> int:
    d = {s: windows(s) for s in ("s4", "news")}
    EXTRA.write_text(json.dumps(d))
    print(f"wrote {EXTRA}: " + ", ".join(f"{k} {len(v):,}" for k, v in d.items()))
    return 0


def dump(a) -> int:
    import torch
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    i, n = (int(x) for x in a.shard.split("/"))
    path = OUT.parent / a.name / a.run / f"{a.set}-shard-{i}-of-{n}.pkl"
    if a.upload and a.skip_done:                # a shard already on the hub (an earlier VM) is not dumped again
        from huggingface_hub import HfApi
        from .gold_eval import HF_REPO
        if HfApi().file_exists(HF_REPO, f"{a.run}/{a.name}/{path.name}"):
            print(f"{path.name}: on the hub, skipped", flush=True)
            return 0
    if dev == "cuda" and a.mem_fraction:
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    lab = Labeller(a.run, a.ver or f"{a.run}-ver3f", ROUTE.get(a.run) if a.route == "auto" else (a.route or None), dev,
                   a.release, a.local)
    ws = windows(a.set)
    ws = ws[i::n][:a.limit] if a.limit else ws[i::n]
    path.parent.mkdir(parents=True, exist_ok=True)
    prog = path.with_suffix(".progress.pkl")
    out, start = {}, 0
    if prog.exists():
        out = pickle.loads(prog.read_bytes())
        start = len(out)
        print(f"resuming at {start:,} from {prog.name}", flush=True)
    t0, peak = time.time(), 0.0
    stats = {"ver": 0, "kept": 0, "open": 0, "open_keep": 0, "route": 0, "oom_retried": 0, "skipped": 0}
    skipped = []
    for c, w in enumerate(ws):
        if c < start:
            continue
        t = None
        for attempt in (0, 1):              # out of memory: free the cache and try the window once more, then skip it
            failed = False
            try:
                t = lab.window(w)
            except torch.OutOfMemoryError:
                failed = True
            if not failed:
                break
            t = None
            if dev == "cuda":
                torch.cuda.empty_cache()
            stats["oom_retried" if attempt == 0 else "skipped"] += 1
        if t is None:
            skipped.append(w["id"])
            print(f"  {w['id']}: skipped, out of memory twice ({len(w['tokens'])} words)", flush=True)
            continue
        out[w["id"]] = t
        stats["ver"] += len(t["ver"]); stats["kept"] += len(t["kept"]); stats["open"] += len(t["open"])
        stats["open_keep"] += sum(f["keep"] for f in t["open"]); stats["route"] += len(t.get("rel_route", [[]])[0])
        if dev == "cuda":
            peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 30)
        if c % 500 == 0 or c == len(ws) - 1:
            print(f"  {c + 1:,}/{len(ws):,} {json.dumps(stats)} peak {peak:.1f} GiB [{time.time() - t0:.0f}s, "
                  f"{(time.time() - t0) / max(c + 1 - start, 1):.2f} s per window]", flush=True)
        if a.save_every and (c + 1) % a.save_every == 0 and c < len(ws) - 1:
            prog.write_bytes(pickle.dumps(out, protocol=5))
    if skipped:
        out["_skipped"] = skipped
    path.write_bytes(pickle.dumps(out, protocol=5))
    prog.unlink(missing_ok=True)
    print(f"wrote {path}: {len(out):,} windows, {path.stat().st_size >> 20} MB [{time.time() - t0:.0f}s]", flush=True)
    if a.upload:
        from huggingface_hub import HfApi
        from .gold_eval import HF_REPO
        HfApi().upload_file(path_or_fileobj=str(path), path_in_repo=f"{a.run}/{a.name}/{path.name}", repo_id=HF_REPO,
                            commit_message=f"{a.run}: v8 teacher labels {path.name}")
        print("uploaded", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--ver", help="verifier run (default RUN-ver3f)")
    d.add_argument("--route", default="auto", help="route run ('auto': the teacher's; '' for none)")
    d.add_argument("--set", default="student", choices=["student", "s4", "news"])
    d.add_argument("--shard", default="0/1")
    d.add_argument("--limit", type=int)
    d.add_argument("--save-every", type=int, default=2000)
    d.add_argument("--device")
    d.add_argument("--release", type=Path)
    d.add_argument("--local", type=Path)
    d.add_argument("--name", default="teacher-labels-v8")
    d.add_argument("--upload", action="store_true")
    d.add_argument("--skip-done", action="store_true", help="with --upload: skip a shard already on the hub")
    d.add_argument("--mem-fraction", type=float, default=0.0, help="this process's share of GPU memory")
    sub.add_parser("extra", help="write the s4 and news windows into the corpus (the VM reads them there)")
    a = ap.parse_args()
    return extra(a) if a.cmd == "extra" else dump(a)


if __name__ == "__main__":
    raise SystemExit(main())
