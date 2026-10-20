"""One teacher's outputs on the student windows (docs/v7-students-plan.md,
step 3), compressed, for distillation.

    python -m v6.train.student_targets dump --run R [--shard i/n] [--limit N] [--upload]
    (on an A100: colab post --jobs st, one VM per teacher and shard)

Per window (``student_windows.jsonl``), every head, in word positions of
the window (``W``: words the teacher's encoder read):

| key | content |
|---|---|
| ``pos`` | ``[W, k]`` top-3 classes and probabilities |
| ``morph``, ``lemma``, ``ner``, ``ent`` | top-5 |
| ``arc`` | top-8 heads (0 = root, j+1 = word j) and probabilities |
| ``deprel`` | top-5 labels at the teacher's tree head (``tree_head``) |
| ``srl_pred`` | ``[W]`` predicate probability |
| ``srl`` | per predicate with probability > 0.3: top-3 roles per word of its sentence |
| ``cls``, ``sentiment`` | per sentence: sigmoid / probabilities |
| ``coref`` | per word: top-8 antecedents (-1 = none) and probabilities |
| ``span`` | with a type schema: margins above ``FLOOR`` per (span, type) |
| ``rel`` | with entities and a relation schema: margins above ``FLOOR`` per (head, tail, relation), entities by their index in the window's list (speakers ``-1-p``) |
| ``attr`` | for facts with margin > -1: polarity, modality, holder and time distributions |

Classes are the vocabulary's indices; relations and types index the
window's own schema lists.
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from ..config import DATA_DIR

WINDOWS = DATA_DIR / "v6-corpus" / "student_windows.jsonl"
OUT = DATA_DIR / "v6-eval" / "student-targets"
FLOOR = -6.0
TOPK = {"pos": 3, "morph": 5, "lemma": 5, "ner": 5, "ent": 5, "arc": 8, "deprel": 5}


def softmax(x, axis=-1):
    x = x.astype(np.float64)
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def topk(p: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, p.shape[-1])
    idx = np.argsort(-p, -1)[..., :k]
    return idx.astype(np.int16), np.take_along_axis(p, idx, -1).astype(np.float16)


class Dumper:
    def __init__(self, k):
        self.k = k                       # a kniv_v7.Kniv over the teacher's graphs

    def window(self, w: dict) -> dict:
        from kniv_v7.runtime import (align, decode_trees, label_text, marker_input, span_candidates,
                                     speaker_turns, type_text, word_ends)
        k = self.k
        tokens, sp = w["tokens"], [tuple(x) for x in w["sentence_spans"]]
        ids, first, W = align(tokens, k.tok, k.meta["max_subwords"])
        sent = np.full(W, -1, dtype=np.int64)
        for j, (s, e) in enumerate(sp):
            sent[s:min(e, W)] = j
        first = np.asarray(first, dtype=np.int64)
        wm = first >= 0
        x = {"input_ids": np.asarray([ids], dtype=np.int64),
             "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
             "first_subword": np.maximum(first, 0)[None], "word_end": word_ends(first, len(ids))[None],
             "word_mask": wm[None], "sent_of_word": sent[None],
             "sent_ids": np.arange(len(sp), dtype=np.int64)}
        o = dict(zip(k.meta["graphs"]["plain"]["outputs"], k.run("plain", **x)))
        t = {"W": W}
        for h in ("pos", "morph", "lemma", "ner", "ent"):
            t[h] = topk(softmax(o[h][0]), TOPK[h])
        arc = softmax(o["arc"][0])
        t["arc"] = topk(arc, TOPK["arc"])
        head = decode_trees(np.log(np.clip(arc, 1e-30, None)), sent, wm)
        lab = softmax(k.run("dep_rel", rel_dep=o["rel_dep"], rel_head=o["rel_head"],
                            head=head[None].astype(np.int64))[0][0])
        t["tree_head"] = head.astype(np.int16)
        t["deprel"] = topk(lab, TOPK["deprel"])
        pp = softmax(o["srl_pred"][0])[:, 1]
        t["srl_pred"] = pp.astype(np.float16)
        t["srl"] = self.srl(o, sent, wm, [int(p) for p in np.nonzero((pp > 0.3) & wm)[0]])
        t["cls"] = (1 / (1 + np.exp(-o["cls"][0].astype(np.float64)))).astype(np.float16)
        t["sentiment"] = softmax(o["sentiment"][0]).astype(np.float16)
        sc = o["coref_scores"][0].astype(np.float64)
        cand = np.concatenate([np.full((W, 1), -1), o["coref_cand"][0]], 1)
        valid = sc > -5e3
        p = softmax(np.where(valid, sc, -1e9))
        idx = np.argsort(-p, 1)[:, :8]
        t["coref"] = (np.take_along_axis(cand, idx, 1).astype(np.int16),
                      np.take_along_axis(p, idx, 1).astype(np.float16))
        if w.get("types"):
            cands = span_candidates(sent, wm)
            if cands:
                temb = k.label_emb([type_text({"name": n}) for n in w["types"]])
                lg = k.run("span", span_words=o["span_words"],
                           start=np.asarray([s for s, _ in cands], dtype=np.int64),
                           end=np.asarray([e for _, e in cands], dtype=np.int64), type_emb=temb)[0]
                m = lg[:, 1:] - lg[:, :1]
                i, j = np.nonzero(m > FLOOR)
                c = np.asarray(cands, dtype=np.int32)
                t["span"] = (c[i, 0].astype(np.int16), c[i, 1].astype(np.int16), j.astype(np.int16),
                             m[i, j].astype(np.float16))
        if w.get("entities") and w.get("relations"):
            t.update(self.relations(w, tokens, sp, first, W))
        return t

    def srl(self, o, sent, wm, preds):
        if not preds:
            return []
        s = o["srl_words"][0]
        same = [(sent == sent[p]) & wm & (sent[p] >= 0) for p in preds]
        start = [int(np.argmax(m)) for m in same]
        end = [len(m) - 1 - int(np.argmax(m[::-1])) for m in same]
        n = [e - a + 1 for a, e in zip(start, end)]
        L = max(n)
        words = np.zeros((len(preds), L, s.shape[-1]), dtype=np.float32)
        mask = np.zeros((len(preds), L), dtype=bool)
        for f, (a, c) in enumerate(zip(start, n)):
            words[f, :c] = s[a:a + c]
            mask[f, :c] = True
        lg = self.k.run("srl_roles", words=words, pred=np.asarray([p - a for p, a in zip(preds, start)],
                                                                  dtype=np.int64), mask=mask)[0]
        out = []
        for f, p in enumerate(preds):
            pr = softmax(lg[f, :n[f]])
            out.append((p, start[f], *topk(pr, 3)))
        return out

    def relations(self, w, tokens, sp, first, W) -> dict:
        from kniv_v7.runtime import label_text, marker_input, speaker_turns
        k = self.k
        ok = [f >= 0 for f in first[:W]]
        speakers = w.get("speakers") or []
        parts = list(dict.fromkeys(s for s in speakers if s))
        turns = [(s, p) for s, p in speaker_turns(sp, speakers, parts) if s < W and ok[s]]
        ids_, spans = [], []
        for p, _ in enumerate(parts):
            ms = [(s, s + 1) for s, q in turns if q == p]
            if ms:
                ids_.append(-1 - p)
                spans.append(ms)
        n_part = len(spans)
        for ei, e in enumerate(w["entities"]):
            ms = [(int(a), int(b)) for a, b in e["spans"] if b <= W and ok[a]]
            if ms:
                ids_.append(ei)
                spans.append(ms)
        if len(spans) < 2:
            return {}
        part_turns = [(s, p) for s, p in turns if p < n_part]
        rid, men_pos = marker_input(tokens, sp, spans[n_part:], k.tok, turns=part_turns,
                                    participants=parts[:n_part])
        E = len(spans)
        pairs = [(h, t) for h in range(E) for t in range(E) if h != t]
        temb = k.label_emb([label_text({"name": n}) for n in w["relations"]])
        logits, pair, labels, entv = k.run(
            "marker", rm_input_ids=np.asarray([rid], dtype=np.int64),
            rm_attention_mask=np.ones((1, len(rid)), dtype=np.int64),
            men_pos=np.asarray([x for kk in range(E) for x in men_pos[kk]], dtype=np.int64),
            men_ent=np.asarray([kk for kk, ms in enumerate(spans) for _ in ms], dtype=np.int64),
            ent_ids=np.arange(E, dtype=np.int64),
            pair_h=np.asarray([h for h, _ in pairs], dtype=np.int64),
            pair_t=np.asarray([t for _, t in pairs], dtype=np.int64), label_emb=temb)
        m = logits[:, 1:] - logits[:, :1]
        pi, r = np.nonzero(m > FLOOR)
        hp = np.asarray([ids_[pairs[q][0]] for q in pi], dtype=np.int32)
        tp = np.asarray([ids_[pairs[q][1]] for q in pi], dtype=np.int32)
        out = {"rel": (hp, tp, r.astype(np.int16), m[pi, r].astype(np.float16)),
               "rel_ents": np.asarray(ids_, dtype=np.int32)}
        a_i, a_r = np.nonzero(m > -1.0)
        if len(a_i):
            pol, mod, hold, tim = k.run("attr", pair=pair[a_i], labels=labels[a_r], ent=entv)
            to_id = lambda d: [(-99 if c == 0 else int(ids_[c - 1])) for c in range(d.shape[1])]
            out["attr"] = {"h": np.asarray([ids_[pairs[q][0]] for q in a_i], dtype=np.int32),
                           "t": np.asarray([ids_[pairs[q][1]] for q in a_i], dtype=np.int32),
                           "r": a_r.astype(np.int16),
                           "polarity": softmax(pol).astype(np.float16),
                           "modality": softmax(mod).astype(np.float16),
                           "holder": softmax(hold).astype(np.float16),
                           "time": softmax(tim).astype(np.float16),
                           "pointer_ids": np.asarray(to_id(hold), dtype=np.int32)}   # -99 = none
        return out


def dump(args) -> int:
    import sys
    import torch
    release = {"ultrabert-380m-v7b8ev": "release/ultrabert-380m",
               "deberta-v3-large-v7b8ev": "release/deberta-v3-large"}
    rel_dir = Path(args.release or release[args.run])
    sys.path.insert(0, str(rel_dir))
    from kniv_v7 import Kniv
    from ..export.torch_backend import TorchBackend
    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = Kniv(rel_dir, backend=TorchBackend.load(args.run, local=args.local, device=dev))
    d = Dumper(k)
    lines = WINDOWS.read_text().splitlines()
    i, n = (int(x) for x in args.shard.split("/"))
    lines = lines[i::n][:args.limit] if args.limit else lines[i::n]
    out, t0 = {}, time.time()
    for c, line in enumerate(lines):
        w = json.loads(line)
        out[w["id"]] = d.window(w)
        if c % 2000 == 0:
            print(f"  {c:,}/{len(lines):,} [{time.time() - t0:.0f}s]", flush=True)
    path = OUT / args.run / f"shard-{i}-of-{n}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out, protocol=5))
    print(f"wrote {path}: {len(out):,} windows, {path.stat().st_size >> 20} MB [{time.time() - t0:.0f}s]",
          flush=True)
    if args.upload:
        from huggingface_hub import HfApi
        from .gold_eval import HF_REPO
        HfApi().upload_file(path_or_fileobj=str(path),
                            path_in_repo=f"{args.run}/student-targets/{path.name}", repo_id=HF_REPO,
                            commit_message=f"{args.run}: student targets {path.name}")
        print("uploaded", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--run", required=True)
    d.add_argument("--release")
    d.add_argument("--local", type=Path)
    d.add_argument("--shard", default="0/1")
    d.add_argument("--limit", type=int)
    d.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    return dump(args)


if __name__ == "__main__":
    raise SystemExit(main())
