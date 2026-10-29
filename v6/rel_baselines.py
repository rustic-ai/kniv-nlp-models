"""GLiREL and GLiNER2 relation predictions for ``v6.train.rel_compare``, run in
their own environments so the project does not depend on them:

    uv run --no-project --python 3.11 --with glirel ... python v6/rel_baselines.py IN.json OUT.json

IN: {"kind": "glirel" | "gliner2", "model": "...", "batch": 8, "device": "mps",
"threshold": 0.0, "jobs": [{"tokens": [...], "labels": [...], "mentions":
[[start, end_exclusive, entity], ...]}]}.

OUT, per job:
- glirel (entities given): ``[head_entity, tail_entity, label, score]``, the
  best mention pair's score for each entity pair and label, scores of at least
  ``min_score`` only;
- gliner2 (finds its own entities): ``[label, head_start, head_end,
  tail_start, tail_end, score]`` in token positions (end exclusive), score =
  head confidence x tail confidence; a span off token boundaries is dropped
  and counted.
"""
import json
import sys
import time


def glirel(spec: dict) -> dict:
    import torch
    from glirel import GLiREL
    model = GLiREL.from_pretrained(spec["model"])
    model.to(torch.device(spec.get("device", "cpu")))
    model.eval()
    min_score = spec.get("min_score", 0.0)
    jobs, out = spec["jobs"], []
    t0 = time.time()
    for i0 in range(0, len(jobs), spec.get("batch", 8)):
        chunk = jobs[i0:i0 + spec.get("batch", 8)]
        by_labels: dict[tuple, list[int]] = {}
        for k, j in enumerate(chunk):
            by_labels.setdefault(tuple(j["labels"]), []).append(k)
        res = [None] * len(chunk)
        for labels, ks in by_labels.items():
            ner = [[[a, b - 1, "ENT", str(e)] for a, b, e in chunk[k]["mentions"]] for k in ks]
            with torch.no_grad():
                pred = model.batch_predict_relations([chunk[k]["tokens"] for k in ks], list(labels),
                                                     threshold=0.0, ner=ner, top_k=-1)
            for k, rels in zip(ks, pred):
                ent = {(a, b): e for a, b, e in chunk[k]["mentions"]}
                best: dict[tuple, float] = {}
                for r in rels:
                    h, t = ent.get(tuple(r["head_pos"])), ent.get(tuple(r["tail_pos"]))
                    if h is None or t is None or h == t or r["score"] < min_score:
                        continue
                    key = (h, t, r["label"])
                    best[key] = max(best.get(key, 0.0), float(r["score"]))
                res[k] = [[h, t, lab, s] for (h, t, lab), s in best.items()]
        out.extend(res)
        if (i0 // spec.get("batch", 8)) % 50 == 0:
            print(f"glirel: {i0 + len(chunk):,}/{len(jobs):,} [{time.time() - t0:.0f}s]", flush=True)
    return {"preds": out}


def gliner2(spec: dict) -> dict:
    import torch
    from gliner2 import GLiNER2
    model = GLiNER2.from_pretrained(spec["model"])
    if spec.get("device", "cpu") != "cpu":
        model.to(torch.device(spec["device"]))
    jobs, out, dropped = spec["jobs"], [None] * len(spec["jobs"]), 0
    bs = spec.get("batch", 8)
    by_labels: dict[tuple, list[int]] = {}
    for k, j in enumerate(jobs):
        by_labels.setdefault(tuple(j["labels"]), []).append(k)
    t0, done = time.time(), 0
    for labels, ks in by_labels.items():
        for i0 in range(0, len(ks), 256):
            chunk = ks[i0:i0 + 256]
            rs = model.batch_extract_relations([" ".join(jobs[k]["tokens"]) for k in chunk], list(labels),
                                               batch_size=bs, threshold=spec.get("threshold", 0.0),
                                               include_confidence=True, include_spans=True)
            for k, r in zip(chunk, rs):
                toks = jobs[k]["tokens"]
                starts, ends, pos = {}, {}, 0
                for i, t in enumerate(toks):
                    starts[pos] = i
                    ends[pos + len(t)] = i + 1
                    pos += len(t) + 1
                p = []
                for lab, pairs in (r.get("relation_extraction") or {}).items():
                    for x in pairs:
                        h, t = x["head"], x["tail"]
                        hs, he = starts.get(h["start"]), ends.get(h["end"])
                        ts, te = starts.get(t["start"]), ends.get(t["end"])
                        if None in (hs, he, ts, te):
                            dropped += 1
                            continue
                        p.append([lab, hs, he, ts, te, float(h["confidence"]) * float(t["confidence"])])
                out[k] = p
            done += len(chunk)
            print(f"gliner2: {done:,}/{len(jobs):,} [{time.time() - t0:.0f}s]", flush=True)
    return {"preds": out, "dropped_off_boundary": dropped}


def main() -> int:
    spec = json.load(open(sys.argv[1]))
    res = {"glirel": glirel, "gliner2": gliner2}[spec["kind"]](spec)
    json.dump(res, open(sys.argv[2], "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
