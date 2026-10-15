"""GLiNER predictions for the span evaluation, run in its own environment so
the project does not depend on it:

    uv run --no-project --with gliner python v6/gliner_baseline.py IN.json OUT.json

IN: {"model": "...", "threshold": 0.5, "sets": {name: {"labels": [...],
"items": [{"tokens": [...]}]}}}. OUT: per set, per item, predicted
``[start, end_exclusive, label]`` in token positions (GLiNER's character
spans mapped back to the whitespace-joined tokens; a span not on token
boundaries is dropped and counted).
"""
import json
import sys

from gliner import GLiNER


def main() -> int:
    spec = json.load(open(sys.argv[1]))
    model = GLiNER.from_pretrained(spec["model"])
    out, dropped = {}, 0
    for name, s in spec["sets"].items():
        preds = []
        for it in s["items"]:
            toks = it["tokens"]
            starts, ends, pos = {}, {}, 0
            for i, t in enumerate(toks):
                starts[pos] = i
                ends[pos + len(t)] = i + 1
                pos += len(t) + 1
            text = " ".join(toks)
            ents = model.predict_entities(text, s["labels"], threshold=spec.get("threshold", 0.5),
                                          flat_ner=True)
            p = []
            for e in ents:
                a, b = starts.get(e["start"]), ends.get(e["end"])
                if a is None or b is None:
                    dropped += 1
                    continue
                p.append([a, b, e["label"]])
            preds.append(p)
        out[name] = preds
        print(f"{name}: {len(preds)} items", flush=True)
    json.dump({"preds": out, "dropped_off_boundary": dropped}, open(sys.argv[2], "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
