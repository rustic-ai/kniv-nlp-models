"""kniv v7 quick start: a conversation and a paragraph, ONNX on CPU.

    python kniv_v7/examples/quickstart.py <release directory>
"""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
sys.path.insert(0, str(root))
from kniv_v7 import Kniv  # noqa: E402

kniv = Kniv(root)
relations = [{"name": "lives in", "description": "the place where the head resides"},
             {"name": "employer", "description": "the organization the head works for"}]

out = kniv.extract(turns=[("user", "I live in Berlin and work for Siemens."),
                          ("assistant", "Nice, how long have you been there?")],
                   relations=relations,
                   entity_types=[{"name": "location"}, {"name": "organization"}, {"name": "person"}])
for w in out:
    ents = w["entities"]
    for f in w["facts"]:
        print(f"{ents[f['head']]['text']} --{f['relation']}--> {ents[f['tail']]['text']}"
              f"  ({f['polarity']}, {f['modality']}, margin {f['margin']})")

full = kniv.analyse("Ada Lovelace was born in London in 1815. She worked with Charles Babbage.")
w = full[0]
print(" ".join(f"{t}/{x['pos']}" for t, x in zip(w["tokens"], w["words"])))
print("NER:", w["ner"])
print("SRL:", json.dumps(w["srl"]))
print("coref:", w["coref"])
