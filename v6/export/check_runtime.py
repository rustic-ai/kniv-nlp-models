"""End-to-end checks of the ``kniv_v7`` runtime on a release directory.

    uv run --extra export python -m v6.export.check_runtime --run ultrabert-380m-v7b8ev \
        --release release/ultrabert-380m

1. **Tokenizer**: ``tokenizers`` with the shipped ``tokenizer.json`` aligns
   words to subwords exactly as the training code's tokenizer did.
2. **Linguistic decoding**: on UD English EWT and PropBank test sentences,
   the runtime (PyTorch graphs) gives the same POS, lemma, morphology, heads,
   labels, NER spans and SRL frames as the scorecard's own decoding.
3. **Extraction**: on prose and conversations, the runtime's entities and
   facts equal ``v6.infer.Extractor``'s (which has no attributes).
4. **ONNX**: the runtime over the ONNX graphs gives the same decoded output
   as over the PyTorch graphs (fp32), and reports how often INT8 differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

PROSE = [
    "Ada Lovelace was born in London in 1815. She worked with Charles Babbage on the Analytical Engine.",
    "Apple announced on Tuesday that Tim Cook will open a new office in Austin, Texas, next spring.",
    "The committee did not approve the merger between Siemens and Alstom, citing competition concerns.",
    "Marie Curie, who won the Nobel Prize twice, was born in Warsaw and later moved to Paris.",
    "If the vendor misses the deadline, we may cancel the contract and request a refund.",
    "The Danube flows through Vienna, Budapest and Belgrade before reaching the Black Sea.",
    "According to Reuters, the central bank raised interest rates by half a point in March.",
    "Python 3.12 removed the distutils module, which many older packages still import.",
    "Dr. Okafor said the trial would start in June, but the hospital has not confirmed it.",
    "The novel follows Elizabeth Bennet, whose sister Jane falls in love with Mr. Bingley.\n\n"
    "Later, Elizabeth visits Pemberley, the estate of Mr. Darcy in Derbyshire.",
    "Tesla's factory in Berlin employs about 11,000 people. It opened in 2022 after long delays.",
    "We are planning to move our servers from AWS to Google Cloud by the end of the year.",
]
TURNS = [
    [("user", "I moved to Lisbon last year and started working at a bank."),
     ("assistant", "That sounds exciting! How are you finding Lisbon?"),
     ("user", "I love it, although my sister Ana still lives in Porto.")],
    [("user", "My daughter Maya starts school in September."),
     ("assistant", "Which school did you pick?"),
     ("user", "Riverside Primary, it's close to our flat in Leeds.")],
]
RELATIONS = [
    {"name": "place of birth", "description": "the place where the head was born"},
    {"name": "employer", "description": "the organization the head works for"},
    {"name": "lives in", "description": "the place where the head resides"},
    {"name": "sibling", "description": "the head and tail share a parent"},
    {"name": "child", "description": "the tail is a child of the head"},
    {"name": "located in", "description": "the head is located in the tail place"},
    {"name": "founded by", "description": "the tail founded the head"},
]


def strip_attrs(out):
    return [{"entities": w["entities"],
             "facts": [{k: f[k] for k in ("head", "relation", "tail")} for f in w["facts"]]}
            for w in out]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--release", type=Path, required=True)
    ap.add_argument("--n-ud", type=int, default=60)
    args = ap.parse_args()
    import sys
    sys.path.insert(0, str(args.release))
    from kniv_v7 import Kniv
    from kniv_v7.runtime import Tok
    from kniv_v7.runtime import align as rt_align
    from ..train.data import align as tr_align
    from ..train.gold_eval import (HF_REPO, bio_constraints, load_run, run_model, viterbi,
                                   window)
    from ..train.vocab import apply_script
    from .torch_backend import TorchBackend
    torch.backends.mha.set_fastpath_enabled(False)
    model, tok, spec, vocab, _ = load_run(args.run, HF_REPO, torch.device("cpu"),
                                          Path("data/v6-eval/checkpoints"))
    tb = TorchBackend(model)
    kt = Kniv(args.release, backend=tb)
    report, ok = {}, True

    # 1. tokenizer
    from ..gold.ud_ewt import load_ud_items
    items = load_ud_items("test", limit=args.n_ud)
    rt = Tok(args.release / "tokenizer.json", tok.pad_token_id, tok.sep_token_id)
    word_lists = [it.tokens for it in items] + [w["tokens"] for t in PROSE
                                                 for w in __import__("kniv_v7").windows(t)]
    bad = sum(tr_align(ws, tok, spec.max_subwords) != rt_align(ws, rt, spec.max_subwords)
              for ws in word_lists)
    report["tokenizer_mismatch"] = f"{bad}/{len(word_lists)}"
    ok &= bad == 0

    # 2. linguistic decoding on UD test sentences
    feats = [window(it.tokens, [(0, len(it.tokens))], tok, spec, vocab) for it in items]
    ref = run_model(model, feats, tok, torch.device("cpu"))
    L = vocab.labels
    nal, nas = bio_constraints(L["ner"])
    mism = {k: 0 for k in ("pos", "lemma", "morph", "head", "deprel", "ner")}
    for it, o in zip(items, ref):
        got = kt.window({"tokens": it.tokens, "sentence_spans": [[0, len(it.tokens)]]}, [], None,
                        True, 4)
        ws = got["words"]
        mism["pos"] += sum(w["pos"] != L["pos"][x] for w, x in zip(ws, o["pos"]))
        mism["lemma"] += sum(w["lemma"] != apply_script(t, L["lemma"][x])
                             for w, t, x in zip(ws, it.tokens, o["lemma"]))
        mism["morph"] += sum(w["morph"] != L["morph"][x] for w, x in zip(ws, o["morph"]))
        mism["head"] += sum(w["head"] != int(x) for w, x in zip(ws, o["head"]))
        mism["deprel"] += sum(w["deprel"] != L["dep"][x] for w, x in zip(ws, o["rel"]))
        from ..entities import bio_spans
        ner = bio_spans([L["ner"][x] for x in viterbi(o["ner"], nal, nas)])
        mism["ner"] += int(sorted(map(tuple, got["ner"])) != sorted(ner))
    # SRL frames: the scorecard's end-to-end decoding (eval_srl_full): the
    # model's predicates, then roles for them.
    tags = L["srl"]
    sal, sas = bio_constraints(tags)
    v = tags.index("V")
    found = [sorted(np.nonzero(o["srl_pred"] == 1)[0].tolist()) for o in ref]
    feats2 = [window(it.tokens, [(0, len(it.tokens))], tok, spec, vocab, predicates=fp)
              for it, fp in zip(items, found)]
    ref2 = run_model(model, feats2, tok, torch.device("cpu"))
    mism["srl"] = 0
    from ..entities import bio_spans as spans_of
    for it, fp, o2 in zip(items, found, ref2):
        want = []
        for p, lp in zip(fp, o2["srl"]):
            keep = np.ones_like(lp, dtype=bool)
            keep[:, v] = False
            keep[p, :] = False
            keep[p, v] = True
            path = [tags[x] for x in viterbi(lp, sal, sas, keep)]
            want.append({"predicate": p, "args": [list(x) for x in spans_of(path) if x[0] != "V"]})
        got = kt.window({"tokens": it.tokens, "sentence_spans": [[0, len(it.tokens)]]}, [], None,
                        True, 4)["srl"]
        mism["srl"] += int(got != want)
    report["linguistic_mismatch"] = mism
    ok &= not any(mism.values())

    # 3. extraction against v6.infer.Extractor
    from ..infer import Extractor
    ex = Extractor(run=args.run, device="cpu")
    ex.frozen_path = False
    diffs = 0
    cases = [{"text": t} for t in PROSE] + [{"turns": t} for t in TURNS]
    for c in cases:
        a = strip_attrs(kt.extract(**c, relations=RELATIONS))
        b = strip_attrs(ex.extract(**c, relations=RELATIONS))
        b = [{"entities": [{k: v for k, v in e.items()} for e in w["entities"]], "facts": w["facts"]}
             for w in b]
        for x, y in zip(a, b):
            ex_ents = [{"type": e["type"], "text": e["text"]} for e in y["entities"]]
            rt_ents = [{"type": e["type"], "text": e["text"]} for e in x["entities"]]
            diffs += int(ex_ents != rt_ents or x["facts"] != y["facts"])
    report["extractor_mismatch"] = f"{diffs}/{len(cases)}"
    ok &= diffs == 0

    # 4. ONNX (fp32, then INT8) against the PyTorch graphs
    for int8 in (False, True):
        ko = Kniv(args.release, int8=int8)
        if int8 and not list((args.release / "onnx").glob("*-int8.onnx")):
            continue
        n = same = 0
        for c in cases:
            a = kt.analyse(**c, relations=RELATIONS)
            b = ko.analyse(**c, relations=RELATIONS)
            for x, y in zip(a, b):
                n += 1
                same += json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True)
        report["onnx_int8_same" if int8 else "onnx_fp32_same"] = f"{same}/{n}"
        if not int8:
            ok &= same == n
    print(json.dumps(report, indent=1))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
