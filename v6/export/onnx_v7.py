"""Export a v7 run to ONNX and check each graph against PyTorch.

    uv run --extra export python -m v6.export.onnx_v7 --run ultrabert-380m-v7b8ev \
        --local data/v6-eval/checkpoints --out release/ultrabert-380m/onnx [--int8]

Writes ``<graph>.onnx`` (+ ``.onnx.data`` external weights for the encoder
graphs) and ``onnx_meta.json`` (inputs, outputs, label lists, constants).
Then runs every graph in onnxruntime on real windows other than the export
example and compares with the PyTorch graph: max absolute difference per
output, at most ``TOL`` in fp32. ``--int8`` adds dynamically quantised
copies (``<graph>-int8.onnx``) of the graphs with encoders.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..train.data import collate
from ..train.gold_eval import HF_REPO, load_run, phase2_window, window
from .check_graphs import diff, plain_inputs, windows
from ..train.model import NEG
from .graphs import COREF_K, GRAPHS

TOL = 2e-3
OPSET = 17
ENCODER_GRAPHS = ("plain", "label", "marker")

# Per graph: input names, output names, dynamic axes.
SPEC = {
    "plain": (["input_ids", "attention_mask", "first_subword", "word_end", "word_mask",
               "sent_of_word", "sent_ids"],
              ["pos", "morph", "lemma", "ner", "arc", "rel_dep", "rel_head", "srl_pred", "srl_words",
               "cls", "sentiment", "coref_words", "coref_scores", "coref_cand", "ent", "span_words"],
              {"input_ids": {1: "T"}, "attention_mask": {1: "T"}, "first_subword": {1: "W"},
               "word_end": {1: "W"}, "word_mask": {1: "W"}, "sent_of_word": {1: "W"},
               "sent_ids": {0: "S"}, "pos": {1: "W"}, "morph": {1: "W"}, "lemma": {1: "W"},
               "ner": {1: "W"}, "arc": {1: "W", 2: "W1"}, "rel_dep": {1: "W"}, "rel_head": {1: "W1"},
               "srl_pred": {1: "W"}, "srl_words": {1: "W"}, "cls": {1: "S"}, "sentiment": {1: "S"},
               "coref_words": {1: "W"}, "coref_scores": {1: "W"}, "coref_cand": {1: "W"},
               "ent": {1: "W"}, "span_words": {1: "W"}}),
    "dep_rel": (["rel_dep", "rel_head", "head"], ["dep_rel"],
                {"rel_dep": {1: "W"}, "rel_head": {1: "W1"}, "head": {1: "W"}, "dep_rel": {1: "W"}}),
    "srl_roles": (["words", "pred", "mask"], ["roles"],
                  {"words": {0: "F", 1: "L"}, "pred": {0: "F"}, "mask": {0: "F", 1: "L"},
                   "roles": {0: "F", 1: "L"}}),
    "coref_spans": (["coref_words", "heads", "sent_of_word"], ["span_logits", "span_pos"],
                    {"coref_words": {1: "W"}, "heads": {0: "M"}, "sent_of_word": {1: "W"},
                     "span_logits": {0: "M"}, "span_pos": {0: "M"}}),
    "label": (["label_ids", "label_mask"], ["label_emb"],
              {"label_ids": {0: "K", 1: "L"}, "label_mask": {0: "K", 1: "L"}, "label_emb": {0: "K"}}),
    "span": (["span_words", "start", "end", "type_emb"], ["span_logits"],
             {"span_words": {1: "W"}, "start": {0: "N"}, "end": {0: "N"}, "type_emb": {0: "K"},
              "span_logits": {0: "N", 1: "K1"}}),
    "marker": (["rm_input_ids", "rm_attention_mask", "men_pos", "men_ent", "ent_ids", "pair_h",
                "pair_t", "label_emb"], ["logits", "pair", "labels", "ent"],
               {"rm_input_ids": {1: "T"}, "rm_attention_mask": {1: "T"}, "men_pos": {0: "M"},
                "men_ent": {0: "M"}, "ent_ids": {0: "E"}, "pair_h": {0: "P"}, "pair_t": {0: "P"},
                "label_emb": {0: "K"}, "logits": {0: "P", 1: "K1"}, "pair": {0: "P"},
                "labels": {0: "K"}, "ent": {0: "E"}}),
    "attr": (["pair", "labels", "ent"], ["polarity", "modality", "holder", "time"],
             {"pair": {0: "A"}, "labels": {0: "A"}, "ent": {0: "E"}, "polarity": {0: "A"},
              "modality": {0: "A"}, "holder": {0: "A", 1: "E1"}, "time": {0: "A", 1: "E1"}}),
}


@torch.no_grad()
def inputs_for(g: dict, model, tok, spec, vocab, w: dict, rels: list[dict]) -> dict[str, tuple]:
    """Every graph's inputs on window ``w``, computed through the PyTorch
    graphs themselves (as the host runtime will chain them)."""
    from ..train.kdata import TypeCollator, label_text, span_targets
    tokens, spans = w["tokens"], w["spans"]
    f = window(tokens, spans, tok, spec, vocab)
    b = collate([f], tok.pad_token_id)
    pin = plain_inputs(b)
    out = g["plain"](*pin)
    W = b["first_subword"].size(1)
    rel_dep, rel_head, s, cw = out[5], out[6], out[8], out[11]
    head = out[4].argmax(-1)
    F_ = 2
    L = min(W, 12)
    srl = (s[0, :L].unsqueeze(0).expand(F_, -1, -1).contiguous(), torch.tensor([0, L - 1]),
           torch.ones(F_, L, dtype=torch.bool))
    heads = torch.arange(0, W, max(1, W // 5))
    enc = tok([label_text(r) for r in rels], padding=True, truncation=True, max_length=48,
              return_tensors="pt")
    lab = (enc["input_ids"], enc["attention_mask"])
    temb = g["label"](*lab)
    types = ["person", "location", "organization"]
    bank = {"entity_types": [{"id": i, "name": t, "description": ""} for i, t in enumerate(types)]}
    fs = window(tokens, spans, tok, spec, vocab)
    fs.update(coref=None, rel=None, ents=span_targets([], fs["_W"], fs["word_ok"],
                                                      list(range(len(types))), False))
    bs = TypeCollator(tok, bank, train=False)(collate([fs], tok.pad_token_id))
    span = (out[15], bs["span_start"], bs["span_end"], g["label"](bs["type_label_ids"], bs["type_label_mask"]))
    fm = phase2_window(tokens, spans, tok, spec, vocab, w["entities"], markers=True)
    fm["coref"] = None
    bm = collate([fm], tok.pad_token_id)
    E = len(bm["ent_batch"])
    marker = (bm["rm_input_ids"], bm["rm_attention_mask"], bm["rm_men_pos"], bm["men_ent"],
              torch.arange(E), bm["pair_h"], bm["pair_t"], temb)
    _, pr, q, ent = g["marker"](*marker)
    A = min(5, len(pr))
    attr = (pr[:A], q[torch.arange(A) % len(q)], ent)
    return {"plain": pin, "dep_rel": (rel_dep, rel_head, head), "srl_roles": srl,
            "coref_spans": (cw, heads, b["sent_of_word"]), "label": lab, "span": span,
            "marker": marker, "attr": attr}


def coref_diff(sa, ca, sb, cb) -> float:
    """Max score difference over the candidates both sides kept, per word;
    infinite if a word's chosen antecedent (or none) differs."""
    sa, ca = np.asarray(sa)[0], np.asarray(ca)[0]
    sb, cb = torch.as_tensor(sb).numpy()[0], torch.as_tensor(cb).numpy()[0]
    worst = 0.0
    for w in range(len(sa)):
        da = {int(c): float(x) for c, x in zip(ca[w], sa[w, 1:]) if x > NEG / 2}
        db = {int(c): float(x) for c, x in zip(cb[w], sb[w, 1:]) if x > NEG / 2}
        for c in da.keys() & db.keys():
            worst = max(worst, abs(da[c] - db[c]))
        pick = lambda d: max(d, key=d.get) if d and max(d.values()) > 0 else None
        if pick(da) != pick(db):
            return float("inf")
    return worst


def export(run: str, local: Path | None, out: Path, int8: bool) -> int:
    import onnxruntime as ort
    torch.backends.mha.set_fastpath_enabled(False)
    model, tok, spec, vocab, _ = load_run(run, HF_REPO, torch.device("cpu"), local)
    g = {k: v(model).eval() for k, v in GRAPHS.items()}
    from ..label_bank import load as load_bank
    rels = [r for r in load_bank()["relations"] if r["source"] == "redocred"][:12]
    ws = windows()
    example = inputs_for(g, model, tok, spec, vocab, ws[-1], rels)
    out.mkdir(parents=True, exist_ok=True)
    for name, (ins, outs, axes) in SPEC.items():
        path = out / f"{name}.onnx"
        torch.onnx.export(g[name], example[name], path, input_names=ins, output_names=outs,
                          dynamic_axes=axes, opset_version=OPSET, dynamo=False,
                          external_data=name in ENCODER_GRAPHS)
        print(f"exported {path.name}", flush=True)
    # Parity on the other windows (not the example the trace saw).
    sess = {k: ort.InferenceSession(str(out / f"{k}.onnx"), providers=["CPUExecutionProvider"])
            for k in SPEC}
    worst: dict[str, float] = {}
    for w in ws[:-1]:
        inp = inputs_for(g, model, tok, spec, vocab, w, rels)
        for name, (ins, outs, _) in SPEC.items():
            with torch.no_grad():
                ref = g[name](*inp[name])
            ref = ref if isinstance(ref, tuple) else (ref,)
            want = {i.name for i in sess[name].get_inputs()}   # an unused input is dropped
            feed = {k: v.numpy() for k, v in zip(ins, inp[name]) if k in want}
            got = sess[name].run(None, feed)
            for o, a, r in zip(outs, got, ref):
                if o in ("coref_scores", "coref_cand"):
                    # Near-tied candidates may rank in either order: compare
                    # each word's candidate -> score map, and its choice.
                    d = coref_diff(got[outs.index("coref_scores")], got[outs.index("coref_cand")],
                                   ref[outs.index("coref_scores")], ref[outs.index("coref_cand")])
                else:
                    d = diff(torch.from_numpy(np.asarray(a)), r)
                worst[f"{name}.{o}"] = max(worst.get(f"{name}.{o}", 0.0), d)
    bad = {k: v for k, v in worst.items() if not v <= TOL}
    for k, v in sorted(worst.items()):
        print(f"{'FAIL' if k in bad else 'ok  '} {k:28s} max |diff| {v:.2e}")
    meta = {"run": run, "encoder": spec.hf_id, "max_subwords": spec.max_subwords,
            "word_pool": model.word_pool, "coref_k": COREF_K, "opset": OPSET,
            "graphs": {k: {"inputs": [i.name for i in sess[k].get_inputs()], "outputs": v[1]}
                       for k, v in SPEC.items()},
            "pad_token_id": tok.pad_token_id, "sep_token_id": tok.sep_token_id,
            "labels": vocab.labels, "parity_max_abs_diff": worst}
    (out / "onnx_meta.json").write_text(json.dumps(meta, indent=1))
    if int8 and not bad:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        for name in ENCODER_GRAPHS:
            quantize_dynamic(str(out / f"{name}.onnx"), str(out / f"{name}-int8.onnx"),
                             weight_type=QuantType.QInt8, use_external_data_format=False)
            print(f"quantised {name}-int8.onnx", flush=True)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    return export(args.run, args.local, args.out, args.int8)


if __name__ == "__main__":
    raise SystemExit(main())
