"""Assemble a release directory for a run (everything but the model card).

    uv run --extra export python -m v6.export.release_dir --run ultrabert-380m-v7b8ev \
        --out release/ultrabert-380m [--onnx]

Writes ``model.pt`` (weights, vocabularies, encoder settings, model config),
``config.json``, ``calibration.json`` (from the runs repo), ``label_bank.json``,
the tokenizer files and the ``kniv_v7`` runtime package; ``--onnx`` also runs
the ONNX export into ``<out>/onnx`` (fp32).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch

from ..train.gold_eval import CKPT_DIR, HF_REPO

REPO = Path(__file__).resolve().parents[2]
# model_from_checkpoint's construction arguments
CONSTRUCT = {"srl_mode", "phase2", "p2_context", "ent_head", "rel_adapter", "rel_markers", "rel_head",
             "label_plain", "legacy_relation", "span_head", "label_adapter", "legacy_rank", "word_pool",
             "attr_head", "srl_pred_hidden", "srl_ctx", "linker", "ner_lstm"}
ENCODER_KEYS = {"name", "hf_id", "max_subwords", "attn_implementation", "trust_remote_code"}


SOURCE_NOTE = re.compile(r"Wikidata|Wikipedia|DocRED|FewRel|Few-NERD|T-REx|\bP\d{2,6}\b|\bQ\d{3,}\b", re.I)


def clean(text: str) -> str:
    """A description without its source's editorial notes: bracketed notes,
    and clauses that cite property or item ids."""
    text = re.sub(r"\s*\[[^\]]*\]", "", text)
    text = re.sub(r"\s*\((?:[^()]*\bP\d+\b[^()]*)\)", "", text)
    parts = [p.strip() for p in re.split(r"(?<=[.;])\s+", text) if p.strip()]
    keep = [p for p in parts if not SOURCE_NOTE.search(p)]
    out = " ".join(keep).strip(" ;,-")
    return re.sub(r"\s{2,}", " ", out)


def public_bank() -> dict:
    """The label bank as released: names, descriptions, aliases, and whether a
    relation has its own calibrated threshold offset (no sources)."""
    bank = json.loads((REPO / "data" / "v6-corpus" / "label_bank.json").read_text())
    rel = [{"name": r["name"], "description": clean(r.get("description", "")),
            **({"aliases": [a for a in r["aliases"] if not SOURCE_NOTE.search(a)]}
               if r.get("aliases") else {}),
            **({"calibrated": True} if r.get("source") == "redocred" else {})}
           for r in bank["relations"]]
    types = [{"name": t["name"], "description": clean(t.get("description", ""))}
             for t in bank["entity_types"]]
    return {"relations": rel, "entity_types": types}


def assemble(run: str, out: Path, onnx: bool) -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT_DIR / run / "best.pt", map_location="cpu", weights_only=False)
    # Only what builds the model: run names, data mixes and holdouts stay internal.
    cfg = {k: v for k, v in ck["model_config"].items() if k in CONSTRUCT}
    enc = {k: v for k, v in ck["encoder"].items() if k in ENCODER_KEYS}
    torch.save({"model": ck["model"], "vocab": ck["vocab"], "encoder": ck["encoder"],
                "model_config": cfg}, out / "model.pt")
    (out / "config.json").write_text(json.dumps({"encoder": enc, "model_config": cfg}, indent=1))
    cal = json.loads(Path(hf_hub_download(HF_REPO, f"{run}/calibration.json")).read_text())
    cal.pop("notes", None)
    (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    (out / "label_bank.json").write_text(json.dumps(public_bank(), indent=1))
    tok = AutoTokenizer.from_pretrained(ck["encoder"]["hf_id"],
                                        trust_remote_code=ck["encoder"].get("trust_remote_code", False))
    tok.save_pretrained(out)
    guide = (REPO / "docs" / "v7-strengths-and-workarounds.md").read_text()
    (out / "GUIDE.md").write_text(guide.replace(
        "(v7-stage-b.md)", "(https://github.com/rustic-ai/kniv-nlp-models/blob/main/docs/v7-stage-b.md)"))
    pkg = out / "kniv_v7"
    if pkg.exists():
        shutil.rmtree(pkg)
    shutil.copytree(REPO / "kniv_v7", pkg, ignore=shutil.ignore_patterns("__pycache__"))
    if onnx:
        from .onnx_v7 import export
        if export(run, CKPT_DIR, out / "onnx", int8=False):
            raise SystemExit("ONNX parity failed")


def minimal(run: str, out: Path) -> Path:
    """What the runtime needs to run ``run`` on PyTorch graphs (no ONNX):
    ``onnx/onnx_meta.json`` (graph inputs and outputs, labels, tokenizer
    ids), tokenizer, calibration and label bank. For VMs that run teachers."""
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from .onnx_v7 import SPEC
    ck = torch.load(Path(hf_hub_download(HF_REPO, f"{run}/best.pt", local_dir=str(CKPT_DIR))),
                    map_location="cpu", weights_only=False)
    enc = ck["encoder"]
    tok = AutoTokenizer.from_pretrained(enc["hf_id"], trust_remote_code=enc.get("trust_remote_code", False))
    (out / "onnx").mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out)
    meta = {"run": run, "encoder": enc["hf_id"], "max_subwords": enc["max_subwords"],
            "graphs": {k: {"inputs": v[0], "outputs": v[1]} for k, v in SPEC.items()},
            "labels": ck["vocab"]["labels"], "pad_token_id": tok.pad_token_id,
            "sep_token_id": tok.sep_token_id}
    (out / "onnx" / "onnx_meta.json").write_text(json.dumps(meta))
    cal = json.loads(Path(hf_hub_download(HF_REPO, f"{run}/calibration.json")).read_text())
    cal.pop("notes", None)
    (out / "calibration.json").write_text(json.dumps(cal))
    (out / "label_bank.json").write_text(json.dumps(public_bank()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--onnx", action="store_true")
    args = ap.parse_args()
    assemble(args.run, args.out, args.onnx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
