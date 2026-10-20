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
import shutil
from pathlib import Path

import torch

from ..train.gold_eval import CKPT_DIR, HF_REPO

REPO = Path(__file__).resolve().parents[2]


def assemble(run: str, out: Path, onnx: bool) -> None:
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CKPT_DIR / run / "best.pt", map_location="cpu", weights_only=False)
    torch.save({"model": ck["model"], "vocab": ck["vocab"], "encoder": ck["encoder"],
                "model_config": ck["model_config"]}, out / "model.pt")
    (out / "config.json").write_text(json.dumps(
        {"run": run, "encoder": ck["encoder"], "model_config": ck["model_config"]}, indent=1))
    cal = json.loads(Path(hf_hub_download(HF_REPO, f"{run}/calibration.json")).read_text())
    cal.pop("notes", None)
    (out / "calibration.json").write_text(json.dumps(cal, indent=1))
    shutil.copy(REPO / "data" / "v6-corpus" / "label_bank.json", out / "label_bank.json")
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
