"""End-to-end v2 distillation pipeline driver.

Runs the published recipe (Stage 1 → Stage 2a → Stage 2b) for one of the
three preset student sizes — xsmall, small, base — using the exact
hyperparameters that produced the published HuggingFace checkpoints.

The actual training implementation lives in ``shared/student_train.py``;
this script is a thin convenience wrapper that:

* picks the right HF encoder id, batch size, and learning rates per size
* points to the canonical local data directories
* emits one ``models/<dir>/{model.pt, metadata.json, trial_result.json}``

Usage:

    uv run python scripts/run_student_pipeline.py xsmall \\
        --output models/kniv-deberta-nlp-base-en-xsmall

    # Stage 1 only (e.g., to dry-run without the Stage 2 datasets):
    uv run python scripts/run_student_pipeline.py xsmall \\
        --output models/xsmall-trial --stages 1

    # Resume from an existing Stage 1 checkpoint:
    uv run python scripts/run_student_pipeline.py xsmall \\
        --output models/xsmall-trial --stages 2a,2b

Per-size presets are sourced from the v2 Colab training notebook in
``docs/Copy of kniv-newdstl-student-train.ipynb`` cells 8/12/13. To
override a preset, pass the matching flag explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from shared.hpo_trial import TrialParams  # noqa: E402
from shared.student_train import StageConfig, run_full_pipeline  # noqa: E402


@dataclass(frozen=True)
class SizePreset:
    encoder: str
    batch_size: int
    lr_encoder: float
    lr_heads: float
    lr_encoder_s2a: float
    lr_cls_s2b: float


# Hyperparameters that produced the published v2 checkpoints. Larger
# encoders use smaller learning rates (more sensitive) and smaller batches
# (memory).
SIZE_PRESETS: dict[str, SizePreset] = {
    "xsmall": SizePreset(
        encoder="microsoft/deberta-v3-xsmall",
        batch_size=64, lr_encoder=2e-5, lr_heads=1e-3,
        lr_encoder_s2a=3e-6, lr_cls_s2b=5e-4,
    ),
    "small": SizePreset(
        encoder="microsoft/deberta-v3-small",
        batch_size=32, lr_encoder=1e-5, lr_heads=5e-4,
        lr_encoder_s2a=2e-6, lr_cls_s2b=3e-4,
    ),
    "base": SizePreset(
        encoder="microsoft/deberta-v3-base",
        batch_size=16, lr_encoder=8e-6, lr_heads=3e-4,
        lr_encoder_s2a=1.5e-6, lr_cls_s2b=2e-4,
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("size", choices=sorted(SIZE_PRESETS),
                    help="Which student size preset to use")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (contains model.pt + metadata.json)")
    ap.add_argument("--distillation-data", type=Path,
                    default=REPO / "data" / "distillation",
                    help="Directory containing shard_*.parquet")
    ap.add_argument("--stage2-data", type=Path,
                    default=REPO / "data" / "prepared" / "kniv-deberta-cascade",
                    help="Directory with srl_train/dev.json + cls_sgd_mwoz_train.json")
    ap.add_argument("--stages", default="1,2a,2b",
                    help="Comma-separated subset of {1, 2a, 2b}")
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs-s1", type=int, default=8)
    ap.add_argument("--epochs-s2a", type=int, default=8)
    ap.add_argument("--epochs-s2b", type=int, default=4)
    ap.add_argument("--silver-limit", type=int, default=80_000)
    ap.add_argument("--no-pkd", action="store_true",
                    help="Disable PKD-style hidden-state distillation")
    ap.add_argument("--no-rdrop", action="store_true",
                    help="Disable R-Drop second-forward consistency loss")

    # Per-size overrides (default = preset)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr-encoder", type=float, default=None)
    ap.add_argument("--lr-heads", type=float, default=None)
    ap.add_argument("--lr-encoder-s2a", type=float, default=None)
    ap.add_argument("--lr-cls-s2b", type=float, default=None)

    args = ap.parse_args()
    preset = SIZE_PRESETS[args.size]

    params = TrialParams(
        lr_encoder=args.lr_encoder if args.lr_encoder is not None else preset.lr_encoder,
        lr_heads=args.lr_heads if args.lr_heads is not None else preset.lr_heads,
        lr_encoder_s2a=(args.lr_encoder_s2a if args.lr_encoder_s2a is not None
                        else preset.lr_encoder_s2a),
        lr_cls_s2b=(args.lr_cls_s2b if args.lr_cls_s2b is not None
                    else preset.lr_cls_s2b),
        batch_size=args.batch_size if args.batch_size is not None else preset.batch_size,
        epochs=args.epochs_s1,
        epochs_s2a=args.epochs_s2a,
        epochs_s2b=args.epochs_s2b,
        use_pkd=not args.no_pkd,
        use_rdrop=not args.no_rdrop,
    )
    cfg = StageConfig(
        student_encoder=preset.encoder,
        distillation_data=args.distillation_data,
        output_dir=args.output,
        device=args.device,
    )
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())

    print(f"Pipeline: {args.size} ({preset.encoder}) → {args.output}")
    print(f"  stages: {stages}")
    print(f"  params: {json.dumps(params.to_dict(), indent=2)}")

    log = run_full_pipeline(
        params, cfg, args.stage2_data,
        silver_limit=args.silver_limit, stages=stages,
    )
    print()
    print("=" * 60)
    print(f"Final scores: {json.dumps(log['final_scores'], indent=2)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
