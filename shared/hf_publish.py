"""Publish a trained ONNX model to HuggingFace Hub.

Usage:
    python -m shared.hf_publish \
        --model-name deberta-v3-nlp-en \
        --onnx-dir onnx-output/deberta-v3-nlp-en \
        --org dragonscale-ai
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def preflight_check(onnx_dir: Path, model_card: Path) -> list[str]:
    """Validate everything exists before publishing."""
    errors = []

    if not onnx_dir.exists():
        errors.append(f"ONNX directory not found: {onnx_dir}")
        return errors

    onnx_files = list(onnx_dir.glob("*.onnx"))
    if not onnx_files:
        errors.append(f"No .onnx files found in {onnx_dir}")

    if not (onnx_dir / "label_maps.json").exists():
        errors.append(f"Missing label_maps.json in {onnx_dir}")

    tokenizer_files = list(onnx_dir.glob("tokenizer*")) + list(onnx_dir.glob("spiece*")) + list(onnx_dir.glob("sentencepiece*"))
    if not tokenizer_files:
        errors.append(f"No tokenizer files found in {onnx_dir}")

    if not model_card.exists():
        errors.append(f"Model card not found: {model_card}")

    return errors


def publish(model_name: str, onnx_dir: Path, model_card: Path, org: str = "dragonscale-ai"):
    """Upload ONNX model, tokenizer, and model card to HuggingFace Hub."""
    repo_id = f"{org}/kniv-{model_name}"

    # Pre-flight check
    errors = preflight_check(onnx_dir, model_card)
    if errors:
        print("Pre-flight check failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # Authenticate
    api = HfApi()
    try:
        user = api.whoami()
        print(f"Authenticated as: {user.get('name', user.get('fullname', 'unknown'))}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Run: huggingface-cli login")
        sys.exit(1)

    # Create or get repo
    create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"Publishing to https://huggingface.co/{repo_id}")

    uploaded = []

    # Upload ONNX files
    for onnx_file in onnx_dir.glob("*.onnx"):
        api.upload_file(
            path_or_fileobj=str(onnx_file),
            path_in_repo=onnx_file.name,
            repo_id=repo_id,
        )
        size_mb = onnx_file.stat().st_size / 1024 / 1024
        uploaded.append(f"{onnx_file.name} ({size_mb:.1f}MB)")

    # Upload tokenizer files (various formats)
    for pattern in ["tokenizer*", "spiece*", "sentencepiece*", "special_tokens*", "added_tokens*"]:
        for f in onnx_dir.glob(pattern):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_id)
            uploaded.append(f.name)

    # Upload config/label files
    for pattern in ["*.json"]:
        for f in onnx_dir.glob(pattern):
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=repo_id)
            uploaded.append(f.name)

    # Upload model card
    api.upload_file(
        path_or_fileobj=str(model_card),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    uploaded.append("README.md (model card)")

    print(f"\nUploaded {len(uploaded)} files:")
    for name in uploaded:
        print(f"  {name}")
    print(f"\nPublished: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Publish model to HuggingFace Hub")
    parser.add_argument("--model-name", required=True, help="Model directory name under models/")
    parser.add_argument("--onnx-dir", type=Path, required=True, help="Directory with ONNX + tokenizer files")
    parser.add_argument("--org", default="dragonscale-ai", help="HuggingFace organization")
    args = parser.parse_args()

    model_card = Path("models") / args.model_name / "README.md"
    publish(args.model_name, args.onnx_dir, model_card, args.org)


if __name__ == "__main__":
    main()
