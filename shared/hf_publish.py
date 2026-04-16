"""Publish a trained ONNX model to HuggingFace Hub.

Usage:
    python -m shared.hf_publish \\
        --model-name distilroberta-nlp-en \\
        --onnx-dir onnx-output/ \\
        --org dragonscale-ai
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def publish(model_name: str, onnx_dir: Path, model_card: Path, org: str = "dragonscale-ai"):
    """Upload ONNX model and model card to HuggingFace Hub."""
    repo_id = f"{org}/uniko-{model_name}"
    api = HfApi()

    # Create or get repo
    create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"Publishing to https://huggingface.co/{repo_id}")

    # Upload ONNX files
    for onnx_file in onnx_dir.glob("*.onnx"):
        api.upload_file(
            path_or_fileobj=str(onnx_file),
            path_in_repo=onnx_file.name,
            repo_id=repo_id,
        )
        print(f"  Uploaded {onnx_file.name}")

    # Upload tokenizer files
    for tokenizer_file in onnx_dir.glob("tokenizer*"):
        api.upload_file(
            path_or_fileobj=str(tokenizer_file),
            path_in_repo=tokenizer_file.name,
            repo_id=repo_id,
        )

    # Upload config files
    for config_file in onnx_dir.glob("*.json"):
        api.upload_file(
            path_or_fileobj=str(config_file),
            path_in_repo=config_file.name,
            repo_id=repo_id,
        )

    # Upload model card
    if model_card.exists():
        api.upload_file(
            path_or_fileobj=str(model_card),
            path_in_repo="README.md",
            repo_id=repo_id,
        )
        print("  Uploaded model card")

    print(f"Published: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Publish model to HuggingFace Hub")
    parser.add_argument("--model-name", required=True, help="Model directory name under models/")
    parser.add_argument("--onnx-dir", type=Path, required=True, help="Directory with ONNX files")
    parser.add_argument("--org", default="dragonscale-ai", help="HuggingFace organization")
    args = parser.parse_args()

    model_card = Path("models") / args.model_name / "README.md"
    publish(args.model_name, args.onnx_dir, model_card, args.org)


if __name__ == "__main__":
    main()
