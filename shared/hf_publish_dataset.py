"""Publish the annotated corpus to HuggingFace as a dataset.

Uploads the final train/dev/test CoNLL-U splits along with a dataset card
and metadata. The dataset can then be loaded with:

    from datasets import load_dataset
    ds = load_dataset("dragonscale-ai/uniko-corpus-en")

Usage:
    python -m shared.hf_publish_dataset \
        --corpus-dir corpus/output/final \
        --org dragonscale-ai \
        --name uniko-corpus-en
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


DATASET_CARD_TEMPLATE = """---
language:
  - en
license: apache-2.0
task_categories:
  - token-classification
task_ids:
  - named-entity-recognition
  - part-of-speech
  - parsing
tags:
  - ner
  - pos
  - dependency-parsing
  - multi-task
  - conllu
size_categories:
  - {size_category}
---

# {dataset_name}

Multi-task NLP corpus for English: NER + POS tagging + dependency parsing annotations.

Built for training [uniko](https://github.com/rustic-ai/uniko) NLP models.

## Dataset Details

- **Format:** CoNLL-U with NER annotations in MISC field
- **Annotation:** spaCy `en_core_web_trf` + GPT validation
- **Total sentences:** {total_sentences:,}

### Splits

| Split | Sentences |
|-------|-----------|
| train | {train_count:,} |
| dev | {dev_count:,} |
| test | {test_count:,} |

### Domains

{domain_table}

### Annotation Fields

Each token has:
- `ID` — token index (1-based)
- `FORM` — word form
- `LEMMA` — lemma
- `UPOS` — Universal POS tag
- `XPOS` — language-specific POS tag
- `HEAD` — head token index
- `DEPREL` — dependency relation
- `MISC` — NER tags in BIO format (e.g., `NER=B-ORG`)

## Source

- **Training code:** [rustic-ai/uniko-nlp-models](https://github.com/rustic-ai/uniko-nlp-models)
- **Trained models:** [dragonscale-ai on HuggingFace](https://huggingface.co/dragonscale-ai)

## License

Apache-2.0 (annotations). See individual domain READMEs for source data licenses.
"""


def generate_dataset_card(metadata: dict, dataset_name: str) -> str:
    """Generate a HuggingFace dataset card from corpus metadata."""
    total = metadata.get("total", 0)

    if total < 1000:
        size_cat = "n<1K"
    elif total < 10_000:
        size_cat = "1K<n<10K"
    elif total < 100_000:
        size_cat = "10K<n<100K"
    elif total < 1_000_000:
        size_cat = "100K<n<1M"
    else:
        size_cat = "1M<n<10M"

    splits = metadata.get("splits", {})
    domains = metadata.get("domains", [])

    domain_lines = [f"| {d} |" for d in domains]
    domain_table = "| Domain |\n|--------|\n" + "\n".join(domain_lines)

    return DATASET_CARD_TEMPLATE.format(
        dataset_name=dataset_name,
        size_category=size_cat,
        total_sentences=total,
        train_count=splits.get("train", 0),
        dev_count=splits.get("dev", 0),
        test_count=splits.get("test", 0),
        domain_table=domain_table,
    )


def preflight_check(corpus_dir: Path) -> list[str]:
    """Validate corpus files exist."""
    errors = []
    if not corpus_dir.exists():
        errors.append(f"Corpus directory not found: {corpus_dir}")
        return errors

    for split in ["train.conllu", "dev.conllu", "test.conllu"]:
        if not (corpus_dir / split).exists():
            errors.append(f"Missing {split} in {corpus_dir}")

    if not (corpus_dir / "metadata.json").exists():
        errors.append(f"Missing metadata.json in {corpus_dir}")

    return errors


def publish_dataset(corpus_dir: Path, org: str, name: str):
    """Upload corpus splits and dataset card to HuggingFace."""
    repo_id = f"{org}/{name}"

    # Pre-flight
    errors = preflight_check(corpus_dir)
    if errors:
        print("Pre-flight check failed:")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    # Auth
    api = HfApi()
    try:
        user = api.whoami()
        print(f"Authenticated as: {user.get('name', user.get('fullname', 'unknown'))}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        print("Run: huggingface-cli login")
        sys.exit(1)

    # Create dataset repo
    create_repo(repo_id, exist_ok=True, repo_type="dataset")
    print(f"Publishing dataset to https://huggingface.co/datasets/{repo_id}")

    # Load metadata
    with open(corpus_dir / "metadata.json") as f:
        metadata = json.load(f)

    uploaded = []

    # Upload splits
    for split_file in ["train.conllu", "dev.conllu", "test.conllu"]:
        path = corpus_dir / split_file
        if path.exists():
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=f"data/{split_file}",
                repo_id=repo_id,
                repo_type="dataset",
            )
            size_mb = path.stat().st_size / 1024 / 1024
            uploaded.append(f"data/{split_file} ({size_mb:.1f}MB)")

    # Upload metadata
    api.upload_file(
        path_or_fileobj=str(corpus_dir / "metadata.json"),
        path_in_repo="metadata.json",
        repo_id=repo_id,
        repo_type="dataset",
    )
    uploaded.append("metadata.json")

    # Generate and upload dataset card
    card = generate_dataset_card(metadata, name)
    card_path = corpus_dir / "README.md"
    card_path.write_text(card)
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    uploaded.append("README.md (dataset card)")

    print(f"\nUploaded {len(uploaded)} files:")
    for name in uploaded:
        print(f"  {name}")
    print(f"\nPublished: https://huggingface.co/datasets/{repo_id}")
    print(f"\nUsage:")
    print(f'  from datasets import load_dataset')
    print(f'  ds = load_dataset("{repo_id}")')


def main():
    parser = argparse.ArgumentParser(description="Publish corpus to HuggingFace Datasets")
    parser.add_argument("--corpus-dir", type=Path, default=Path("corpus/output/final"),
                        help="Directory with train/dev/test CoNLL-U files")
    parser.add_argument("--org", default="dragonscale-ai", help="HuggingFace organization")
    parser.add_argument("--name", default="uniko-corpus-en", help="Dataset name on HuggingFace")
    args = parser.parse_args()

    publish_dataset(args.corpus_dir, args.org, args.name)


if __name__ == "__main__":
    main()
