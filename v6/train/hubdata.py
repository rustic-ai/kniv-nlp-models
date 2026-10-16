"""Prepared training data on the Hugging Face hub, so a VM downloads it
instead of preparing it.

    # locally: build every cache a run needs, then push them
    uv run python -m v6.train.loop <run args> --prepare-only
    uv run python -m v6.train.hubdata push --manifest manifest-<run>.json
    # on the VM (colab.py does this with --hub-data):
    python -m v6.train.hubdata pull --manifest manifest-<run>.json

The repo (private dataset ``dragonscale-ai/kniv-v7-data``) mirrors
``data/v6-corpus/``: the knowledge sources and label bank at their paths, and
the feature caches a manifest lists under ``features/``. Cache keys hash file
contents, not timestamps, so a cache built here is the one a VM looks up from
the same files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import DATA_DIR

REPO = "dragonscale-ai/kniv-v7-data"
CORPUS = DATA_DIR / "v6-corpus"
SIDE = ["label_bank.json", "vocab.json", "MANIFEST.json", "kg_holdout.json", "kg_teacher.json",
        "kg_pretrain.json", "kg_trex.json", "fewrel_rel.json", "fewnerd_spans.json",
        "docred_distant_rel.json", "redocred_rel.json", "entities_docred.json"]


def push(manifest: str, repo: str = REPO) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    man = json.loads((CORPUS / "features" / manifest).read_text())
    files = [f for f in SIDE if (CORPUS / f).exists()] + ["features/" + manifest] + \
        ["features/" + c for c in man["caches"]]
    have = {f.path: f.size for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True)
            if hasattr(f, "size")}
    todo = [f for f in files if have.get(f) != (CORPUS / f).stat().st_size]
    print(f"{len(files)} files, {len(todo)} to upload "
          f"({sum((CORPUS / f).stat().st_size for f in todo) / 2**30:.2f} GB)", flush=True)
    if todo:
        api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(CORPUS),
                          allow_patterns=todo,
                          commit_message=f"Prepared data for {man['encoder']} ({manifest})")
    print(f"pushed to {repo}")


def pull(manifest: str, repo: str = REPO, dest: Path = CORPUS) -> None:
    from huggingface_hub import hf_hub_download
    dest.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(repo, "features/" + manifest, repo_type="dataset", local_dir=str(dest))
    man = json.loads(Path(p).read_text())
    files = SIDE + ["features/" + c for c in man["caches"]]
    for f in files:
        try:
            hf_hub_download(repo, f, repo_type="dataset", local_dir=str(dest))
        except Exception as e:                  # a side file this run never needed
            if not f.startswith("features/"):
                print(f"skipped {f}: {type(e).__name__}", flush=True)
                continue
            raise
    print(f"pulled {len(files)} files from {repo} for {man['encoder']}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["push", "pull"])
    ap.add_argument("--manifest", required=True, help="features/manifest-<run>.json file name")
    ap.add_argument("--repo", default=REPO)
    a = ap.parse_args()
    (push if a.cmd == "push" else pull)(a.manifest, a.repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
