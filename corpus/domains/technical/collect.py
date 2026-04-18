"""Collect technical text from Wikipedia and Python documentation.

Usage:
    python -m corpus.domains.technical.collect
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import requests
import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "technical"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "uniko-nlp-models/1.0 (https://github.com/rustic-ai/uniko-nlp-models; research@dragonscale.ai)",
}


def load_config() -> dict:
    with open(DOMAIN_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _wiki_category_members(category: str, limit: int) -> list[str]:
    titles = []
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmlimit": min(limit, 500),
        "cmtype": "page", "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        for member in resp.json().get("query", {}).get("categorymembers", []):
            titles.append(member["title"])
    except Exception as e:
        print(f"  ⚠ Category '{category}' failed: {e}", flush=True)
    return titles[:limit]


def _wiki_article_text(title: str) -> str | None:
    params = {
        "action": "query", "titles": title, "prop": "extracts",
        "explaintext": "true", "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        for page in resp.json().get("query", {}).get("pages", {}).values():
            return page.get("extract", "")
    except Exception:
        return None


def collect_wikipedia(config: dict):
    """Fetch technical Wikipedia articles."""
    cfg = config["sources"]["wikipedia"]
    out_dir = OUTPUT_DIR / "wikipedia"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "articles.jsonl"
    if output_file.exists():
        print("Wikipedia: already collected.", flush=True)
        return

    max_per_cat = cfg.get("max_articles_per_category", 80)
    all_titles = set()
    for category in cfg["categories"]:
        titles = _wiki_category_members(category, max_per_cat)
        all_titles.update(titles)
        time.sleep(0.1)
        print(f"  Category '{category}': {len(titles)} articles", flush=True)

    print(f"Wikipedia: fetching {len(all_titles)} unique articles...", flush=True)
    articles = []
    for i, title in enumerate(sorted(all_titles)):
        text = _wiki_article_text(title)
        time.sleep(0.05)
        if text and len(text) > 200:
            articles.append({"title": title, "source": "wikipedia", "text": text})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_titles)} articles fetched...", flush=True)

    with open(output_file, "w") as f:
        for article in articles:
            f.write(json.dumps(article) + "\n")
    print(f"Wikipedia: collected {len(articles)} articles → {output_file}", flush=True)


def strip_rst(text: str) -> str:
    text = re.sub(r"\.\.\s+\w[\w-]*::.*?\n", "\n", text)
    text = re.sub(r":\w+:`([^`]*)`", r"\1", text)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"``([^`]*)``", r"\1", text)
    text = re.sub(r"^[=\-~^\"'+*]+$", "", text, flags=re.MULTILINE)
    return text


def collect_python_docs(config: dict):
    """Clone CPython repo and extract documentation RST files."""
    cfg = config["sources"]["python_docs"]
    out_dir = OUTPUT_DIR / "python_docs"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "docs.jsonl"
    if output_file.exists():
        print("Python docs: already collected.", flush=True)
        return

    clone_dir = out_dir / "cpython"
    if not clone_dir.exists():
        print("Python docs: cloning CPython (docs only)...", flush=True)
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
             cfg["repo"], str(clone_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ⚠ Clone failed: {result.stderr[:200]}", flush=True)
            return
        # Sparse checkout only the Doc directory
        subprocess.run(
            ["git", "sparse-checkout", "set", cfg["docs_path"]],
            cwd=str(clone_dir), capture_output=True,
        )

    docs_dir = clone_dir / cfg["docs_path"]
    rst_files = sorted(docs_dir.rglob("*.rst"))
    print(f"Python docs: processing {len(rst_files)} RST files...", flush=True)

    docs = []
    for rst_path in rst_files:
        try:
            raw = rst_path.read_text(encoding="utf-8", errors="replace")
            text = strip_rst(raw)
            if len(text) > 200:
                rel = rst_path.relative_to(docs_dir)
                docs.append({"path": str(rel), "source": "python_docs", "text": text})
        except Exception:
            continue

    with open(output_file, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Python docs: collected {len(docs)} documents → {output_file}", flush=True)


def main():
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, collector in [("wikipedia", collect_wikipedia), ("python_docs", collect_python_docs)]:
        print(f"\n{'=' * 50}\nCollecting: {name}\n{'=' * 50}", flush=True)
        try:
            collector(config)
        except Exception as e:
            print(f"⚠ {name} failed: {e}", flush=True)

    print(f"\nCollection complete. Raw data in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
