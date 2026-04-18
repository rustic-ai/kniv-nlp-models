"""Collect encyclopedic text from Wikipedia (general knowledge).

Usage:
    python -m corpus.domains.encyclopedic.collect
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "encyclopedic"
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


def main():
    config = load_config()
    cfg = config["sources"]["wikipedia"]
    out_dir = OUTPUT_DIR / "wikipedia"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "articles.jsonl"
    if output_file.exists():
        print("Wikipedia: already collected.", flush=True)
        return

    max_per_cat = cfg.get("max_articles_per_category", 60)
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


if __name__ == "__main__":
    main()
