"""Collect news text from Wikinews and Wikipedia.

Usage:
    python -m corpus.domains.news.collect
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "news"
WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKINEWS_API = "https://en.wikinews.org/w/api.php"
HEADERS = {
    "User-Agent": "uniko-nlp-models/1.0 (https://github.com/rustic-ai/uniko-nlp-models; research@dragonscale.ai)",
}


def load_config() -> dict:
    with open(DOMAIN_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def _api_category_members(api_url: str, category: str, limit: int) -> list[str]:
    titles = []
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmlimit": min(limit, 500),
        "cmtype": "page", "format": "json",
    }
    try:
        resp = requests.get(api_url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        for member in resp.json().get("query", {}).get("categorymembers", []):
            titles.append(member["title"])
    except Exception as e:
        print(f"  ⚠ Category '{category}' failed: {e}", flush=True)
    return titles[:limit]


def _api_article_text(api_url: str, title: str) -> str | None:
    params = {
        "action": "query", "titles": title, "prop": "extracts",
        "explaintext": "true", "format": "json",
    }
    try:
        resp = requests.get(api_url, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        for page in resp.json().get("query", {}).get("pages", {}).values():
            return page.get("extract", "")
    except Exception:
        return None


def collect_wikinews(config: dict):
    """Fetch recent Wikinews articles."""
    cfg = config["sources"]["wikinews"]
    out_dir = OUTPUT_DIR / "wikinews"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "articles.jsonl"
    if output_file.exists():
        print("Wikinews: already collected.", flush=True)
        return

    api = cfg["api"]
    max_articles = cfg.get("max_articles", 1000)

    # Get recent articles via allpages
    print(f"Wikinews: fetching up to {max_articles} articles...", flush=True)
    titles = []
    apcontinue = None
    while len(titles) < max_articles:
        params = {
            "action": "query", "list": "allpages",
            "apnamespace": "0", "aplimit": 500, "format": "json",
        }
        if apcontinue:
            params["apcontinue"] = apcontinue
        try:
            resp = requests.get(api, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            pages = data.get("query", {}).get("allpages", [])
            titles.extend(p["title"] for p in pages)
            apcontinue = data.get("continue", {}).get("apcontinue")
            if not apcontinue:
                break
        except Exception as e:
            print(f"  ⚠ Wikinews list failed: {e}", flush=True)
            break

    titles = titles[:max_articles]
    print(f"Wikinews: fetching text for {len(titles)} articles...", flush=True)

    articles = []
    for i, title in enumerate(titles):
        text = _api_article_text(api, title)
        time.sleep(0.05)
        if text and len(text) > 200:
            articles.append({"title": title, "source": "wikinews", "text": text})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(titles)} articles fetched...", flush=True)

    with open(output_file, "w") as f:
        for article in articles:
            f.write(json.dumps(article) + "\n")
    print(f"Wikinews: collected {len(articles)} articles → {output_file}", flush=True)


def collect_wikipedia(config: dict):
    """Fetch news-related Wikipedia articles."""
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
        titles = _api_category_members(WIKI_API, category, max_per_cat)
        all_titles.update(titles)
        time.sleep(0.1)
        print(f"  Category '{category}': {len(titles)} articles", flush=True)

    print(f"Wikipedia: fetching {len(all_titles)} unique articles...", flush=True)
    articles = []
    for i, title in enumerate(sorted(all_titles)):
        text = _api_article_text(WIKI_API, title)
        time.sleep(0.05)
        if text and len(text) > 200:
            articles.append({"title": title, "source": "wikipedia", "text": text})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_titles)} articles fetched...", flush=True)

    with open(output_file, "w") as f:
        for article in articles:
            f.write(json.dumps(article) + "\n")
    print(f"Wikipedia: collected {len(articles)} articles → {output_file}", flush=True)


def main():
    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for name, collector in [("wikinews", collect_wikinews), ("wikipedia", collect_wikipedia)]:
        print(f"\n{'=' * 50}\nCollecting: {name}\n{'=' * 50}", flush=True)
        try:
            collector(config)
        except Exception as e:
            print(f"⚠ {name} failed: {e}", flush=True)

    print(f"\nCollection complete. Raw data in {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
