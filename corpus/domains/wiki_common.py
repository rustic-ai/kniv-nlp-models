"""Shared Wikipedia collection utilities for all domains.

Handles rate limiting, subcategory recursion, and article fetching.
"""

from __future__ import annotations

import time

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {
    "User-Agent": "kniv-nlp-models/1.0 (https://github.com/rustic-ai/kniv-nlp-models; research@dragonscale.ai)",
}


def wiki_category_members(
    category: str,
    limit: int,
    recurse_depth: int = 1,
    api_url: str = WIKI_API,
) -> list[str]:
    """Get article titles from a category, optionally recursing into subcategories."""
    titles = set()
    _collect_from_category(api_url, category, limit, recurse_depth, titles)
    return sorted(titles)[:limit]


def _collect_from_category(
    api_url: str, category: str, limit: int, depth: int, titles: set,
):
    if len(titles) >= limit:
        return

    # Get pages in this category
    params = {
        "action": "query", "list": "categorymembers",
        "cmtitle": f"Category:{category}", "cmlimit": 500,
        "cmtype": "page", "format": "json",
    }
    time.sleep(0.2)
    try:
        resp = requests.get(api_url, params=params, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        for member in resp.json().get("query", {}).get("categorymembers", []):
            titles.add(member["title"])
    except Exception as e:
        print(f"  ⚠ Category '{category}' pages failed: {e}", flush=True)

    # Recurse into subcategories
    if depth > 0 and len(titles) < limit:
        params["cmtype"] = "subcat"
        time.sleep(0.2)
        try:
            resp = requests.get(api_url, params=params, headers=WIKI_HEADERS, timeout=15)
            resp.raise_for_status()
            subcats = resp.json().get("query", {}).get("categorymembers", [])
            for subcat in subcats[:10]:  # limit subcategory exploration
                subcat_name = subcat["title"].replace("Category:", "")
                _collect_from_category(api_url, subcat_name, limit, depth - 1, titles)
                if len(titles) >= limit:
                    break
        except Exception:
            pass


def wiki_article_text(title: str, api_url: str = WIKI_API) -> str | None:
    """Get plain text extract of a Wikipedia article."""
    params = {
        "action": "query", "titles": title, "prop": "extracts",
        "explaintext": "true", "format": "json",
    }
    time.sleep(0.1)
    try:
        resp = requests.get(api_url, params=params, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        for page in resp.json().get("query", {}).get("pages", {}).values():
            return page.get("extract", "")
    except Exception:
        return None


def collect_wiki_articles(
    categories: list[str],
    max_per_category: int,
    output_file,
    source_name: str = "wikipedia",
    min_text_length: int = 200,
    recurse_depth: int = 1,
    api_url: str = WIKI_API,
) -> int:
    """Collect articles from multiple categories and save to JSONL.

    Returns the number of articles collected.
    """
    import json

    all_titles = set()
    for category in categories:
        titles = wiki_category_members(category, max_per_category, recurse_depth, api_url)
        all_titles.update(titles)
        print(f"  Category '{category}': {len(titles)} articles", flush=True)

    print(f"Fetching {len(all_titles)} unique articles...", flush=True)

    articles = []
    for i, title in enumerate(sorted(all_titles)):
        text = wiki_article_text(title, api_url)
        if text and len(text) > min_text_length:
            articles.append({"title": title, "source": source_name, "text": text})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_titles)} articles fetched ({len(articles)} kept)...", flush=True)

    with open(output_file, "w") as f:
        for article in articles:
            f.write(json.dumps(article) + "\n")

    print(f"Collected {len(articles)} articles → {output_file}", flush=True)
    return len(articles)
