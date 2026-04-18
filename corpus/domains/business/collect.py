"""Collect business/finance/ERP text from multiple open sources.

Sources:
  1. SEC EDGAR 10-K filings (public domain)
  2. Enron email corpus (public domain)
  3. OpenStax business textbooks (CC BY-4.0)
  4. Odoo documentation (CC BY-SA 3.0)
  5. Wikipedia business articles (CC BY-SA 3.0)
  6. CUAD commercial contracts (CC BY-4.0)
  7. Semantic Scholar abstracts (ODC-BY)

Each source saves to its own file/dir under output/raw/business/.
Sources are collected independently — failures don't block others.

Usage:
    python -m corpus.domains.business.collect
    python -m corpus.domains.business.collect --source sec_edgar
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tarfile
import time
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree

import requests
import yaml

DOMAIN_DIR = Path(__file__).parent
OUTPUT_DIR = Path(__file__).parent.parent.parent / "output" / "raw" / "business"

EDGAR_BASE = "https://data.sec.gov"
EDGAR_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
WIKI_API = "https://en.wikipedia.org/w/api.php"
S2_API = "https://api.semanticscholar.org/graph/v1"
OPENALEX_API = "https://api.openalex.org"


def load_config() -> dict:
    with open(DOMAIN_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


# ── Helpers ──────────────────────────────────────────────────────

class _HTMLStripper(HTMLParser):
    """Strip HTML tags and return plain text."""

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip = True
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self._parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def strip_html(html: str) -> str:
    """Strip HTML tags and return plain text."""
    stripper = _HTMLStripper()
    stripper.feed(html)
    return stripper.get_text()


def strip_rst(text: str) -> str:
    """Strip reStructuredText markup to plain text."""
    # Remove directives (.. directive:: ...)
    text = re.sub(r"\.\.\s+\w[\w-]*::.*?\n", "\n", text)
    # Remove inline roles (:role:`text`)
    text = re.sub(r":\w+:`([^`]*)`", r"\1", text)
    # Remove emphasis and strong (*text* and **text**)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    # Remove literal backticks (``text``)
    text = re.sub(r"``([^`]*)``", r"\1", text)
    # Remove section underlines (===, ---, ~~~, etc.)
    text = re.sub(r"^[=\-~^\"'+*]+$", "", text, flags=re.MULTILINE)
    # Remove image/figure references
    text = re.sub(r"\.\.\s+(image|figure)::\s*\S+", "", text)
    # Remove comment blocks (.. comment)
    text = re.sub(r"\.\.\s*\n(\s+\S.*\n)*", "\n", text)
    return text


def rate_limit(min_interval: float = 0.12):
    """Sleep to respect rate limits (default: ~8 req/s for EDGAR)."""
    time.sleep(min_interval)


def _get(url: str, headers: dict | None = None, timeout: int = 30) -> requests.Response | None:
    """GET with error handling."""
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        return resp
    except Exception as e:
        print(f"  ⚠ GET failed: {url}: {e}")
        return None


# ── 1. SEC EDGAR 10-K filings ────────────────────────────────────

def collect_sec_edgar(config: dict):
    """Fetch 10-K filings for listed companies via EDGAR API."""
    cfg = config["sources"]["sec_edgar"]
    out_dir = OUTPUT_DIR / "sec_edgar"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "filings.jsonl"
    if output_file.exists():
        print("SEC EDGAR: already collected.")
        return

    headers = {"User-Agent": cfg["user_agent"]}
    max_per_company = cfg.get("max_filings_per_company", 2)
    filings = []

    print(f"SEC EDGAR: fetching 10-K filings for {len(cfg['ciks'])} companies...")
    for cik in cfg["ciks"]:
        padded = str(cik).zfill(10)
        url = f"{EDGAR_BASE}/submissions/CIK{padded}.json"
        rate_limit()

        resp = _get(url, headers=headers)
        if not resp:
            continue

        data = resp.json()
        company_name = data.get("name", f"CIK{cik}")
        recent = data.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        filing_dates = recent.get("filingDate", [])

        # Find 10-K filings
        found = 0
        for i, form in enumerate(forms):
            if form != "10-K" or found >= max_per_company:
                continue

            accession = accessions[i]
            primary_doc = primary_docs[i]
            accession_nd = accession.replace("-", "")

            doc_url = f"{EDGAR_ARCHIVES}/{cik}/{accession_nd}/{primary_doc}"
            rate_limit()

            doc_resp = _get(doc_url, headers=headers, timeout=60)
            if not doc_resp:
                continue

            text = strip_html(doc_resp.text)
            if len(text) < 1000:
                continue

            filings.append({
                "company": company_name,
                "cik": cik,
                "form": "10-K",
                "date": filing_dates[i],
                "text": text,
            })
            found += 1
            print(f"  ✓ {company_name} ({filing_dates[i]}): {len(text)} chars")

    with open(output_file, "w") as f:
        for filing in filings:
            f.write(json.dumps(filing) + "\n")

    print(f"SEC EDGAR: collected {len(filings)} filings → {output_file}")


# ── 2. Enron email corpus ────────────────────────────────────────

def _parse_email(raw: str) -> dict | None:
    """Parse a raw email file into header + body."""
    parts = raw.split("\n\n", 1)
    if len(parts) < 2:
        return None

    header_text, body = parts
    body = body.strip()
    if not body or len(body) < 20:
        return None

    # Parse basic headers
    headers = {}
    for line in header_text.split("\n"):
        if ": " in line:
            key, _, value = line.partition(": ")
            headers[key.strip()] = value.strip()

    return {
        "subject": headers.get("Subject", ""),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "body": body,
    }


def collect_enron(config: dict):
    """Download and extract Enron email corpus."""
    cfg = config["sources"]["enron"]
    out_dir = OUTPUT_DIR / "enron"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "emails.jsonl"
    if output_file.exists():
        print("Enron: already collected.")
        return

    archive_path = out_dir / "enron_mail.tar.gz"

    # Download archive
    if not archive_path.exists():
        print("Enron: downloading corpus (423MB, may take a few minutes)...")
        resp = requests.get(cfg["url"], stream=True, timeout=600)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(archive_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {downloaded / 1024 / 1024:.0f}MB / {total / 1024 / 1024:.0f}MB ({pct:.0f}%)", end="", flush=True)
        print()

    # Extract emails
    print("Enron: extracting emails...")
    emails = []
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile() or member.size > 100_000:
                continue
            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                raw = f.read().decode("utf-8", errors="replace")
                parsed = _parse_email(raw)
                if parsed:
                    emails.append(parsed)
            except Exception:
                continue

            if len(emails) % 10000 == 0 and len(emails) > 0:
                print(f"  {len(emails)} emails extracted...")

    with open(output_file, "w") as f:
        for email in emails:
            f.write(json.dumps(email) + "\n")

    print(f"Enron: extracted {len(emails)} emails → {output_file}")

    # Remove archive to save disk space
    archive_path.unlink(missing_ok=True)


# ── 3. OpenStax business textbooks ───────────────────────────────

def _extract_cnxml_text(cnxml_path: Path) -> str:
    """Extract plain text from a CNXML file."""
    try:
        tree = ElementTree.parse(cnxml_path)
    except ElementTree.ParseError:
        return ""

    root = tree.getroot()
    # CNXML uses namespaces — strip them for easier traversal
    text_parts = []
    for elem in root.iter():
        if elem.text:
            text_parts.append(elem.text)
        if elem.tail:
            text_parts.append(elem.tail)

    return " ".join(text_parts)


def collect_openstax(config: dict):
    """Clone OpenStax book repos and extract text from CNXML files."""
    cfg = config["sources"]["openstax"]
    out_dir = OUTPUT_DIR / "openstax"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "books.jsonl"
    if output_file.exists():
        print("OpenStax: already collected.")
        return

    books = []
    for book in cfg["books"]:
        repo = book["repo"]
        title = book["title"]
        clone_dir = out_dir / repo.split("/")[-1]

        # Clone if not present
        if not clone_dir.exists():
            print(f"  Cloning {repo}...")
            result = subprocess.run(
                ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(clone_dir)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print(f"  ⚠ Failed to clone {repo}: {result.stderr[:200]}")
                continue

        # Find CNXML files
        cnxml_files = sorted(clone_dir.rglob("*.cnxml"))
        if not cnxml_files:
            # Try .xhtml files as fallback
            cnxml_files = sorted(clone_dir.rglob("*.xhtml"))

        book_text = []
        for cnxml_path in cnxml_files:
            text = _extract_cnxml_text(cnxml_path)
            if text and len(text) > 100:
                book_text.append(text)

        if book_text:
            full_text = "\n\n".join(book_text)
            books.append({
                "title": title,
                "source": "openstax",
                "text": full_text,
            })
            print(f"  ✓ {title}: {len(cnxml_files)} files, {len(full_text)} chars")
        else:
            print(f"  ⚠ {title}: no content files found in {clone_dir}")

    with open(output_file, "w") as f:
        for book in books:
            f.write(json.dumps(book) + "\n")

    print(f"OpenStax: collected {len(books)} books → {output_file}")


# ── 4. Odoo documentation ────────────────────────────────────────

def collect_odoo(config: dict):
    """Clone Odoo docs repo and extract text from RST files."""
    cfg = config["sources"]["odoo"]
    out_dir = OUTPUT_DIR / "odoo"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "docs.jsonl"
    if output_file.exists():
        print("Odoo: already collected.")
        return

    clone_dir = out_dir / "documentation"

    if not clone_dir.exists():
        print(f"Odoo: cloning documentation ({cfg['branch']})...")
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", cfg["branch"],
             cfg["repo"], str(clone_dir)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  ⚠ Failed to clone: {result.stderr[:200]}")
            return

    # Extract text from RST files
    rst_files = sorted(clone_dir.rglob("*.rst"))
    print(f"Odoo: processing {len(rst_files)} RST files...")

    docs = []
    for rst_path in rst_files:
        try:
            raw = rst_path.read_text(encoding="utf-8", errors="replace")
            text = strip_rst(raw)
            if len(text) > 200:
                rel_path = rst_path.relative_to(clone_dir)
                docs.append({
                    "path": str(rel_path),
                    "source": "odoo",
                    "text": text,
                })
        except Exception:
            continue

    with open(output_file, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Odoo: collected {len(docs)} documents → {output_file}")


# ── 5. Wikipedia business articles ───────────────────────────────

WIKI_HEADERS = {
    "User-Agent": "kniv-nlp-models/1.0 (https://github.com/rustic-ai/kniv-nlp-models; research@dragonscale.ai)",
}


def _wiki_category_members(category: str, limit: int) -> list[str]:
    """Get article titles in a Wikipedia category."""
    titles = []
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category}",
        "cmlimit": min(limit, 500),
        "cmtype": "page",
        "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for member in data.get("query", {}).get("categorymembers", []):
            titles.append(member["title"])
    except Exception as e:
        print(f"  ⚠ Wikipedia category '{category}' failed: {e}")

    return titles[:limit]


def _wiki_article_text(title: str) -> str | None:
    """Get plain text extract of a Wikipedia article."""
    params = {
        "action": "query",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",
        "format": "json",
    }
    try:
        resp = requests.get(WIKI_API, params=params, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page in pages.values():
            return page.get("extract", "")
    except Exception:
        return None


def collect_wikipedia(config: dict):
    """Fetch business-related Wikipedia articles by category."""
    cfg = config["sources"]["wikipedia"]
    out_dir = OUTPUT_DIR / "wikipedia"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "articles.jsonl"
    if output_file.exists():
        print("Wikipedia: already collected.")
        return

    max_per_cat = cfg.get("max_articles_per_category", 100)

    # Gather unique article titles from all categories
    all_titles = set()
    for category in cfg["categories"]:
        titles = _wiki_category_members(category, max_per_cat)
        all_titles.update(titles)
        rate_limit(0.1)
        print(f"  Category '{category}': {len(titles)} articles")

    print(f"Wikipedia: fetching {len(all_titles)} unique articles...")

    articles = []
    for i, title in enumerate(sorted(all_titles)):
        text = _wiki_article_text(title)
        rate_limit(0.05)

        if text and len(text) > 200:
            articles.append({
                "title": title,
                "source": "wikipedia",
                "text": text,
            })

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(all_titles)} articles fetched...")

    with open(output_file, "w") as f:
        for article in articles:
            f.write(json.dumps(article) + "\n")

    print(f"Wikipedia: collected {len(articles)} articles → {output_file}")


# ── 6. CUAD contracts ────────────────────────────────────────────

def collect_cuad(config: dict):
    """Load CUAD contract dataset from HuggingFace."""
    out_dir = OUTPUT_DIR / "cuad"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "contracts.jsonl"
    if output_file.exists():
        print("CUAD: already collected.")
        return

    print("CUAD: loading from HuggingFace...")
    from datasets import load_dataset
    dataset = load_dataset("cuad", revision="refs/convert/parquet", split="test")

    # CUAD has contract text in the 'context' field
    seen_contexts = set()
    contracts = []
    for item in dataset:
        context = item.get("context", "")
        if not context or len(context) < 200:
            continue

        # Deduplicate (CUAD repeats context across QA pairs)
        key = context[:500]
        if key in seen_contexts:
            continue
        seen_contexts.add(key)

        contracts.append({
            "title": item.get("title", ""),
            "source": "cuad",
            "text": context,
        })

    with open(output_file, "w") as f:
        for contract in contracts:
            f.write(json.dumps(contract) + "\n")

    print(f"CUAD: collected {len(contracts)} contract passages → {output_file}")


# ── 7. Academic abstracts via OpenAlex ────────────────────────────

# OpenAlex concept IDs for business domains
OPENALEX_CONCEPTS = {
    "Supply chain management": "C44104985",
    "Corporate finance": "C80515813",
    "Operations management": "C21547014",
    "Marketing management": "C192975520",
    "Accounting": "C121955636",
}


def _reconstruct_abstract(inverted_index: dict) -> str:
    """Reconstruct abstract text from OpenAlex inverted index format."""
    positions = {}
    for word, pos_list in inverted_index.items():
        for pos in pos_list:
            positions[pos] = word
    return " ".join(positions[i] for i in sorted(positions.keys()))


def collect_s2orc(config: dict):
    """Fetch business/finance paper abstracts from OpenAlex API."""
    cfg = config["sources"]["s2orc"]
    out_dir = OUTPUT_DIR / "s2orc"
    out_dir.mkdir(parents=True, exist_ok=True)

    output_file = out_dir / "abstracts.jsonl"
    if output_file.exists():
        print("S2ORC: already collected.", flush=True)
        return

    max_papers = cfg.get("max_papers", 2000)
    papers_per_concept = max_papers // len(OPENALEX_CONCEPTS)
    mailto = "research@dragonscale.ai"

    abstracts = []
    for concept_name, concept_id in OPENALEX_CONCEPTS.items():
        print(f"  OpenAlex: fetching '{concept_name}'...", flush=True)
        cursor = "*"
        concept_count = 0

        while concept_count < papers_per_concept:
            params = {
                "filter": f"concepts.id:C{concept_id},has_abstract:true,language:en",
                "per_page": 200,
                "cursor": cursor,
                "mailto": mailto,
                "select": "title,abstract_inverted_index,publication_year",
            }

            rate_limit(0.2)  # OpenAlex allows ~10 req/s with mailto

            try:
                resp = requests.get(f"{OPENALEX_API}/works", params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"    ⚠ API error: {e}", flush=True)
                break

            results = data.get("results", [])
            if not results:
                break

            for work in results:
                inv_idx = work.get("abstract_inverted_index")
                if not inv_idx:
                    continue
                abstract = _reconstruct_abstract(inv_idx)
                if len(abstract) > 100:
                    abstracts.append({
                        "title": work.get("title", ""),
                        "year": work.get("publication_year"),
                        "field": concept_name,
                        "source": "s2orc",
                        "text": abstract,
                    })
                    concept_count += 1
                    if concept_count >= papers_per_concept:
                        break

            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor:
                break

            print(f"    {concept_name}: {concept_count} abstracts so far...", flush=True)

        print(f"    {concept_name}: {concept_count} abstracts (done)", flush=True)

    with open(output_file, "w") as f:
        for abstract in abstracts:
            f.write(json.dumps(abstract) + "\n")

    print(f"S2ORC: collected {len(abstracts)} abstracts → {output_file}", flush=True)


# ── Main ─────────────────────────────────────────────────────────

COLLECTORS = {
    "sec_edgar": collect_sec_edgar,
    "enron": collect_enron,
    "openstax": collect_openstax,
    "odoo": collect_odoo,
    "wikipedia": collect_wikipedia,
    "cuad": collect_cuad,
    "s2orc": collect_s2orc,
}


def main():
    parser = argparse.ArgumentParser(description="Collect business corpus")
    parser.add_argument("--source", choices=list(COLLECTORS.keys()),
                        help="Collect a specific source only")
    args = parser.parse_args()

    config = load_config()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.source:
        sources = {args.source: COLLECTORS[args.source]}
    else:
        sources = COLLECTORS

    for name, collector in sources.items():
        print(f"\n{'=' * 60}")
        print(f"Collecting: {name}")
        print(f"{'=' * 60}")
        try:
            collector(config)
        except Exception as e:
            print(f"⚠ {name} failed: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nCollection complete. Raw data in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
