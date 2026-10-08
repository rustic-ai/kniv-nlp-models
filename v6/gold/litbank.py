"""LitBank coreference gold (CC-BY-4.0), chunked into fixed-size windows.

LitBank is the usable gold for coreference: OntoNotes coref is LDC-restricted
and Maverick — the current SOTA system — is CC-BY-NC-SA, so neither is
available for commercial work. LitBank is 100 literary documents in
CoNLL-2012 coref format.

Documents run to thousands of tokens, so they are chunked into windows that
match the v6 encoder budget. **Mentions and clusters that cross a window
boundary are dropped**, and singleton clusters left behind are discarded —
so these numbers measure *within-window* coreference only. Cross-window
linking is a separate problem that no model here is being asked to solve.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

from ..config import DATA_DIR
from .ud_ewt import GoldItem

LITBANK_DIR = DATA_DIR / "litbank" / "coref"
RAW_BASE = "https://raw.githubusercontent.com/dbamman/litbank/master/coref/conll"
LISTING = "https://api.github.com/repos/dbamman/litbank/contents/coref/conll"


def download(limit_docs: int | None = None) -> list[Path]:
    LITBANK_DIR.mkdir(parents=True, exist_ok=True)
    have = sorted(LITBANK_DIR.glob("*.conll"))
    if limit_docs and len(have) >= limit_docs:
        return have[:limit_docs]
    with urllib.request.urlopen(LISTING, timeout=30) as r:
        names = [f["name"] for f in json.load(r) if f["name"].endswith(".conll")]
    names.sort()
    if limit_docs:
        names = names[:limit_docs]
    out = []
    for n in names:
        dst = LITBANK_DIR / n
        if not dst.exists():
            with urllib.request.urlopen(f"{RAW_BASE}/{n}", timeout=60) as r:
                dst.write_bytes(r.read())
        out.append(dst)
    return out


def parse_conll(path: Path) -> tuple[list[str], dict[int, list[tuple[int, int]]]]:
    """Return (tokens, {cluster_id: [(start, end_inclusive), ...]})."""
    tokens: list[str] = []
    clusters: dict[int, list[tuple[int, int]]] = {}
    open_spans: dict[int, list[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        idx = len(tokens)
        tokens.append(cols[3])
        markup = cols[-1].strip()
        if not markup or markup == "-":
            continue
        # Markup looks like "(0", "0)", "(2)", or "(3|(4" for nesting.
        for part in markup.split("|"):
            part = part.strip()
            if not part:
                continue
            cid = int(re.sub(r"[()]", "", part))
            if part.startswith("(") and part.endswith(")"):
                clusters.setdefault(cid, []).append((idx, idx))
            elif part.startswith("("):
                open_spans.setdefault(cid, []).append(idx)
            elif part.endswith(")"):
                if open_spans.get(cid):
                    start = open_spans[cid].pop()
                    clusters.setdefault(cid, []).append((start, idx))
    return tokens, clusters


def load_coref_items(limit: int | None = None, window: int = 400,
                     limit_docs: int | None = None,
                     min_clusters: int = 2) -> list[GoldItem]:
    """Chunk LitBank documents into windows carrying self-contained clusters."""
    items: list[GoldItem] = []
    for path in download(limit_docs):
        tokens, clusters = parse_conll(path)
        doc = path.stem
        for w0 in range(0, len(tokens), window):
            w1 = min(w0 + window, len(tokens))
            win_tokens = tokens[w0:w1]
            if len(win_tokens) < 50:
                continue
            win_clusters = []
            for spans in clusters.values():
                inside = [(s - w0, e - w0) for s, e in spans if s >= w0 and e < w1]
                if len(inside) >= 2:            # a singleton is not coreference
                    win_clusters.append(sorted(inside))
            if len(win_clusters) < min_clusters:
                continue
            items.append(GoldItem(
                id=f"litbank-{doc}-{w0 // window:03d}",
                tokens=win_tokens,
                layers={"coref": win_clusters},
            ))
            if limit and len(items) >= limit:
                return items
    return items
