"""Build 512-token windows from whole documents in ``corpus/output/raw/``.

This is the step v5 never had. ``corpus/domains/*/preprocess.py`` emits
``{"text", "source", "domain"}`` per sentence with no document id and no
index, which is why the published corpus has document structure only in
name: measured on ``corpus/gold/test.parquet``, 0 of 65,731 rows had a
``prev_text`` matching the previous row. Coreference is close to vacuous
below the document, so v6 reads ``raw/`` directly and never runs
``preprocess.py``.

Each domain writes ``raw/`` in its own shape, so a document adapter per
domain is unavoidable:

* conversation — one JSONL row per utterance, grouped by ``conv_id`` and
  ordered by ``turn_idx``; a document is a whole conversation.
* narrative — one plain ``.txt`` per book; a document is a chapter-sized
  chunk, since a whole book is far beyond any window.
* technical / news / encyclopedic — one JSONL row per article.
* business — one row per SEC filing, Enron email or OpenStax section.

**Windows never cross a document boundary**, and a sentence is never split
across two windows: both would manufacture false adjacency, which is the
failure this module exists to avoid.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "corpus" / "output" / "raw"

# Sentence splitting is used ONLY to record spans inside a window. It is
# never a unit of training, and never decides what the model sees.
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")

# Canonical tokenization, fixed here once for every annotator. Whitespace
# splitting is not adequate: it leaves "coffee.", "str.find" and
# 'parser.add_argument("x",' as single tokens, and annotators then guess at
# malformed units and guess differently — measured at 0.41 POS agreement
# between two taggers that agree 0.977 on UD EWT.
#
# UD-style: clitics and contractions split off, punctuation separated,
# but URLs, emails and decimals kept whole.
_URLISH = re.compile(r"""(?:https?://|www\.)\S+|\S+@\S+\.\w+""")
_CLITIC = re.compile(
    r"(?i)(.+?)(n't|'s|'re|'ve|'ll|'d|'m)$")
_TOKEN = re.compile(r"""
      \d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?   # numbers, percents, ordinals
    | \w+(?:[-']\w+)*                     # words, hyphenated, internal '
    | \.\.\.|[^\w\s]                     # ellipsis, single punctuation
""", re.VERBOSE)


def tokenize(text: str) -> list[str]:
    """Deterministic UD-ish word tokenizer.

    Deliberately dependency-free and reproducible: the corpus must be
    re-derivable without pinning a toolkit, and every annotator must see
    byte-identical tokens.
    """
    out: list[str] = []
    for chunk in text.split():
        if _URLISH.fullmatch(chunk):
            out.append(chunk)
            continue
        for tok in _TOKEN.findall(chunk):
            m = _CLITIC.match(tok)
            if m and m.group(1):
                out.extend([m.group(1), m.group(2)])
            else:
                out.append(tok)
    return out


# Non-prose rejection. The technical domain carries Python documentation,
# which mixes prose with code blocks and reStructuredText tables:
#
#     import argparse; parser.add_argument("x", type=int, help="the base")
#     | Searching and Replacing | str.find | +--------------+------------+
#
# Measured POS agreement between two taggers on such windows is 0.04-0.06 —
# they are not annotating language, they are guessing. Dropped at the unit
# (paragraph) level so surrounding prose in the same document survives.
_CODEY = re.compile(r"""
      ^\s{4,}\S                      # indented block
    | [{}();]\s*$                     # statement punctuation at line end
    | \b(?:import|def|class|return|lambda|elif|None|True|False)\b
    | [A-Za-z_]\w*\s*=\s*\S          # assignment
    | [A-Za-z_]\w*\.\w+\(            # method call
    | \+[-+]{3,}                      # ASCII table rule
    | ^\s*[|:]                        # table / directive line
    | ::\s*$                          # RST literal block marker
""", re.VERBOSE | re.MULTILINE)


# Model-artifact cleaning. v6 reads raw/ directly and skips
# preprocess.py — correctly, because that is where document structure was
# destroyed — but preprocess.py also cleaned text, and that was lost with
# it. Measured on the collected conversation domain: 8.4% of utterances
# carry <|endoftext|> markers and 2.5% contain function-call JSON, all from
# the glaive source. A CLS smoke test duly labelled a tool-call payload
# "Directive", which is meaningless.
_ARTIFACT = re.compile(r"""
      <\|[a-z_]+\|>                        # <|endoftext|>, <|im_start|>
    | <\s*/?\s*functioncall\s*>            # tool-call wrapper
""", re.VERBOSE | re.IGNORECASE)
_JSON_BLOB = re.compile(r"""\{\s*"[^"]+"\s*:.*?\}""", re.DOTALL)


def clean_text(text: str) -> str:
    """Strip model artifacts and inline JSON payloads."""
    text = _ARTIFACT.sub(" ", text)
    text = _JSON_BLOB.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def looks_like_prose(text: str, min_words: int = 5) -> bool:
    """True if a unit reads as natural language rather than code or markup."""
    words = text.split()
    if len(words) < min_words:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in text) / max(len(text), 1)
    if alpha < 0.75:                       # punctuation/symbol heavy
        return False
    if len(_CODEY.findall(text)) >= 2:     # one hit can be ordinary prose
        return False
    # A real sentence has function words; code and tables rarely do.
    lowered = {w.strip(".,;:!?()[]\"'").lower() for w in words}
    if not (lowered & {"the", "a", "an", "is", "are", "was", "were", "to",
                       "of", "and", "or", "in", "on", "for", "that", "this",
                       "it", "you", "we", "i", "be", "with", "as", "but"}):
        return False
    return True


@dataclass
class Document:
    doc_id: str
    domain: str
    source: str
    units: list[str] = field(default_factory=list)   # turns, paragraphs
    speakers: list[str | None] = field(default_factory=list)

    def __post_init__(self):
        keep = [i for i, u in enumerate(self.units) if u and u.strip()]
        if len(keep) != len(self.units):
            self.units = [self.units[i] for i in keep]
            if self.speakers:
                self.speakers = [self.speakers[i] for i in keep]
        if not self.speakers:
            self.speakers = [None] * len(self.units)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


# ── adapters ─────────────────────────────────────────────────────

def _conversation(domain_dir: Path):
    """One document per conversation, turns in turn_idx order."""
    for f in sorted(domain_dir.rglob("*.jsonl")):
        convs: dict[str, list[dict]] = {}
        for line in f.open():
            r = json.loads(line)
            convs.setdefault(r.get("conv_id") or f.stem, []).append(r)
        for cid, turns in convs.items():
            turns.sort(key=lambda t: t.get("turn_idx", 0))
            idx = [t.get("turn_idx", 0) for t in turns]
            # A gap means turns were dropped; windowing over it would splice
            # non-adjacent turns. Collection no longer filters, but a source
            # could reintroduce this, so it is checked rather than assumed.
            if idx != list(range(idx[0], idx[0] + len(idx))):
                continue
            yield Document(
                doc_id=cid, domain="conversation",
                source=turns[0].get("source", f.parent.name),
                units=[clean_text(t["text"]) for t in turns],
                speakers=[t.get("speaker") for t in turns],
            )


def _articles(domain_dir: Path, domain: str):
    """One document per JSONL row (article, filing, email, section)."""
    for f in sorted(domain_dir.rglob("*.jsonl")):
        for i, line in enumerate(f.open()):
            r = json.loads(line)
            text = (r.get("text") or "").strip()
            if not text:
                continue
            key = r.get("title") or r.get("path") or r.get("id") or f"{f.stem}-{i}"
            units = [c for c in (clean_text(q) for q in text.split("\n\n"))
                     if c and looks_like_prose(c)]
            if not units:
                continue
            yield Document(
                doc_id=f"{f.parent.name}/{key}", domain=domain,
                source=r.get("source", f.parent.name),
                units=units,
            )


def _narrative(domain_dir: Path, chunk_paragraphs: int = 40):
    """Books are far longer than any window, so chunk into pseudo-chapters.

    ``all_books.txt`` is skipped: it concatenates the per-book files, and
    including both would duplicate every document.
    """
    for f in sorted(domain_dir.glob("*.txt")):
        if f.stem == "all_books":
            continue
        paras = [c for c in (clean_text(q) for q in
                             f.read_text(errors="replace").split("\n\n"))
                 if c and looks_like_prose(c)]
        for c in range(0, len(paras), chunk_paragraphs):
            block = paras[c:c + chunk_paragraphs]
            yield Document(doc_id=f"{f.stem}-{c // chunk_paragraphs:04d}",
                           domain="narrative", source=f"gutenberg/{f.stem}",
                           units=block)


ADAPTERS = {
    "conversation": _conversation,
    "narrative": _narrative,
    "technical": lambda d: _articles(d, "technical"),
    "news": lambda d: _articles(d, "news"),
    "encyclopedic": lambda d: _articles(d, "encyclopedic"),
    "business": lambda d: _articles(d, "business"),
}


# ── windowing ────────────────────────────────────────────────────

def build_windows(doc: Document, tokenize, max_tokens: int = 512,
                  min_tokens: int = 32) -> list[dict]:
    """Pack a document's units into windows without crossing its boundary.

    A unit longer than ``max_tokens`` on its own is split at sentence
    boundaries rather than dropped or truncated.
    """
    windows: list[dict] = []
    cur_tokens: list[str] = []
    cur_spans: list[list[int]] = []
    cur_units: list[int] = []

    def flush():
        if len(cur_tokens) >= min_tokens:
            windows.append({
                "window_id": hashlib.sha1(
                    f"{doc.doc_id}:{len(windows)}".encode()).hexdigest()[:16],
                "doc_id": doc.doc_id, "domain": doc.domain, "source": doc.source,
                "window_idx": len(windows),
                "tokens": list(cur_tokens),
                "sentence_spans": [list(s) for s in cur_spans],
                "unit_idx": sorted(set(cur_units)),
                "n_tokens": len(cur_tokens),
            })
        cur_tokens.clear(); cur_spans.clear(); cur_units.clear()

    for ui, unit in enumerate(doc.units):
        for sent in split_sentences(unit):
            toks = tokenize(sent)
            if not toks:
                continue
            if len(toks) > max_tokens:          # pathological single sentence
                toks = toks[:max_tokens]
            if len(cur_tokens) + len(toks) > max_tokens:
                flush()
            start = len(cur_tokens)
            cur_tokens.extend(toks)
            cur_spans.append([start, len(cur_tokens)])
            cur_units.append(ui)
    flush()
    return windows


def iter_documents(domain: str):
    d = RAW / domain
    if not d.exists():
        return
    yield from ADAPTERS[domain](d)
