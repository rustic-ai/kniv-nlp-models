"""Text to windows: sentences, words, and windows of up to 512 words.

A window never crosses a document (a text, or a conversation) and a sentence
is never split across two windows.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[A-Z0-9\"'(\[])")


_URLISH = re.compile(r"""(?:https?://|www\.)\S+|\S+@\S+\.\w+""")


_CLITIC = re.compile(
    r"(?i)(.+?)(n't|'s|'re|'ve|'ll|'d|'m)$")


_TOKEN = re.compile(r"""
      \d+(?:[.,]\d+)*(?:%|st|nd|rd|th)?   # numbers, percents, ordinals
    | \w+(?:[-']\w+)*                     # words, hyphenated, internal '
    | \.\.\.|[^\w\s]                     # ellipsis, single punctuation
""", re.VERBOSE)


def tokenize(text: str) -> list[str]:
    """Deterministic word tokenizer: clitics and contractions split off,
    punctuation separated, URLs, emails and decimals kept whole."""
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
                # Per sentence: its unit (turn, paragraph) and that unit's
                # speaker. Additive -- the window id does not depend on them.
                "sentence_units": list(cur_units),
                "sentence_speakers": [doc.speakers[u] for u in cur_units],
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

