"""Sentence splitter and tokenizer regressions (runtime ``kniv_v7.windows``;
``v6.windows.split_sentences`` follows it). Names with titles and initials,
dotted acronyms and abbreviated months must not be cut into sentences or
tokens.

    uv run python -m tests.test_splitter        # or: pytest tests/test_splitter.py
"""
from __future__ import annotations

ONE_SENTENCE = [
    "Sen. Robert F. Kennedy spoke today.",
    "Dr. Smith met J. K. Rowling.",
    "He joined the U.S. Army in 1990.",
    "It was Dec. 13 when Lt. Col. Smith arrived.",
    "Mr. and Mrs. Lee met Prof. Chan at St. Mary hospital.",
    "Gov. Brown signed it on Jan. 5 in Washington, D.C. after a vote.",
    "The case was Roe vs. Wade.",
    "Michael D. Papagiannis wrote No. 7.",
]
SPLITS = [
    ("The vote was 5 to 4. Justice Kennedy wrote it.", 2),
    ("She left. He stayed.", 2),
    ("Is it done? Yes! It is.", 3),
    ('He said "stop." Then he left.', 2),
    ("Sen. Robert F. Kennedy spoke today. He left.", 2),
]
TOKENS = [
    ("Sen. Robert F. Kennedy spoke.", ["Sen.", "Robert", "F.", "Kennedy", "spoke", "."]),
    ("J. K. Rowling", ["J.", "K.", "Rowling"]),
    ("the U.S. Army", ["the", "U.S.", "Army"]),
    ("Acme Inc. said", ["Acme", "Inc.", "said"]),
]


def _check(split, label):
    bad = []
    for t in ONE_SENTENCE:
        got = split(t)
        if len(got) != 1:
            bad.append(f"{label}: {t!r} -> {got}")
    for t, n in SPLITS:
        got = split(t)
        if len(got) != n:
            bad.append(f"{label}: {t!r} -> {got} (want {n})")
    return bad


def test_runtime_splitter():
    from kniv_v7.windows import split_sentences
    assert not _check(split_sentences, "kniv_v7"), _check(split_sentences, "kniv_v7")


def test_v6_splitter():
    from v6.windows import split_sentences
    assert not _check(split_sentences, "v6"), _check(split_sentences, "v6")


def test_runtime_tokenizer():
    from kniv_v7.windows import tokenize
    bad = [(t, tokenize(t), w) for t, w in TOKENS if tokenize(t) != w]
    assert not bad, bad


if __name__ == "__main__":
    fails = 0
    for f in (test_runtime_splitter, test_v6_splitter, test_runtime_tokenizer):
        try:
            f()
            print(f"PASS {f.__name__}")
        except AssertionError as e:
            fails += 1
            print(f"FAIL {f.__name__}: {e}")
    raise SystemExit(1 if fails else 0)
