"""Mention proposals from the linguistic layer (v8 plan r2, pilot P1).

Fact arguments on short text are mostly not named entities: noun phrases,
values, roles (E8). The model's own NER (OntoNotes, which covers money,
dates, quantities and percentages) and its parse see them, but the
knowledge layer proposed only the span head's and the DocRED entity head's
spans. ``propose`` adds, per window:

- ``ner``: the OntoNotes NER head's spans (Viterbi over its BIO tags);
- ``np``: noun phrases from the dependency parse. For each NOUN, PROPN or
  NUM word: its *core* phrase (the word and the subtrees of its compound,
  flat, adjectival, numeric, possessive and determiner dependents) and its
  *full* subtree, each with and without a leading determiner, when the span
  stays inside the sentence and within ``MAX_WORDS``.

The span head then types every proposal; proposals compete with its own
spans in the greedy decode with the calibrated ``span_boost``.
"""
from __future__ import annotations

MAX_WORDS = 12
NP_HEADS = {"NOUN", "PROPN", "NUM"}
CORE = {"compound", "flat", "flat:name", "flat:foreign", "amod", "nummod", "nmod:poss",
        "fixed", "goeswith", "det", "det:poss", "case", "advmod", "nmod:npmod", "obl:npmod"}
LEADING = {"det", "det:poss", "nmod:poss"}
EDGE_POS = {"PUNCT", "CCONJ", "ADP"}


def ner_spans(o: dict, vocab, W: int) -> set[tuple[int, int]]:
    from .entities import bio_spans
    from .train.gold_eval import bio_constraints, viterbi
    tags = vocab.labels["ner"]
    allowed, start = bio_constraints(tags)
    return {(a, b) for _, a, b in bio_spans([tags[x] for x in viterbi(o["ner"][:W], allowed, start)])}


def np_spans(sp, o: dict, vocab, W: int) -> set[tuple[int, int]]:
    pos = [vocab.labels["pos"][int(x)] for x in o["pos"][:W]]
    rel = [vocab.labels["dep"][int(x)] for x in o["dep_rel"][:W]]
    head = [int(h) - 1 for h in o["dep_head"][:W]]          # -1: root
    kids: dict[int, list[int]] = {}
    for i, h in enumerate(head):
        if h >= 0:
            kids.setdefault(h, []).append(i)
    sent = {}
    for k, (s, e) in enumerate(sp):
        for i in range(s, min(e, W)):
            sent[i] = (s, min(e, W))

    def subtree(i, keep=None):
        out, stack = [], [i]
        while stack:
            j = stack.pop()
            out.append(j)
            for c in kids.get(j, []):
                if keep is None or (j == i and rel[c] in keep) or (j != i and keep is CORE):
                    stack.append(c)
        return sorted(out)

    out: set[tuple[int, int]] = set()

    def add(words, i):
        s0, e0 = sent.get(i, (0, W))
        a, b = min(words), max(words) + 1
        while a < b and pos[a] in EDGE_POS:
            a += 1
        while b > a and pos[b - 1] in EDGE_POS:
            b -= 1
        if a >= b or a < s0 or b > e0 or b - a > MAX_WORDS:
            return
        out.add((a, b))
        if rel[a] in LEADING and a != i and b - a > 1:      # without "the", "its", "our"
            out.add((a + 1, b))

    for i in range(W):
        if pos[i] not in NP_HEADS or i not in sent:
            continue
        add(subtree(i, CORE), i)
        add(subtree(i), i)
    return out


def propose(tokens: list[str], sp, o: dict, vocab, sources) -> set[tuple[int, int]]:
    W = len(o["ner"])
    out: set[tuple[int, int]] = set()
    if "ner" in sources:
        out |= ner_spans(o, vocab, W)
    if "np" in sources and o.get("dep_head") is not None:
        out |= np_spans([tuple(x) for x in sp], o, vocab, W)
    return out
