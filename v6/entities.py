"""Entity clusters from NER spans plus coref chains.

Shared by the relation probe and the corpus builder, because the relation
annotator does not find entities — it classifies relations over clusters we
supply. Getting this wrong shows up as duplicate graph nodes rather than as
an error, so the logic lives in one place.

The merge is two-stage:

1. A coref chain absorbs every NER span it overlaps.
2. Exact normalised surface forms are merged for PER/ORG/LOC/MISC.

Stage 2 is not redundant. Measured on 177 corpus windows, coref alone left
**12.6% of entities as unmerged duplicates** of the same surface form —
26.8% on conversation — so `George Orwell` became three entities and
produced three identical `author` triples: three graph nodes where there
should be one. Adding it drops duplicates to 0.5% and removed 293 triples,
28% of the raw output.

NUM and TIME are excluded from stage 2 by design: two occurrences of `1` or
`Monday` are not the same entity.
"""
from __future__ import annotations

# OntoNotes 18 -> DocRED's six coarse types, which is what the relation
# annotator was trained to condition on.
TYPE_MAP = {
    "PERSON": "PER", "ORG": "ORG",
    "GPE": "LOC", "LOC": "LOC", "FAC": "LOC",
    "DATE": "TIME", "TIME": "TIME",
    "PERCENT": "NUM", "MONEY": "NUM", "QUANTITY": "NUM",
    "ORDINAL": "NUM", "CARDINAL": "NUM",
    "NORP": "MISC", "PRODUCT": "MISC", "EVENT": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
}
MERGEABLE = {"PER", "ORG", "LOC", "MISC"}

# OntoNotes tags every bare count and rank -- "two", "12", "first" -- as
# CARDINAL or ORDINAL. DocRED does not annotate those as entities at all, and
# they are not relation arguments: across Re-DocRED train, NUM appears in 30 of
# 85,932 gold relations (0.03%), and those few look like mistyped entities. So
# ATLOP has effectively never seen a NUM argument and cannot predict one.
#
# They are not rare here. Share of all mentions, measured over the full corpus
# (NER is complete for all 583,824 sentences):
#
#   domain          PER    ORG    LOC   TIME    NUM   MISC   bare CARDINAL/ORDINAL
#   conversation   5.2%   8.8%  20.1%  27.6%  27.6%  10.7%   21.4%
#   news          17.6%  16.5%  22.3%  17.3%  13.9%  12.5%   10.2%
#   technical     10.9%  14.3%   2.8%  10.0%  31.5%  30.4%   28.4%
#   encyclopedic  20.5%   8.5%  21.7%  15.9%  16.8%  16.5%   13.7%
#   narrative     47.4%   3.3%  11.1%  13.7%  19.0%   5.5%   17.4%
#   Re-DocRED     15.6%  13.5%  29.3%  19.3%   6.4%  15.9%
#
# Dropping them leaves NUM at roughly Re-DocRED's 6.4% -- the remaining MONEY,
# PERCENT and QUANTITY are the DocRED-like members and stay. Pairs grow with
# the square of the vertex count, so this is also most of the ATLOP cost.
#
# The inflation is worse per VERTEX than per mention, because MERGEABLE excludes
# NUM: PER/ORG/LOC/MISC mentions collapse by string match while every bare
# numeral stays its own vertex. On the conversation prefix that turns 27.6% of
# mentions into 43.3% of vertices.
DROP_TYPES = {"CARDINAL", "ORDINAL"}


def bio_spans(tags: list[str]) -> list[tuple[str, int, int]]:
    """(type, start, end_exclusive) from BIO tags."""
    spans, cur, start = [], None, 0
    for i, t in enumerate(list(tags) + ["O"]):
        if not t.startswith("I-"):
            if cur:
                spans.append((cur, start, i))
            cur = t[2:] if t.startswith("B-") else None
            start = i
        elif cur is None:
            cur, start = t[2:], i
    return spans


def build_entities(tokens: list[str], sentence_spans: list[list[int]],
                   ner_by_sentence: list[list[str] | None],
                   coref_chains: list[list[list[int]]] | None):
    """Return DocRED-format ``(sents, vertexSet)``, or ``(None, None)``.

    ``None`` when fewer than two entities survive — a relation needs a pair,
    and a window with one entity is not evidence of anything.
    """
    spans: list[tuple[str, int, int]] = []
    for si, (s, e) in enumerate(sentence_spans):
        tags = ner_by_sentence[si] if si < len(ner_by_sentence) else None
        if not tags:
            continue
        for ty, a, b in bio_spans(tags):
            if ty in DROP_TYPES:
                continue
            spans.append((TYPE_MAP.get(ty, "MISC"), s + a, s + b))
    if len(spans) < 2:
        return None, None

    used, groups = set(), []
    for chain in (coref_chains or []):
        members = []
        for cs, ce in chain:
            for i, (ty, a, b) in enumerate(spans):
                if i not in used and a <= ce and cs <= b - 1:
                    members.append(i); used.add(i)
        if members:
            groups.append(members)
    for i in range(len(spans)):
        if i not in used:
            groups.append([i])

    by_name: dict[tuple[str, str], int] = {}
    merged: list[list[int]] = []
    for g in groups:
        ty = spans[g[0]][0]
        key = (ty, " ".join(tokens[spans[g[0]][1]:spans[g[0]][2]]).lower())
        if ty in MERGEABLE and len(key[1]) > 2 and key in by_name:
            merged[by_name[key]].extend(g)
        else:
            if ty in MERGEABLE:
                by_name[key] = len(merged)
            merged.append(list(g))

    offs = [s for s, _ in sentence_spans]
    sents = [tokens[s:e] for s, e in sentence_spans]

    def sent_of(pos: int) -> int:
        return max(i for i, o in enumerate(offs) if o <= pos)

    vertex = []
    for g in merged:
        mentions = []
        for i in g:
            ty, a, b = spans[i]
            si = sent_of(a)
            if b - offs[si] > len(sents[si]):     # mention crosses a sentence
                continue
            mentions.append({"name": " ".join(tokens[a:b]),
                             "pos": [a - offs[si], b - offs[si]],
                             "sent_id": si, "type": ty})
        if mentions:
            vertex.append(mentions)
    if len(vertex) < 2:
        return None, None
    return sents, vertex
