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
DOCRED_TYPES = {"LOC", "MISC", "NUM", "ORG", "PER", "TIME"}

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


_FUNCTION = {"the", "a", "an", "of", "and", "&", "'s", "’s", ",", ".", "-", "inc", "inc.",
             "ltd", "ltd.", "co", "co.", "corp", "corp.", "mr", "mr.", "mrs", "mrs.", "dr", "dr."}


def _name_tokens(tokens: list[str], a: int, b: int) -> tuple[str, ...]:
    return tuple(t.lower() for t in tokens[a:b] if t.lower() not in _FUNCTION)


def _acronym(short: tuple[str, ...], raw_short: list[str], long_raw: list[str]) -> bool:
    s = "".join(raw_short).replace(".", "")
    caps = [t for t in long_raw if t[:1].isupper() and t.lower() not in _FUNCTION]
    return len(s) >= 2 and s.isupper() and len(caps) >= 2 and s == "".join(t[0] for t in caps)


def names_compatible(tokens: list[str], m1: tuple[int, int], m2: tuple[int, int]) -> bool:
    """Could two mentions name the same thing: equal names, one name's
    content words inside the other's ("Obama" / "Barack Obama", "Apple" /
    "Apple Inc."), or an acronym of it ("UN" / "United Nations")."""
    n1, n2 = _name_tokens(tokens, *m1), _name_tokens(tokens, *m2)
    if not n1 or not n2:
        return False
    if n1 == n2:
        return True
    short, long_, rs, rl = ((n1, n2, m1, m2) if len(n1) <= len(n2) else (n2, n1, m2, m1))
    if set(short) <= set(long_):
        return True
    return _acronym(short, tokens[rs[0]:rs[1]], tokens[rl[0]:rl[1]])


def build_entities(tokens: list[str], sentence_spans: list[list[int]],
                   ner_by_sentence: list[list[str] | None],
                   coref_chains: list[list[list[int]]] | None,
                   coref: str = "all", partial_names: bool = False, link=None,
                   link_merge=None):
    """Return DocRED-format ``(sents, vertexSet)``, or ``(None, None)``.

    ``coref``: how coreference chains group the entity mentions -- ``all``
    (every chain, split by type), ``compatible`` (within a chain, only
    mentions whose names are compatible, ``names_compatible``), ``none``.
    Measured on Re-DocRED (gold_eval ``redocred_oracle``), chains fused two
    gold entities in 3-8% of predicted entities. ``partial_names``: also merge
    an entity into the one other entity whose name contains it or is its
    acronym (several candidates: ambiguous, left alone). ``link``: a function
    from the typed mention spans ``[(type, start, end_exclusive)]`` to groups
    of their indices (``entity_link.Linker``); it replaces both the chains and
    the name rules. ``link_merge``: a function from the typed spans and the
    rules' groups to coarser groups (``entity_link.Linker.merge``, the hybrid).

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
            # DocRED types (from the entity-mention head) pass through as is.
            spans.append((ty if ty in DOCRED_TYPES else TYPE_MAP.get(ty, "MISC"),
                          s + a, s + b))
    if len(spans) < 2:
        return None, None

    if link is not None:
        merged = [list(g) for g in link(spans)]
        partial_names = False
        coref_chains, coref = None, "none"
    used, groups = set(), []
    for chain in (coref_chains if coref != "none" else None) or []:
        members = []
        for cs, ce in chain:
            for i, (ty, a, b) in enumerate(spans):
                if i not in used and a <= ce and cs <= b - 1:
                    members.append(i); used.add(i)
        # A coref chain is not evidence that two mentions are the same ENTITY
        # when the NER head calls them different things. LingMess merges
        # across types -- ("tomorrow" TIME, "Team Meeting" MISC),
        # ("2010" TIME, "Toyota" ORG, "Camry" MISC) -- and the vertex then
        # took the FIRST mention's type and wore it for all of them.
        # Measured before this split: 4.71% of entities had mentions
        # disagreeing on type, and 9.51% of relation triples touched one.
        #
        # Splitting by type keeps the coref evidence where the two annotators
        # agree and discards it only where they contradict each other, which
        # is the same principle as masking a sentence whose parse is not a
        # tree rather than repairing it.
        bytype: dict[str, list[int]] = {}
        for i in members:
            bytype.setdefault(spans[i][0], []).append(i)
        if coref == "compatible":
            for ms in bytype.values():
                parent = {i: i for i in ms}

                def find(i):
                    while parent[i] != i:
                        parent[i] = parent[parent[i]]
                        i = parent[i]
                    return i
                for x in range(len(ms)):
                    for y in range(x + 1, len(ms)):
                        if names_compatible(tokens, spans[ms[x]][1:], spans[ms[y]][1:]):
                            parent[find(ms[x])] = find(ms[y])
                comp: dict[int, list[int]] = {}
                for i in ms:
                    comp.setdefault(find(i), []).append(i)
                groups.extend(comp.values())
        else:
            groups.extend(bytype.values())
    for i in range(len(spans)):
        if i not in used:
            groups.append([i])

    by_name: dict[tuple[str, str], int] = {}
    if link is None:
        merged = []
    for g in (groups if link is None else []):
        ty = spans[g[0]][0]
        key = (ty, " ".join(tokens[spans[g[0]][1]:spans[g[0]][2]]).lower())
        if ty in MERGEABLE and len(key[1]) > 2 and key in by_name:
            merged[by_name[key]].extend(g)
        else:
            if ty in MERGEABLE:
                by_name[key] = len(merged)
            merged.append(list(g))

    if link_merge is not None and link is None:
        merged = link_merge(spans, merged)

    if partial_names:
        # A short name joins the single longer entity it is part of; with two
        # or more candidates ("Smith" beside "John Smith" and "Mary Smith")
        # it stays apart.
        order = sorted(range(len(merged)), key=lambda k: -max(spans[i][2] - spans[i][1]
                                                               for i in merged[k]))
        alive = set(range(len(merged)))
        for k in reversed(order):                       # shortest first
            if k not in alive or spans[merged[k][0]][0] not in MERGEABLE:
                continue
            ty = spans[merged[k][0]][0]
            cand = [j for j in alive if j != k and spans[merged[j][0]][0] == ty
                    and any(spans[j_][2] - spans[j_][1] > spans[i][2] - spans[i][1]
                            and names_compatible(tokens, spans[i][1:], spans[j_][1:])
                            for i in merged[k] for j_ in merged[j])]
            if len(cand) == 1:
                merged[cand[0]].extend(merged[k])
                alive.discard(k)
        merged = [merged[k] for k in sorted(alive)]

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
