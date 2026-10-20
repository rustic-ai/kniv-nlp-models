"""Entities from typed mention spans and coreference chains.

Mentions of one entity are merged in two stages: a coreference chain (in the
mode chosen, ``coref``) absorbs the spans it overlaps, then exact normalised
surface forms are merged for person, organisation, location and other names.
Numbers and times are never merged by surface form: two occurrences of `1`
or `Monday` are not the same entity.
"""
from __future__ import annotations

# Fine NER types -> the six coarse types.
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
COARSE_TYPES = {"LOC", "MISC", "NUM", "ORG", "PER", "TIME"}

# Counts and ordinals are not entities for relation extraction.
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
    """Return ``(sents, vertexSet)`` (entities as lists of mentions), or ``(None, None)``.

    ``coref``: how coreference chains group the entity mentions -- ``all``
    (every chain, split by type), ``compatible`` (within a chain, only
    mentions whose names are compatible, ``names_compatible``), ``none``.
    ``partial_names``: also merge
    an entity into the one other entity whose name contains it or is its
    acronym (several candidates: ambiguous, left alone). ``link``: a function
    from the typed mention spans ``[(type, start, end_exclusive)]`` to groups
    of their indices; it replaces both the chains and
    the name rules. ``link_merge``: a function from the typed spans and the
    rules' groups to coarser groups.

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
            # Coarse types (from the entity head) pass through as is.
            spans.append((ty if ty in COARSE_TYPES else TYPE_MAP.get(ty, "MISC"),
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
