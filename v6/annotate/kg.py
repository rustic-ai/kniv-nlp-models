"""The ``kg`` teacher: entities, facts and fact attributes against a schema
sampled per window.

The shared ``LLMAnnotator`` builds one JSON schema per layer. Here the schema
-- which entity types and relations a window is asked for -- changes with
every item, so it travels with the item: ``annotate`` puts the item in a
context variable for the duration of its call (asyncio gives each task its
own context, so concurrent items never see each other's), and the request
builder and validator read it from there. Retries, caching, the
response-format ladder and repairs are the parent's.

Mentions come back as ``(sentence, text)`` and are aligned to window tokens
here, deterministically; a mention that cannot be found is dropped and
counted, so the alignment rate is measurable (a gate in ``kg_teacher``).
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field

from ..prompts import SYSTEM, kg_user_message
from ..schemas import MODALITIES, POLARITIES, kg_schema
from .llm import LLMAnnotator

_ITEM: contextvars.ContextVar = contextvars.ContextVar("kg_item")


@dataclass
class KGItem:
    id: str                       # window id + schema digest
    tokens: list[str]
    sentence_spans: list[list[int]]
    speakers: list[str | None] | None
    participants: list[str]       # entity ids 0..k-1
    types: list[dict]             # label-bank entity types (name, description, temporal)
    relations: list[dict]         # label-bank relations (name, description)
    domain: str = ""
    predicate_idx: int | None = None
    entities: list | None = None
    layers: dict = field(default_factory=dict)


def _norm(t: str) -> str:
    return t.lower().strip("\"'`.,;:!?()[]")


def align_mention(tokens: list[str], span: list[int], text: str, taken: set) -> tuple[int, int] | None:
    """The first unused occurrence of ``text`` (tokenised the corpus's way)
    inside the sentence ``span``, as window-absolute ``(start, end_excl)``.

    Both sides are compared without their punctuation-only tokens -- the
    corpus writes '2:55' as '2 : 55' and 'U.S.' as 'U . S .', and a teacher
    copying either form must still match -- and the match is mapped back to
    the sentence's own token positions."""
    from ..windows import tokenize
    want = [w for w in (_norm(x) for x in tokenize(text)) if w]
    if not want:
        return None
    s0, e0 = span
    kept = [(s0 + i, w) for i, w in enumerate(_norm(t) for t in tokens[s0:e0]) if w]
    n = len(want)
    for i in range(len(kept) - n + 1):
        if [w for _, w in kept[i:i + n]] == want:
            sp = (kept[i][0], kept[i + n - 1][0] + 1)
            if sp not in taken:
                return sp
    return None


def validate_kg(payload, item: KGItem):
    if not isinstance(payload, dict):
        return None, "knowledge is not an object", "parse"
    if "stats" in payload:                     # a cached, already-aligned payload
        return payload, None, None
    k = len(item.participants)
    n_sent = len(item.sentence_spans)
    types = {t["name"]: t for t in item.types}
    rels = {r["name"] for r in item.relations}
    ents, idmap, taken = [], {}, set()
    n_men = n_ok = 0
    misses = []
    for e in payload.get("entities") or []:
        if not isinstance(e, dict) or e.get("type") not in types:
            continue
        spans = []
        for m in e.get("mentions") or []:
            n_men += 1
            try:
                j = int(m["sentence"]) - 1
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= j < n_sent:
                continue
            sp = align_mention(item.tokens, item.sentence_spans[j], str(m.get("text", "")), taken)
            if sp:
                taken.add(sp); spans.append(list(sp)); n_ok += 1
            elif len(misses) < 8:
                s0, e0 = item.sentence_spans[j]
                misses.append({"text": str(m.get("text", "")),
                               "sentence": " ".join(item.tokens[s0:e0])[:300]})
        if spans:
            try:
                idmap[int(e["id"])] = k + len(ents)
            except (KeyError, TypeError, ValueError):
                continue
            ents.append({"type": e["type"], "spans": sorted(spans)})
    valid = set(range(k)) | set(idmap.values())
    temporal = {k + i for i, e in enumerate(ents) if types[e["type"]].get("temporal")}

    def ref(x, allow_none=False):
        try:
            x = int(x)
        except (TypeError, ValueError):
            return None
        if allow_none and x == -1:
            return -1
        x = x if x < k else idmap.get(x)
        return x if x in valid else None

    facts, seen = [], set()
    for f in payload.get("facts") or []:
        if not isinstance(f, dict) or f.get("relation") not in rels:
            continue
        h, t = ref(f.get("head")), ref(f.get("tail"))
        if h is None or t is None or h == t:
            continue
        key = (h, t, f["relation"])
        if key in seen:
            continue
        seen.add(key)
        holder = ref(f.get("holder"), allow_none=True)
        time = ref(f.get("time"), allow_none=True)
        facts.append({
            "head": h, "tail": t, "relation": f["relation"],
            "polarity": f.get("polarity") if f.get("polarity") in POLARITIES else "positive",
            "modality": f.get("modality") if f.get("modality") in MODALITIES else "asserted",
            "holder": -1 if holder is None else holder,
            "time": time if time in temporal else -1,
            "evidence": sorted({int(x) - 1 for x in (f.get("evidence") or [])
                                if isinstance(x, int) and 1 <= x <= n_sent}),
        })
    out = {"entities": ents, "facts": facts, "participants": item.participants,
           "stats": {"mentions": n_men, "aligned": n_ok, "misses": misses}}
    return out, None, None


class KGAnnotator(LLMAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        sents = [" ".join(item.tokens[s:e]) for s, e in item.sentence_spans]
        return [{"role": "system", "content": SYSTEM["kg"]},
                {"role": "user", "content": kg_user_message(
                    sents, item.speakers, item.participants, item.types, item.relations)
                 + self._json_hint(layer)}]

    def _build_kwargs(self, layer: str, messages: list[dict]) -> dict:
        kwargs = super()._build_kwargs(layer, messages)
        if layer == "kg" and kwargs.get("response_format", {}).get("type") == "json_schema":
            item = _ITEM.get()
            kwargs["response_format"] = {"type": "json_schema", "json_schema": kg_schema(
                [t["name"] for t in item.types], [r["name"] for r in item.relations])}
        return kwargs

    async def annotate(self, layer: str, item):
        token = _ITEM.set(item)
        try:
            return await super().annotate(layer, item)
        finally:
            _ITEM.reset(token)


def current_item():
    return _ITEM.get(None)
