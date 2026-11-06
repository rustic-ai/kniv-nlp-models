"""Short-text knowledge teacher (v8 W1 + W4): 1-3 sentence texts with their own
small schemas, and minimal pairs for negation and modality.

    uv run python -m v6.kg_short dry --texts 500          # no API: items, 5 prompts, overlap check
    uv run python -m v6.kg_short pilot --texts 500        # part A: annotate the short texts
    uv run python -m v6.kg_short variants --bases 100     # part B: rewrite facts, annotate the rewrites
    uv run python -m v6.kg_short check                    # judge precision, second-annotator recall, pair agreement

Why short texts: every v7 relation source is long (Re-DocRED and Wikipedia
documents, ~200-word teacher windows), and v7 fails on 25-word sentences with
their own 5-relation schema (CoNLL04 zero-shot 0.17-0.28 end to end).

- **Texts** are 1-3 consecutive sentences cut from corpus *train* windows (no
  new source), kept only when they carry at least two entity or value cues.
- **Schemas** have 3-15 relations: half "normal" (domain prior, neighbours,
  random, as the v7 sampler), 30% leaning on value and list relations (price,
  amount, count, uses, ...), 20% "empty" (relations unlikely in the domain), so
  the model learns to return nothing.
- **Minimal pairs** take plain asserted facts from part A and ask for six
  rewrites changing as little as possible: negated, reported, planned,
  desired, questioned, hypothetical. The expected polarity and modality are
  known from the rewrite instruction; the teacher annotates each rewrite too,
  and the two must agree.

The ``kgs`` system prompt is the v7 ``kg`` prompt plus rules for values, lists
and empty answers. Its responses are cached under their own prompt version
(``SHORT_VERSION``), so the v7 teacher cache (``PROMPT_VERSION``) is untouched.
CoNLL04, CrossRE and the FewRel validation relations stay out: relations with
``train: false`` never enter a schema, and ``dry`` checks 8-gram overlap with
the CoNLL04 texts when the dataset is available.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .annotate.base import CacheStore, RunLogger
from .annotate.kg import KGAnnotator, KGItem
from .config import DATA_DIR, RUNS_DIR, load_annotators
from .kg_teacher import CACHE, SPEAKERS, SchemaSampler, _rng, participants_of
from .label_bank import load as load_bank
from .prompts import SYSTEM, kg_user_message

# Round 1 (the first pilot) and round 2 (v7-ranked schemas, the negation rule,
# filtered bases, judged rewrites) keep their own items, cache and run logs.
ROUND = 1
SHORT_VERSION = "s1"
OUT = RUNS_DIR / "kg-short"
ITEMS = DATA_DIR / "v6-corpus" / "kg_short_items.json"
RANK = DATA_DIR / "v6-corpus" / "kg_short_rank.json"
RANK_TOP = 20


# Round 3: round 2 plus value relations ranked by v7, main-clause rewrites with
# a stricter judge; part A only on value-leaning texts with a number, and the
# rewrite bases taken from round 2's part A.
BASE_ITEMS, BASE_VERSION = ITEMS, SHORT_VERSION


# Per-run request settings: the teacher's reasoning effort (None = the
# deployment default, as in the pilots) and the concurrency cap. The effort
# changes the answers, so it is part of the cache version.
TEACHER_EFFORT: str | None = None
SPLIT = "train"
CONCURRENCY: int | None = None


def _spec(name: str, effort: str | None):
    import dataclasses
    spec = load_annotators([name])[name]
    extra = dict(spec.extra or {})
    if effort:
        extra["reasoning_effort"] = effort
    return dataclasses.replace(spec, extra=extra, max_concurrency=CONCURRENCY or spec.max_concurrency)


def teacher_spec(name: str):
    return _spec(name, TEACHER_EFFORT)


def judge_spec(name: str):
    return _spec(name, None)


def store(version: str) -> CacheStore:
    return CacheStore(CACHE, f"{version}-{TEACHER_EFFORT}" if TEACHER_EFFORT else version)


def set_round(n: int) -> None:
    global ROUND, SHORT_VERSION, OUT, ITEMS, BASE_ITEMS, BASE_VERSION
    ROUND = n
    if n >= 2:
        SHORT_VERSION = f"s{n}"
        OUT = RUNS_DIR / f"kg-short-r{n}"
        ITEMS = DATA_DIR / "v6-corpus" / f"kg_short_items_r{n}.json"
    BASE_ITEMS, BASE_VERSION = ITEMS, SHORT_VERSION
    if n == 3:
        BASE_ITEMS, BASE_VERSION = DATA_DIR / "v6-corpus" / "kg_short_items_r2.json", "s2"
    if n == 4:
        # the full run: round-2 schemas and prompt, several slices per window;
        # items and ranks per corpus split (dev texts calibrate, never train)
        global RANK
        SHORT_VERSION = "s4"
        ITEMS = DATA_DIR / "v6-corpus" / f"kg_short_items_r4_{SPLIT}.json"
        RANK = DATA_DIR / "v6-corpus" / f"kg_short_rank_r4_{SPLIT}.json"
        BASE_ITEMS, BASE_VERSION = ITEMS, SHORT_VERSION

# Relations whose tail is usually a value or one item of a list.
VALUE_RE = re.compile(r"\b(price|prices|monetary|costs?|count|value|revenue|amount|duration|percent|"
                      r"percentage|salary|stake|budget|fee|fees|uses|skill|skills|expertise|ingredient|capacity|"
                      r"population|age|rate|valuation|funding|quantity|number)\b", re.I)
CUE_RE = re.compile(r"\d|[$€£¥%]")
NUM_RE = re.compile(r"\d")
# value relations whose tail is one item of a list rather than a number
LIST_RELATIONS = {"uses", "has expertise in", "has ingredient", "uses payment method"}
MAX_WORDS = 120
MIN_SENT_WORDS = 5   # drops reference lists, headings and table fragments

# The types users put in their own schemas, which neither DocRED's six nor
# Few-NERD's 66 cover (no unnamed financial items, roles or skills). Written to
# be mutually exclusive, so every entity has one right type. Not yet in the
# label bank: added to it (with ids) when the W1 data is prepared.
GENERAL_TYPES = [{"name": n, "description": d, "source": "general-short"} for n, d in [
    ("person", "a human being, named or described ('my sister', 'the CEO')"),
    ("organization", "a company, institution, team, government body or other group"),
    ("location", "a place: a country, city, region, building, address or natural feature"),
    ("product or service", "something made or offered: a device, software product, plan, dish or service"),
    ("financial item", "a loan, credit facility, account, fund, obligation, share class or other financial "
                       "instrument or balance"),
    ("document or agreement", "a contract, licence, policy, law, report, filing or other written instrument"),
    ("event", "a meeting, deal, launch, trip, election, incident or other occurrence"),
    ("job title", "a role or position someone holds, such as 'data engineer' or 'chief financial officer'"),
    ("skill or technology", "a skill, programming language, tool, method or technical field"),
    ("activity", "something people do: a sport, hobby, task or practice, such as 'surfing' or 'hiking'"),
]]


def is_value_relation(r: dict) -> bool:
    return bool(VALUE_RE.search(r["name"]))


def cues(tokens: list[str], spans: list[list[int]]) -> int:
    """Entity or value cues: capitalised words not starting a sentence, and
    numbers / currency / percent tokens."""
    n = 0
    for s, e in spans:
        for i in range(s, e):
            t = tokens[i]
            if CUE_RE.search(t):
                n += 1
            elif i > s and t[:1].isupper() and t.lower() not in {"i", "i'm", "i've", "i'll", "i'd"}:
                n += 1
    return n


class ShortSampler:
    """3-15 relations per text, from the v7 sampler's pool and near-synonym
    conflicts; modes 'normal' (0.5), 'value' (0.3) and 'empty' (0.2)."""

    def __init__(self, base: SchemaSampler, rank: dict | None = None):
        self.b = base
        self.value = np.array([is_value_relation(r) for r in base.relations], dtype=float)
        self.listy = np.array([r["name"] in LIST_RELATIONS for r in base.relations], dtype=float)
        self.numeric = self.value * (1 - self.listy)
        self.index = {r["name"]: i for i, r in enumerate(base.relations)}
        # round 2: per text, the pool relations v7 ranks highest for it (best first)
        self.rank = rank

    def mode(self, key: str) -> str:
        x = _rng(f"smode:{key}").random()
        return "value" if x < 0.3 else ("empty" if x < 0.5 else "normal")

    def relations_for(self, key: str, domain: str, has_number: bool = False) -> tuple[list[int], str]:
        if self.rank is not None:
            return self.ranked_for(key, domain, has_number)
        b, rng, mode = self.b, _rng(f"srel:{key}"), self.mode(key)
        n = int(rng.integers(3, 16))
        pick: list[int] = []

        def draw(weights, k):
            for _ in range(k):
                w = np.array(weights, dtype=float)
                for i in pick:
                    w[i] = 0
                    for j in b.conflicts[i]:
                        w[j] = 0
                if w.sum() <= 0:
                    return
                pick.append(int(rng.choice(len(w), p=w / w.sum())))

        prior = b.prior(domain)
        if mode == "empty":
            # relations this domain's text rarely supports: the bottom half of its prior
            low = (prior <= np.median(prior)).astype(float)
            draw(low, n)
        else:
            if mode == "value":
                draw(self.value + 1e-9, round(n * 0.5))
            draw(prior, round(n * 0.5) if mode == "normal" else n - len(pick) - round(n * 0.2))
            neigh = np.zeros(len(b.relations))
            for i in list(pick):
                for j in b.by_cat[b.relations[i]["category"]]:
                    neigh[j] = 1.0
            draw(neigh + 1e-9, round(n * 0.2))
            draw(np.ones(len(b.relations)), n - len(pick))
        return sorted(pick), mode

    def ranked_for(self, key: str, domain: str, has_number: bool) -> tuple[list[int], str]:
        """Round 2. 'normal': 40% from v7's top 20 for this text, 20% domain
        prior, 20% neighbours of the picks (hard negatives), the rest random.
        'value': 40% numeric value relations if the text has a number, else
        list relations; 20% v7's top 20; 20% neighbours; the rest random.
        'empty': as round 1."""
        b, rng, mode = self.b, _rng(f"srel2:{key}"), self.mode(key)
        n = int(rng.integers(3, 16))
        pick: list[int] = []

        def draw(weights, k):
            for _ in range(k):
                w = np.array(weights, dtype=float)
                for i in pick:
                    w[i] = 0
                    for j in b.conflicts[i]:
                        w[j] = 0
                if w.sum() <= 0:
                    return
                pick.append(int(rng.choice(len(w), p=w / w.sum())))

        prior = b.prior(domain)
        if mode == "empty":
            draw((prior <= np.median(prior)).astype(float), n)
            return sorted(pick), mode
        top = np.zeros(len(b.relations))
        entry = self.rank.get(key) or []
        ranked = entry["top"] if isinstance(entry, dict) else entry
        for name in ranked[:RANK_TOP]:
            if name in self.index:
                top[self.index[name]] = 1.0
        if mode == "value":
            kind = self.numeric if has_number else self.listy
            if ROUND >= 3 and isinstance(entry, dict):
                # the value relations v7 ranks highest for this text, best first
                order = [self.index[r] for r in entry["value"] if r in self.index and kind[self.index[r]]]
                w = np.zeros(len(b.relations))
                w[order[:5]] = 1.0
                draw(w + 1e-9 * kind, round(n * 0.4))
            else:
                draw(kind + 1e-9, round(n * 0.4))
            draw(top + 1e-9, round(n * 0.2))
        else:
            draw(top + 1e-9, round(n * 0.4))
            draw(prior, round(n * 0.2))
        neigh = np.zeros(len(b.relations))
        for i in list(pick):
            for j in b.by_cat[b.relations[i]["category"]]:
                neigh[j] = 1.0
        draw(neigh + 1e-9, round(n * 0.2))
        draw(np.ones(len(b.relations)), n - len(pick))
        return sorted(pick), mode

    def types_for(self, key: str) -> list[dict]:
        """Types that can cover a short text: the coarse six (0.6), or 4-10
        fine types plus the fine catch-alls ('other person', ...), so every
        entity still has exactly one right type. Never coarse and fine
        together (an entity would have two right answers)."""
        b, rng = self.b, _rng(f"stype:{key}")
        x = rng.random()
        if x < 0.4:
            return [t for t in b.coarse if t["code"] not in ("TIME", "NUM")] + b.always
        if x < 0.7:
            return GENERAL_TYPES + b.always
        catch = [t for t in b.fine if t["name"].startswith("other ")]
        rest = [t for t in b.fine if not t["name"].startswith("other ")]
        k = int(rng.integers(4, 11))
        chosen = [rest[i] for i in sorted(rng.choice(len(rest), k, replace=False))]
        return chosen + catch + b.always


def _slices(r: dict) -> list[tuple[int, int]]:
    """Candidate (first sentence, length) slices of one window. Rounds 1-3:
    one random 1-3 sentence slice. Round 4: the whole window cut into
    consecutive, non-overlapping 1-3 sentence pieces, so a domain with few
    windows (encyclopedic: 1.6k train windows, 28k sentences) can still fill
    its share."""
    n = len(r["sentence_spans"])
    if not n:
        return []
    if ROUND < 4:
        rng = _rng(f"slice:{r['window_id']}")
        L = int(rng.integers(1, min(3, n) + 1))
        return [(int(rng.integers(0, n - L + 1)), L)]
    rng, out, a = _rng(f"slices4:{r['window_id']}"), [], 0
    while a < n:
        L = min(int(rng.integers(1, 4)), n - a)
        out.append((a, L))
        a += L
    return out


def short_items(n_texts: int, sampler: ShortSampler, split: str = "train") -> list[KGItem]:
    """``n_texts`` short texts, spread evenly over the corpus domains. Slices
    are taken one per window first, then second slices, and so on, so a
    domain uses as many different windows as it can."""
    from .train.data import load_rows
    rows = load_rows(split, columns=["window_id", "domain", "tokens", "sentence_spans"])
    speakers = json.loads(SPEAKERS.read_text()) if SPEAKERS.exists() else {}
    by = defaultdict(list)
    for r in rows:
        by[r["domain"]].append(r)
    per = max(1, n_texts // len(by))
    items = []
    for d in sorted(by):
        got = 0
        cand = []
        for r in by[d]:
            h = hashlib.sha1(f"kgshort:{r['window_id']}".encode()).hexdigest()
            for j, (a, L) in enumerate(sorted(_slices(r), key=lambda x: _rng(f"o:{r['window_id']}:{x}").random())):
                cand.append((j, h, a, L, r))
        cand.sort(key=lambda x: (x[0], x[1], x[2]))
        for _, _, a, L, r in cand:
            if got >= per:
                break
            sp_all = [list(x) for x in r["sentence_spans"]]
            s0, e0 = sp_all[a][0], sp_all[a + L - 1][1]
            if e0 - s0 > MAX_WORDS or any(e - s < MIN_SENT_WORDS for s, e in sp_all[a:a + L]):
                continue
            tokens = r["tokens"][s0:e0]
            spans = [[s - s0, e - s0] for s, e in sp_all[a:a + L]]
            spk = speakers.get(r["window_id"])
            spk = spk[a:a + L] if spk else None
            need = 1 if spk and any(spk) else 2
            if cues(tokens, spans) < need:
                continue
            key = f"{r['window_id']}#{a}-{a + L}"
            if sampler.rank is not None and key not in sampler.rank:
                continue                      # round 2: only texts v7 has ranked
            has_number = any(NUM_RE.search(t) for t in tokens)
            if ROUND == 3 and not (sampler.mode(key) == "value" and has_number):
                continue                      # round 3: value-leaning texts with a number only
            rel_ids, mode = sampler.relations_for(key, d, has_number)
            types = sampler.types_for(key)
            rels = [sampler.b.relations[i] for i in rel_ids]
            digest = hashlib.sha1(json.dumps([[x["name"] for x in rels], [t["name"] for t in types]]
                                             ).encode()).hexdigest()[:10]
            items.append(KGItem(
                id=f"{key}:{digest}", tokens=tokens, sentence_spans=spans, speakers=spk,
                participants=participants_of(spk), types=types,
                relations=[{"name": x["name"], "description": x["description"]} for x in rels],
                domain=d, layers={"mode": mode, "window_id": r["window_id"]}))
            got += 1
    return items


# ── prompts ──────────────────────────────────────────────────────────
SYSTEM["kgs"] = SYSTEM["kg"] + (
    "\n\nSHORT TEXTS, VALUES AND LISTS.\n"
    "- The passage may be a single sentence. Extract from it exactly as from a long one.\n"
    "- Amounts, prices, percentages, counts, ages, dates and durations are entities (type "
    "number or time) whenever a listed relation takes them: in 'Northwind will buy CareLoop "
    "for $1.2 billion', '$1.2 billion' is an entity and the tail of a price relation.\n"
    "- Each item of a list is its own entity and its own fact: 'skilled in Python and dbt' "
    "gives one fact for Python and one for dbt.\n"
    "- If no listed relation holds, return no facts. Never stretch a relation to fit the text."
)

SYSTEM["kgs2"] = SYSTEM["kgs"] + (
    "\n- A negated statement is still a fact to list, with polarity negative: 'Kaili was not "
    "released from detention', 'Sarah Kim was not charged', 'I don't eat meat'. Do not leave it "
    "out because it does not hold.\n"
    "- Planned, wished, asked, conditional and reported statements are facts to list too, with "
    "their modality; never drop a fact because it is not asserted."
)

SYSTEM["kgvarjudge"] = (
    "You check rewrites of one sentence. Each rewrite should express the same fact (the same "
    "head, relation and tail) with a different status: negated (the text says it does not "
    "hold), reported (someone else is said to claim it), planned, desired (someone wants it), "
    "questioned (it is asked about) or hypothetical (conditional or possible). A rewrite is ok "
    "if it reads naturally, keeps the head and tail mentions, and a careful reader would see "
    "that exact fact with that exact status. It is not ok if the status attaches to a "
    "different part of the sentence, the fact itself changed, or the sentence is unnatural."
)

SYSTEM["kgvarjudge3"] = SYSTEM["kgvarjudge"] + (
    " Also not ok: the status is placed in a relative or subordinate clause attached to one of "
    "the entities ('X, who was scheduled to become a general, ...') instead of on the "
    "statement of the fact itself; or the sentence states the fact twice, once with and once "
    "without the status."
)

SYSTEM["kgvar"] = (
    "You rewrite one sentence so that one fact in it changes status, changing as few words as "
    "possible. Keep the two entity mentions exactly as written (same spelling and case), keep "
    "the rest of the sentence natural and grammatical, and do not add other facts. Write one "
    "rewrite for each kind:\n"
    "- negated: the text says the fact does NOT hold ('X does not work for Y', 'X was not "
    "charged with Y').\n"
    "- reported: someone else, named or described in the sentence, is said to claim it ('Her "
    "manager says X works for Y'); give that person's mention as 'source'.\n"
    "- planned: it is intended or scheduled, not yet true ('X will join Y next month').\n"
    "- desired: someone wants or wishes it ('X would love to work for Y').\n"
    "- questioned: it is asked about ('Does X work for Y?').\n"
    "- hypothetical: it is conditional or possible ('If the deal closes, X might work for Y')."
)
SYSTEM["kgvar3"] = SYSTEM["kgvar"] + (
    "\nPut the change on the clause that states the fact, never in a relative clause attached "
    "to one of the entities ('X, who was scheduled to ...'). If the fact is stated inside a "
    "longer sentence, you may shorten the sentence to the clause that states it, keeping both "
    "mentions exactly as written."
)
VAR_KINDS = ("negated", "reported", "planned", "desired", "questioned", "hypothetical")
# kind -> (polarity, modality) a correct annotation of the rewrite must carry
VAR_EXPECT = {"negated": ("negative", "asserted"), "reported": ("positive", "reported"),
              "planned": ("positive", "planned"), "desired": ("positive", "desired"),
              "questioned": ("positive", "questioned"), "hypothetical": ("positive", "hypothetical")}


def var_user(sentence: str, head: str, relation: str, description: str, tail: str) -> str:
    return (f"Sentence: {sentence}\n"
            f"Fact: [{head}] {relation} [{tail}]  ({relation}: {description})\n"
            f"Head mention: {head}\nTail mention: {tail}")


class ShortKG(KGAnnotator):
    """The ``kg`` layer (same schema, alignment and validation) with the
    short-text system prompt."""

    def build_messages(self, layer: str, item) -> list[dict]:
        sents = [" ".join(item.tokens[s:e]) for s, e in item.sentence_spans]
        return [{"role": "system", "content": SYSTEM["kgs2" if ROUND >= 2 else "kgs"]},
                {"role": "user", "content": kg_user_message(
                    sents, item.speakers, item.participants, item.types, item.relations)
                 + self._json_hint(layer)}]


class RewriteJudge(KGAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        return [{"role": "system", "content": SYSTEM["kgvarjudge3" if ROUND >= 3 else "kgvarjudge"]},
                {"role": "user", "content": item.layers["judge_prompt"] + self._json_hint(layer)}]


class Rewriter(KGAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        return [{"role": "system", "content": SYSTEM["kgvar3" if ROUND >= 3 else "kgvar"]},
                {"role": "user", "content": item.layers["var_prompt"] + self._json_hint(layer)}]


def render(item: KGItem) -> str:
    sents = [" ".join(item.tokens[s:e]) for s, e in item.sentence_spans]
    return ("=== SYSTEM ===\n" + SYSTEM["kgs2" if ROUND >= 2 else "kgs"] + "\n=== USER ===\n"
            + kg_user_message(sents, item.speakers, item.participants, item.types, item.relations))


# ── overlap with held-out evaluation text ───────────────────────────
def _grams(tokens: list[str], n: int = 8) -> set[int]:
    w = [t.lower() for t in tokens if any(c.isalnum() for c in t)]
    return {hash(tuple(w[i:i + n])) for i in range(len(w) - n + 1)}


def heldout_grams() -> tuple[set[int], str]:
    try:
        from datasets import load_dataset
        g: set[int] = set()
        for split in ("train", "validation", "test"):
            for x in load_dataset("DFKI-SLT/conll04", split=split):
                g |= _grams(x["tokens"])
        return g, "CoNLL04 (all splits)"
    except Exception as e:  # noqa: BLE001 - optional check
        return set(), f"skipped ({type(e).__name__})"


# ── commands ─────────────────────────────────────────────────────────
def dry(n: int, sampler: ShortSampler) -> int:
    items = short_items(n, sampler, SPLIT)
    ITEMS.write_text(json.dumps([it.__dict__ for it in items]))
    c = Counter(it.domain for it in items)
    modes = Counter(it.layers["mode"] for it in items)
    words = [len(it.tokens) for it in items]
    nsent = Counter(len(it.sentence_spans) for it in items)
    nrel = [len(it.relations) for it in items]
    val = sum(any(is_value_relation(r) for r in it.relations) for it in items)
    print(f"{len(items)} texts -> {ITEMS}")
    print("domains:", dict(sorted(c.items())))
    print("modes:", dict(modes), f"| schemas with a value relation: {val / len(items):.0%}")
    print(f"words: median {int(np.median(words))}, max {max(words)} | sentences: {dict(sorted(nsent.items()))}"
          f" | relations per schema: median {int(np.median(nrel))}, {min(nrel)}-{max(nrel)}")
    print(f"value relations in the pool: {int(sampler.value.sum())}, e.g. "
          + ", ".join(r["name"] for r, v in zip(sampler.b.relations, sampler.value) if v)[:400])
    g, src = heldout_grams()
    hits = [it.id for it in items if _grams(it.tokens) & g] if g else []
    print(f"8-gram overlap with {src}: {len(hits)} texts{(' ' + str(hits[:5])) if hits else ''}")
    chars = [len(render(it)) for it in items]
    print(f"prompt size: median {int(np.median(chars))} chars (~{int(np.median(chars)) // 4} tokens)")
    out = OUT / "dry-prompts.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    pick = []
    for want in [("mode", "normal"), ("mode", "value"), ("mode", "empty"),
                 ("domain", "conversation"), ("domain", "technical"), ("domain", "business")]:
        it = next((x for x in items if (x.layers["mode"] if want[0] == "mode" else x.domain) == want[1]
                   and x not in pick), None)
        if it is not None and len(pick) < 5:
            pick.append(it)
    out.write_text("\n\n".join(f"##### {it.id} [{it.domain}, {it.layers['mode']}]\n{render(it)}"
                               for it in pick))
    print(f"5 rendered prompts -> {out}")
    return 0


def rank(n: int, release: str, shard: tuple[int, int] = (0, 1), threads: int = 0) -> int:
    """Round 2, local and free: v7 ranks the whole relation pool against each
    text (its best margin over the text's entity pairs), for ``n`` texts.

    Ranking needs no fact attributes, so the attribute graph is skipped, and
    each pair keeps its 40 best relations: 8x faster than a plain ``window``
    call (1.18 -> 0.14 s per text on xlarge-u INT8), with the same top-20 on
    every one of 30 test texts. ``shard`` = (i, n): this process ranks every
    n-th text from i, into its own file; ``rank-merge`` joins them.
    ``threads``: onnxruntime threads per process (4 processes x 3 threads ran
    1.75x one default process on a 12-core M2 Max).

    Which model ranks: the medium student (ettin-68m) put the teacher's
    relation in its top 20 for 48% of round-1 facts against xlarge-u's 52%
    (254 relations, within noise), at 3.5x the speed, so the full run ranks
    with it."""
    import sys
    import time
    sys.path.insert(0, release)
    if threads:
        import onnxruntime as ort
        _session = ort.InferenceSession

        def session(path, providers=None, **kw):
            so = ort.SessionOptions()
            so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
            return _session(path, sess_options=so, providers=providers)
        ort.InferenceSession = session
    from kniv_v7 import Kniv
    base = ShortSampler(SchemaSampler(load_bank()))      # round-1 schemas only pick the texts
    texts = short_items(n, base, SPLIT)
    pool = [{"name": r["name"], "description": r["description"]} for r in base.b.relations]
    value_names = {r["name"] for r in base.b.relations if is_value_relation(r)}
    k = Kniv(release, int8=True)
    k.zs_offset -= 30.0                                  # every pair's top relations reach the output
    graph = k.run

    def run(name, **x):
        if name == "attr":                               # attributes are not needed to rank
            m = len(x["pair"])
            return [np.zeros((m, 2)), np.zeros((m, 6)), np.zeros((m, 2)), np.zeros((m, 2))]
        return graph(name, **x)
    k.run = run
    i0, n_sh = shard
    texts = texts[i0::n_sh]
    out, t0 = {}, time.time()
    for i, it in enumerate(texts):
        key = it.id.rsplit(":", 1)[0]
        w = {"tokens": it.tokens, "sentence_spans": it.sentence_spans,
             "sentence_speakers": it.speakers or []}
        o = k.window(w, pool, base.types_for(key), False, 40)
        best: dict[str, float] = {}
        for f in o["facts"]:
            best[f["relation"]] = max(best.get(f["relation"], -1e9), f["margin"])
        order = sorted(best, key=lambda r: -best[r])
        out[key] = {"top": order[:40], "value": [r for r in order if r in value_names]}
        if i % 50 == 0:
            print(f"{i}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    dst = RANK if n_sh == 1 else RANK.with_suffix(f".shard{i0}of{n_sh}.json")
    dst.write_text(json.dumps(out))
    print(f"{len(out)} texts ranked -> {dst}")
    return 0


def rank_merge(n_sh: int) -> int:
    out = {}
    for i in range(n_sh):
        out.update(json.loads(RANK.with_suffix(f".shard{i}of{n_sh}.json").read_text()))
    RANK.write_text(json.dumps(out))
    print(f"{len(out)} texts ranked -> {RANK}")
    return 0


def load_items() -> list[KGItem]:
    return [KGItem(**d) for d in json.loads(ITEMS.read_text())]


async def _run(ann, layer: str, items: list, tag: str) -> dict:
    logger = RunLogger(OUT / tag, every=50)
    out = {}
    try:
        for coro in asyncio.as_completed([ann.annotate_and_cache(layer, it) for it in items]):
            r = await coro
            logger.record(r, (0, 0), len(items)); out[r.item_id] = r
    finally:
        logger.close()
    return out


def pilot(annotator: str) -> int:
    items = load_items()
    spec = teacher_spec(annotator)
    res = asyncio.run(_run(ShortKG(spec, store(SHORT_VERSION)), "kg", items, f"pilot-{annotator}"))
    agg = defaultdict(Counter)
    for it in items:
        r = res.get(it.id)
        c = agg[it.domain]
        c["texts"] += 1
        if r is None or not r.ok:
            c["failed"] += 1
            continue
        p = r.payload
        c["mentions"] += p["stats"]["mentions"]; c["aligned"] += p["stats"]["aligned"]
        c["facts"] += len(p["facts"]); c["empty"] += not p["facts"]
        c[f"empty_{it.layers['mode']}"] += not p["facts"]; c[f"mode_{it.layers['mode']}"] += 1
        rels = {x["name"] for x in it.relations if is_value_relation(x)}
        c["value_facts"] += sum(f["relation"] in rels for f in p["facts"])
        c["neg"] += sum(f["polarity"] == "negative" for f in p["facts"])
    print(f"{'domain':13s} {'texts':>5s} {'fail':>4s} {'align':>6s} {'facts/t':>7s} {'empty%':>6s} "
          f"{'empty%|empty-mode':>17s} {'value facts':>11s}")
    tot = Counter()
    for d in sorted(agg):
        tot.update(agg[d])
    for d, c in sorted(list(agg.items()) + [("ALL", tot)], key=lambda x: x[0] == "ALL"):
        ok = max(c["texts"] - c["failed"], 1)
        print(f"{d:13s} {c['texts']:5d} {c['failed']:4d} {c['aligned'] / max(c['mentions'], 1):6.1%} "
              f"{c['facts'] / ok:7.2f} {c['empty'] / ok:6.0%} "
              f"{c['empty_empty'] / max(c['mode_empty'], 1):17.0%} {c['value_facts']:11d}")
    return 0


def part_a(annotator: str, bases_of: bool = False) -> tuple[list[KGItem], dict]:
    """Part A items and their (cached) annotations; ``bases_of``: the round
    whose part A supplies the rewrite bases (round 3 uses round 2's)."""
    global ROUND
    path, version = (BASE_ITEMS, BASE_VERSION) if bases_of else (ITEMS, SHORT_VERSION)
    items = [KGItem(**d) for d in json.loads(path.read_text())]
    spec = teacher_spec(annotator)
    keep, ROUND = ROUND, (2 if bases_of and ROUND == 3 else ROUND)   # round 2's prompt for its cache
    try:
        res = asyncio.run(_run(ShortKG(spec, store(version)), "kg", items, f"pilot-{annotator}"))
    finally:
        ROUND = keep
    return items, res


def _text(item: KGItem, spans) -> str:
    a, b = spans[0]
    return " ".join(item.tokens[a:b])


def _norm_words(s: str) -> list[str]:
    from .windows import tokenize
    return [w.lower() for w in tokenize(s) if any(c.isalnum() for c in w)]


def _contains(sentence: str, mention: str) -> bool:
    s, m = _norm_words(sentence), _norm_words(mention)
    return bool(m) and any(s[i:i + len(m)] == m for i in range(len(s) - len(m) + 1))


PRONOUN_SOURCES = {"he", "she", "him", "her", "they", "them", "it", "the group", "the company",
                   "the institute", "the authors", "the author"}
DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "its", "their", "his", "her",
               "our", "my", "your", "each", "every", "some", "such"}


def named(mention: str) -> bool:
    """A name or a value, not a description: a capitalised word that is not a
    sentence-initial determiner, or a number."""
    w = mention.split()
    return any(c.isdigit() for c in mention) or any(
        t[:1].isupper() and not (i == 0 and t.lower() in DETERMINERS) for i, t in enumerate(w))


def bases(items, res, n: int) -> list[tuple[KGItem, dict, str, str, int]]:
    """Up to one plain fact per text: positive, asserted, the author's, both
    arguments named (not participants) in one sentence. Spread over domains."""
    by = defaultdict(list)
    for it in items:
        r = res.get(it.id)
        if r is None or not r.ok:
            continue
        p, k = r.payload, len(it.participants)
        for f in p["facts"]:
            if (f["polarity"], f["modality"], f["holder"]) != ("positive", "asserted", -1):
                continue
            if f["head"] < k or f["tail"] < k:
                continue
            hs, ts = p["entities"][f["head"] - k]["spans"], p["entities"][f["tail"] - k]["spans"]
            for j, (s, e) in enumerate(it.sentence_spans):
                h = [x for x in hs if s <= x[0] < e]
                t = [x for x in ts if s <= x[0] < e]
                if ROUND >= 2 and (e - s >= 40 or not h or not t
                                   or not named(_text(it, h)) or not named(_text(it, t))):
                    continue
                if h and t:
                    by[it.domain].append((it, f, _text(it, h), _text(it, t), j))
                    break
            else:
                continue
            break
    per = max(1, n // max(len(by), 1))
    return [x for d in sorted(by) for x in by[d][:per]][:n]


def variants(annotator: str, n: int, judge: str = "astra") -> int:
    items, res = part_a(annotator, bases_of=True)
    picked = bases(items, res, n)
    spec = teacher_spec(annotator)
    cache = store(SHORT_VERSION)
    reqs = []
    for it, f, head, tail, j in picked:
        s, e = it.sentence_spans[j]
        sentence = " ".join(it.tokens[s:e])
        desc = next(r["description"] for r in it.relations if r["name"] == f["relation"])
        q = KGItem(**{**it.__dict__})
        q.id = f"{it.id}:var:{f['head']}-{f['tail']}-{f['relation']}"
        q.layers = {**it.layers, "var_prompt": var_user(sentence, head, f["relation"], desc, tail),
                    "base": [sentence, head, f["relation"], tail]}
        reqs.append(q)
    rw = asyncio.run(_run(Rewriter(spec, cache), "kgvar", reqs, f"rewrite-{annotator}"))
    from .windows import tokenize
    verdict = {}
    if ROUND >= 2:
        # one judge call per base, over all its rewrites
        jspec = judge_spec(judge)
        jq = []
        for q in reqs:
            r = rw.get(q.id)
            if r is None or not r.ok:
                continue
            sentence, head, rel, tail = q.layers["base"]
            lines = [f"Original: {sentence}", f"Fact: [{head}] {rel} [{tail}]", "", "Rewrites:"]
            lines += [f"- {kind}: {r.payload[kind]['sentence']}" for kind in VAR_KINDS if kind in r.payload]
            j = KGItem(**{**q.__dict__}); j.id = f"{q.id}:vjudge"
            j.layers = {"judge_prompt": "\n".join(lines), "base": q.id}
            jq.append(j)
        jr = asyncio.run(_run(RewriteJudge(jspec, cache), "kgvarjudge", jq, f"rewrite-judge-{judge}"))
        for j in jq:
            r = jr.get(j.id)
            verdict[j.layers["base"]] = r.payload if r is not None and r.ok else {}
    out, var_items, bad = [], [], Counter()
    for q in reqs:
        r = rw.get(q.id)
        if r is None or not r.ok:
            bad["rewrite failed"] += 1
            continue
        sentence, head, rel, tail = q.layers["base"]
        for kind in VAR_KINDS:
            v = r.payload.get(kind)
            if not v:
                bad[f"{kind}: missing"] += 1
                continue
            if not (_contains(v["sentence"], head) and _contains(v["sentence"], tail)):
                bad[f"{kind}: mention changed"] += 1
                continue
            if ROUND >= 2 and not verdict.get(q.id, {}).get(kind, False):
                bad[f"{kind}: judged not ok"] += 1
                continue
            toks = tokenize(v["sentence"])
            vi = KGItem(id=f"{q.id}:{kind}", tokens=toks, sentence_spans=[[0, len(toks)]], speakers=None,
                        participants=[], types=q.types, relations=q.relations, domain=q.domain,
                        layers={"kind": kind, "head": head, "tail": tail, "relation": rel,
                                "source": v["source"]})
            var_items.append(vi)
    ann = asyncio.run(_run(ShortKG(spec, cache), "kg", var_items, f"variants-{annotator}"))
    agree, c = Counter(), Counter()
    for vi in var_items:
        kind, L = vi.layers["kind"], vi.layers
        c[kind] += 1
        r = ann.get(vi.id)
        if r is None or not r.ok:
            agree[f"{kind}:failed"] += 1
            continue
        p = r.payload

        def name(i):  # no participants here, so entity ids index p["entities"]
            if i < 0:
                return ""
            a, b = p["entities"][i]["spans"][0]
            return " ".join(vi.tokens[a:b])

        def same(x, y):
            return bool(x) and bool(y) and (_contains(x, y) or _contains(y, x))

        hit = [f for f in p["facts"] if f["relation"] == L["relation"]
               and same(name(f["head"]), L["head"]) and same(name(f["tail"]), L["tail"])]
        if not hit:
            agree[f"{kind}:fact missing"] += 1
            continue
        f = hit[0]
        ok = (f["polarity"], f["modality"]) == VAR_EXPECT[kind]
        if kind == "reported" and ok and L["source"]:
            src = L["source"].lower()
            if src in PRONOUN_SOURCES or src.split()[0] in {"his", "her", "its", "their"}:
                # 'him', 'the group', 'his spokesperson': any holder other than head and tail
                ok = f["holder"] >= 0 and not same(name(f["holder"]), L["head"]) \
                    and not same(name(f["holder"]), L["tail"])
            else:
                ok = same(name(f["holder"]), L["source"])
        agree[f"{kind}:{'agree' if ok else 'disagree'}"] += 1
    print(f"{len(reqs)} base facts, {len(var_items)} rewrites kept; dropped: {dict(bad)}")
    print(f"{'kind':13s} {'n':>4s} {'agree':>6s} {'disagree':>8s} {'missing':>7s}")
    tot = Counter()
    for kind in VAR_KINDS:
        a, d, m = agree[f"{kind}:agree"], agree[f"{kind}:disagree"], agree[f"{kind}:fact missing"]
        tot.update({"n": c[kind], "a": a, "d": d, "m": m})
        print(f"{kind:13s} {c[kind]:4d} {a / max(c[kind], 1):6.0%} {d / max(c[kind], 1):8.0%} {m / max(c[kind], 1):7.0%}")
    print(f"{'ALL':13s} {tot['n']:4d} {tot['a'] / max(tot['n'], 1):6.0%} {tot['d'] / max(tot['n'], 1):8.0%} "
          f"{tot['m'] / max(tot['n'], 1):7.0%}")
    (OUT / "variants.json").write_text(json.dumps([{**vi.__dict__} for vi in var_items]))
    return 0


def check(annotator: str, judge: str, n_recall: int = 100, n_judge_facts: int = 300) -> int:
    """Precision by the judge on part A facts; recall against a second,
    stronger annotator on ``n_recall`` texts, with its extra facts judged too,
    so 'missed' means a supported fact the teacher left out."""
    from .prompt_tune import Judge, _match, judge_prompt
    items, res = part_a(annotator)
    cache = store(SHORT_VERSION)
    jspec = judge_spec(judge)
    # second annotator on a fixed, domain-balanced sample
    by = defaultdict(list)
    for it in items:
        by[it.domain].append(it)
    sample = [it for d in sorted(by) for it in by[d][:max(1, n_recall // len(by))]]
    res2 = asyncio.run(_run(ShortKG(jspec, cache), "kg", sample, f"second-{judge}"))
    jitems, meta, nf = [], [], 0
    for it in items:
        r = res.get(it.id)
        if r is None or not r.ok or not r.payload["facts"] or nf >= n_judge_facts:
            continue
        j = KGItem(**{**it.__dict__}); j.id = f"{it.id}:judge"
        j.layers = {"judge_prompt": judge_prompt(it, r.payload)}
        jitems.append(j); meta.append(("teacher", it, r.payload["facts"])); nf += len(r.payload["facts"])
    extra = {}
    for it in sample:
        r, r2 = res.get(it.id), res2.get(it.id)
        if r is None or r2 is None or not r.ok or not r2.ok:
            continue
        p, p2, k = r.payload, r2.payload, len(it.participants)
        m = _match([{tuple(s) for s in e["spans"]} for e in p2["entities"]],
                   [{tuple(s) for s in e["spans"]} for e in p["entities"]])
        idx = lambda x: x if x < k else (None if m.get(x - k) is None else m[x - k] + k)
        mine = {(f["head"], f["tail"], f["relation"]) for f in p["facts"]}
        only = [f for f in p2["facts"] if (idx(f["head"]), idx(f["tail"]), f["relation"]) not in mine]
        extra[it.id] = (len(p2["facts"]) - len(only), len(only))
        if only:
            j = KGItem(**{**it.__dict__}); j.id = f"{it.id}:judge2"
            j.layers = {"judge_prompt": judge_prompt(it, {**p2, "facts": only})}
            jitems.append(j); meta.append(("second-only", it, only))
    jres = asyncio.run(_run(Judge(jspec, cache), "kgjudge", jitems, f"judge-{judge}"))
    sup = Counter()
    for j, (who, it, facts) in zip(jitems, meta):
        r = jres.get(j.id)
        if r is None or not r.ok:
            continue
        sup[f"{who}:judged"] += len(facts)
        sup[f"{who}:supported"] += sum(1 for i in range(len(facts)) if r.payload.get(i, False))
    prec = sup["teacher:supported"] / max(sup["teacher:judged"], 1)
    both = sum(a for a, _ in extra.values())
    teacher_in_sample = sum(len(res[i].payload["facts"]) for i in extra)
    real_miss = sup["second-only:supported"]
    recall = (teacher_in_sample * prec) / max(teacher_in_sample * prec + real_miss, 1e-9)
    print(f"judged precision of the teacher: {prec:.1%} ({sup['teacher:judged']} facts)")
    print(f"second annotator on {len(extra)} texts: {both} shared facts, "
          f"{sum(b for _, b in extra.values())} only theirs, of which {real_miss} judged supported")
    print(f"estimated teacher recall: {recall:.1%}")
    return 0


# ── synthetic value and list facts ───────────────────────────────────
# Natural short text rarely states prices, stakes, quantities or skill lists
# (rounds 1-3), so these are written to order: the LLM writes a passage that
# states facts for the requested relations and returns them; the teacher then
# annotates the passage blind, against a schema with distractors, and must
# recover them. Each scenario's first relations are its value / list ones.
SCENARIOS = {
    "acquisition": (["has price", "has monetary value"],
                    ["acquires", "board member of", "investor", "owned by", "headquartered in"]),
    "company results": (["total revenue", "net profit", "employees", "has count"],
                        ["headquartered in", "subsidiary of", "has tax rate"]),
    "person profile": (["has expertise in", "has age"],
                       ["employer", "holds position", "educated at", "located in"]),
    "product listing": (["has price", "has capacity", "has feature"],
                        ["made of", "runs on", "supports payment method"]),
    "travel booking": (["has price", "has duration"],
                       ["has destination", "scheduled for", "booked for", "uses payment method"]),
    "recipe": (["has ingredient", "has duration", "has count"], ["made of"]),
    "software": (["has default value", "has value", "has feature"], ["runs on", "uses"]),
    "finance": (["has monetary value", "has tax rate", "has budget"], ["due on", "owes payment to", "funded by"]),
}
STYLES = ["a news sentence", "a press release", "a short professional bio", "a chat message from a customer",
          "a product listing", "a work email", "an excerpt from an annual report", "a forum post",
          "a meeting note", "a booking confirmation"]
LIST_HINT = {"has expertise in", "has ingredient", "has feature", "supports payment method", "uses"}

SYSTEM["kgsyn"] = (
    "You write short, realistic passages for training an information-extraction model. Given a "
    "scenario, a style and some relations (each with a definition), write ONE passage of 1-3 "
    "sentences in that style that states facts for each listed relation, then list those facts.\n"
    "- Invent realistic, varied names, places, amounts and dates (fictional is fine; avoid famous "
    "real people). Vary phrasing; do not copy the relation names into the text.\n"
    "- State each fact explicitly enough that a careful reader would extract it from the passage alone.\n"
    "- Amounts, prices, percentages, counts, ages and durations are written as in real text "
    "('$1.2 billion', '35%', '3 hours', '12 employees').\n"
    "- For a relation marked LIST, state 2-3 tail items in one list ('skilled in Python, SQL and "
    "dbt') and give one fact per item.\n"
    "- In the facts, give head and tail exactly as written in the passage, head to tail as the "
    "definition reads. List every fact the passage states for the listed relations, and nothing else."
)


def syn_requests(n: int, base: SchemaSampler) -> list[KGItem]:
    by = {r["name"]: r for r in base.relations}
    out = []
    names = sorted(SCENARIOS)
    for i in range(n):
        rng = _rng(f"syn:{i}")
        sc = names[i % len(names)]
        values, others = SCENARIOS[sc]
        values = [v for v in values if v in by]
        others = [o for o in others if o in by]
        k_val = int(rng.integers(1, min(2, len(values)) + 1))
        k_oth = int(rng.integers(1, min(2, len(others)) + 1))
        chosen = (list(rng.choice(values, k_val, replace=False))
                  + list(rng.choice(others, k_oth, replace=False)))
        style = STYLES[int(rng.integers(len(STYLES)))]
        lines = [f"Scenario: {sc}", f"Style: {style}", "Relations (head -> tail):"]
        for r in chosen:
            mark = " [LIST]" if r in LIST_HINT else ""
            lines.append(f"- {r}{mark}: {by[r]['description']}")
        out.append(KGItem(id=f"syn{i:04d}:{sc}:{'|'.join(chosen)}", tokens=[], sentence_spans=[],
                          speakers=None, participants=[], types=[], relations=[], domain=sc,
                          layers={"syn_prompt": "\n".join(lines), "relations": chosen, "style": style}))
    return out


class Synthesizer(KGAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        return [{"role": "system", "content": SYSTEM["kgsyn"]},
                {"role": "user", "content": item.layers["syn_prompt"] + self._json_hint(layer)}]


def synth(annotator: str, n: int) -> int:
    """Pilot of the synthetic value and list set: generate ``n`` passages,
    annotate each blind with the round-2 prompt against its scenario's
    relations plus distractors, and score how many written facts the teacher
    recovers."""
    from .windows import tokenize
    global ROUND
    base = SchemaSampler(load_bank())
    by = {r["name"]: r for r in base.relations}
    spec = teacher_spec(annotator)
    cache = store("y1")
    reqs = syn_requests(n, base)
    gen = asyncio.run(_run(Synthesizer(spec, cache), "kgsyn", reqs, f"synth-gen-{annotator}"))
    items, bad = [], Counter()
    for q in reqs:
        r = gen.get(q.id)
        if r is None or not r.ok:
            bad["generation failed"] += 1
            continue
        text, facts = r.payload["text"], r.payload["facts"]
        want = set(q.layers["relations"])
        ok = [f for f in facts if f["relation"] in want and _contains(text, f["head"]) and _contains(text, f["tail"])]
        bad["facts dropped (relation or mention)"] += len(facts) - len(ok)
        vals, _ = SCENARIOS[q.domain]
        if not any(f["relation"] in vals for f in ok):
            bad["no value or list fact"] += 1
            continue
        # sentences: split the tokens at sentence-final punctuation
        toks = tokenize(text)
        spans, s0 = [], 0
        for j, t in enumerate(toks):
            if t in {".", "!", "?"} and j + 1 < len(toks):
                spans.append([s0, j + 1]); s0 = j + 1
        spans.append([s0, len(toks)])
        rng = _rng(f"synschema:{q.id}")
        values, others = SCENARIOS[q.domain]
        menu = [x for x in values + others if x in by]
        pool = [r2["name"] for r2 in base.relations if r2["name"] not in menu]
        distract = list(rng.choice(pool, 3, replace=False))
        schema = sorted(menu + distract, key=lambda x: _rng(f"o:{q.id}:{x}").random())
        items.append(KGItem(id=f"{q.id}:ann", tokens=toks, sentence_spans=spans, speakers=None,
                            participants=[], types=GENERAL_TYPES + base.always,
                            relations=[{"name": x, "description": by[x]["description"]} for x in schema],
                            domain=q.domain, layers={"gold": ok, "text": text, "style": q.layers["style"]}))
    keep, ROUND = ROUND, 2
    try:
        ann = asyncio.run(_run(ShortKG(spec, cache), "kg", items, f"synth-ann-{annotator}"))
    finally:
        ROUND = keep
    c = defaultdict(Counter)
    examples = []
    for it in items:
        r = ann.get(it.id)
        cc = c[it.domain]
        cc["texts"] += 1
        gold = it.layers["gold"]
        cc["gold"] += len(gold)
        if r is None or not r.ok:
            cc["failed"] += 1
            continue
        p = r.payload

        def name(i):
            a, b = p["entities"][i]["spans"][0]
            return " ".join(it.tokens[a:b])

        def same(x, y):
            return bool(x) and bool(y) and (_contains(x, y) or _contains(y, x))
        pred = [(name(f["head"]), f["relation"], name(f["tail"])) for f in p["facts"]]
        found = [g for g in gold if any(rel == g["relation"] and same(h, g["head"]) and same(t, g["tail"])
                                         for h, rel, t in pred)]
        cc["found"] += len(found)
        cc["value_gold"] += sum(g["relation"] in SCENARIOS[it.domain][0] for g in gold)
        cc["value_found"] += sum(g["relation"] in SCENARIOS[it.domain][0] for g in found)
        cc["extra"] += max(0, len(pred) - len(found))
        if len(examples) < 12 and len(found) < len(gold):
            miss = [g for g in gold if g not in found]
            examples.append(f"[{it.domain}] {it.layers['text'][:200]}\n    missed: "
                            + "; ".join(f"{g['head']} | {g['relation']} | {g['tail']}" for g in miss[:3]))
    print(f"{len(reqs)} requests, {len(items)} passages kept; dropped: {dict(bad)}")
    print(f"{'scenario':17s} {'texts':>5s} {'facts':>5s} {'recovered':>9s} {'value rec.':>10s} {'extra/t':>7s}")
    tot = Counter()
    for d in sorted(c):
        tot.update(c[d])
    for d, cc in sorted(list(c.items()) + [("ALL", tot)], key=lambda x: x[0] == "ALL"):
        print(f"{d:17s} {cc['texts']:5d} {cc['gold']:5d} {cc['found'] / max(cc['gold'], 1):9.0%} "
              f"{cc['value_found'] / max(cc['value_gold'], 1):10.0%} {cc['extra'] / max(cc['texts'], 1):7.2f}")
    print("\n".join(examples))
    (OUT / "synth.json").write_text(json.dumps([{**it.__dict__} for it in items]))
    return 0


def effort_compare(annotator: str, judge: str, efforts: list[str], n_syn: int = 40, n_nat: int = 40) -> int:
    """The teacher at each reasoning effort on one fixed set: synthetic
    passages (recall of the written facts, exact) and round-2 natural texts
    (facts per text, precision judged by ``judge`` at its default effort),
    with latency and tokens. 'default' = no effort sent (the pilots), served
    from their caches."""
    global TEACHER_EFFORT, ROUND
    import statistics as st
    from .prompt_tune import Judge, judge_prompt
    syn_all = [KGItem(**d) for d in json.loads((RUNS_DIR / "kg-short-synth" / "synth.json").read_text())]
    by = defaultdict(list)
    for it in syn_all:
        by[it.domain].append(it)
    syn = [it for d in sorted(by) for it in by[d][:max(1, n_syn // len(by))]]
    nat_all = [KGItem(**d) for d in json.loads((DATA_DIR / "v6-corpus" / "kg_short_items_r2.json").read_text())]
    by = defaultdict(list)
    for it in nat_all:
        if it.layers["mode"] != "empty":
            by[it.domain].append(it)
    nat = [it for d in sorted(by) for it in by[d][:max(1, n_nat // len(by))]]
    ROUND = 2
    rows = []
    for eff in efforts:
        TEACHER_EFFORT = None if eff == "default" else eff
        spec = teacher_spec(annotator)
        rs = asyncio.run(_run(ShortKG(spec, store("y1")), "kg", syn, f"effort-syn-{eff}"))
        rn = asyncio.run(_run(ShortKG(spec, store("s2")), "kg", nat, f"effort-nat-{eff}"))
        # recall on the synthetic passages
        gold = found = 0
        for it in syn:
            r = rs.get(it.id)
            g = it.layers["gold"]
            gold += len(g)
            if r is None or not r.ok:
                continue
            p = r.payload
            nm = lambda i: " ".join(it.tokens[slice(*p["entities"][i]["spans"][0])])
            same = lambda x, y: bool(x) and bool(y) and (_contains(x, y) or _contains(y, x))
            pred = [(nm(f["head"]), f["relation"], nm(f["tail"])) for f in p["facts"]]
            found += sum(any(rel == x["relation"] and same(h, x["head"]) and same(t, x["tail"])
                             for h, rel, t in pred) for x in g)
        # judged precision on the natural texts
        jitems, nfacts = [], 0
        for it in nat:
            r = rn.get(it.id)
            if r is None or not r.ok or not r.payload["facts"]:
                continue
            j = KGItem(**{**it.__dict__}); j.id = f"{it.id}:judge"
            j.layers = {"judge_prompt": judge_prompt(it, r.payload), "n": len(r.payload["facts"])}
            jitems.append(j); nfacts += len(r.payload["facts"])
        jr = asyncio.run(_run(Judge(judge_spec(judge), store("s2")), "kgjudge", jitems, f"effort-judge-{eff}"))
        judged = sup = 0
        for j in jitems:
            r = jr.get(j.id)
            if r is not None and r.ok:
                judged += j.layers["n"]
                sup += sum(1 for i in range(j.layers["n"]) if r.payload.get(i, False))
        live = [r for r in list(rs.values()) + list(rn.values()) if not r.cached and r.ok]
        lat = sorted(r.latency_ms / 1000 for r in live)
        rows.append({
            "effort": eff, "syn_recall": found / max(gold, 1), "facts_per_text": nfacts / len(nat),
            "precision": sup / max(judged, 1),
            "lat_p50": st.median(lat) if lat else None, "lat_p90": lat[int(0.9 * len(lat))] if lat else None,
            "out_tokens": st.mean(r.tokens_out for r in live) if live else None,
            "reasoning_tokens": st.mean(r.tokens_reasoning for r in live) if live else None,
            "live_calls": len(live)})
    TEACHER_EFFORT = None
    print(f"{len(syn)} synthetic passages ({sum(len(i.layers['gold']) for i in syn)} facts), {len(nat)} natural texts")
    print(f"{'effort':8s} {'syn recall':>10s} {'facts/text':>10s} {'precision':>9s} {'p50 s':>6s} {'p90 s':>6s} "
          f"{'out tok':>7s} {'reasoning':>9s} {'calls':>5s}")
    f = lambda x, fmt: (format(x, fmt) if x is not None else "cached")
    for r in rows:
        print(f"{r['effort']:8s} {r['syn_recall']:10.1%} {r['facts_per_text']:10.2f} {r['precision']:9.1%} "
              f"{f(r['lat_p50'], '6.1f'):>6s} {f(r['lat_p90'], '6.1f'):>6s} {f(r['out_tokens'], '7.0f'):>7s} "
              f"{f(r['reasoning_tokens'], '9.0f'):>9s} {r['live_calls']:5d}")
    (RUNS_DIR / "kg-short-effort.json").write_text(json.dumps(rows, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["rank", "rank-merge", "dry", "pilot", "variants", "check", "synth", "effort"])
    ap.add_argument("--effort", default=None, choices=["low", "medium", "high", "xhigh"],
                    help="the teacher's reasoning effort (default: the deployment's, as in the pilots)")
    ap.add_argument("--efforts", default="default,low,medium,high,xhigh", help="effort: levels to compare")
    ap.add_argument("--concurrency", type=int, default=None, help="override the annotators' concurrency cap")
    ap.add_argument("--shard", default="0/1", help="rank: i/n, this process ranks every n-th text from i")
    ap.add_argument("--threads", type=int, default=0, help="rank: onnxruntime threads per process")
    ap.add_argument("--round", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--split", default="train", choices=["train", "dev"], help="dry / rank: corpus split")
    ap.add_argument("--release", default="release/ultrabert-380m", help="rank: the v7 model that ranks")
    ap.add_argument("--texts", type=int, default=500)
    ap.add_argument("--bases", type=int, default=100)
    ap.add_argument("--annotator", default="sol61")
    ap.add_argument("--judge", default="astra")
    ap.add_argument("--recall-texts", type=int, default=100, help="check: texts the second annotator labels")
    ap.add_argument("--judge-facts", type=int, default=300, help="check: teacher facts judged")
    args = ap.parse_args()
    global TEACHER_EFFORT, CONCURRENCY, SPLIT
    SPLIT = args.split
    set_round(args.round)
    TEACHER_EFFORT, CONCURRENCY = args.effort, args.concurrency
    if args.cmd == "effort":
        return effort_compare(args.annotator, args.judge, args.efforts.split(","))
    if args.cmd == "synth":
        global OUT
        OUT = RUNS_DIR / "kg-short-synth"
        OUT.mkdir(parents=True, exist_ok=True)
        return synth(args.annotator, args.texts)
    if args.cmd == "rank":
        i, n = (int(x) for x in args.shard.split("/"))
        return rank(args.texts, args.release, (i, n), args.threads)
    if args.cmd == "rank-merge":
        return rank_merge(int(args.shard.split("/")[1]))
    rank_map = json.loads(RANK.read_text()) if ROUND >= 2 else None
    sampler = ShortSampler(SchemaSampler(load_bank()), rank_map)
    if args.cmd == "dry":
        return dry(args.texts, sampler)
    if args.cmd == "pilot":
        return pilot(args.annotator)
    if args.cmd == "variants":
        return variants(args.annotator, args.bases, args.judge)
    return check(args.annotator, args.judge, args.recall_texts, args.judge_facts)


if __name__ == "__main__":
    raise SystemExit(main())
