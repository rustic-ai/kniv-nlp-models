"""Tune the ``kg`` teacher prompt for one annotator -- MIPRO-style, in house.

    uv run python -m v6.prompt_tune eval --variants base,recall --split search
    uv run python -m v6.prompt_tune propose --from recall --n 3
    uv run python -m v6.prompt_tune eval --variants <best> --split held

A prompt variant is a system instruction, optional worked examples (few-shot
demonstrations: a window, its schema and the full answer) and optional
request parameters. Each is scored by running the annotator on fixed
evaluation sets, split once into a search half (to choose) and a held-out
half (to confirm, so the choice is not fitted to its own test):

- ``redocred``: Re-DocRED dev documents with sampled schemas of its
  relations. Human-annotated and exhaustive, so precision and recall are
  real; the entities are the teacher's own, matched to gold by mention span.
- ``pilot``: windows from the other domains (conversation, business,
  technical, narrative, news), from the astra pilot. No gold exists, so the
  answer is scored three ways: facts per window; recall of astra's facts
  (astra misses many, so this is a floor on coverage, not recall); and
  precision as judged by a stronger model (``kgjudge``: is each fact
  supported by the passage?).

DSPy's optimisers do the same search -- instructions proposed from failures,
demonstrations bootstrapped and selected by a metric -- but assume their own
prompt and output formats; here the schema changes per window, the output is
aligned to tokens after the call, and calls go through the project's cached
annotator, so the loop is written out.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .annotate.base import CacheStore, RunLogger
from .annotate.kg import KGAnnotator, KGItem
from .config import DATA_DIR, RUNS_DIR, load_annotators
from .kg_teacher import CACHE, SchemaSampler, _rng, load_export, make_item
from .label_bank import load as load_bank
from .prompts import PROMPT_VERSION, SYSTEM, kg_user_message

VARIANTS = Path(__file__).parent / "kg_prompts.json"
# The astra pilot, kept apart from kg_teacher.json (which each export rewrites).
PILOT = DATA_DIR / "v6-corpus" / "kg_pilot_astra.json"
OUT = RUNS_DIR / "prompt-tune"
PILOT_DOMAINS = ("business", "conversation", "narrative", "news", "technical")
N_RD = {"search": (0, 30), "held": (30, 80)}
N_PILOT = {"search": (0, 6), "held": (6, 14)}         # per domain


@dataclass
class Variant:
    name: str
    system: str
    demos: list[str] = field(default_factory=list)   # demo ids (see demo_pool)
    params: dict = field(default_factory=dict)        # extra request kwargs
    note: str = ""

    @property
    def digest(self) -> str:
        return hashlib.sha1(json.dumps([self.system, self.demos, self.params],
                                       sort_keys=True).encode()).hexdigest()[:8]


def load_variants() -> dict[str, Variant]:
    out = {"base": Variant("base", SYSTEM["kg"], note=f"the teacher prompt ({PROMPT_VERSION})")}
    if VARIANTS.exists():
        for v in json.loads(VARIANTS.read_text()):
            out[v["name"]] = Variant(**v)
    return out


def save_variant(v: Variant) -> None:
    have = json.loads(VARIANTS.read_text()) if VARIANTS.exists() else []
    have = [x for x in have if x["name"] != v.name] + [v.__dict__]
    VARIANTS.write_text(json.dumps(have, indent=1))


# ── evaluation sets ──────────────────────────────────────────────────
def _describe(item: KGItem, bank: dict) -> KGItem:
    rel = {r["name"]: r for r in bank["relations"]}
    typ = {t["name"]: t for t in bank["entity_types"]}
    item.relations = [{"name": r["name"], "description": rel[r["name"]]["description"]}
                      for r in item.relations if r["name"] in rel]
    item.types = [typ[t["name"]] for t in item.types if t["name"] in typ]
    return item


def redocred_set(split: str, sampler: SchemaSampler, source: str = "dev"):
    """``[(item, gold)]``: Re-DocRED documents, each with a schema of 8-40 of
    its relations, about half of them present in the document."""
    from .gold.redocred import load_rel_items
    a, b = N_RD[split] if source == "dev" else (0, 400)
    golds = load_rel_items(source, max_tokens=10 ** 6)[a:b]
    wiki = [i for i, r in enumerate(sampler.relations) if r["source"] == "redocred"]
    out = []
    for g in golds:
        rng = _rng(f"bake:{g.id}")
        present = sorted({i for i in wiki
                          if any(t[2] == sampler.relations[i]["name"] for t in g.layers["rel"])})
        k = int(rng.integers(8, 41))
        rel = set(rng.choice(present, size=min(len(present), k // 2), replace=False).tolist()) \
            if present else set()
        rest = [i for i in wiki if i not in rel]
        rel |= set(rng.choice(rest, size=k - len(rel), replace=False).tolist())
        out.append((make_item(g.id, g.tokens, g.layers["sentence_spans"], "encyclopedic", None,
                              sampler, rel_ids=sorted(rel), types=list(sampler.coarse)), g))
    return out


def pilot_set(split: str, bank: dict):
    """``[(item, astra payload)]``: pilot windows, a fixed hash order per domain."""
    by = defaultdict(list)
    for it, p in load_export(PILOT):
        if it.domain in PILOT_DOMAINS and ":s0:" in it.id:
            by[it.domain].append((it, p))
    a, b = N_PILOT[split]
    out = []
    for d in PILOT_DOMAINS:
        xs = sorted(by[d], key=lambda x: hashlib.sha1(f"tune:{x[0].id}".encode()).hexdigest())
        out += [(_describe(it, bank), p) for it, p in xs[a:b]]
    return out


# ── worked examples ──────────────────────────────────────────────────
def render_answer(item: KGItem, payload: dict) -> str:
    """An aligned payload back in the teacher's own answer format."""
    k = len(item.participants)
    sent_of = {}
    for j, (s, e) in enumerate(item.sentence_spans):
        for i in range(s, e):
            sent_of[i] = j + 1
    ents = [{"id": k + i, "type": e["type"],
             "mentions": [{"sentence": sent_of[a], "text": " ".join(item.tokens[a:b])}
                          for a, b in e["spans"]]}
            for i, e in enumerate(payload["entities"])]
    facts = [{"head": f["head"], "relation": f["relation"], "tail": f["tail"],
              "polarity": f["polarity"], "modality": f["modality"], "holder": f["holder"],
              "time": f["time"], "evidence": [j + 1 for j in f["evidence"]]}
             for f in payload["facts"]]
    return json.dumps({"knowledge": {"entities": ents, "facts": facts}}, ensure_ascii=False)


def gold_payload(item: KGItem, g) -> dict:
    """A Re-DocRED training document's gold as an aligned payload."""
    from .label_bank import DOCRED_TYPES
    names = {r["name"] for r in item.relations}
    sent_of = {}
    for j, (s, e) in enumerate(item.sentence_spans):
        for i in range(s, e):
            sent_of[i] = j
    ents = [{"type": DOCRED_TYPES[e["type"]][0], "spans": [[a, b + 1] for a, b in e["mentions"]]}
            for e in g.entities]
    facts = []
    for h, t, r in g.layers["rel"]:
        if r not in names:
            continue
        hs = {sent_of[a] for a, _ in g.entities[h]["mentions"]}
        ts = {sent_of[a] for a, _ in g.entities[t]["mentions"]}
        facts.append({"head": h, "tail": t, "relation": r, "polarity": "positive",
                      "modality": "asserted", "holder": -1, "time": -1,
                      "evidence": sorted(hs & ts) or sorted({min(hs), min(ts)})})
    return {"entities": ents, "facts": facts}


def demo_pool(sampler: SchemaSampler, bank: dict) -> dict[str, tuple[KGItem, dict]]:
    """Demonstrations: Re-DocRED train documents (gold), and pilot windows of
    each domain outside both evaluation halves (astra's answers), short ones
    first. Ids: ``rd0..``, ``<domain>0..``."""
    pool = {}
    rd = sorted(redocred_set("search", sampler, source="train"), key=lambda x: len(x[0].tokens))
    rd = [x for x in rd if 120 <= len(x[0].tokens) <= 220 and len(x[1].layers["rel"]) >= 8][:3]
    for i, (it, g) in enumerate(rd):
        pool[f"rd{i}"] = (it, gold_payload(it, g))
    used = {it.id for sp in N_PILOT for it, _ in pilot_set(sp, bank)}
    by = defaultdict(list)
    for it, p in load_export(PILOT):
        if it.domain in PILOT_DOMAINS and it.id not in used and len(p["facts"]) >= 4 \
                and len(it.tokens) <= 260:
            by[it.domain].append((_describe(it, bank), p))
    for d, xs in by.items():
        xs.sort(key=lambda x: hashlib.sha1(f"demo:{x[0].id}".encode()).hexdigest())
        for i, x in enumerate(xs[:3]):
            pool[f"{d}{i}"] = x
    return pool


# ── running a variant ────────────────────────────────────────────────
def _user(item: KGItem) -> str:
    sents = [" ".join(item.tokens[s:e]) for s, e in item.sentence_spans]
    return kg_user_message(sents, item.speakers, item.participants, item.types, item.relations)


class TunedKG(KGAnnotator):
    def __init__(self, spec, cache, variant: Variant, demos: dict):
        super().__init__(spec, cache)
        self.variant, self.demos = variant, demos

    def build_messages(self, layer: str, item) -> list[dict]:
        msgs = [{"role": "system", "content": self.variant.system}]
        for d in self.variant.demos:
            it, p = self.demos[d]
            msgs += [{"role": "user", "content": _user(it)},
                     {"role": "assistant", "content": render_answer(it, p)}]
        return msgs + [{"role": "user", "content": _user(item) + self._json_hint(layer)}]

    def _build_kwargs(self, layer: str, messages: list[dict]) -> dict:
        return {**super()._build_kwargs(layer, messages), **self.variant.params}


class Judge(KGAnnotator):
    def build_messages(self, layer: str, item) -> list[dict]:
        return [{"role": "system", "content": SYSTEM["kgjudge"]},
                {"role": "user", "content": item.layers["judge_prompt"] + self._json_hint(layer)}]


def _with_id(item: KGItem, suffix: str) -> KGItem:
    it = KGItem(**{**item.__dict__})
    it.id = f"{item.id}:{suffix}"
    return it


async def _run(ann, layer: str, items: list, tag: str) -> dict:
    logger = RunLogger(OUT / tag, every=25)
    out = {}
    try:
        for coro in asyncio.as_completed([ann.annotate_and_cache(layer, it) for it in items]):
            r = await coro
            logger.record(r, (0, 0), len(items)); out[r.item_id] = r
    finally:
        logger.close()
    return out


def _mentions(item: KGItem, spans) -> str:
    return " / ".join(" ".join(item.tokens[a:b]) for a, b in spans[:3])


def judge_prompt(item: KGItem, payload: dict) -> str:
    k = len(item.participants)
    defs = {r["name"]: r["description"] for r in item.relations}

    def name(i):
        return item.participants[i] if i < k else _mentions(item, payload["entities"][i - k]["spans"])
    sents = [" ".join(item.tokens[s:e]) for s, e in item.sentence_spans]
    lines = ["Passage:"]
    for j, s in enumerate(sents):
        who = f"[{item.speakers[j]}] " if item.speakers and item.speakers[j] else ""
        lines.append(f"{j + 1}\t{who}{s}")
    lines += ["", "Facts:"]
    for n, f in enumerate(payload["facts"]):
        neg = "NOT " if f["polarity"] == "negative" else ""
        lines.append(f"{n}\t[{name(f['head'])}] {neg}{f['relation']} [{name(f['tail'])}]"
                     f"  ({f['relation']}: {defs.get(f['relation'], '')})")
    return "\n".join(lines)


def _match(pred_spans: list, ref_spans: list) -> dict:
    """Predicted entity -> reference entity by the most shared mention spans."""
    out = {}
    for i, pm in enumerate(pred_spans):
        ov = [len(pm & x) for x in ref_spans]
        m = max(ov) if ov else 0
        out[i] = ov.index(m) if m and ov.count(m) == 1 else None
    return out


def score(variant: Variant, split: str, annotator: str, judge: str | None,
          sampler: SchemaSampler, bank: dict, demos: dict) -> dict:
    spec = load_annotators([annotator])[annotator]
    cache = CacheStore(CACHE, PROMPT_VERSION)
    ann = TunedKG(spec, cache, variant, demos)
    rd = redocred_set(split, sampler)
    pl = pilot_set(split, bank)
    rd_items = [_with_id(it, f"tune{variant.digest}") for it, _ in rd]
    pl_items = [_with_id(it, f"tune{variant.digest}") for it, _ in pl]
    res = asyncio.run(_run(ann, "kg", rd_items + pl_items, f"{variant.name}-{split}-{annotator}"))
    rep: dict = {"variant": variant.name, "digest": variant.digest, "split": split,
                 "failed": sum(not r.ok for r in res.values())}
    # Re-DocRED: gold precision and recall, entities matched by span.
    tp = npred = ngold = 0
    misses = []
    for it, (_, g) in zip(rd_items, rd):
        names = {r["name"] for r in it.relations}
        gold = {(h, t, r) for h, t, r in g.layers["rel"] if r in names}
        ngold += len(gold)
        r = res.get(it.id)
        if r is None or not r.ok:
            continue
        p = r.payload
        gm = [{(a, e + 1) for a, e in en["mentions"]} for en in g.entities]
        m = _match([{tuple(s) for s in e["spans"]} for e in p["entities"]], gm)
        pred = {(m.get(f["head"]), m.get(f["tail"]), f["relation"]) for f in p["facts"]}
        tp += len(pred & gold); npred += len(p["facts"])
        for h, t, rel in sorted(gold - pred)[:2]:
            misses.append(f"{g.entities[h]['aliases'][0]} --{rel}--> {g.entities[t]['aliases'][0]}")
    rep["rd_p"], rep["rd_r"] = tp / max(npred, 1), tp / max(ngold, 1)
    rep["rd_f1"] = 2 * tp / max(npred + ngold, 1)
    # Pilot: density, astra coverage, judged precision.
    per = defaultdict(Counter)
    jitems = []
    for it, (_, ref) in zip(pl_items, pl):
        c = per[it.domain]
        r = res.get(it.id)
        c["windows"] += 1
        c["ref_facts"] += len(ref["facts"])
        if r is None or not r.ok:
            continue
        p = r.payload
        c["facts"] += len(p["facts"]); c["entities"] += len(p["entities"])
        k = len(it.participants)
        m = _match([{tuple(s) for s in e["spans"]} for e in p["entities"]],
                   [{tuple(s) for s in e["spans"]} for e in ref["entities"]])
        idx = lambda x: x if x < k else (None if m.get(x - k) is None else m[x - k] + k)
        pred = {(idx(f["head"]), idx(f["tail"]), f["relation"]) for f in p["facts"]}
        refs = {(f["head"], f["tail"], f["relation"]) for f in ref["facts"]}
        c["ref_found"] += len(pred & refs)
        if p["facts"] and judge:
            j = _with_id(it, f"judge{variant.digest}")
            j.layers = {"judge_prompt": judge_prompt(it, p)}
            jitems.append((j, it.domain, len(p["facts"])))
    if jitems:
        jspec = load_annotators([judge])[judge]
        jres = asyncio.run(_run(Judge(jspec, cache), "kgjudge", [j for j, _, _ in jitems],
                                f"{variant.name}-{split}-judge"))
        for j, d, n in jitems:
            r = jres.get(j.id)
            if r is not None and r.ok:
                per[d]["judged"] += n
                per[d]["supported"] += sum(1 for i in range(n) if r.payload.get(i, False))
    tot = Counter()
    for c in per.values():
        tot.update(c)
    per["ALL"] = tot
    rep["pilot"] = {d: {"facts_w": c["facts"] / max(c["windows"], 1),
                        "ents_w": c["entities"] / max(c["windows"], 1),
                        "astra_cov": c["ref_found"] / max(c["ref_facts"], 1),
                        "judged_p": c["supported"] / max(c["judged"], 1) if c["judged"] else None}
                    for d, c in per.items()}
    rep["rd_misses"] = misses[:20]
    return rep


def print_rep(rep: dict) -> None:
    a = rep["pilot"]["ALL"]
    jp = f"{a['judged_p']:.2f}" if a["judged_p"] is not None else "  - "
    print(f"{rep['variant']:16s} {rep['split']:6s} RD P {rep['rd_p']:.3f} R {rep['rd_r']:.3f} "
          f"F1 {rep['rd_f1']:.3f} | pilot facts/w {a['facts_w']:5.1f} astra-cov {a['astra_cov']:.2f} "
          f"judged-P {jp} | failed {rep['failed']}")
    for d in PILOT_DOMAINS:
        x = rep["pilot"].get(d)
        if x:
            jp = f"{x['judged_p']:.2f}" if x["judged_p"] is not None else "-"
            print(f"    {d:13s} facts/w {x['facts_w']:5.1f} ents/w {x['ents_w']:5.1f} "
                  f"astra-cov {x['astra_cov']:.2f} judged-P {jp}")


# ── proposing instructions ───────────────────────────────────────────
PROPOSE = (
    "You improve the system prompt of an information-extraction model. It is "
    "given a passage and a schema (entity types and relations with "
    "definitions) and must return every entity and every schema relation the "
    "passage supports, with attributes. Its answers are training labels for a "
    "smaller model, so missing facts and unsupported facts both hurt; today it "
    "misses far more than it invents. Below: the current prompt, its scores, "
    "and facts it missed. Write {n} distinct improved system prompts. Keep the "
    "output contract (entity ids, mentions as sentence + exact text, the fact "
    "fields and their meanings) unchanged; change how the model is told to "
    "work (e.g. a procedure, what counts as implied, what not to skip, how to "
    "handle conversations and technical text). Return JSON: "
    '{{"prompts": [{{"name": "...", "rationale": "...", "system": "..."}}]}}.'
)


def propose(base: Variant, rep: dict, n: int, proposer: str) -> list[Variant]:
    spec = load_annotators([proposer])[proposer]
    ann = KGAnnotator(spec, CacheStore(CACHE, PROMPT_VERSION))
    msg = (f"CURRENT PROMPT:\n{base.system}\n\nSCORES:\n{json.dumps(rep['pilot'], indent=1)}\n"
           f"Re-DocRED precision {rep['rd_p']:.3f}, recall {rep['rd_r']:.3f}\n\n"
           f"MISSED (Re-DocRED gold):\n" + "\n".join(rep["rd_misses"]))
    client = ann.client

    async def go():
        r = await client.chat.completions.create(
            model=spec.model, response_format={"type": "json_object"},
            messages=[{"role": "system", "content": PROPOSE.format(n=n)},
                      {"role": "user", "content": msg}])
        return json.loads(r.choices[0].message.content)
    out = asyncio.run(go())
    return [Variant(f"{base.name}.{p['name']}"[:40].replace(" ", "_"), p["system"],
                    list(base.demos), dict(base.params), p.get("rationale", ""))
            for p in out["prompts"][:n]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["eval", "propose", "show"])
    ap.add_argument("--variants", default="base")
    ap.add_argument("--split", choices=["search", "held"], default="search")
    ap.add_argument("--annotator", default="sol61")
    ap.add_argument("--judge", default="astra")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--from", dest="src", default="base")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args()
    bank = load_bank()
    sampler = SchemaSampler(bank)
    variants = load_variants()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.cmd == "show":
        for v in variants.values():
            print(f"{v.name} [{v.digest}] demos={v.demos} params={v.params}\n  {v.note}")
        return 0
    demos = demo_pool(sampler, bank)
    if args.cmd == "propose":
        rep_path = OUT / f"{args.src}-search-{args.annotator}.json"
        rep = json.loads(rep_path.read_text())
        for v in propose(variants[args.src], rep, args.n, args.judge):
            save_variant(v)
            print(f"saved {v.name}: {v.note}")
        return 0
    for name in args.variants.split(","):
        v = variants[name]
        missing = [d for d in v.demos if d not in demos]
        if missing:
            raise SystemExit(f"{name}: unknown demos {missing}; have {sorted(demos)}")
        rep = score(v, args.split, args.annotator, None if args.no_judge else args.judge,
                    sampler, bank, demos)
        (OUT / f"{name}-{args.split}-{args.annotator}.json").write_text(json.dumps(rep, indent=1))
        print_rep(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
