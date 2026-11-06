"""Name swap for the synthetic value and list passages.

The generator reused a few names ("Mara Ellison" heads 1,517 of 38k facts,
"Harbor..." companies everywhere) and always wrote people in full. Training
on that would teach a name prior and never show a short reference. This swap
runs when the features are built, with a fresh draw per (item, epoch):

* people get a new first and last name, the first name of the original's
  gender (read from the corpus's pronouns), so "she"/"he" still fit;
* organisations keep their kind word ("Bank", "Analytics") under a new stem;
  CamelCase products keep their tail ("...Sync") under a new stem;
* later mentions are rewritten to the forms real text uses, by register:
  surname and "Ms. / Dr. Surname" in formal styles, first name in chats,
  emails and notes, the stem alone for an organisation; in informal styles
  the first mention can be the first name alone;
* values in the facts' tails get fresh numbers in the same format (prices,
  amounts, rates, counts, ages), dates and times (one year shift and month
  rotation per passage), card endings and operating systems; a value that
  repeats keeps one new value; a passage where one number is derived from
  others ("3 at $5, $15 in total") keeps its numbers;
* real brands ("PayPal", "PostgreSQL") are kept;
* every teacher span is remapped to the new tokens. A span that started or
  ended inside a rewritten name is dropped (and an entity left with no span,
  with its facts); the counts are reported.

    python -m v6.train.name_swap            # report over the 10k passages
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FEMALE = ("Amara Elif Farah Hana Ines Lucia Nia Priya Sana Uma Ximena Zara Anouk Dalia Freya Ilse "
          "Leila Noor Olga Rania Thandi Valeria Yara Zeynep Aiko Chiara Esra Greta Isla Lina Paula "
          "Rhea Tove Wanjiru Ama Carmen Maya Ingrid Sofia Aisha Mei Keiko Fatima Elena Hanne Ayla "
          "Grace Anika Ruth Imani Selin Mira Clara Dina Rosa Beatriz Signe Adaeze Yuki Marisol").split()
MALE = ("Bilal Dario Goran Jomo Kaito Mateo Oren Rafael Tomas Viktor Yusuf Cyrus Emeka Gael Hiroshi "
        "Kofi Malik Pavel Soren Ugo Wiktor Bruno Dmitri Hamid Jasper Kemal Mihai Omar Vikram Yosef "
        "Zoltan Bjorn Lukas Arjun Tariq Diego Henrik Kwame Felix Ivan Ravi Samuel Nikhil Hugo Andre "
        "Kenji Tobias Ahmed Leon Marco Elias Chidi Anders Rohan Mattias Teodor Baraka Joaquin").split()
NEUTRAL = ("Quinn Chen Wen Bao Jun Femi Sami Deniz Alex Robin Sasha Kai Remy Ari Eden Rowan Jordan "
           "Avery Noa Sam Morgan Jesse Riley Taylor Emerson").split()
LAST = ("Okafor Lindqvist Ferreira Nakamura Haddad Kowalski Mensah Duarte Varga Ozturk Bianchi Petrov "
        "Achebe Sato Moreau Novak Ibrahim Castillo Larsen Kumar Abara Reyes Tanaka Dubois Horvat Aydin "
        "Rossi Nwosu Eriksen Mehta Szabo Alvarez Fischer Banda Lopes Yilmaz Costa Ivanova Mwangi "
        "Schmidt Takahashi Silva Popescu Rahman Jensen Morales Hoang Kariuki Weber Oyelaran Fontaine "
        "Bauer Delgado Kaur Lund Ortega Sorensen Zhou Adeyemi Vasquez Brennan Halloran Quist Whitfield "
        "Pereira Oduya Nguyen Chandra Abbasi Kovac Lindgren Moretti Salazar Takeda Wojcik Hartmann "
        "Ndiaye Calder Iyer Bose Forde Gallo Strand Kaplan Osei Marchetti Aguilar Sundberg Okonkwo "
        "Rasmussen Ferrante Kapoor Mbatha Lachance Hendricks Volkov Inoue Barros Ekwueme").split()
STEM = ("Alder Basalt Cinder Ember Fjord Garnet Helio Indigo Juniper Kestrel Lumen Meridian Nimbus "
        "Onyx Pinnacle Quartz Rivet Sable Tundra Umber Vertex Willow Yarrow Zephyr Brook Cobalt Drift "
        "Flint Grove Halcyon Kiln Lattice Maple Orbit Pebble Quill Ridge Sprout Tidal Vale Wren Beacon "
        "Crest Dune Fern Glint Hive Ingot Loom Moss Opal Prism Thistle Corvid Larch Saffron Talus "
        "Marrow Bramble Cairn Osprey Lantern Fennel Granite Heron Isle Jasmine Kelp Mistral Nettle "
        "Ochre Plover Rook Sorrel Tamarack Verity Wicket").split()
# kind words that can stand in for each other without changing what the
# organisation is (a "Bank" stays a bank)
KINDS = [("Analytics", "Data", "Insights", "Intelligence"),
         ("Systems", "Technologies", "Software", "Digital", "Labs", "Tech"),
         ("Holdings", "Group", "Partners", "Capital", "Ventures"),
         ("Logistics", "Freight", "Shipping", "Transport"),
         ("Health", "Healthcare", "Medical", "Care"),
         ("Foods", "Kitchens", "Provisions"),
         ("Consulting", "Advisory", "Associates", "Solutions")]
KIND_OF = {w: g for g in KINDS for w in g}
KNOWN_FIRST = set(FEMALE) | set(MALE) | set(NEUTRAL) | set(
    "Marta Mara Nadia Mira Lena Leena Nina Maren Tessa Nora Leona Marisa Elena Priya Maya Leila".split())
REAL_BRANDS = set("""PostgreSQL PayPal OpenTelemetry RabbitMQ OpenSSL GitHub GitLab MySQL SQLite MongoDB
    DynamoDB BigQuery GraphQL NoSQL OAuth JavaScript TypeScript PyTorch TensorFlow NumPy YouTube LinkedIn
    WhatsApp FaceTime MacBook AirPods PlayStation QuickBooks WordPress SharePoint OneDrive PowerPoint
    HubSpot NetSuite ChatGPT OpenAI DeepMind CrowdStrike FedEx DoorDash SoftBank BlackRock ExxonMobil
    MasterCard ClickHouse CockroachDB InfluxDB OpenSearch ElasticSearch CloudFlare DigitalOcean SendGrid
    MailChimp AutoCAD SolidWorks PowerBI LastPass BambooHR ZipRecruiter JetBlue UberEats GrubHub
    DevOps WooCommerce BigCommerce QuickTime AirDrop CarPlay HomeKit SmartThings PagerDuty NewRelic
    DataDog ServiceNow SuccessFactors WebEx GoPro TikTok SnapChat InDesign IntelliJ PhpStorm VMware
    OpenShift CentOS AlmaLinux FreeBSD OpenBSD MariaDB TimescaleDB DuckDB ScyllaDB OpenAPI""".split())
FUNCTION = set("""Desk Sync Pay Cart Queue Trace Patch Book View Bridge Dock Vault Task Market Relay Route
    Stock Till Slot Fleet Flow Link Pilot Grid Note Box Metric Suite Lane Path Scope Forge Meet Stack
    Seat Stay Shelf Parcel Venue Point Loop Log Stream Tally Vend Data Insight Track Shift Fund Pulse
    Board Cast Mail Sign Hire Form Shop Wallet Ticket Plan Chart""".split())
FUNCTION_L = sorted(FUNCTION)
FORMAL = {"a news sentence", "a press release", "an excerpt from an annual report",
          "a short professional bio", "a booking confirmation", "a product listing"}
SHE, HE = {"she", "her", "hers", "herself"}, {"he", "him", "his", "himself"}
TO_MALE = {"she": "he", "hers": "his", "herself": "himself"}   # "her": by case, below
HER_CASES = Path(__file__).resolve().parents[2] / "data" / "v6-corpus" / "synth_her_case.json"
CAP = re.compile(r"^[A-Z][a-z]+$")
CAMEL = re.compile(r"^([A-Z][a-z]+)([A-Z][a-zA-Z]+)$")
SYNTH = Path(__file__).resolve().parents[1] / "runs" / "kg-short-synth-r4"


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def _span_text(tokens, s):
    return tuple(tokens[s[0]:s[1]])


def _people(tokens, payload):
    """Two-word capitalised person names from the teacher's person spans."""
    out = set()
    for e in payload["entities"]:
        if e["type"] != "person":
            continue
        for s in e["spans"]:
            t = _span_text(tokens, s)
            if len(t) == 2 and all(CAP.match(x) for x in t):
                out.add(t)
    return out


def gender_table(items, payloads) -> dict[str, str]:
    """Gender of each original first name, from passages that name one
    person: 'f' / 'm' when at least 80% of 2+ pronouns agree, else 'n'."""
    c = defaultdict(Counter)
    for it in items:
        p = payloads.get(it["id"])
        if p is None:
            continue
        ppl = _people(it["tokens"], p)
        if len(ppl) != 1:
            continue
        low = [t.lower() for t in it["tokens"]]
        f, m = sum(t in SHE for t in low), sum(t in HE for t in low)
        c[next(iter(ppl))[0]].update(f=f, m=m)
    out = {}
    for name, x in c.items():
        n = x["f"] + x["m"]
        out[name] = "f" if n >= 2 and x["f"] >= 0.8 * n else "m" if n >= 2 and x["m"] >= 0.8 * n else "n"
    return out


def _pick(rng, p: dict):
    keys = list(p)
    return keys[rng.choice(len(keys), p=np.array([p[k] for k in keys]) / sum(p.values()))]


def _person_form(rng, k: int, formal: bool):
    if k == 0:
        return _pick(rng, {"full": 0.9, "title_full": 0.1} if formal else {"full": 0.75, "first": 0.25})
    return _pick(rng, {"full": 0.25, "last": 0.45, "title_last": 0.25, "first": 0.05} if formal
                 else {"full": 0.2, "first": 0.65, "last": 0.1, "title_last": 0.05})


def her_cases(model: str = "ettin-68m") -> int:
    """Case of every "her" in the synthetic passages (possessive -> "his",
    object -> "him"), from kniv's own morphology; written once to
    ``HER_CASES``. A passage whose tokens the runtime splits differently is
    left out (it is then never gender-flipped)."""
    from kniv_v7.runtime import Kniv
    root = Path(__file__).resolve().parents[2]
    k = Kniv(root / "release" / model, int8=True)
    items, _ = load_synth()
    out, skipped = {}, 0
    for it in items:
        if not any(t.lower() == "her" for t in it["tokens"]):
            continue
        ws = k.analyse(it["layers"]["text"], relations=[], linguistic=True)
        toks = [t for w in ws for t in w["tokens"]]
        words = [x for w in ws for x in w["words"]]
        if toks != it["tokens"]:
            skipped += 1
            continue
        out[it["id"]] = {str(i): ("poss" if "Poss=Yes" in words[i]["morph"] else "acc")
                         for i, t in enumerate(toks) if t.lower() == "her"}
    HER_CASES.write_text(json.dumps(out))
    print(f"{len(out)} passages with 'her' ({sum(len(v) for v in out.values())} tokens; "
          f"{dict(Counter(c for v in out.values() for c in v.values()))}); {skipped} skipped (tokenisation)")
    return 0


# ---- values -------------------------------------------------------------
MONTHS = "January February March April May June July August September October November December".split()
OSES = [["Ubuntu", "22.04"], ["Ubuntu", "24.04", "LTS"], ["Debian", "12"], ["Debian", "11"],
        ["Windows", "10"], ["Windows", "11"], ["Windows", "Server", "2022"], ["macOS", "14"],
        ["macOS", "15"], ["Rocky", "Linux", "9"], ["Fedora", "40"], ["Alpine", "Linux", "3.19"],
        ["Red", "Hat", "Enterprise", "Linux", "9"], ["Android", "14"], ["Android", "13"], ["iOS", "17"],
        ["FreeBSD", "14"], ["Amazon", "Linux", "2023"], ["openSUSE", "Leap", "15.6"], ["ChromeOS"]]
OS_RE = re.compile(r"^(Ubuntu( Server)? \d+\.\d+( LTS)?|Windows( Server)? \d+|Debian \d+|Android \d+|"
                   r"macOS \d+|iOS \d+|Fedora \d+)$")
NUM = re.compile(r"^(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(%?)$")
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "k": 1e3, "m": 1e6, "bn": 1e9}


def _fmt_like(x: float, tok: str) -> str:
    """``x`` written the way ``tok`` writes a number: same decimals, commas
    when the original had them, trailing zeros of a round integer kept."""
    body = NUM.match(tok).group(1)
    pct = tok.endswith("%")
    if "." in body:
        d = len(body.split(".")[1])
        out = f"{x:.{d}f}"
    else:
        digits = body.replace(",", "")
        zeros = len(digits) - len(digits.rstrip("0")) if len(digits) > 1 else 0
        q = 10 ** min(zeros, max(len(str(int(round(x)))) - 1, 0))
        n = max(int(round(x / q)) * q, 1)
        out = f"{n:,}" if "," in body or n >= 10000 else str(n)
    return out + ("%" if pct else "")


def _quantities(toks) -> list[float]:
    out = []
    for i, t in enumerate(toks):
        m = NUM.match(t)
        if not m:
            continue
        v = float(m.group(1).replace(",", ""))
        if i + 1 < len(toks) and toks[i + 1].lower() in SCALE:
            v *= SCALE[toks[i + 1].lower()]
        out.append(v)
    return out


def derived(toks) -> bool:
    """True when one quantity is the sum, difference, product or percentage
    of two others (within 0.5%): perturbing numbers would break the passage."""
    q = sorted({v for v in _quantities(toks) if v >= 2})
    for c in q:
        if c < 10:
            continue
        for i, a in enumerate(q):
            for b in q[i:]:
                if a == c or b == c:
                    continue
                for r in (a + b, a * b, abs(b - a), a * b / 100):
                    if r > 0 and abs(r - c) <= 0.005 * c:
                        return True
    return False


def value_edits(toks, payload, rng, claim, stats, tol: bool = True) -> None:
    """New numbers, dates, times and operating systems for the value facts'
    tails: every annotated span of a tail entity whose text holds a digit,
    and every other occurrence of the same multi-word value string. One
    mapping per passage, so a repeated value stays the same; years move by
    one shift and months by one rotation, so their order mostly survives."""
    rel_of = defaultdict(set)
    for f in payload["facts"]:
        rel_of[f["tail"]].add(f["relation"])
    spans, ages = set(), set()
    for i, rels in rel_of.items():
        for sp in payload["entities"][i]["spans"]:
            if sp[1] > len(toks):
                stats["value span past the end"] += 1
                continue
            t = toks[sp[0]:sp[1]]
            if any(c.isdigit() for x in t for c in x) or OS_RE.match(" ".join(t)):
                spans.add(tuple(sp))
                if "has age" in rels:
                    ages.add(" ".join(t))
    if not spans:
        return
    strings = {tuple(toks[a:b]) for a, b in spans if b - a >= 2}
    for st in strings:                                   # unannotated repeats of the same value
        n = len(st)
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) == st:
                spans.add((i, i + n))
    frozen = derived(toks)
    stats["passages with derived values (numbers kept)"] += frozen
    year_shift = int(rng.integers(-3, 4))
    month_rot = int(rng.integers(-4, 5))
    memo, os_memo = {}, {}

    def new_tok(i, t, kind):
        key = (kind, t)
        if key not in memo:
            m = NUM.match(t)
            v = float(m.group(1).replace(",", "")) if m else None
            if kind == "year":
                out = str(int(v) + year_shift)
            elif kind == "day":
                out = str(int(rng.integers(1, 29)))
            elif kind == "hour":
                out = str(int(rng.integers(1, 13)))
            elif kind == "minute":
                out = ["00", "15", "30", "45"][rng.integers(4)]
            elif kind == "card":
                out = f"{int(rng.integers(1000, 10000))}"
            elif kind == "age":
                out = str(int(rng.integers(19, 75)))
            else:
                for _ in range(8):
                    x = v * float(np.exp(rng.uniform(-1.0, 1.0)))
                    if t.endswith("%") and v < 100:
                        x = min(x, 95.0)
                    if v >= 2 and "." not in t:
                        x = max(x, 2.0)          # never "1 hours"
                    out = _fmt_like(x, t)
                    if out != t:
                        break
            memo[key] = out
        return memo[key]

    for a, b in sorted(spans):
        t = toks[a:b]
        is_age = " ".join(t) in ages
        if OS_RE.match(" ".join(t)):
            k = " ".join(t)
            if k not in os_memo:
                os_memo[k] = OSES[rng.integers(len(OSES))]
            claim(a, b, os_memo[k], "value:os")
            continue
        for i in range(a, b):
            x = toks[i]
            prev = toks[i - 1] if i > 0 else ""
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            if x in MONTHS:
                claim(i, i + 1, [MONTHS[(MONTHS.index(x) + month_rot) % 12]], "value:month")
                continue
            if not NUM.match(x):
                continue
            n = float(NUM.match(x).group(1).replace(",", ""))
            if x.isdigit() and len(x) == 4 and 1990 <= n <= 2039 and "%" not in x:
                kind = "year"
            elif x.isdigit() and n <= 31 and (prev in MONTHS or nxt in MONTHS):
                kind = "day"
            elif x.isdigit() and nxt == ":":
                kind = "hour"
            elif x.isdigit() and prev == ":":
                kind = "minute"
            elif x.isdigit() and len(x) == 4 and prev == "in" and i >= 2 and toks[i - 2] == "ending":
                kind = "card"
            elif frozen:
                continue
            elif x.isdigit() and is_age:
                kind = "age"
            else:
                kind = "num"
            claim(i, i + 1, [new_tok(i, x, kind)], f"value:{kind}")


def swap_names(item: dict, payload: dict, seed: str, gender: dict[str, str],
               stats: Counter | None = None, her: dict | None = None,
               flip: float = 0.5, values: bool = True) -> tuple[dict, dict]:
    """``item`` (a KGItem dict) and its teacher ``payload`` with every person,
    organisation and CamelCase product renamed and later mentions varied; the
    draw depends only on ``seed`` (e.g. f"{item_id}:{epoch}").

    Gender flip: with probability ``flip``, a passage whose named people are
    all women (or that names none) and has no male pronoun becomes male: male
    first names, "Mr", and she/her/hers/herself -> he/his or him/his/himself.
    Only when every "her" has a known case (``her``: item id -> {position:
    'poss'|'acc'})."""
    stats = Counter() if stats is None else stats
    rng = _rng(f"swap:{seed}")
    toks = item["tokens"]
    low = [t.lower() for t in toks]
    formal = item.get("layers", {}).get("style") in FORMAL
    claimed = np.zeros(len(toks), bool)
    edits = []          # (start, end, new tokens, form)

    def claim(s, e, new, form):
        if claimed[s:e].any():
            return
        claimed[s:e] = True
        edits.append((s, e, list(new), form))
        stats[form] += 1

    # people: one new name each, first names and surnames distinct in the passage
    ppl = sorted(_people(toks, payload))
    parts = {x for n in ppl for x in n}
    singles = sorted({toks[a] for e in payload["entities"] if e["type"] == "person"
                      for a, b in e["spans"] if b - a == 1 and b <= len(toks) and CAP.match(toks[a])
                      and toks[a] not in parts and (toks[a] in KNOWN_FIRST or toks[a] in gender)})
    has_she, has_he = any(t in SHE for t in low), any(t in HE for t in low)
    gof = {f: gender.get(f, "n") for f, _ in ppl}
    gof.update({f: gender.get(f, "n") for f in singles})
    if len(ppl) == 1 and gof[ppl[0][0]] == "n" and has_she != has_he:
        gof[ppl[0][0]] = "f" if has_she else "m"      # the passage's own pronouns decide
    cases = (her or {}).get(item["id"], {})
    hers_known = all(str(i) in cases for i, t in enumerate(low) if t == "her")
    male = (her is not None and not has_he and hers_known and all(g == "f" for g in gof.values())
            and (ppl or has_she) and rng.random() < flip)
    if male:
        gof = {f: "m" for f in gof}
        stats["passages flipped to male"] += 1
    firsts = {"f": FEMALE, "m": MALE, "n": NEUTRAL}
    used_f, used_l = set(), set()
    for f, l in ppl:
        g = gof[f]
        pool = [x for x in firsts[g] if x not in used_f]
        nf = pool[rng.integers(len(pool))]
        pool = [x for x in LAST if x not in used_l]
        nl = pool[rng.integers(len(pool))]
        used_f.add(nf); used_l.add(nl)
        title = "Dr" if g == "n" or rng.random() < 0.3 else ("Ms" if g == "f" else "Mr")
        forms = {"full": [nf, nl], "first": [nf], "last": [nl],
                 "title_full": [title, ".", nf, nl], "title_last": [title, ".", nl]}
        k = 0
        for i in range(len(toks) - 1):
            if toks[i] == f and toks[i + 1] == l:
                form = _person_form(rng, k, formal)
                claim(i, i + 2, forms[form], f"person:{form}")
                k += 1
        for i, t in enumerate(toks):         # references the generator already shortened
            if not claimed[i] and t == l:
                claim(i, i + 1, [nl], "person:kept_last")
            elif not claimed[i] and t == f:
                claim(i, i + 1, [nf], "person:kept_first")
    for f in singles:                        # people named by first name only ("Hi Marta")
        pool = [x for x in firsts[gof[f]] if x not in used_f]
        nf = pool[rng.integers(len(pool))]
        used_f.add(nf)
        for i, t in enumerate(toks):
            if not claimed[i] and t == f:
                claim(i, i + 1, [nf], "person:first_only")

    # organisations: new stem, same kind word; later mentions may drop it
    used_s = set()
    orgs, prods = set(), set()
    for e in payload["entities"]:
        for s in e["spans"]:
            t = _span_text(toks, s)
            if e["type"] == "organization" and 2 <= len(t) <= 3 and all(CAP.match(x) for x in t):
                orgs.add(t)
            elif e["type"] in ("product or service", "organization") and t and CAMEL.match(t[0]) \
                    and (len(t) == 1 or e["type"] == "product or service" and len(t) <= 3):
                prods.add(t[0])                  # "HarborSync", "TraceNest Pro"
    def new_stem():
        pool = [x for x in STEM if x not in used_s]
        a = pool[rng.integers(len(pool))]
        used_s.add(a)
        if rng.random() < 0.5:                       # compound: "Alder" + "brook"
            b = STEM[rng.integers(len(STEM))].lower()
            return a + b if b[0] != a[-1] else a + b[1:]
        return a

    for org in sorted(orgs, key=len, reverse=True):
        stem = new_stem()
        kind = KIND_OF.get(org[-1])
        tail = [*org[1:-1], kind[rng.integers(len(kind))] if kind else org[-1]]
        org_new = [stem, *tail]
        n, k = len(org), 0
        for i in range(len(toks) - n + 1):
            if tuple(toks[i:i + n]) == org:
                short = k > 0 and rng.random() < 0.5
                claim(i, i + n, org_new[:-1] if short else org_new, "org:stem" if short else "org:full")
                k += 1
        for i, t in enumerate(toks):
            if not claimed[i] and t == org[0]:
                claim(i, i + 1, [stem], "org:kept_stem")
    for p in sorted(prods):
        if p in REAL_BRANDS:
            stats["product:real brand kept"] += 1
            continue
        head, tail = CAMEL.match(p).groups()
        # keep the part that says what it is ("Desk", "Pay"); the generator's
        # brand-ish parts ("Harbor", "Nest", "Ledger") go
        func = tail if tail in FUNCTION else head if head in FUNCTION else FUNCTION_L[rng.integers(len(FUNCTION_L))]
        pool = [x for x in STEM if x not in used_s]
        stem = pool[rng.integers(len(pool))]
        used_s.add(stem)
        for i, t in enumerate(toks):
            if not claimed[i] and t == p:
                claim(i, i + 1, [stem + func], "product")

    if male:
        for i, t in enumerate(low):
            if t in SHE and not claimed[i]:
                rep = ("his" if cases[str(i)] == "poss" else "him") if t == "her" else TO_MALE[t]
                claim(i, i + 1, [rep.capitalize() if toks[i][0].isupper() else rep], "pronoun")

    if values:
        value_edits(toks, payload, _rng(f"value:{seed}"), claim, stats)

    # apply, building the boundary map old -> new
    edits.sort()
    new, bmap, j, prev = [], {}, 0, 0
    for s, e, rep, _ in edits:
        for b in range(prev, s):
            bmap[b] = len(new); new.append(toks[b])
        bmap[s] = len(new)
        if len(rep) == e - s:                    # same length: inner boundaries map one to one
            for d in range(1, e - s):
                bmap[s + d] = len(new) + d
        new.extend(rep)
        prev = e
    for b in range(prev, len(toks)):
        bmap[b] = len(new); new.append(toks[b])
    bmap[len(toks)] = len(new)

    def remap(sp):
        a, b = bmap.get(sp[0]), bmap.get(sp[1])
        return None if a is None or b is None or b <= a else [a, b]

    edited = np.zeros(len(toks) + 1, bool)
    for s_, e_, _, _ in edits:
        edited[s_:e_] = True
    nedited = np.zeros(len(new) + 1, bool)
    for s_, e_, rep, _ in edits:
        nedited[bmap[s_]:bmap[s_] + len(rep)] = True

    def same_outside(old, nw):
        return [toks[i] for i in range(*old) if not edited[i]] == [new[i] for i in range(*nw) if not nedited[i]]

    keep_idx, ents = {}, []
    for i, e in enumerate(payload["entities"]):
        spans = []
        for s0 in e["spans"]:
            x = remap(s0)
            if x is not None:
                stats["span check failed"] += not same_outside(s0, x)
                spans.append(x)
        stats["spans dropped"] += len(e["spans"]) - len(spans)
        if spans:
            keep_idx[i] = len(ents)
            ents.append({**e, "spans": spans})
        else:
            stats["entities dropped"] += 1
    facts = []
    for f in payload["facts"]:
        if f["head"] in keep_idx and f["tail"] in keep_idx:
            ref = lambda x: keep_idx.get(x, -1) if x >= 0 else x
            facts.append({**f, "head": keep_idx[f["head"]], "tail": keep_idx[f["tail"]],
                          "holder": ref(f.get("holder", -1)), "time": ref(f.get("time", -1))})
        else:
            stats["facts dropped"] += 1
    sents = [remap(s) for s in item["sentence_spans"]]
    assert all(s is not None for s in sents), "a sentence boundary fell inside a name"
    layers = {**item.get("layers", {}), "text": " ".join(new)}
    return ({**item, "tokens": new, "sentence_spans": sents, "layers": layers},
            {**payload, "entities": ents, "facts": facts})


def load_synth(run: Path = SYNTH):
    items = json.loads((run / "synth.json").read_text())
    payloads = {}
    for line in (run / "synth-ann-sol61" / "events.jsonl").open():
        e = json.loads(line)
        if e["ok"]:
            payloads[e["item_id"]] = e["payload"]
    return items, payloads


def report(epochs: int = 2) -> int:
    items, payloads = load_synth()
    gender = gender_table(items, payloads)
    print(f"{len(items)} passages, {len(payloads)} annotated; first names by gender: "
          f"{dict(Counter(gender.values()))}")

    def heads(its, pls):
        c = Counter()
        for it in its:
            p = pls.get(it["id"])
            if p is None:
                continue
            for f in p["facts"]:
                for k in ("head", "tail"):
                    e = p["entities"][f[k]]
                    t = _span_text(it["tokens"], e["spans"][0])
                    if e["type"] in ("person", "organization") and all(x[:1].isupper() for x in t):
                        c[" ".join(t)] += 1
        return c

    her = json.loads(HER_CASES.read_text()) if HER_CASES.exists() else None
    before = heads(items, payloads)
    for ep in range(epochs):
        stats, its, pls, pron = Counter(), [], {}, Counter()
        for it in items:
            p = payloads.get(it["id"])
            if p is None:
                continue
            a, b = swap_names(it, p, f"{it['id']}:{ep}", gender, stats, her)
            again = swap_names(it, p, f"{it['id']}:{ep}", gender, her=her)
            assert again[0]["tokens"] == a["tokens"], "not deterministic"
            low = [t.lower() for t in a["tokens"]]
            pron.update(f=sum(t in SHE for t in low), m=sum(t in HE for t in low))
            its.append(a); pls[a["id"]] = b
        after = heads(its, pls)
        top = lambda c: ", ".join(f"{k} {v}" for k, v in c.most_common(4))
        print(f"\nepoch {ep}: named person/org fact arguments distinct {len(before) / sum(before.values()):.0%} -> "
              f"{len(after) / sum(after.values()):.0%}; most frequent before: {top(before)}; after: {top(after)}")
        print("  forms:", dict(sorted(stats.items())))
        print(f"  pronouns after: she/her {pron['f']}, he/him {pron['m']}")
    shown = 0
    for it in items:
        p = payloads.get(it["id"])
        if p is None or len(_people(it["tokens"], p)) == 0:
            continue
        txt = " ".join(it["tokens"])
        if sum(txt.count(" ".join(n)) for n in _people(it["tokens"], p)) < 2:
            continue
        a, _ = swap_names(it, p, f"{it['id']}:0", gender, her=her)
        print(f"\n[{it['layers']['style']}]\n  before: {txt[:300]}\n  after:  {' '.join(a['tokens'])[:300]}")
        shown += 1
        if shown == 4:
            break
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--her-cases", action="store_true", help="compute HER_CASES with kniv's morphology")
    a = ap.parse_args()
    raise SystemExit(her_cases() if a.her_cases else report(a.epochs))
