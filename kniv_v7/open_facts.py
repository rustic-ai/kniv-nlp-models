"""Schema-free facts from the linguistic layer: subject, predicate, object.

Decoded from one window of ``Kniv.analyse`` (``tokens``, ``words`` with
``pos`` / ``lemma`` / ``head`` / ``deprel``, ``srl``, ``coref``); no model
call. Sources:

- **role frames**: the subject is the frame's ARG0 (else its ARG1, for an
  object other than ARG1); objects are ARG1-ARG4 and the location and
  direction arguments. A leading preposition of the object joins the
  predicate ("push for", "derived from"); a particle joins it too ("fill
  out").
- **copulas**: "X is Y", Y a noun phrase with a subject and a copula.
- **appositions**: "X, Y" read as "X is Y".

Arguments are cut to their noun phrase (no conjuncts of a noun, clauses,
relative clauses, adverbs, trailing possessive marks); a pronoun becomes the
name of its coreference cluster, or the fact is dropped. Arguments headed
by a verb, adjective, adverb or function word are dropped, as are facts
whose two arguments overlap.

Each fact: ``subject`` / ``object`` (text, span ``[start, end)`` in window
words, ``names``: the cut text and the names of its coreference cluster),
``predicate`` (the words, their lemma), ``polarity`` (negative when the
frame has a negation or the copula's noun a "not"), ``modality`` (the modal
verb, if any), ``time`` (the frame's temporal argument, if any), ``holder``
(the speaker of a speech or belief verb whose content holds the fact, else
the turn's speaker in a conversation, else none), ``source``. In a
conversation "I" is the turn's speaker.
These attributes come from the roles; a verifier, where one is run, decides
the fact and its polarity and modality.
"""
from __future__ import annotations

NOT_THING = {"VERB", "ADJ", "ADV", "AUX", "ADP", "SCONJ", "CCONJ", "PART", "INTJ", "DET", "PUNCT", "SYM"}
NOMINAL = {"NOUN", "PROPN", "NUM", "SYM", "PRON"}
# dependents cut from a noun phrase (``conj`` / ``cc`` only on nominal heads)
CUT = {"acl", "acl:relcl", "advcl", "advmod", "parataxis", "appos", "dep", "discourse", "vocative",
       "mark", "cop", "nsubj", "nsubj:pass", "csubj", "aux", "aux:pass", "expl", "reparandum",
       "list", "orphan", "dislocated"}
CUT_NOMINAL = {"conj", "cc"}
OBJECT_ROLES = ("ARG1", "ARG2", "ARG3", "ARG4", "ARGM-LOC", "ARGM-DIR")
URLISH = ("http", "www", "://", ".com", " / /", "\\")
FIRST_PERSON = {"i", "me", "my", "myself", "mine"}
SPEECH = {"say", "tell", "report", "claim", "state", "announce", "believe", "think", "argue", "add", "note",
          "warn", "write", "explain", "suggest", "insist", "deny", "confirm", "expect", "allege", "estimate",
          "predict", "assert", "admit", "acknowledge", "reveal", "disclose", "testify", "mention", "promise",
          "according", "indicate", "conclude", "speculate", "fear", "hope", "doubt", "assume", "recall"}
NEGATORS = {"not", "n't", "never", "no"}


class _Window:
    def __init__(self, a: dict):
        self.tokens = a["tokens"]
        self.words = a["words"]
        self.W = len(self.words)
        self.head = [w["head"] - 1 for w in self.words]          # -1: root
        self.rel = [w["deprel"] for w in self.words]
        self.pos = [w["pos"] for w in self.words]
        self.lemma = [w["lemma"] for w in self.words]
        self.kids: list[list[int]] = [[] for _ in range(self.W)]
        for i, h in enumerate(self.head):
            if 0 <= h < self.W:
                self.kids[h].append(i)
        self.clusters = [[(int(s), int(e) + 1) for s, e in c] for c in a.get("coref") or []]
        self.speaker = [None] * self.W
        for d in a.get("sentences") or []:
            for i in range(d["start"], min(d["end"], self.W)):
                self.speaker[i] = d.get("speaker")

    def text(self, a: int, b: int) -> str:
        return " ".join(self.tokens[a:b])

    def span_head(self, a: int, b: int) -> int | None:
        for i in range(a, min(b, self.W)):
            if not (a <= self.head[i] < b):
                return i
        return None

    def noun_phrase(self, a: int, b: int, run: bool = False) -> tuple[int, int] | None:
        """The span ``[a, b)`` cut to its head's noun phrase, inside the span."""
        b = min(b, self.W)
        while a < b and (self.pos[a] in ("ADP", "PUNCT", "CCONJ") or self.rel[a] in ("case", "cc:preconj", "mark")):
            a += 1
        h = self.span_head(a, b)
        if h is None:
            return None
        keep, todo = set(), [h]
        while todo:
            i = todo.pop()
            keep.add(i)
            for k in self.kids[i]:
                if not (a <= k < b):
                    continue
                r = self.rel[k]
                if r in CUT or (r in CUT_NOMINAL and self.pos[i] in NOMINAL):
                    continue
                if r in ("nmod", "obl") and i != h:    # one level of modifier: "a variety of alterations"
                    continue
                if r == "punct" and not (a <= k < b):
                    continue
                todo.append(k)
        # a role's span: from the first to the last kept word. A copula's or an
        # apposition's (a subtree, less bounded): the contiguous run around the
        # head, a bracketed group between kept words included.
        s, e = min(keep), max(keep)
        if run:
            s, e = self._run(h, keep, a, b)
        while s <= e and self.pos[s] == "PUNCT":
            s += 1
        while e >= s and (self.pos[e] == "PUNCT" or self.rel[e] == "case"):   # "China 's"
            e -= 1
        return (s, e + 1) if s <= e else None

    def _run(self, h, keep, a, b):
        s = e = h
        while True:
            if e + 1 < b and e + 1 in keep:
                e += 1
            elif e + 1 < b and self.tokens[e + 1] == "(":
                c = next((j for j in range(e + 2, b) if self.tokens[j] == ")"), None)
                if c is None or c + 1 not in keep:
                    break
                e = c + 1
            else:
                break
        while True:
            if s - 1 >= a and s - 1 in keep:
                s -= 1
            elif s - 1 >= a and self.tokens[s - 1] == ")":
                c = next((j for j in range(s - 2, a - 1, -1) if self.tokens[j] == "("), None)
                if c is None or c - 1 not in keep:
                    break
                s = c - 1
            else:
                break
        return s, e


    def cluster_names(self, a: int, b: int) -> list[str]:
        h = self.span_head(a, b)
        out = []
        for c in self.clusters:
            if any(s <= h < e for s, e in c):
                for s, e in c:
                    hh = self.span_head(s, e)
                    if hh is not None and self.pos[hh] != "PRON":
                        out.append(self.text(s, e))
        return out

    def argument(self, a: int, b: int, run: bool = False) -> dict | None:
        np_ = self.noun_phrase(a, b, run)
        if np_ is None:
            return None
        s, e = np_
        h = self.span_head(s, e)
        names = self.cluster_names(s, e)
        if self.pos[h] == "PRON" and self.tokens[h].lower() in FIRST_PERSON and self.speaker[h] and e - s == 1:
            text = self.speaker[h]                       # conversations: "I" is the speaker
            names = [text]
        elif self.pos[h] == "PRON":
            if not names:
                return None
            text = names[0]
        elif self.pos[h] in NOT_THING:
            return None
        else:
            text = self.text(s, e)
        if sum(ch.isalpha() for ch in text) < 2 or any(u in text.lower() for u in URLISH):   # "s", "$ 3", links
            return None
        low = []
        for n in [text] + names:
            if n.lower() not in low:
                low.append(n.lower())
        return {"text": text, "span": [s, e], "head": h, "names": low}


def _overlap(x: dict, y: dict) -> bool:
    (a, b), (c, d) = x["span"], y["span"]
    return a < d and c < b or x["text"].lower() == y["text"].lower()


def _fact(w: _Window, s: dict, pred: list[int], o: dict, source: str, neg: bool = False,
          modal: str | None = None, time: str | None = None) -> dict:
    words = [w.tokens[i] for i in pred]
    lem = [w.lemma[i] if i < w.W and w.pos[i] in ("VERB", "AUX", "NOUN") else w.tokens[i].lower() for i in pred]
    return {"subject": s, "object": o,
            "predicate": {"text": " ".join(words), "lemma": " ".join(lem).lower(), "words": pred},
            "polarity": "negative" if neg else "positive", "modality": modal, "time": time,
            "holder": None, "source": source}


def decode(a: dict) -> list[dict]:
    """Open facts of one analysed window (``Kniv.analyse`` with ``linguistic``)."""
    if not a.get("words"):
        return []
    w = _Window(a)
    out: list[dict] = []
    seen = set()
    # speech and belief frames: who asserts what their ARG1 (or ARG2) holds
    said = []
    for fr in a.get("srl") or []:
        q = fr["predicate"]
        if q < w.W and w.lemma[q].lower() in SPEECH:
            who = next((x for r, s_, e_ in fr["args"] if r == "ARG0" and (x := w.argument(s_, e_))), None)
            if who:
                for r, s_, e_ in fr["args"]:
                    if r in ("ARG1", "ARG2"):
                        said.append((s_, e_, who["text"]))

    def holder(i: int) -> str | None:
        inner = [x for x in said if x[0] <= i < x[1]]
        if inner:
            return min(inner, key=lambda x: x[1] - x[0])[2]
        return w.speaker[i] if 0 <= i < w.W else None

    def add(f):
        anchor = f["predicate"]["words"][0] if f["predicate"]["words"] else f["subject"]["head"]
        f["holder"] = holder(anchor)
        key = (f["subject"]["text"].lower(), f["predicate"]["text"].lower(), f["object"]["text"].lower())
        if key not in seen and not _overlap(f["subject"], f["object"]):
            seen.add(key)
            out.append(f)

    for fr in a.get("srl") or []:
        p = fr["predicate"]
        if p >= w.W or w.pos[p] == "AUX" or w.rel[p] == "cop":     # copulas: below
            continue
        args = [(r, s, e) for r, s, e in fr["args"]]
        neg = any(r == "ARGM-NEG" for r, _, _ in args)
        modal = next((w.text(s, e) for r, s, e in args if r == "ARGM-MOD"), None)
        time = next((w.text(s, e) for r, s, e in args if r == "ARGM-TMP"), None)
        prt = [k for k in w.kids[p] if w.rel[k] == "compound:prt"]

        def first(role):
            for r, s, e in args:
                if r == role:
                    x = w.argument(s, e)
                    if x is not None:
                        return x
            return None
        a0, a1 = first("ARG0"), first("ARG1")
        for r, s, e in args:
            if r not in OBJECT_ROLES:
                continue
            subj = a0 if a0 is not None else (a1 if r != "ARG1" else None)
            if subj is None:
                continue
            o = w.argument(s, e)
            if o is None or any(x["span"][0] <= p < x["span"][1] for x in (subj, o)):
                continue
            prep = [i for i in range(s, o["span"][0]) if w.pos[i] == "ADP" or w.rel[i] == "case"][:1]
            add(_fact(w, subj, [p] + prt + prep, o, f"srl:{r}", neg, modal, time))

    for i in range(w.W):
        if w.rel[i] == "cop" and 0 <= w.head[i] < w.W:
            n = w.head[i]
            if w.pos[n] not in ("NOUN", "PROPN"):
                continue
            sub = [k for k in w.kids[n] if w.rel[k] in ("nsubj", "nsubj:pass")]
            if not sub:
                continue
            lo, hi = _subtree(w, sub[0])
            s = w.argument(lo, hi, run=True)
            lo, hi = _subtree(w, n, skip={sub[0]})
            o = w.argument(max(lo, i + 1), hi, run=True)
            if s is None or o is None:
                continue
            neg = any(w.tokens[k].lower() in NEGATORS for k in w.kids[n] if w.rel[k] == "advmod")
            add({**(f := _fact(w, s, [i], o, "cop", neg)),
                 "predicate": {**f["predicate"], "phrase": _phrase(w, w.tokens[i], o)}})
        if w.rel[i] == "appos" and 0 <= w.head[i] < w.W:
            n = w.head[i]
            lo, hi = _subtree(w, i)
            n_lo, _ = _subtree(w, n, skip={i})
            s = w.argument(n_lo, lo, run=True) if n < lo else None      # the noun before its apposition

            o = w.argument(lo, hi, run=True)
            if s is None or o is None or o["span"][0] - s["span"][1] > 3:   # an apposition is next to its noun
                continue
            add({**_fact(w, s, [], o, "appos"),
                 "predicate": {"text": "is", "lemma": "be", "words": [], "phrase": _phrase(w, "is", o)}})
    return out


def _phrase(w: _Window, cop: str, o: dict) -> str:
    """"is the chief executive of": the copula, the object up to its head and
    the head's next preposition (for the nearest known relation)."""
    a, h = o["span"][0], o["head"]
    words = [cop] + w.tokens[a:h + 1]
    if h + 1 < o["span"][1] and w.pos[h + 1] == "ADP":
        words.append(w.tokens[h + 1])
    return " ".join(words)


def _subtree(w: _Window, h: int, skip: set = frozenset()) -> tuple[int, int]:
    seen, todo = {h}, [h]
    while todo:
        i = todo.pop()
        for k in w.kids[i]:
            if k not in skip and k not in seen:
                seen.add(k)
                todo.append(k)
    lo = hi = h
    while lo - 1 in seen:
        lo -= 1
    while hi + 1 in seen:
        hi += 1
    return lo, hi + 1
