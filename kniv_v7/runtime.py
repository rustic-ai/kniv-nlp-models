"""kniv v7 inference: text (or conversation turns) to a full analysis.

    from kniv_v7 import Kniv
    k = Kniv("path/to/kniv-v7-ultrabert-380m-en")            # ONNX, fp32
    out = k.analyse("Ada Lovelace was born in London in 1815.",
                    relations=[{"name": "place of birth", "description": "where the head was born"}])

Per window (text is split into sentences and packed into windows of up to
512 words): tokens; per word POS, morphology, lemma, dependency head and
label; named entities; semantic-role frames; per sentence an utterance class
and sentiment; coreference clusters; entities (DocRED's six coarse types, or
the caller's entity types) and facts against the caller's relations, each
with polarity, modality, holder and time.

The model runs as eight graphs (``onnx/``); this module is the host code
around them -- windows, subword maps, marker text, and every decoding step
-- and needs only numpy, onnxruntime and tokenizers. ``backend`` swaps the
graphs for another implementation of the same interface (the PyTorch
reference, ``v6.export.torch_backend``).
"""
from __future__ import annotations

import bisect
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .entities import bio_spans, build_entities
from .windows import Document, build_windows, tokenize

NEG = -1e4
ENT_TYPES = sorted({"LOC", "MISC", "NUM", "ORG", "PER", "TIME"})
ENT_TAGS = ["O"] + [f"{p}-{t}" for t in ENT_TYPES for p in ("B", "I")]
POLARITIES = ["positive", "negative"]
MODALITIES = ["asserted", "hypothetical", "planned", "desired", "questioned", "reported"]
MARKER = "*"
MARKER_MAX_SUBWORDS = 1024
LABEL_MAX_SUBWORDS = 48
SPAN_MAX_WIDTH = 12


# ── backends ─────────────────────────────────────────────────────────
class OnnxBackend:
    """The eight graphs in onnxruntime. ``int8``: the quantised encoder
    graphs (``plain``, ``label``, ``marker``) where present."""

    def __init__(self, onnx_dir: Path, int8: bool = False, providers=None):
        import onnxruntime as ort
        self.dir, self.int8 = Path(onnx_dir), int8
        self.providers = providers or ["CPUExecutionProvider"]
        self._ort, self._sess = ort, {}

    def _session(self, name: str):
        if name not in self._sess:
            path = self.dir / f"{name}-int8.onnx"
            if not (self.int8 and path.exists()):
                path = self.dir / f"{name}.onnx"
            self._sess[name] = self._ort.InferenceSession(str(path), providers=self.providers)
        return self._sess[name]

    def run(self, name: str, **inputs) -> list[np.ndarray]:
        s = self._session(name)
        want = {i.name for i in s.get_inputs()}
        return s.run(None, {k: v for k, v in inputs.items() if k in want})


# ── text to windows and features ─────────────────────────────────────
def windows(text: str | None = None, turns=None) -> list[dict]:
    """Paragraphs (split at blank lines) or ``(speaker, text)`` turns into
    windows of up to 512 words; sentences never cross a window."""
    if turns is not None:
        doc = Document("doc", "conversation", "infer", units=[t for _, t in turns],
                       speakers=[s for s, _ in turns])
    else:
        doc = Document("doc", "text", "infer",
                       units=[p for p in (text or "").split("\n\n") if p.strip()])
    return build_windows(doc, tokenize, max_tokens=512, min_tokens=1)


class Tok:
    """The encoder's tokenizer (``tokenizer.json``), as the training code
    called it."""

    def __init__(self, path: Path, pad_id: int, sep_id: int):
        from tokenizers import Tokenizer
        self.t = Tokenizer.from_file(str(path))
        self.t.no_padding()
        self.t.no_truncation()
        self.pad, self.sep = pad_id, sep_id

    def encode(self, text: str):
        e = self.t.encode(text, add_special_tokens=True)
        return e.ids, e.offsets

    def labels(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Label texts, each cut to 48 subwords (keeping the final special
        token), padded: ``(ids, mask) [K,L]``."""
        rows = []
        for t in texts:
            ids, _ = self.encode(t)
            if len(ids) > LABEL_MAX_SUBWORDS:
                ids = ids[:LABEL_MAX_SUBWORDS - 1] + ids[-1:]
            rows.append(ids)
        L = max(len(r) for r in rows)
        ids = np.full((len(rows), L), self.pad, dtype=np.int64)
        mask = np.zeros((len(rows), L), dtype=np.int64)
        for i, r in enumerate(rows):
            ids[i, :len(r)], mask[i, :len(r)] = r, 1
        return ids, mask


def align(words: list[str], tok: Tok, max_subwords: int):
    """``(ids, first_subword, limit)`` as ``v6.train.data.align``."""
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    ids, offs = tok.encode(" ".join(words))
    first = [-1] * len(words)
    for ti, (s, e) in enumerate(offs):
        if e <= s:
            continue
        wi = bisect.bisect_right(starts, e - 1) - 1
        if first[wi] < 0:
            first[wi] = ti
    limit = len(words)
    if len(ids) > max_subwords:
        cut = max_subwords - 1
        ids = list(ids[:cut]) + [tok.sep]
        for i, f in enumerate(first):
            if f >= cut:
                limit = i
                break
    return list(ids), first[:limit], limit


def word_ends(first: np.ndarray, n: int) -> np.ndarray:
    end = np.ones(len(first), dtype=np.int64)
    nxt = n - 1
    for i in range(len(first) - 1, -1, -1):
        f = int(first[i])
        if 0 <= f < nxt:
            end[i], nxt = nxt, f
        else:
            end[i] = min(max(f, 0) + 1, n)
    return end


def marker_input(tokens, sentence_spans, entities, tok: Tok, turns=None, participants=None):
    """``v6.train.data.marker_features``: the window with ``*`` around every
    mention and ``* role * :`` opening each turn; each mention's start-marker
    position (-1 past the limit), speakers' turns first."""
    starts, ends = set(), set()
    for ms in entities:
        for a, b in ms:
            starts.add(int(a)); ends.add(int(b) - 1)
    words, marker_of = [], {}
    turn_at = dict(turns or [])
    spk_word: list[list[int]] = [[] for _ in (participants or [])]
    for i, w in enumerate(tokens):
        if i in turn_at:
            spk_word[turn_at[i]].append(len(words))
            words += [MARKER, participants[turn_at[i]], MARKER, ":"]
        if i in starts:
            marker_of[i] = len(words)
            words.append(MARKER)
        words.append(w)
        if i in ends:
            words.append(MARKER)
    ids, first, limit = align(words, tok, MARKER_MAX_SUBWORDS)
    pos = [[first[marker_of[int(a)]] if marker_of[int(a)] < limit else -1 for a, _ in ms]
           for ms in entities]
    spk = [[first[w] if w < limit else -1 for w in ws] for ws in spk_word]
    return ids, spk + pos


def speaker_turns(sentence_spans, speakers, participants):
    out, last = [], None
    for j, (s, _) in enumerate(sentence_spans):
        who = speakers[j] if speakers and j < len(speakers) else None
        if who and who != last and who in participants:
            out.append((s, participants.index(who)))
        last = who
    return out


# ── decoding ─────────────────────────────────────────────────────────
def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return (x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))).astype(np.float32)


def bio_constraints(tags: list[str]):
    C = len(tags)
    allowed = np.ones((C, C), dtype=bool)
    start = np.ones(C, dtype=bool)
    for j, t in enumerate(tags):
        if t.startswith("I-"):
            start[j] = False
            for i, s in enumerate(tags):
                allowed[i, j] = s in (f"B-{t[2:]}", t)
    return allowed, start


def viterbi(logp, allowed, start, keep=None) -> np.ndarray:
    NEGV = -1e9
    T, C = logp.shape
    lp = logp.copy()
    if keep is not None:
        lp[~keep] = NEGV
    trans = np.where(allowed, 0.0, NEGV)
    score = np.where(start, lp[0], NEGV)
    back = np.zeros((T, C), dtype=np.int32)
    for t in range(1, T):
        cand = score[:, None] + trans
        back[t] = cand.argmax(0)
        score = cand.max(0) + lp[t]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def _cycle(heads):
    state = [0] * len(heads)
    for start in range(1, len(heads)):
        path, j = [], start
        while j > 0 and state[j] == 0:
            state[j] = 1
            path.append(j)
            j = int(heads[j])
        if j > 0 and state[j] == 1:
            return path[path.index(j):]
        for k in path:
            state[k] = 2
    return None


def _chu_liu_edmonds(S):
    N = S.shape[0]
    heads = S.argmax(1)
    heads[0] = -1
    cyc = _cycle(heads)
    if cyc is None:
        return heads
    inC = set(cyc)
    rest = [i for i in range(N) if i not in inC]
    c = len(rest)
    S2 = np.full((c + 1, c + 1), -np.inf)
    S2[:c, :c] = S[np.ix_(rest, rest)]
    held = S[cyc, heads[cyc]]
    enter = S[np.ix_(cyc, rest)] - held[:, None]
    S2[c, :c] = enter.max(0)
    leave = S[np.ix_(rest, cyc)]
    S2[:c, c] = leave.max(1)
    np.fill_diagonal(S2, -np.inf)
    sub = _chu_liu_edmonds(S2)
    out = heads.copy()
    for k, d in enumerate(rest):
        if d == 0:
            continue
        out[d] = cyc[leave[k].argmax()] if sub[k] == c else rest[sub[k]]
    h = sub[c]
    out[cyc[enter[:, h].argmax()]] = rest[h]
    return out


def tree_heads(S):
    heads = _chu_liu_edmonds(S)
    roots = [i for i in range(1, len(heads)) if heads[i] == 0]
    if len(roots) <= 1:
        return heads
    best, best_score = heads, -np.inf
    for r in roots:
        T = S.copy()
        T[1:, 0] = -np.inf
        T[r, 0] = S[r, 0]
        h = _chu_liu_edmonds(T)
        score = T[np.arange(1, len(h)), h[1:]].sum()
        if score > best_score:
            best, best_score = h, score
    return best


def decode_trees(arc: np.ndarray, sent: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """One tree per sentence over ``arc [W, W+1]`` (column 0 the root):
    head columns, as ``v6.train.model.decode_trees``."""
    out = arc.argmax(-1)
    lp = log_softmax(arc)
    for s in np.unique(sent[wm & (sent >= 0)]):
        w = np.nonzero((sent == s) & wm)[0]
        n = len(w)
        S = np.full((n + 1, n + 1), -np.inf)
        S[1:, 0] = lp[w, 0]
        S[1:, 1:] = lp[np.ix_(w, w + 1)]
        np.fill_diagonal(S, -np.inf)
        h = tree_heads(S)
        out[w] = np.where(h[1:] == 0, 0, w[np.maximum(h[1:] - 1, 0)] + 1)
    return out


def apply_script(word: str, script: str) -> str:
    case, strip, append = script.split("|", 2)
    w = word.lower() if case else word
    return w[:len(w) - int(strip)] + append


def link_clusters(antecedent: dict[int, int]) -> list[list[int]]:
    parent: dict[int, int] = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for w, a in antecedent.items():
        parent[find(w)] = find(a)
    groups: dict[int, list[int]] = defaultdict(list)
    for w in parent:
        groups[find(w)].append(w)
    return sorted(sorted(g) for g in groups.values() if len(g) > 1)


def label_text(rel: dict) -> str:
    return f"relation: {rel['name']}. {rel['description']}" if rel.get("description") \
        else f"relation: {rel['name']}"


def type_text(t: dict) -> str:
    return f"entity type: {t['name']}. {t['description']}" if t.get("description") \
        else f"entity type: {t['name']}"


def span_candidates(sent: np.ndarray, word_ok: np.ndarray) -> list[tuple[int, int]]:
    """Every span of up to 12 words inside one sentence (``TypeCollator``)."""
    out, n = [], len(sent)
    for s in range(n):
        if not word_ok[s] or sent[s] < 0:
            continue
        for e in range(s + 1, min(s + SPAN_MAX_WIDTH, n) + 1):
            if sent[e - 1] != sent[s]:
                break
            if word_ok[e - 1]:
                out.append((s, e))
    return out


def decode_spans(logits: np.ndarray, spans, offset: float = 0.0) -> list[tuple[int, int, int]]:
    """Greedy flat decoding (``kdata.decode_spans``), one window."""
    if not len(logits):
        return []
    margin = logits[:, 1:] - logits[:, :1] - offset
    best, j = margin.max(1), margin.argmax(1)
    taken = np.zeros(max(e for _, e in spans) + 1, dtype=bool)
    out = []
    for i in np.argsort(-best, kind="stable"):
        if best[i] <= 0:
            break
        s, e = spans[i]
        if taken[s:e].any():
            continue
        taken[s:e] = True
        out.append((s, e, int(j[i])))
    return sorted(out)


# ── the pipeline ─────────────────────────────────────────────────────
class Kniv:
    """A released kniv v7 model directory (``onnx/``, ``tokenizer.json``,
    ``calibration.json``, ``label_bank.json``), or any ``backend`` with the
    same graphs."""

    def __init__(self, path, backend=None, int8: bool = False, providers=None):
        self.path = Path(path)
        self.meta = json.loads((self.path / "onnx" / "onnx_meta.json").read_text())
        self.labels = self.meta["labels"]
        self.tok = Tok(self.path / "tokenizer.json", self.meta["pad_token_id"],
                       self.meta["sep_token_id"])
        self.backend = backend or OnnxBackend(self.path / "onnx", int8, providers)
        cal_path = self.path / "calibration.json"
        self.cal = json.loads(cal_path.read_text()) if cal_path.exists() else {}
        bank_path = self.path / "label_bank.json"
        rd = []
        if bank_path.exists():
            rd = [r["name"] for r in json.loads(bank_path.read_text())["relations"]
                  if r.get("source") == "redocred"]
        # Re-DocRED's relations: their own dev-tuned offset, else the global
        # one; every other relation: unseen_offset (the two never stack).
        self.rel_offsets = ({n: float(self.cal["rel_offset"]) for n in rd}
                            if "rel_offset" in self.cal else {})
        self.rel_offsets.update(self.cal.get("rel_offsets") or {})
        self.unseen_offset = float(self.cal.get("unseen_offset", 0.0))
        self.zs_offset = float(self.cal.get("offset", 0.0))
        self.top = int(self.cal.get("rel_top", 4))
        self.ent_o_bias = float(self.cal.get("ent_o_bias", 0.0))
        self.coref_bias = float(self.cal.get("coref_link_bias", 0.0))
        self.entity_coref = self.cal.get("entity_coref", "all")
        self.span_offset = float(self.cal.get("span_offset", 0.0))
        self._label_cache: dict[tuple, np.ndarray] = {}

    def run(self, name, **x):
        return self.backend.run(name, **x)

    # ── public ──────────────────────────────────────────────────────
    def analyse(self, text: str | None = None, turns=None, relations=(), entity_types=None,
                linguistic: bool = True, top: int | None = None) -> list[dict]:
        """``text`` (paragraphs separated by blank lines) or ``turns``
        (``(speaker, text)`` pairs); ``relations`` and ``entity_types`` as
        ``{"name", "description"?}``. One result per window."""
        return [self.window(w, list(relations), entity_types, linguistic, top or self.top)
                for w in windows(text, turns)]

    def extract(self, text=None, turns=None, relations=(), entity_types=None, top=None):
        """Entities and facts only (no linguistic layer in the output)."""
        return self.analyse(text, turns, relations, entity_types, linguistic=False, top=top)

    # ── one window ──────────────────────────────────────────────────
    def window(self, w: dict, relations, entity_types, linguistic: bool, top: int) -> dict:
        tokens, sp = w["tokens"], [tuple(x) for x in w["sentence_spans"]]
        ids, first, W = align(tokens, self.tok, self.meta["max_subwords"])
        sent = np.full(W, -1, dtype=np.int64)
        for j, (s, e) in enumerate(sp):
            sent[s:min(e, W)] = j
        first = np.asarray(first, dtype=np.int64)
        wm = first >= 0
        x = {"input_ids": np.asarray([ids], dtype=np.int64),
             "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
             "first_subword": np.maximum(first, 0)[None], "word_end": word_ends(first, len(ids))[None],
             "word_mask": wm[None], "sent_of_word": sent[None],
             "sent_ids": np.arange(len(sp), dtype=np.int64)}
        o = dict(zip(self.meta["graphs"]["plain"]["outputs"], self.run("plain", **x)))
        out = {"tokens": tokens, "analysed_words": W,
               "sentences": [{"start": s, "end": e} for s, e in sp]}
        if w.get("sentence_speakers") and any(w["sentence_speakers"]):
            for d, who in zip(out["sentences"], w["sentence_speakers"]):
                d["speaker"] = who
        coref = self.coref(o, sent, wm, W)
        if linguistic:
            out.update(self.linguistic(o, tokens, sp, sent, wm, W))
            out["coref"] = [[list(m) for m in c] for c in coref]
        ents = self.entities(o, tokens, sp, sent, wm, W, coref, entity_types)
        out["entities"], out["facts"] = self.relations(w, tokens, sp, ents, relations, top)
        return out

    def linguistic(self, o, tokens, sp, sent, wm, W) -> dict:
        L = self.labels
        pos = o["pos"][0].argmax(-1)
        morph = o["morph"][0].argmax(-1)
        lemma = o["lemma"][0].argmax(-1)
        head = decode_trees(o["arc"][0], sent, wm)
        rel = self.run("dep_rel", rel_dep=o["rel_dep"], rel_head=o["rel_head"],
                       head=head[None].astype(np.int64))[0][0].argmax(-1)
        words = [{"pos": L["pos"][pos[i]], "morph": L["morph"][morph[i]],
                  "lemma": apply_script(tokens[i], L["lemma"][lemma[i]]),
                  "head": int(head[i]), "deprel": L["dep"][rel[i]]} for i in range(W)]
        ner_tags = L["ner"]
        al, st = bio_constraints(ner_tags)
        ner = [[ner_tags[k] for k in viterbi(log_softmax(o["ner"][0, s:min(e, W)]), al, st)]
               for s, e in sp if s < W]
        ner_spans = [(t, s + a, s + b) for (s, _), tags in zip(sp, ner) for t, a, b in bio_spans(tags)]
        cls = 1 / (1 + np.exp(-o["cls"][0].astype(np.float64)))
        senti = o["sentiment"][0].argmax(-1)
        sentences = [{"cls": [L["cls"][k] for k in np.nonzero(cls[j] > 0.5)[0]],
                      "sentiment": L["sentiment"][senti[j]], "whole": bool(e <= W)}
                     for j, (s, e) in enumerate(sp)]
        return {"words": words, "ner": [list(x) for x in ner_spans],
                "srl": self.srl(o, sent, wm, W), "sentence_layers": sentences}

    def srl(self, o, sent, wm, W) -> list[dict]:
        tags = self.labels["srl"]
        preds = [int(p) for p in np.nonzero((o["srl_pred"][0].argmax(-1) == 1) & wm)[0]]
        if not preds:
            return []
        s = o["srl_words"][0]
        same = [(sent == sent[p]) & wm & (sent[p] >= 0) for p in preds]
        start = [int(np.argmax(m)) for m in same]
        end = [len(m) - 1 - int(np.argmax(m[::-1])) for m in same]
        n = [e - a + 1 for a, e in zip(start, end)]
        L = max(n)
        words = np.zeros((len(preds), L, s.shape[-1]), dtype=np.float32)
        mask = np.zeros((len(preds), L), dtype=bool)
        for f, (a, k) in enumerate(zip(start, n)):
            words[f, :k] = s[a:a + k]
            mask[f, :k] = True
        lg = self.run("srl_roles", words=words, pred=np.asarray(
            [p - a for p, a in zip(preds, start)], dtype=np.int64), mask=mask)[0]
        al, st = bio_constraints(tags)
        v = tags.index("V")
        frames = []
        for f, p in enumerate(preds):
            role = np.full((W, len(tags)), NEG, dtype=np.float32)
            idx = start[f] + np.arange(n[f])
            role[idx] = lg[f, :n[f]]
            role[~same[f]] = NEG
            lp = log_softmax(role)
            keep = np.ones_like(lp, dtype=bool)
            keep[:, v] = False
            keep[p, :] = False
            keep[p, v] = True
            path = [tags[k] for k in viterbi(lp, al, st, keep)]
            frames.append({"predicate": p, "args": [list(x) for x in bio_spans(path) if x[0] != "V"]})
        return frames

    def coref(self, o, sent, wm, W) -> list[list[tuple[int, int]]]:
        """Clusters of inclusive word spans, as ``gold_eval.run_phase2``."""
        sc = o["coref_scores"][0].astype(np.float32).copy()
        sc[:, 0] += self.coref_bias
        best = sc.argmax(-1)
        cand = o["coref_cand"][0]
        ante = {w: int(cand[w, best[w] - 1]) for w in range(W) if best[w] > 0}
        clusters = link_clusters(ante)
        heads = [w for c in clusters for w in c]
        if not heads:
            return []
        lg, pos = self.run("coref_spans", coref_words=o["coref_words"],
                           heads=np.asarray(heads, dtype=np.int64), sent_of_word=sent[None])
        h = np.asarray(heads)[:, None]
        st = np.where(pos > h, -1e9, lg[..., 0]).argmax(1)
        en = np.where(pos < h, -1e9, lg[..., 1]).argmax(1)
        span = {w: (int(pos[i, st[i]]), int(pos[i, en[i]])) for i, w in enumerate(heads)}
        return [[span[w] for w in c] for c in clusters]

    def label_emb(self, texts: list[str]) -> np.ndarray:
        key = tuple(texts)
        if key not in self._label_cache:
            if len(self._label_cache) >= 256:
                self._label_cache.clear()
            ids, mask = self.tok.labels(texts)
            self._label_cache[key] = self.run("label", label_ids=ids, label_mask=mask)[0]
        return self._label_cache[key]

    def entities(self, o, tokens, sp, sent, wm, W, coref, entity_types) -> list[dict]:
        """Typed spans (the caller's types through the span head, else the
        DocRED entity head), grouped by ``build_entities``."""
        span_type: dict[tuple[int, int], str] = {}
        if entity_types:
            names = [t["name"] for t in entity_types]
            cands = span_candidates(sent, wm)
            if cands:
                temb = self.label_emb([type_text({"name": n, "description": ""}) for n in names])
                lg = self.run("span", span_words=o["span_words"],
                              start=np.asarray([s for s, _ in cands], dtype=np.int64),
                              end=np.asarray([e for _, e in cands], dtype=np.int64), type_emb=temb)[0]
                pred = decode_spans(lg, cands, self.span_offset)
            else:
                pred = []
            ner = []
            for s, e in sp:
                tags = ["O"] * (e - s)
                for a, b, t in pred:
                    if s <= a and b <= e:
                        span_type[(a, b)] = names[t]
                        tags[a - s] = "B-MISC"
                        for i in range(a + 1, b):
                            tags[i - s] = "I-MISC"
                ner.append(tags if s < W else None)
        else:
            al, st = bio_constraints(ENT_TAGS)
            lp = log_softmax(o["ent"][0])
            lp[:, 0] += self.ent_o_bias
            ner = [[ENT_TAGS[x] for x in viterbi(lp[s:min(e, W)], al, st)] if s < W else None
                   for s, e in sp]
        _, vertex = build_entities(tokens, [list(x) for x in sp], ner,
                                   [[list(m) for m in c] for c in coref], coref=self.entity_coref)
        ents = []
        for v in vertex or []:
            ms = sorted({(sp[m["sent_id"]][0] + m["pos"][0], sp[m["sent_id"]][0] + m["pos"][1])
                         for m in v})
            ty = [span_type[m] for m in ms if m in span_type]
            ents.append({"type": max(set(ty), key=ty.count) if ty else v[0].get("type"),
                         "mentions": ms})
        return ents

    def relations(self, w, tokens, sp, ents, relations, top):
        """Entities (speakers first in a conversation) and facts with their
        attributes, as ``v6.infer.Extractor`` plus the attribute head."""
        _, first, W = align(tokens, self.tok, self.meta["max_subwords"])
        ok = [f >= 0 for f in first]
        speakers = w.get("sentence_speakers") or []
        parts = list(dict.fromkeys(s for s in speakers if s))
        turns = [(s, p) for s, p in speaker_turns(sp, speakers, parts) if s < W and ok[s]]
        nodes, spans = [], []
        for p, who in enumerate(parts):
            ms = [(s, s + 1) for s, q in turns if q == p]
            if ms:
                nodes.append({"type": "speaker", "text": who, "speaker": True})
                spans.append(ms)
        n_part = len(nodes)
        for e in ents:
            ms = [(a, b) for a, b in e["mentions"] if b <= W and ok[a]]
            if ms:
                nodes.append({"type": e["type"], "text": " ".join(tokens[ms[0][0]:ms[0][1]]),
                              "mentions": [" ".join(tokens[a:b]) for a, b in ms],
                              "spans": [list(m) for m in ms]})
                spans.append(ms)
        if len(spans) < 2 or not relations:
            return nodes, []
        part_turns = [(s, p) for s, p in turns if p < n_part]
        ids, men_pos = marker_input(tokens, sp, spans[n_part:], self.tok, turns=part_turns,
                                    participants=parts[:n_part])
        men_ent = [k for k, ms in enumerate(spans) for _ in ms]
        pos = [x for k in range(len(spans)) for x in men_pos[k]]
        E = len(spans)
        pairs = [(h, t) for h in range(E) for t in range(E) if h != t]
        temb = self.label_emb([label_text(r) for r in relations])
        logits, pair, labels, entv = self.run(
            "marker", rm_input_ids=np.asarray([ids], dtype=np.int64),
            rm_attention_mask=np.ones((1, len(ids)), dtype=np.int64),
            men_pos=np.asarray(pos, dtype=np.int64), men_ent=np.asarray(men_ent, dtype=np.int64),
            ent_ids=np.arange(E, dtype=np.int64),
            pair_h=np.asarray([h for h, _ in pairs], dtype=np.int64),
            pair_t=np.asarray([t for _, t in pairs], dtype=np.int64), label_emb=temb)
        off = np.array([self.rel_offsets.get(r["name"], -self.unseen_offset) for r in relations])
        facts, keep = [], []
        for pi, ((h, t), row) in enumerate(zip(pairs, logits)):
            m = row[1:] - (row[0] + self.zs_offset) + off
            for k in np.argsort(-m, kind="stable")[:top]:
                if m[k] > 0:
                    facts.append({"head": h, "relation": relations[k]["name"], "tail": t,
                                  "margin": round(float(m[k]), 3)})
                    keep.append((pi, int(k)))
        if facts:
            pol, mod, hold, tim = self.run(
                "attr", pair=pair[[p for p, _ in keep]], labels=labels[[k for _, k in keep]], ent=entv)
            for f, a, b, c, d in zip(facts, pol, mod, hold, tim):
                f["polarity"] = POLARITIES[int(a.argmax())]
                f["modality"] = MODALITIES[int(b.argmax())]
                f["holder"] = None if c.argmax() == 0 else int(c.argmax()) - 1
                f["time"] = None if d.argmax() == 0 else int(d.argmax()) - 1
        return nodes, facts
