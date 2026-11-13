"""Knowledge extraction from raw text against a caller's schema.

    from v6.infer import Extractor
    ex = Extractor()                                  # the v6.2 candidate
    out = ex.extract(turns=[("user", "I moved to Lisbon last year."),
                            ("assistant", "Nice! Do you like it?")],
                     relations=[{"name": "lives in", "description": "the head resides in the tail place"}],
                     entity_types=[{"name": "location"}, {"name": "person"}])

Text, or speaker turns, are split into sentences and packed into windows of
up to 512 words (``windows.build_windows``, the corpus's own). Per window:

1. **Entities.** The span head finds mentions of the requested entity types
   (or, without types, the frozen DocRED entity head finds people,
   organisations, places, times, numbers and other names); coreference
   clusters merge mentions of one thing (``entities.build_entities``). In a
   conversation each speaker is an entity too, mentioned by the turn tags.
2. **Relations.** Every ordered entity pair is scored against the requested
   relations through the hybrid: a relation the bank knows as one of
   Re-DocRED's 96 is answered by the frozen ATLOP path, every other by the
   label path, read from its name and description, with the run's calibrated
   threshold offset. A relation holds where its margin over the threshold is
   positive; at most ``top`` per pair.

Returns, per window, entities (type, mention texts and word spans) and facts
(head, relation, tail, margin). Facts are per window; merging across windows
(and across documents) is the consumer's.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

UNSEEN_ID = 10 ** 6          # columns the bank does not know: never routed to the frozen path


class Extractor:
    def __init__(self, run: str = "deberta-v3-large-p2-zs-v62-ftr", device=None,
                 local: Path | None = None, zs_offset: float | None = None,
                 span_offset: float | None = None, span_boost: float | None = None):
        from .train.gold_eval import HF_REPO, load_run
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                              "mps" if torch.backends.mps.is_available() else "cpu"))
        self.run = run
        self.model, self.tok, self.spec, self.vocab, _ = load_run(run, HF_REPO, self.device, local)
        if self.device.type == "mps":
            from .train import data as _data
            _data.SEQ_BUCKET = _data.SEQ_BUCKET or 64      # see data.SEQ_BUCKET
        cal = {}
        if local is not None and (Path(local) / run / "calibration.json").exists():
            cal = json.loads((Path(local) / run / "calibration.json").read_text())
        elif local is None:
            try:
                from huggingface_hub import hf_hub_download
                cal = json.loads(Path(hf_hub_download(HF_REPO, f"{run}/calibration.json")).read_text())
            except Exception:
                cal = {}
        self.model.zs_offset = float(zs_offset if zs_offset is not None else cal.get("offset", 0.0))
        self.span_offset = float(span_offset if span_offset is not None
                                 else cal.get("span_offset", 0.0))
        # Typed entities: the entity-mention head's spans also proposed to
        # the span head, their margin raised by this (None: span head alone).
        self.span_boost = span_boost if span_boost is not None else cal.get("span_boost")
        # How coreference groups mentions into entities, chosen per run on
        # Re-DocRED dev (gold_eval redocred_dev_oracle).
        self.entity_coref = cal.get("entity_coref", "all")
        self.link_threshold = float(cal.get("link_threshold", 0.5))
        # Decoding chosen on dev (gold_eval): per-relation threshold offsets
        # for relations the bank knows from Re-DocRED (others keep the
        # label-path offset), facts per pair, the entity head's O bias and
        # the coreference link bias.
        from .label_bank import load as _load_bank
        rd = {r["name"] for r in _load_bank()["relations"] if r["source"] == "redocred"}
        # Re-DocRED relations: their own offset, else the global one tuned
        # with them; any other relation: 0 (the label path's own threshold).
        self.rel_offsets = {n: float(cal.get("rel_offset", 0.0)) for n in rd} if "rel_offset" in cal else {}
        self.rel_offsets.update(cal.get("rel_offsets") or {})
        # Every other relation: calibrate_zs's offset (held-out teacher
        # windows), "predict when margin > unseen_offset".
        self.unseen_offset = float(cal.get("unseen_offset", 0.0))
        self.top = int(cal.get("rel_top", 4))
        self.ent_o_bias = float(cal.get("ent_o_bias", 0.0))
        self.model.coref_link_bias = float(cal.get("coref_link_bias", 0.0))
        from .label_bank import load as load_bank
        self.bank_ids = {r["name"]: r["id"] for r in load_bank()["relations"]}
        self.frozen_path = True
        # Extra mention proposals for typed entities (v8 P1): "ner" (OntoNotes
        # spans), "np" (noun phrases from the parse); see v6.mentions.
        self.proposers: tuple[str, ...] = ()
        self.nested = False      # typed entities may nest (v8 P1)
        # With nested entities one fact comes back once per span variant
        # ("Northwind" / "Northwind 's plant"); keep the highest-margin one (v8 M1).
        self.merge_variants = False
        # v8 plan r4 V1: a fact verifier decides the relations it was trained
        # on (candidates down to ``ver_floor`` under the bar, two per pair,
        # kept when its probability of "expressed" reaches ``ver_threshold``);
        # every other relation keeps the margin rule. See ``use_verifier``.
        self.verifier = None
        self.ver_threshold = 0.5
        self.ver_floor = 3.0
        self.ver_trained: set[str] = set()
        self.ver_on = True
        # Evaluators add a constant to every offset to see all candidates;
        # they set it here so the verifier's floor stays relative to the bar.
        self.margin_shift = 0.0
        # Plan r4 T2: the new-relation bar raised by this for relations that
        # have no offset of their own and that the verifier does not decide.
        self.new_bar_shift = 0.0
        # Relations the frozen ATLOP path answers in prose (Re-DocRED's 96):
        # there the margin decides and the verifier only vetoes, below this
        # probability (stage 3: deciding them by the verifier cost 8 F1 on
        # Re-DocRED documents; the veto gains about 1).
        self.ver_veto = 0.1
        self.rd_names = rd

    def use_verifier(self, run: str, threshold: float = 0.5, floor: float = 3.0,
                     trained: set[str] | None = None) -> None:
        """Attach the verifier trained on this extractor's run (its base must
        be the same run: it reuses this model, frozen)."""
        from .train.gold_eval import CKPT_DIR, HF_REPO
        from .train.verifier import Verifier, trained_relations
        path = CKPT_DIR / run / "verifier.pt"
        if not path.exists():
            from huggingface_hub import hf_hub_download
            path = Path(hf_hub_download(HF_REPO, f"{run}/verifier.pt", local_dir=str(CKPT_DIR)))
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = Verifier(self.model, r=ck["rank"]).to(self.device)
        m.ver.load_state_dict(ck["ver"]); m.head.load_state_dict(ck["head"])
        m.eval()
        from .label_bank import load as load_bank
        self.verifier = m
        self.ver_bank = {r["name"]: r for r in load_bank()["relations"]}
        self.ver_threshold, self.ver_floor = float(threshold), float(floor)
        self.ver_trained = set(trained) if trained is not None else trained_relations()

    @torch.no_grad()
    def _verify(self, w: dict, nodes: list[dict], spans: list, parts: list, cands: list) -> list:
        """Probability that each candidate (head, tail, relation name) is
        expressed (supported, negated or hedged); None where the window and
        the candidate do not fit the verifier's input (the margin decides)."""
        from .train.verifier import collate as vcollate, encode
        rec = {"tokens": w["tokens"], "sentence_spans": w["sentence_spans"],
               "speakers": w.get("sentence_speakers"), "participants": parts,
               "entities": [[tuple(int(x) for x in m) for m in ms] for ms in spans],
               "names": [n["text"] for n in nodes]}
        cache: dict = {}
        exs, idx = [], []
        for j, (h, t, rel) in enumerate(cands):
            e = encode(rec, {"h": h, "t": t, "rel": rel, "y": 0}, self.tok, self.ver_bank, None, None, cache=cache)
            if e is not None:
                exs.append(e); idx.append(j)
        out: list[float | None] = [None] * len(cands)     # None: too long to verify
        bs = 16 if len(rec["tokens"]) < 200 else 4       # long windows: 1,024-subword inputs
        for i in range(0, len(exs), bs):
            lg = self.verifier(vcollate(exs[i:i + bs], self.tok.pad_token_id, self.device))
            for j, p in zip(idx[i:i + bs], (1 - torch.softmax(lg, -1)[:, 1]).tolist()):
                out[j] = p
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return out

    # ── text to windows ──────────────────────────────────────────────
    @staticmethod
    def windows(text: str | None = None, turns=None) -> list[dict]:
        from .windows import Document, build_windows, tokenize
        if turns is not None:
            doc = Document("doc", "conversation", "infer", units=[t for _, t in turns],
                           speakers=[s for s, _ in turns])
        else:
            units = [p for p in (text or "").split("\n\n") if p.strip()]
            doc = Document("doc", "text", "infer", units=units)
        return build_windows(doc, tokenize, max_tokens=512, min_tokens=1)

    # ── entities ─────────────────────────────────────────────────────
    @torch.no_grad()
    def _entities(self, w: dict, types: list | None):
        """``types``: the caller's entity types, ``{"name", "description"?}``
        (bare names also accepted); None: the frozen DocRED entity head."""
        from .entities import build_entities
        from .train.data import ENT_TAGS
        from .train.gold_eval import bio_constraints, phase2_window, run_phase2, viterbi
        tokens, sp = w["tokens"], [tuple(x) for x in w["sentence_spans"]]
        f = phase2_window(tokens, sp, self.tok, self.spec, self.vocab, None, markers=False)
        o = run_phase2(self.model, [f], self.tok, self.device, batch=1)[0]
        W = f["_W"]
        span_type: dict[tuple[int, int], str] = {}
        if types:
            from .train.span_eval import kniv_predict
            types = [t if isinstance(t, dict) else {"name": t} for t in types]
            desc = {t["name"]: t.get("description") or "" for t in types}
            pred = kniv_predict(self.model, self.tok, self.spec, self.vocab, self.device,
                                [{"tokens": tokens[:W], "spans": []}], [t["name"] for t in types],
                                batch=1, offset=self.span_offset, describe=desc.get,
                                boost=self.span_boost,
                                extra=self._proposals(tokens[:W], sp, o) if self.proposers else None,
                                nested=self.nested)[0]
            ner = []
            for s, e in sp:
                tags = ["O"] * (e - s)
                for a, b, t in sorted(pred):
                    if s <= a and b <= e:
                        span_type[(a, b)] = t
                        tags[a - s] = "B-MISC"
                        for i in range(a + 1, b):
                            tags[i - s] = "I-MISC"
                ner.append(tags if s < W else None)
        else:
            allowed, start = bio_constraints(ENT_TAGS)
            lp = o["ent"].copy()
            lp[:, 0] += self.ent_o_bias                 # ENT_TAGS[0] is O
            ner = [[ENT_TAGS[x] for x in viterbi(lp[s:min(e, W)], allowed, start)]
                   if s < W else None for s, e in sp]
        link = None
        if getattr(self.model, "linker", None) is not None and o.get("link_words") is not None:
            from .train.entity_link import Linker
            link = Linker(self.model, tokens, sp, o["link_words"], o["coref"], self.link_threshold)
        given = None
        if types and self.nested:
            given = [("MISC", a, b) for (a, b) in sorted(span_type)]
        _, vertex = build_entities(tokens, [list(x) for x in sp], ner,
                                   [[list(m) for m in c] for c in o["coref"]],
                                   coref=self.entity_coref, link=link, min_entities=1, spans=given)
        ents = []
        for v in vertex or []:
            ms = sorted({(sp[m["sent_id"]][0] + m["pos"][0], sp[m["sent_id"]][0] + m["pos"][1])
                         for m in v})
            ty = [span_type[m] for m in ms if m in span_type]
            ents.append({"type": max(set(ty), key=ty.count) if ty else v[0].get("type"),
                         "mentions": ms})
        return ents

    def _proposals(self, tokens, sp, o) -> list[set]:
        from .mentions import propose
        return [propose(tokens, sp, o, self.vocab, self.proposers)]

    # ── relations ────────────────────────────────────────────────────
    @torch.no_grad()
    def _relations(self, w: dict, ents: list[dict], relations: list[dict], top: int):
        from .train.data import blank_window, collate, marker_features, speaker_turns
        from .train.kdata import label_text
        tokens, sp = w["tokens"], [tuple(x) for x in w["sentence_spans"]]
        f = blank_window(tokens, sp, self.tok, self.spec.max_subwords, self.vocab.size("cls"))
        W = f["_W"]
        speakers = w.get("sentence_speakers") or []
        parts = list(dict.fromkeys(s for s in speakers if s))
        turns = [(s, p) for s, p in speaker_turns(sp, speakers, parts) if s < W and f["word_ok"][s]]
        nodes, spans = [], []
        for p, who in enumerate(parts):
            ms = [(s, s + 1) for s, q in turns if q == p]
            if ms:
                nodes.append({"type": "speaker", "text": who, "speaker": True})
                spans.append(np.asarray(ms, dtype=np.int32))
        n_part = len(nodes)
        for e in ents:
            ms = [(a, b) for a, b in e["mentions"] if b <= W and f["word_ok"][a]]
            if ms:
                nodes.append({"type": e["type"], "text": " ".join(tokens[ms[0][0]:ms[0][1]]),
                              "mentions": [" ".join(tokens[a:b]) for a, b in ms], "spans": ms})
                spans.append(np.asarray(ms, dtype=np.int32))
        if len(spans) < 2:
            return nodes, []
        f.update(coref=None, _n_rel=1, rel={"spans": spans, "types": np.zeros(len(spans), np.int32),
                                            "triples": []})
        part_turns = [(s, p) for s, p in turns if p < n_part]
        f["rm"] = marker_features(tokens, sp, spans[n_part:], self.tok, turns=part_turns,
                                  participants=parts[:n_part])
        b = collate([f], self.tok.pad_token_id)
        texts = [label_text(r) for r in relations]
        from .train import data as _data
        enc = self.tok(texts, padding=True, truncation=True, max_length=48, return_tensors="pt",
                       pad_to_multiple_of=16 if _data.SEQ_BUCKET else None)
        # The frozen ATLOP path answers Re-DocRED's relations in prose; it was
        # trained on Wikipedia, before speaker tags, so in a conversation every
        # relation goes to the label path.
        frozen = self.frozen_path and not parts
        cols = [self.bank_ids.get(r["name"], UNSEEN_ID + i) if frozen else UNSEEN_ID + i
                for i, r in enumerate(relations)]
        b = {k: v.to(self.device) for k, v in b.items() if hasattr(v, "to")}
        b["rel_label_emb"] = self.model.label_embeddings(enc["input_ids"].to(self.device),
                                                         enc["attention_mask"].to(self.device))
        b["rel_label_cols"] = torch.tensor(cols, device=self.device)
        if self.model.early_fusion:
            from .train.kdata import fuse_rows
            b["rel_fuse_ids"], b["rel_fuse_mask"] = [
                v.to(self.device) for v in fuse_rows(self.tok, [r["name"] for r in relations])]
        o = self.model(b, relation_only=True)
        lg = o["relation"].float().cpu().numpy()
        facts = []
        ver = self.verifier is not None and self.ver_on
        off = np.array([self.rel_offsets[r["name"]] if r["name"] in self.rel_offsets else
                        -self.unseen_offset - (self.new_bar_shift if not (ver and r["name"] in self.ver_trained)
                                               else 0.0) for r in relations])
        vc, vm = [], []
        for (h, t), row in zip(zip(b["pair_h"].tolist(), b["pair_t"].tolist()), lg):
            m = row[1:] - row[0] + off
            order = np.argsort(-m)
            if ver:     # trained relations: the pair's two best candidates near the bar go to the verifier
                decide = lambda k: (relations[k]["name"] in self.ver_trained
                                    and not (frozen and relations[k]["name"] in self.rd_names))
                tr = [k for k in order if decide(k)][:2]
                for k in tr:
                    if m[k] - self.margin_shift > -self.ver_floor:
                        vc.append((h, t, relations[k]["name"], False)); vm.append(float(m[k]))
                for k in order[:top]:           # frozen-path relations above the bar: the verifier may veto
                    if (frozen and relations[k]["name"] in self.rd_names and m[k] > 0
                            and relations[k]["name"] in self.ver_trained):
                        vc.append((h, t, relations[k]["name"], True)); vm.append(float(m[k]))
                order = [k for k in order if not decide(k)
                         and not (frozen and relations[k]["name"] in self.rd_names
                                  and relations[k]["name"] in self.ver_trained)]
            for k in order[:top]:
                if m[k] > 0:
                    facts.append({"head": h, "relation": relations[k]["name"], "tail": t,
                                  "margin": round(float(m[k]), 3)})
        if vc:
            # A verified fact's margin is its log-odds over the threshold:
            # positive exactly when kept, ranked by the verifier.
            lt = float(np.log(self.ver_threshold) - np.log1p(-self.ver_threshold))
            lv = float(np.log(self.ver_veto) - np.log1p(-self.ver_veto))
            probs = self._verify(w, nodes, spans, parts, [c[:3] for c in vc])
            for (h, t, rel, veto), m, p in zip(vc, vm, probs):
                if p is not None:
                    q = min(max(p, 1e-6), 1 - 1e-6)
                    lo = float(np.log(q) - np.log1p(-q))
                    # a vetoable fact keeps its margin unless the verifier is below the veto
                    m = min(m, self.margin_shift + lo - lv) if veto else self.margin_shift + lo - lt
                if m > 0:
                    facts.append({"head": h, "relation": rel, "tail": t, "margin": round(m, 3),
                                  "p": None if p is None else round(p, 4)})
        return nodes, facts

    def extract(self, text: str | None = None, turns=None, relations: list[dict] = (),
                entity_types: list[dict] | None = None, top: int | None = None) -> list[dict]:
        types = list(entity_types) if entity_types else None
        out = []
        for w in self.windows(text, turns):
            ents = self._entities(w, types)
            nodes, facts = self._relations(w, ents, list(relations), top or self.top)
            if self.merge_variants:
                facts = merge_variants(nodes, facts)
            out.append({"tokens": w["tokens"], "entities": nodes, "facts": facts})
        return out


def _nests(a: dict, b: dict) -> bool:
    """One node's mentions contain or lie inside the other's: by span, or by
    text (the same words mentioned again elsewhere in the window)."""
    if a is b:
        return True
    if a.get("speaker") or b.get("speaker"):
        return False
    if any((x0 <= y0 and y1 <= x1) or (y0 <= x0 and x1 <= y1)
           for x0, x1 in a["spans"] for y0, y1 in b["spans"]):
        return True
    def inside(x, y):                      # word sequence x within word sequence y
        return any(y[i:i + len(x)] == x for i in range(len(y) - len(x) + 1))
    A = [m.lower().split() for m in a["mentions"]]
    B = [m.lower().split() for m in b["mentions"]]
    return any(inside(x, y) or inside(y, x) for x in A for y in B)


def merge_variants(nodes: list[dict], facts: list[dict], bar: float = 0.0) -> list[dict]:
    """Facts above ``bar`` with the same relation whose heads nest and whose
    tails nest are one fact said with longer or shorter spans: keep the one
    with the highest margin. Facts at or below the bar pass through."""
    keep, out = [], [f for f in facts if f["margin"] <= bar]
    for f in sorted((f for f in facts if f["margin"] > bar), key=lambda f: -f["margin"]):
        if not any(g["relation"] == f["relation"] and _nests(nodes[g["head"]], nodes[f["head"]])
                   and _nests(nodes[g["tail"]], nodes[f["tail"]]) for g in keep):
            keep.append(f)
    return keep + out
