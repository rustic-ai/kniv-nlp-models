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
                 span_offset: float | None = None):
        from .train.gold_eval import HF_REPO, load_run
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else
                                              "mps" if torch.backends.mps.is_available() else "cpu"))
        self.model, self.tok, self.spec, self.vocab, _ = load_run(run, HF_REPO, self.device, local)
        cal = {}
        if local is None:
            try:
                from huggingface_hub import hf_hub_download
                cal = json.loads(Path(hf_hub_download(HF_REPO, f"{run}/calibration.json")).read_text())
            except Exception:
                cal = {}
        self.model.zs_offset = float(zs_offset if zs_offset is not None else cal.get("offset", 0.0))
        self.span_offset = float(span_offset if span_offset is not None
                                 else cal.get("span_offset", 0.0))
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
    def _entities(self, w: dict, types: list[str] | None):
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
            pred = kniv_predict(self.model, self.tok, self.spec, self.vocab, self.device,
                                [{"tokens": tokens[:W], "spans": []}], types, batch=1,
                                offset=self.span_offset)[0]
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
        _, vertex = build_entities(tokens, [list(x) for x in sp], ner,
                                   [[list(m) for m in c] for c in o["coref"]],
                                   coref=self.entity_coref, link=link)
        ents = []
        for v in vertex or []:
            ms = sorted({(sp[m["sent_id"]][0] + m["pos"][0], sp[m["sent_id"]][0] + m["pos"][1])
                         for m in v})
            ty = [span_type[m] for m in ms if m in span_type]
            ents.append({"type": max(set(ty), key=ty.count) if ty else v[0].get("type"),
                         "mentions": ms})
        return ents

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
        enc = self.tok(texts, padding=True, truncation=True, max_length=48, return_tensors="pt")
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
        o = self.model(b, relation_only=True)
        lg = o["relation"].float().cpu().numpy()
        facts = []
        off = np.array([self.rel_offsets.get(r["name"], -self.unseen_offset) for r in relations])
        for (h, t), row in zip(zip(b["pair_h"].tolist(), b["pair_t"].tolist()), lg):
            m = row[1:] - row[0] + off
            for k in np.argsort(-m)[:top]:
                if m[k] > 0:
                    facts.append({"head": h, "relation": relations[k]["name"], "tail": t,
                                  "margin": round(float(m[k]), 3)})
        return nodes, facts

    def extract(self, text: str | None = None, turns=None, relations: list[dict] = (),
                entity_types: list[dict] | None = None, top: int | None = None) -> list[dict]:
        types = [t["name"] for t in entity_types] if entity_types else None
        out = []
        for w in self.windows(text, turns):
            ents = self._entities(w, types)
            nodes, facts = self._relations(w, ents, list(relations), top or self.top)
            out.append({"tokens": w["tokens"], "entities": nodes, "facts": facts})
        return out
