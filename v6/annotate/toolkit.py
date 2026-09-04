"""Supervised UD pipelines (Stanza, Trankit) as bake-off annotators.

These are the incumbent specialists for the structural layers. Published
numbers are not comparable to v5's: toolkit scores are end-to-end from raw
text, so their own tokenizer and sentence-splitter errors (Trankit 98.67 /
90.49 on English EWT) are baked in, while v5 was measured on gold tokens.

Both wrappers therefore feed **pre-tokenized, pre-split** input — one gold
sentence at a time — which is also how they would run in the v6 pipeline,
where canonical tokens are fixed before any annotator sees the text. That
makes the resulting numbers directly comparable to v5 and to the LLMs.

Each toolkit needs its own virtualenv (Trankit pins transformers<4.40 and
breaks on Python 3.13; the main venv holds the transformers==5.6.2 that v5
requires). The shared response cache is the integration point, so a run from
a different venv still lands in the same report.
"""
from __future__ import annotations

import time

from ..schemas import DEPRELS
from .base import AnnotationResult, tree_is_wellformed

# Toolkit deprels occasionally fall outside the EWT inventory the schema
# declares (language-general relations, or subtypes EWT does not use).
_DEPREL_SET = set(DEPRELS)


def _norm_deprel(rel: str) -> str:
    if rel in _DEPREL_SET:
        return rel
    base = rel.split(":")[0]                 # obl:arg -> obl
    return base if base in _DEPREL_SET else "dep"


def _norm_feats(feats) -> str:
    """Normalise a FEATS value to canonical UD form: alphabetical, '_' if empty.

    UFeats is scored as whole-string equality, so ordering matters.
    """
    if not feats or feats in ("_", "None"):
        return "_"
    if isinstance(feats, dict):
        pairs = [f"{k}={v}" for k, v in feats.items()]
    else:
        pairs = [p for p in str(feats).split("|") if "=" in p]
    return "|".join(sorted(pairs)) if pairs else "_"


class _ToolkitAnnotator:
    """Shared cache/result plumbing; subclasses supply :meth:`_analyse`."""

    name = "toolkit"
    LAYERS = ("pos", "lemma", "morph", "dep", "ner")

    # The driver instantiates one annotator per (annotator, layer), so a
    # sentence would otherwise be parsed once per layer. These pipelines
    # produce all four layers in a single pass, so memoise across instances.
    _MEMO: dict[tuple[str, str], dict] = {}

    def __init__(self, spec, cache, **kwargs):
        self.spec = spec
        self.cache = cache
        self.name = spec.name
        self.opts = kwargs
        self._pipe = None

    def _analyse_memo(self, item) -> dict:
        key = (self.name, item.id)
        if key not in self._MEMO:
            if len(self._MEMO) > 4096:          # bound: one sweep's worth
                self._MEMO.clear()
            self._MEMO[key] = self._analyse(item.tokens)
        return self._MEMO[key]

    def _load(self):
        raise NotImplementedError

    def _analyse(self, tokens: list[str]) -> dict:
        """Return {'pos','lemma','morph','dep'} for one pre-tokenized sentence."""
        raise NotImplementedError

    async def annotate_and_cache(self, layer: str, item) -> AnnotationResult:
        if layer not in self.LAYERS:
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=False,
                error=f"{self.name} does not produce {layer}",
                error_kind="unsupported")

        cached = self.cache.get(self.name, layer, item.id)
        if cached is not None:
            p = cached["payload"]
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=True,
                payload=p, cached=True,
                well_formed=(tree_is_wellformed(p["heads"])
                             if layer == "dep" else None))

        if self._pipe is None:
            self._load()

        t0 = time.time()
        try:
            out = self._analyse_memo(item)
        except Exception as exc:                                # noqa: BLE001
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=False,
                error=f"{type(exc).__name__}: {exc}", error_kind="api",
                latency_ms=(time.time() - t0) * 1000)

        payload = out.get(layer)
        if payload is None:
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=False,
                error=f"{self.name} produced no {layer}", error_kind="unsupported")
        n = len(item.tokens)
        got = len(payload["heads"]) if layer == "dep" else len(payload)
        if got != n:
            # The toolkit re-segmented despite pre-tokenized input. Same
            # contract as the LLMs: recorded, never padded to fit.
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=False,
                error=f"expected {n} entries, got {got}", error_kind="length",
                latency_ms=(time.time() - t0) * 1000)

        self.cache.put(self.name, layer, item.id,
                       {"payload": payload, "repairs": 0})
        return AnnotationResult(
            item_id=item.id, layer=layer, annotator=self.name, ok=True,
            payload=payload, latency_ms=(time.time() - t0) * 1000,
            well_formed=(tree_is_wellformed(payload["heads"])
                         if layer == "dep" else None))


class StanzaAnnotator(_ToolkitAnnotator):
    """Stanza (Apache-2.0), UD v2.8 English models, pretokenized mode."""

    # Stanza's English NER model uses the OntoNotes 18-type inventory — the
    # same schema v5's head was trained on, so the two are directly comparable.
    PROCESSORS = "tokenize,pos,lemma,depparse,ner"

    def _load(self):
        import stanza
        try:
            self._pipe = stanza.Pipeline(
                lang="en", processors=self.PROCESSORS,
                tokenize_pretokenized=True, download_method=None,
                logging_level="WARN")
        except Exception:                                       # noqa: BLE001
            stanza.download("en", logging_level="WARN")
            self._pipe = stanza.Pipeline(
                lang="en", processors=self.PROCESSORS,
                tokenize_pretokenized=True, logging_level="WARN")
        print(f"  [{self.name}] stanza pipeline ready", flush=True)

    def _analyse(self, tokens):
        doc = self._pipe([list(tokens)])
        words = [w for s in doc.sentences for w in s.words]
        # Stanza attaches NER to tokens (not words); with pretokenized input
        # the two align 1:1.
        toks = [t for s in doc.sentences for t in s.tokens]
        ner = [(t.ner or "O") for t in toks]
        ner = [t if t == "O" or t[:2] in ("B-", "I-") else
               ("B-" + t[2:] if t.startswith("S-") else
                "I-" + t[2:] if t.startswith("E-") else t)
               for t in ner]                       # BIOES -> BIO
        return {
            "pos": [w.upos or "X" for w in words],
            "lemma": [w.lemma or w.text for w in words],
            "morph": [_norm_feats(w.feats) for w in words],
            "ner": ner,
            "dep": {"heads": [int(w.head) for w in words],
                    "rels": [_norm_deprel(w.deprel or "dep") for w in words]},
        }


class TrankitAnnotator(_ToolkitAnnotator):
    """Trankit (Apache-2.0 code), XLM-R-large English models, pretokenized."""

    LAYERS = ("pos", "lemma", "morph", "dep")   # no NER in the UD pipeline

    def _load(self):
        from trankit import Pipeline
        self._pipe = Pipeline(lang="english", gpu=self.opts.get("gpu", False),
                              cache_dir=self.opts.get("cache_dir", "./cache"))
        print(f"  [{self.name}] trankit pipeline ready", flush=True)

    def _analyse(self, tokens):
        doc = self._pipe([list(tokens)])
        words = [w for s in doc["sentences"] for w in s["tokens"]]
        # Trankit nests expanded multi-word tokens under "expanded".
        flat = []
        for w in words:
            if w.get("expanded"):
                flat.extend(w["expanded"])
            else:
                flat.append(w)
        return {
            "pos": [w.get("upos") or "X" for w in flat],
            "lemma": [w.get("lemma") or w.get("text") for w in flat],
            "morph": [_norm_feats(w.get("feats")) for w in flat],
            "dep": {"heads": [int(w.get("head", 0)) for w in flat],
                    "rels": [_norm_deprel(w.get("deprel") or "dep") for w in flat]},
        }


TOOLKITS = {"stanza": StanzaAnnotator, "trankit": TrankitAnnotator}
