"""fastcoref (MIT) as a coreference annotator.

Two checkpoints: ``FCoref`` (fast) and ``LingMessCoref`` (accurate). Both are
MIT-licensed, which is why they are here rather than Maverick — the current
SOTA system is CC-BY-NC-SA and cannot be used commercially.

fastcoref returns character offsets, so predictions are mapped back to token
indices against the same whitespace-joined text the tokens were rendered
from. A span whose offsets do not land on token boundaries is dropped rather
than approximated.
"""
from __future__ import annotations

import contextlib
import time

from .base import AnnotationResult


@contextlib.contextmanager
def _eager_attention():
    """Force eager attention while a fastcoref checkpoint loads.

    LingMess is Longformer-based, and current transformers defaults to SDPA
    and raises for architectures that do not implement it. fastcoref does not
    expose ``attn_implementation``, so the default is injected here for the
    duration of the load only — narrower than setting it process-wide, and it
    leaves FCoref (distilroberta, SDPA-capable) untouched in every other
    code path.
    """
    from transformers import PreTrainedModel
    # fastcoref's model classes predate `all_tied_weights_keys`, which
    # transformers 5.x reads during weight loading. Supplying an empty
    # mapping is faithful: these checkpoints tie no weights.
    # A plain class attribute, not a property: transformers' own post_init
    # *assigns* to this on models that do call it, and a read-only property
    # would break them. This only supplies a fallback for fastcoref's custom
    # classes, which tie no weights and never set it.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    original = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def patched(cls, *args, **kwargs):
        kwargs.setdefault("attn_implementation", "eager")
        return original(cls, *args, **kwargs)

    PreTrainedModel.from_pretrained = patched
    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = classmethod(original)


class FastCorefAnnotator:
    LAYERS = ("coref",)

    def __init__(self, spec, cache, model: str = "fcoref", device: str = "cpu", **_):
        self.spec = spec
        self.cache = cache
        self.name = spec.name
        self.model_kind = model
        self.device = device
        self._pipe = None

    def _load(self):
        from fastcoref import FCoref, LingMessCoref
        cls = LingMessCoref if self.model_kind == "lingmess" else FCoref
        with _eager_attention():
            self._pipe = cls(device=self.device)
        print(f"  [{self.name}] fastcoref ({self.model_kind}) ready", flush=True)

    @staticmethod
    def _offsets(tokens: list[str]) -> tuple[str, dict, dict]:
        """The rendered text plus char offset -> token index, for both ends."""
        start_of, end_of, pos = {}, {}, 0
        for i, t in enumerate(tokens):
            start_of[pos] = i
            pos += len(t)
            end_of[pos] = i
            pos += 1                                    # the joining space
        return " ".join(tokens), start_of, end_of

    @staticmethod
    def _to_clusters(pred, start_of: dict, end_of: dict) -> list[list[list[int]]]:
        """Map one prediction's char spans onto token indices.

        A span whose offsets do not land on token boundaries is dropped, and
        a cluster left with fewer than two mentions is not a cluster.
        """
        clusters = []
        for cl in pred.get_clusters(as_strings=False):
            spans = []
            for cs, ce in cl:
                s, e = start_of.get(cs), end_of.get(ce)
                if s is not None and e is not None and e >= s:
                    spans.append([s, e])
            if len(spans) >= 2:
                clusters.append(spans)
        return clusters

    def _analyse(self, tokens: list[str]) -> list[list[list[int]]]:
        text, start_of, end_of = self._offsets(tokens)
        preds = self._pipe.predict(texts=[text])
        return self._to_clusters(preds[0], start_of, end_of)

    # fastcoref packs several documents into one forward pass. Measured on
    # 24 windows: 0.68 -> 0.95 windows/s at 4096 tokens per batch, with
    # clusters identical on 24/24. A 16384-token batch was *slower* (0.55),
    # so the useful window is narrow and 4096 is the operating point.
    MAX_TOKENS_IN_BATCH = 4096

    async def annotate_batch_and_cache(self, layer: str,
                                       items: list) -> list[AnnotationResult]:
        """Annotate many windows per forward pass, preserving input order."""
        if layer != "coref":
            return [await self.annotate_and_cache(layer, it) for it in items]

        results: dict[str, AnnotationResult] = {}
        todo = []
        for it in items:
            cached = self.cache.get(self.name, layer, it.id)
            if cached is not None:
                results[it.id] = AnnotationResult(
                    item_id=it.id, layer=layer, annotator=self.name, ok=True,
                    payload=cached["payload"], cached=True)
            else:
                todo.append(it)

        if todo:
            if self._pipe is None:
                self._load()
            rendered = [self._offsets(it.tokens) for it in todo]
            t0 = time.time()
            try:
                preds = self._pipe.predict(
                    texts=[r[0] for r in rendered],
                    max_tokens_in_batch=self.MAX_TOKENS_IN_BATCH)
            except Exception as exc:                    # noqa: BLE001
                # One bad document fails the whole call, so fall back to
                # per-item analysis to isolate it rather than losing the batch.
                for it in todo:
                    results[it.id] = await self.annotate_and_cache(layer, it)
                preds = None
            if preds is not None:
                per = (time.time() - t0) * 1000 / max(len(todo), 1)
                for it, (_, start_of, end_of), pred in zip(todo, rendered, preds):
                    payload = self._to_clusters(pred, start_of, end_of)
                    self.cache.put(self.name, layer, it.id,
                                   {"payload": payload, "repairs": 0})
                    results[it.id] = AnnotationResult(
                        item_id=it.id, layer=layer, annotator=self.name,
                        ok=True, payload=payload, latency_ms=per)

        return [results[it.id] for it in items]

    async def annotate_and_cache(self, layer: str, item) -> AnnotationResult:
        if layer != "coref":
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=False,
                error=f"{self.name} only produces coref", error_kind="unsupported")
        cached = self.cache.get(self.name, layer, item.id)
        if cached is not None:
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=True,
                payload=cached["payload"], cached=True)
        if self._pipe is None:
            self._load()
        t0 = time.time()
        try:
            payload = self._analyse(item.tokens)
        except Exception as exc:                        # noqa: BLE001
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=False,
                error=f"{type(exc).__name__}: {exc}", error_kind="api",
                latency_ms=(time.time() - t0) * 1000)
        self.cache.put(self.name, layer, item.id, {"payload": payload, "repairs": 0})
        return AnnotationResult(
            item_id=item.id, layer=layer, annotator=self.name, ok=True,
            payload=payload, latency_ms=(time.time() - t0) * 1000)
