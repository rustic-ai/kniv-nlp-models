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

    def _analyse(self, tokens: list[str]) -> list[list[list[int]]]:
        text = " ".join(tokens)
        # char offset -> token index, for both span ends
        start_of, end_of, pos = {}, {}, 0
        for i, t in enumerate(tokens):
            start_of[pos] = i
            pos += len(t)
            end_of[pos] = i
            pos += 1                                    # the joining space
        preds = self._pipe.predict(texts=[text])
        clusters = []
        for cl in preds[0].get_clusters(as_strings=False):
            spans = []
            for cs, ce in cl:
                s, e = start_of.get(cs), end_of.get(ce)
                if s is not None and e is not None and e >= s:
                    spans.append([s, e])
            if len(spans) >= 2:
                clusters.append(spans)
        return clusters

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
