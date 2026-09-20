"""LLM annotator over an OpenAI-compatible endpoint.

Covers Azure OpenAI deployments and any OpenAI-compatible API (xAI, MAI,
self-hosted) through one code path — the difference is confined to client
construction.

Structured outputs are used wherever the endpoint supports them, so the
model cannot return prose instead of a payload. Where they are not
supported the response is parsed as JSON and validated identically; the
difference shows up as a higher ``parse`` failure count in the report rather
than as silently degraded data.
"""
from __future__ import annotations

import asyncio
import json
import random
import time

from ..config import AnnotatorSpec
from ..prompts import PROMPT_VERSION, SYSTEM, user_message
from ..schemas import PAYLOAD_KEY, schema_for
from .base import (
    AnnotationResult, CacheStore, validate_payload, tree_is_wellformed,
)

_RETRYABLE = ("rate", "timeout", "timed out", "429", "500", "502", "503",
              "504", "overloaded", "connection")


class EmptyResponse(Exception):
    """Model returned no usable content.

    Distinct from a transport error: the call succeeded but produced nothing,
    which happens on content filtering, a truncated generation, or a refusal.
    Retryable, and counted under its own error kind so it never hides inside
    the generic API bucket.
    """


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, EmptyResponse):
        return True
    msg = str(exc).lower()
    return any(t in msg for t in _RETRYABLE)


class LLMAnnotator:
    """One annotator endpoint, one layer at a time."""

    def __init__(self, spec: AnnotatorSpec, cache: CacheStore,
                 max_retries: int = 4, max_repairs: int = 1):
        self.spec = spec
        self.cache = cache
        self.max_retries = max_retries
        self.max_repairs = max_repairs
        self._client = None
        self._sem = asyncio.Semaphore(spec.max_concurrency)
        # Parameters this endpoint has rejected; dropped for the rest of the
        # run so one probe failure does not repeat on every item.
        self._unsupported: set[str] = set()
        if not spec.supports_temperature:
            self._unsupported.add("temperature")
        if not spec.supports_seed:
            self._unsupported.add("seed")
        # response_format ladder: 0 = json_schema (strict), 1 = json_object,
        # 2 = omit entirely and ask for JSON in the prompt. Some endpoints
        # (e.g. reasoning models) reject the whole parameter, not just the
        # strict form, so one fallback step is not enough.
        self._rf_level = 0 if spec.supports_structured_output else 1

    # ── client ───────────────────────────────────────────────────
    @property
    def client(self):
        if self._client is None:
            if self.spec.kind == "azure":
                from openai import AsyncAzureOpenAI
                kwargs = {
                    "azure_endpoint": self.spec.base_url,
                    "api_version": self.spec.api_version or "2024-10-21",
                    "timeout": self.spec.request_timeout,
                    # Retries belong to our loop, which counts and reports
                    # them; the SDK's own would multiply against it silently.
                    "max_retries": 0,
                }
                if self.spec.auth == "azure_ad":
                    from .azure_auth import get_token
                    kwargs["azure_ad_token_provider"] = get_token
                else:
                    kwargs["api_key"] = self.spec.api_key
                self._client = AsyncAzureOpenAI(**kwargs)
            else:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(
                    base_url=self.spec.base_url or None,
                    api_key=self.spec.api_key,
                    timeout=self.spec.request_timeout,
                    max_retries=0,
                )
        return self._client

    # ── one call ─────────────────────────────────────────────────
    def _build_kwargs(self, layer: str, messages: list[dict]) -> dict:
        # Re-apply the prompt-level contract in case the ladder stepped down
        # after these messages were built.
        hint = self._json_hint(layer)
        if hint and not messages[-1]["content"].endswith(hint):
            messages = messages[:-1] + [
                {**messages[-1], "content": messages[-1]["content"] + hint}]
        kwargs: dict = {"model": self.spec.model, "messages": messages}
        if "temperature" not in self._unsupported:
            kwargs["temperature"] = self.spec.temperature
        if "seed" not in self._unsupported and self.spec.seed is not None:
            kwargs["seed"] = self.spec.seed
        if self._rf_level == 0:
            kwargs["response_format"] = {
                "type": "json_schema", "json_schema": schema_for(layer),
            }
        elif self._rf_level == 1:
            kwargs["response_format"] = {"type": "json_object"}
        # level 2: no response_format at all — the prompt carries the
        # contract instead (see _json_hint).
        return kwargs

    def _json_hint(self, layer: str) -> str:
        """Prompt-level output contract, used when response_format is gone."""
        if self._rf_level < 2:
            return ""
        key = PAYLOAD_KEY[layer]
        return (f"\n\nRespond with a single JSON object and nothing else — "
                f"no prose, no markdown fences. Shape: "
                f'{{"{key}": [ ... ]}}')

    _RF_STEP = {0: "json_schema -> json_object",
                1: "json_object -> no response_format"}

    _PARAM_ERR = ("unsupported", "not supported", "unrecognized",
                  "invalid_request", "not enabled", "does not support")

    @classmethod
    def _is_param_error(cls, exc: Exception) -> bool:
        return any(t in str(exc).lower() for t in cls._PARAM_ERR)

    def _note_unsupported(self, exc: Exception) -> bool:
        """Drop a parameter the endpoint rejected. True if something changed.

        Returning False does NOT mean the request is hopeless: under
        concurrency, many in-flight items hit the same rejection at once and
        only the first one changes state. The others must still retry,
        because ``_build_kwargs`` will now omit the offending parameter. The
        caller handles that via :meth:`_is_param_error`.
        """
        msg = str(exc).lower()
        if not self._is_param_error(exc):
            return False
        if "response_format" in msg and self._rf_level < 2:
            step = self._RF_STEP[self._rf_level]
            self._rf_level += 1
            print(f"  [{self.spec.name}] endpoint rejected response_format; "
                  f"stepping down {step}", flush=True)
            return True
        for param in ("temperature", "seed"):
            if param in msg and param not in self._unsupported:
                self._unsupported.add(param)
                print(f"  [{self.spec.name}] endpoint rejected {param!r}; "
                      f"dropping it for the rest of the run", flush=True)
                return True
        return False

    # A request that blew its deadline is not going to succeed on the fourth
    # attempt, and each retry costs another full timeout. Two attempts is the
    # allowance: one for a genuine transient stall, then give up. Left
    # unbounded this is the dominant cost of a stuck item — measured at ~20
    # minutes of wall-clock per item against a 300s deadline.
    MAX_TIMEOUT_ATTEMPTS = 2

    @staticmethod
    def _is_timeout(exc: Exception) -> bool:
        return any(t in str(exc).lower() for t in ("timeout", "timed out"))

    async def _call(self, layer: str, messages: list[dict]) -> tuple[str, int, int]:
        last: Exception | None = None
        timeouts = 0
        for attempt in range(self.max_retries):
            kwargs = self._build_kwargs(layer, messages)
            try:
                resp = await self.client.chat.completions.create(**kwargs)
                usage = getattr(usage_src := resp, "usage", None)
                choices = getattr(resp, "choices", None) or []
                if not choices:
                    raise EmptyResponse("model returned no choices")
                choice = choices[0]
                content = getattr(choice.message, "content", None) or ""
                if not content.strip():
                    # finish_reason is the useful diagnostic here: "length"
                    # means the output was truncated, "content_filter" means
                    # it was blocked.
                    raise EmptyResponse(
                        f"empty content (finish_reason="
                        f"{getattr(choice, 'finish_reason', '?')})")
                del usage_src
                return (
                    content,
                    getattr(usage, "prompt_tokens", 0) or 0,
                    getattr(usage, "completion_tokens", 0) or 0,
                )
            except Exception as exc:                       # noqa: BLE001
                last = exc
                self._note_unsupported(exc)
                if self._is_param_error(exc):
                    # Retry regardless of whether *this* task was the one that
                    # recorded the parameter: the rebuilt kwargs differ either
                    # way. Without this, every concurrently in-flight item
                    # fails permanently while one of them heals.
                    if attempt == self.max_retries - 1:
                        raise
                    continue
                if self._is_timeout(exc):
                    timeouts += 1
                    if timeouts >= self.MAX_TIMEOUT_ATTEMPTS:
                        raise
                if not _is_retryable(exc) or attempt == self.max_retries - 1:
                    raise
                # Exponential backoff with jitter.
                await asyncio.sleep(min(2 ** attempt, 30) * (0.5 + random.random()))
        raise last                                          # pragma: no cover

    def build_messages(self, layer: str, item) -> list[dict]:
        """System + user turn for one item.

        Factored out so prompt experiments (few-shot, pair enumeration) can
        vary the wording without touching the bake-off path or the retry,
        cache and validation machinery around it.
        """
        return [
            {"role": "system", "content": SYSTEM[layer]},
            {"role": "user", "content": user_message(
                layer, item.tokens, item.predicate_idx,
                getattr(item, "entities", None)) + self._json_hint(layer)},
        ]

    # ── one item ─────────────────────────────────────────────────
    async def annotate(self, layer: str, item) -> AnnotationResult:
        n = len(item.tokens)
        n_ent = len(item.entities) if getattr(item, "entities", None) else None
        cached = self.cache.get(self.spec.name, layer, item.id)
        if cached is not None:
            payload, err, kind = validate_payload(
                layer, cached.get("payload"), n, n_ent)
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.spec.name,
                ok=err is None, payload=payload, error=err, error_kind=kind,
                repairs=cached.get("repairs", 0),
                well_formed=(tree_is_wellformed(payload["heads"])
                             if layer == "dep" and payload else None),
                cached=True,
            )

        messages = self.build_messages(layer, item)

        # Every task for a layer is created up front, so most of them sit on
        # the semaphore before doing anything. Timing from here would fold
        # that queue wait into "latency" and make a fast endpoint look slow
        # purely because it was scheduled late.
        t_queued = time.time()
        tok_in = tok_out = 0
        repairs = 0
        payload = None
        err: str | None = "no attempt"
        kind: str | None = "api"

        async with self._sem:
            t0 = time.time()
            for attempt in range(self.max_repairs + 1):
                try:
                    content, ti, to = await self._call(layer, messages)
                except Exception as exc:                    # noqa: BLE001
                    return AnnotationResult(
                        item_id=item.id, layer=layer, annotator=self.spec.name,
                        ok=False, error=f"{type(exc).__name__}: {exc}",
                        error_kind=("empty" if isinstance(exc, EmptyResponse)
                                    else "api"),
                        repairs=repairs,
                        latency_ms=(time.time() - t0) * 1000,
                        queue_ms=(t0 - t_queued) * 1000,
                    )
                tok_in += ti
                tok_out += to

                try:
                    obj = json.loads(content)
                    raw = obj.get(PAYLOAD_KEY[layer])
                except (json.JSONDecodeError, AttributeError):
                    payload, err, kind = None, "response was not valid JSON", "parse"
                else:
                    payload, err, kind = validate_payload(layer, raw, n, n_ent)

                if err is None:
                    break
                if attempt < self.max_repairs:
                    # One corrective turn. If the model still cannot satisfy
                    # the contract, that is a real result about the model, so
                    # record it rather than papering over it.
                    repairs += 1
                    messages += [
                        {"role": "assistant", "content": content},
                        {"role": "user", "content":
                            f"That response was rejected: {err}. Return "
                            f"exactly {n} entries, one per token, in order, "
                            f"and nothing else."},
                    ]

        return AnnotationResult(
            item_id=item.id, layer=layer, annotator=self.spec.name,
            ok=err is None, payload=payload, error=err, error_kind=kind,
            repairs=repairs,
            well_formed=(tree_is_wellformed(payload["heads"])
                         if layer == "dep" and payload else None),
            tokens_in=tok_in, tokens_out=tok_out,
            latency_ms=(time.time() - t0) * 1000,
            queue_ms=(t0 - t_queued) * 1000,
        )

    async def annotate_and_cache(self, layer: str, item) -> AnnotationResult:
        res = await self.annotate(layer, item)
        if not res.cached and res.ok:
            self.cache.put(self.spec.name, layer, item.id, {
                "payload": (res.payload if layer != "dep" else
                            [{"head": h, "rel": r} for h, r
                             in zip(res.payload["heads"], res.payload["rels"])]),
                "repairs": res.repairs,
                "prompt_version": PROMPT_VERSION,
            })
        return res
