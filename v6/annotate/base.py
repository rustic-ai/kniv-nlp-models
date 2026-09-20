"""Annotator infrastructure: caching, validation, logging, cost accounting.

The operational contract carried forward from the v5 work:

* **Resume** — every response is cached on disk keyed by
  ``(annotator, layer, prompt_version, item)``. Killing a run and restarting
  it re-reads the cache and only issues the calls that are actually missing.
* **Observability** — a JSONL event log records one line per item with
  latency, token counts and failure kind; the console gets unbuffered
  progress with a running rate and ETA.
* **No silent failures** — a malformed response is counted by *kind* and
  surfaced in the report. Nothing is ever padded, truncated, or coerced to a
  default label to make it fit.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

from ..schemas import UPOS_TAGS, DEPRELS, SRL_TAGS

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_id(item_id: str) -> str:
    """Filesystem-safe, collision-resistant cache filename for an item id."""
    stem = _SAFE.sub("_", item_id)[:80]
    digest = hashlib.sha1(item_id.encode("utf-8")).hexdigest()[:10]
    return f"{stem}.{digest}"


@dataclass
class AnnotationResult:
    item_id: str
    layer: str
    annotator: str
    ok: bool
    payload: object | None = None
    error: str | None = None
    error_kind: str | None = None      # api | refusal | parse | length | range
    repairs: int = 0
    well_formed: bool | None = None    # dep only: single root + acyclic
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0      # time in the API call itself
    queue_ms: float = 0.0        # time waiting on the concurrency semaphore
    cached: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ── Validation ───────────────────────────────────────────────────

def tree_is_wellformed(heads: list[int]) -> bool:
    """True if ``heads`` (1-indexed, 0 = root) forms a single rooted tree.

    Reported per annotator because a dependency parse that is not a tree is
    unusable downstream no matter how many individual arcs are right.
    """
    n = len(heads)
    if sum(1 for h in heads if h == 0) != 1:
        return False
    for i in range(n):
        seen, cur, steps = set(), i + 1, 0
        while cur != 0:
            if cur in seen or steps > n:
                return False
            seen.add(cur)
            cur = heads[cur - 1]
            steps += 1
    return True


def validate_payload(layer: str, payload: object, n: int,
                     n_entities: int | None = None) -> tuple[object, str | None, str | None]:
    """Return ``(normalised_payload, error, error_kind)``.

    Length and value-range violations are errors, not something to repair
    silently. ``error`` is None when the payload is usable.
    """
    if layer == "rel":
        if not isinstance(payload, list):
            return None, "triples is not a list", "parse"
        if n_entities is None:
            return None, "rel validation requires n_entities", "parse"
        out, seen = [], set()
        for ti, tr in enumerate(payload):
            if isinstance(tr, dict):
                if not {"h", "t", "r"} <= set(tr):
                    return None, f"triple {ti} missing h/t/r", "parse"
                h, t, r = tr["h"], tr["t"], tr["r"]
            elif isinstance(tr, (list, tuple)) and len(tr) == 3:
                h, t, r = tr                      # cached form
            else:
                return None, f"triple {ti} malformed", "parse"
            try:
                h, t = int(h), int(t)
            except (TypeError, ValueError):
                return None, f"triple {ti} entity id not an integer", "parse"
            if not (0 <= h < n_entities and 0 <= t < n_entities):
                return None, (f"triple {ti} entity id out of range "
                              f"0..{n_entities - 1}"), "range"
            if h == t:
                continue                          # a self-relation is not a fact
            key = (h, t, str(r))
            if key not in seen:                   # duplicates are not new facts
                seen.add(key)
                out.append([h, t, str(r)])
        return out, None, None

    if layer == "coref":
        if not isinstance(payload, list):
            return None, "clusters is not a list", "parse"
        out = []
        for ci, cl in enumerate(payload):
            if not isinstance(cl, list):
                return None, f"cluster {ci} is not a list", "parse"
            spans = []
            for m in cl:
                if isinstance(m, dict):
                    if "start" not in m or "end" not in m:
                        return None, f"cluster {ci} mention missing start/end", "parse"
                    a, b = int(m["start"]) - 1, int(m["end"]) - 1   # 1-indexed in
                elif isinstance(m, (list, tuple)) and len(m) == 2:
                    a, b = int(m[0]), int(m[1])                     # cached: 0-indexed
                else:
                    return None, f"cluster {ci} mention malformed", "parse"
                if not (0 <= a <= b < n):
                    return None, f"mention [{a},{b}] out of range 0..{n-1}", "range"
                spans.append([a, b])
            if len(spans) >= 2:          # singletons are not coreference
                out.append(spans)
        return out, None, None

    if layer == "dep":
        if not isinstance(payload, list):
            return None, "arcs is not a list", "parse"
        if len(payload) != n:
            return None, f"expected {n} arcs, got {len(payload)}", "length"
        heads, rels = [], []
        for i, arc in enumerate(payload):
            if not isinstance(arc, dict) or "head" not in arc or "rel" not in arc:
                return None, f"arc {i} malformed", "parse"
            try:
                h = int(arc["head"])
            except (TypeError, ValueError):
                return None, f"arc {i} head not an integer", "parse"
            if not (0 <= h <= n):
                return None, f"arc {i} head {h} out of range 0..{n}", "range"
            rel = str(arc["rel"])
            if rel not in DEPRELS:
                return None, f"arc {i} unknown deprel {rel!r}", "range"
            heads.append(h)
            rels.append(rel)
        return {"heads": heads, "rels": rels}, None, None

    if not isinstance(payload, list):
        return None, f"{layer} payload is not a list", "parse"
    if len(payload) != n:
        return None, f"expected {n} entries, got {len(payload)}", "length"
    if not all(isinstance(x, str) for x in payload):
        return None, f"{layer} entries must be strings", "parse"

    allowed = {"pos": UPOS_TAGS, "srl": SRL_TAGS}.get(layer)
    if allowed:
        bad = next((x for x in payload if x not in allowed), None)
        if bad is not None:
            return None, f"unknown {layer} label {bad!r}", "range"
    return payload, None, None


# ── Cache ────────────────────────────────────────────────────────

class CacheStore:
    """On-disk response cache. Presence of a file is what makes runs resumable."""

    MARKER = "EVAL_ONLY_DO_NOT_TRAIN.md"
    _MARKER_TEXT = (
        "# Evaluation output — never training data\n\n"
        "This directory holds annotator predictions over **public gold test "
        "sets** (UD English EWT, PropBank EWT), cached so bake-off runs can "
        "resume.\n\n"
        "It is not a corpus. Feeding it to training would put licensed "
        "treebank text into v6 and contaminate the very test sets v6 is "
        "measured on. v6 training data comes from our own annotated corpus "
        "only.\n"
    )

    def __init__(self, root: Path, prompt_version: str):
        self.root = Path(root)
        self.prompt_version = prompt_version
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / self.MARKER
        if not marker.exists():
            marker.write_text(self._MARKER_TEXT)

    def path(self, annotator: str, layer: str, item_id: str) -> Path:
        return (self.root / annotator / layer / self.prompt_version
                / f"{sanitize_id(item_id)}.json")

    def get(self, annotator: str, layer: str, item_id: str) -> dict | None:
        p = self.path(annotator, layer, item_id)
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except json.JSONDecodeError:
            # A corrupt cache entry (interrupted write) is discarded, not
            # treated as an empty result.
            p.unlink(missing_ok=True)
            return None

    def put(self, annotator: str, layer: str, item_id: str, record: dict) -> None:
        p = self.path(annotator, layer, item_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record))
        tmp.replace(p)                      # atomic: no half-written entries


# ── Logging ──────────────────────────────────────────────────────

@dataclass
class Counters:
    total: int = 0
    ok: int = 0
    cached: int = 0
    repairs: int = 0
    failures: dict = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0


class RunLogger:
    """JSONL event log + unbuffered console progress.

    ``flush=True`` everywhere: these runs are long and often watched over an
    ssh or Colab session that buffers stdout into uselessness otherwise.
    """

    def __init__(self, run_dir: Path, every: int = 25):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events = (self.run_dir / "events.jsonl").open("a", buffering=1)
        self.every = every
        self.counters: dict[tuple[str, str], Counters] = {}
        self.t0 = time.time()

    def counter(self, annotator: str, layer: str) -> Counters:
        return self.counters.setdefault((annotator, layer), Counters())

    def record(self, res: AnnotationResult, spec_price: tuple[float, float],
               total_items: int) -> None:
        c = self.counter(res.annotator, res.layer)
        c.total += 1
        c.tokens_in += res.tokens_in
        c.tokens_out += res.tokens_out
        c.repairs += res.repairs
        if not res.cached:
            pin, pout = spec_price
            c.cost_usd += (res.tokens_in * pin + res.tokens_out * pout) / 1e6
        if res.cached:
            c.cached += 1
        if res.ok:
            c.ok += 1
        else:
            kind = res.error_kind or "unknown"
            c.failures[kind] = c.failures.get(kind, 0) + 1

        self.events.write(json.dumps(res.to_dict()) + "\n")

        if c.total % self.every == 0 or c.total == total_items:
            elapsed = max(time.time() - self.t0, 1e-6)
            rate = c.total / elapsed
            remaining = max(total_items - c.total, 0)
            eta = remaining / rate if rate > 0 else 0.0
            fails = ", ".join(f"{k}={v}" for k, v in sorted(c.failures.items()))
            print(
                f"  [{res.annotator}/{res.layer}] {c.total}/{total_items} "
                f"ok={c.ok} cached={c.cached} "
                f"{'fail(' + fails + ') ' if fails else ''}"
                f"{rate:.1f}/s eta={eta / 60:.1f}m ${c.cost_usd:.2f}",
                flush=True,
            )

    def close(self) -> None:
        self.events.close()
