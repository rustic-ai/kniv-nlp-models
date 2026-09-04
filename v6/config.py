"""Paths, run configuration, and the annotator registry.

Annotators are declared in ``v6/annotators.yaml``. Secrets are never stored
there — a field whose value looks like ``${ENV_VAR}`` is resolved from the
environment at load time, and a missing variable is a hard error rather than
a silent empty string.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

DATA_DIR = REPO / "data"
UD_DIR = DATA_DIR / "ud-english-ewt"
PREPARED_DIR = DATA_DIR / "prepared" / "kniv-deberta-cascade"
RUNS_DIR = REPO / "v6" / "runs"

ANNOTATORS_YAML = Path(__file__).parent / "annotators.yaml"

# Layers the bake-off can measure, in report order.
LAYERS = ("pos", "lemma", "morph", "dep", "ner", "srl", "coref")

_ENV_RE = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def _resolve(value: object, where: str) -> object:
    """Expand ``${ENV_VAR}`` placeholders; fail loudly if unset."""
    if not isinstance(value, str):
        return value
    m = _ENV_RE.match(value.strip())
    if not m:
        return value
    var = m.group(1)
    resolved = os.environ.get(var)
    if not resolved:
        raise RuntimeError(
            f"{where}: environment variable {var} is referenced in "
            f"{ANNOTATORS_YAML.name} but is not set."
        )
    return resolved


@dataclass(frozen=True)
class AnnotatorSpec:
    """One annotator endpoint.

    ``family`` records the underlying model lineage. Consensus only
    decorrelates errors across *different* families, so two specs sharing a
    family are counted as a single vote by the agreement logic.
    """

    name: str
    kind: str                       # "openai" | "azure" | "kniv-v5"
    family: str                     # e.g. "openai", "xai", "microsoft"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    api_version: str = ""
    max_concurrency: int = 8
    temperature: float = 0.0
    seed: int | None = 7
    # USD per 1M tokens; used for the cost column in the run report.
    price_in: float = 0.0
    price_out: float = 0.0
    supports_structured_output: bool = True
    # Reasoning models typically reject sampling parameters. These are hints:
    # the client also strips a parameter the API explicitly rejects and
    # remembers that for the rest of the run.
    supports_temperature: bool = True
    supports_seed: bool = True
    extra: dict = field(default_factory=dict)


def _read_registry(path: Path | None = None) -> dict[str, dict]:
    path = path or ANNOTATORS_YAML
    if not path.exists():
        raise FileNotFoundError(
            f"No annotator registry at {path}. Copy annotators.example.yaml "
            f"to annotators.yaml and fill in your deployments."
        )
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("annotators") or {}
    if not entries:
        raise RuntimeError(f"{path} declares no annotators.")
    return entries


def annotator_names(path: Path | None = None) -> list[str]:
    """Declared annotator names, without touching the environment."""
    return list(_read_registry(path))


def load_annotators(select: list[str] | None = None,
                    path: Path | None = None) -> dict[str, AnnotatorSpec]:
    """Build specs for ``select`` (default: all).

    Environment lookups happen only for the annotators actually requested, so
    running a single endpoint does not require every other endpoint's key.
    """
    entries = _read_registry(path)
    wanted = list(entries) if select is None else select
    unknown = [n for n in wanted if n not in entries]
    if unknown:
        raise KeyError(f"Unknown annotators {unknown}; declared: {sorted(entries)}")
    specs: dict[str, AnnotatorSpec] = {}
    for name in wanted:
        cfg = {k: _resolve(v, f"annotator '{name}'")
               for k, v in (entries[name] or {}).items()}
        specs[name] = AnnotatorSpec(name=name, **cfg)
    return specs
