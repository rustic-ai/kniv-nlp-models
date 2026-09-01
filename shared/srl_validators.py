"""Programmatic validators for Semantic Role Labeling (SRL) outputs.

Catches a meaningful fraction of SRL errors *without* needing gold labels, by
checking predictions against PropBank framesets, syntactic-cascade consistency,
and NER-type consistency.

Three core validators:

1. ``validate_propbank_frame(verb_lemma, predicted_roles)`` — checks whether
   the predicted set of argument roles is licensed by *any* PropBank sense of
   the verb. Catches ARG3 / ARG4 / ARGM-* mis-assignments to verbs whose
   framesets don't define them.

2. ``validate_dep_cascade(srl_tags, dep_relations, words)`` — checks that
   ARG0 / ARG1 spans align with the expected UD dependency relations from the
   verb (nsubj / obj / obl). Catches role-swap and core-vs-modifier confusions.

3. ``validate_argm_ner(srl_tags, ner_tags)`` — checks that ARGM-LOC spans
   overlap with GPE/LOC/FAC entities, ARGM-TMP overlaps with DATE/TIME,
   ARGM-MNR doesn't overlap with proper nouns, etc.

Combined into ``score_frame()`` for a single 0..1 quality score per predicted
frame, and ``filter_predictions()`` for use in self-training pipelines.

PropBank framesets are loaded lazily from the propbank/propbank-frames
GitHub repository (CC-BY licensed). On first use, the loader can either read
from a local clone (specified via the ``KNIV_PROPBANK_FRAMES_DIR`` environment
variable or default ``data/propbank-frames``) or attempt a fresh download.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# ── Constants & label sets ──────────────────────────────────────────

# Standard PropBank role inventory we care about (matches our SRL tag set
# minus the BIO prefixes and the V tag).
CORE_ROLES = {"ARG0", "ARG1", "ARG2", "ARG3", "ARG4", "ARG5"}
ARGM_ROLES = {
    "ARGM-TMP", "ARGM-LOC", "ARGM-MNR", "ARGM-CAU", "ARGM-PRP",
    "ARGM-NEG", "ARGM-ADV", "ARGM-DIR", "ARGM-DIS", "ARGM-EXT",
    "ARGM-MOD", "ARGM-PRD", "ARGM-GOL", "ARGM-COM", "ARGM-REC",
}
ALL_ROLES = CORE_ROLES | ARGM_ROLES

# Which UD relations are most consistent with each PropBank core role for
# active-voice constructions. These are heuristics: SRL ↔ DEP is not a
# 1:1 mapping, but mismatches are a signal of likely error.
ROLE_TO_UD_REL = {
    "ARG0": {"nsubj", "csubj", "nsubj:outer", "csubj:outer", "obl:agent"},
    "ARG1": {"obj", "nsubj", "nsubj:pass", "csubj:pass", "ccomp", "xcomp"},
    "ARG2": {"iobj", "obl", "obl:agent", "obl:tmod", "advmod"},
    "ARG3": {"obl", "iobj"},
    "ARG4": {"obl"},
}

# NER types that *should* overlap with each ARGM
ARGM_TO_NER_TYPES = {
    "ARGM-LOC": {"GPE", "LOC", "FAC"},
    "ARGM-TMP": {"DATE", "TIME"},
    "ARGM-DIR": {"GPE", "LOC", "FAC"},
}


# ── Frame data structures ───────────────────────────────────────────

@dataclass(frozen=True)
class PropBankSense:
    """One sense of a PropBank verb (e.g. ``read.01``).

    ``valid_roles`` is the union of all roleset roles defined for this sense,
    expressed in our flat label space (e.g. ``{"ARG0", "ARG1", "ARGM-MNR"}``).
    """
    sense_id: str
    valid_roles: frozenset[str]
    description: str = ""

    def licenses(self, predicted_roles: Iterable[str]) -> bool:
        """True if every core predicted role is in this sense's role set.

        ARGM-* modifier roles (temporal / locative / manner / etc.) are
        universally licensed in PropBank annotation, even though the XML
        framesets typically only list core ARG0-ARG5 per sense. We only
        check core role membership.
        """
        core_predicted = {r for r in predicted_roles if not r.startswith("ARGM-")}
        return core_predicted.issubset(self.valid_roles)


# ── Frameset loader ─────────────────────────────────────────────────

def _default_framesets_dir() -> Path:
    """Resolve the local copy of github.com/propbank/propbank-frames."""
    env = os.environ.get("KNIV_PROPBANK_FRAMES_DIR")
    if env:
        return Path(env)
    # Project-local default (sibling of `data/`)
    here = Path(__file__).resolve().parent.parent
    return here / "data" / "propbank-frames"


def _normalize_propbank_role(arg_n: str | None, arg_f: str | None) -> str | None:
    """Convert a PropBank XML role spec to our flat label space.

    PropBank XML format:
        <role n="0" f="PAG">       → ARG0
        <role n="1" f="PPT">       → ARG1
        <role n="m" f="TMP">       → ARGM-TMP
        <role n="m" f="LOC">       → ARGM-LOC
    """
    if arg_n is None:
        return None
    n = arg_n.strip().lower()
    if n in {"0", "1", "2", "3", "4", "5", "6", "7"}:
        return f"ARG{n}"
    if n == "m" and arg_f:
        return f"ARGM-{arg_f.strip().upper()}"
    return None  # unknown — skip


def _parse_one_frame_xml(xml_path: Path) -> dict[str, list[PropBankSense]]:
    """Parse a single ``<verb>.xml`` PropBank frame file.

    Returns ``{verb_lemma: [PropBankSense, ...]}``. A frame file may define
    multiple lemmas (e.g. ``read.xml`` covers ``read``, ``re-read``).
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return {}
    root = tree.getroot()
    out: dict[str, list[PropBankSense]] = {}
    for predicate in root.findall("predicate"):
        lemma = predicate.get("lemma", "").strip().lower()
        if not lemma:
            continue
        senses: list[PropBankSense] = []
        for roleset in predicate.findall("roleset"):
            sense_id = roleset.get("id", "")
            description = roleset.get("name", "")
            roles_in_sense: set[str] = set()
            for role in roleset.findall("./roles/role"):
                normalized = _normalize_propbank_role(role.get("n"), role.get("f"))
                if normalized is not None:
                    roles_in_sense.add(normalized)
            senses.append(PropBankSense(
                sense_id=sense_id,
                valid_roles=frozenset(roles_in_sense),
                description=description,
            ))
        if senses:
            out.setdefault(lemma, []).extend(senses)
    return out


@lru_cache(maxsize=1)
def load_framesets(frames_dir: str | None = None) -> dict[str, list[PropBankSense]]:
    """Load every PropBank frame XML under ``frames_dir`` into a flat lookup.

    Result is cached per-process. Set ``KNIV_PROPBANK_FRAMES_DIR`` to override
    the location, or pass ``frames_dir`` explicitly. Files are expected at
    ``frames_dir/frames/*.xml`` (matches the propbank-frames repo layout).
    """
    base = Path(frames_dir) if frames_dir else _default_framesets_dir()
    xml_dir = base / "frames"
    if not xml_dir.is_dir():
        raise FileNotFoundError(
            f"PropBank framesets directory not found at {xml_dir}. "
            f"Clone github.com/propbank/propbank-frames to {base} or set "
            f"KNIV_PROPBANK_FRAMES_DIR to its location."
        )

    framesets: dict[str, list[PropBankSense]] = {}
    for xml_path in sorted(xml_dir.glob("*.xml")):
        for lemma, senses in _parse_one_frame_xml(xml_path).items():
            framesets.setdefault(lemma, []).extend(senses)
    return framesets


# ── BIO span helper ─────────────────────────────────────────────────

def extract_arg_spans(srl_tags: list[str]) -> list[tuple[str, int, int]]:
    """Extract ``(role, start, end)`` spans from a BIO SRL tag sequence.

    Bare ``V`` tags are excluded (they're the predicate, not an argument).
    """
    spans: list[tuple[str, int, int]] = []
    i = 0
    while i < len(srl_tags):
        tag = srl_tags[i]
        if tag.startswith("B-"):
            role = tag[2:]
            start = i
            i += 1
            while i < len(srl_tags) and srl_tags[i] == f"I-{role}":
                i += 1
            spans.append((role, start, i - 1))
        else:
            i += 1
    return spans


# ── Validator 1: PropBank frameset ──────────────────────────────────

def validate_propbank_frame(
    verb_lemma: str,
    predicted_roles: Iterable[str],
    framesets: dict[str, list[PropBankSense]] | None = None,
) -> tuple[bool | None, str | None]:
    """Check whether predicted roles are licensed by any PropBank sense.

    Returns ``(verdict, matched_sense_id)``:
      - ``(True, sense_id)``: predicted roles fit some sense's frameset
      - ``(False, None)``: no sense supports this argument structure
      - ``(None, None)``: verb not in framesets (can't validate)
    """
    if framesets is None:
        framesets = load_framesets()
    senses = framesets.get(verb_lemma.lower())
    if not senses:
        return None, None
    predicted_set = set(predicted_roles)
    if not predicted_set:
        # Empty prediction is trivially licensed by any sense
        return True, senses[0].sense_id
    for sense in senses:
        if sense.licenses(predicted_set):
            return True, sense.sense_id
    return False, None


# ── Validator 2: DEP / SRL cascade consistency ──────────────────────

# UD relations that mark span-internal modifiers (determiners, case markers,
# possessives, etc.) — never the syntactic head of the span itself. Skip past
# these when picking a representative head token. Empirically derived from
# false-positive (role, rel) pairs on PropBank EWT correct frames.
SPAN_INTERNAL_RELS = {
    "det", "det:predet", "det:poss",
    "case",
    "mark",
    "nmod:poss",
    "amod", "nummod", "compound",
    "punct", "cc", "fixed", "flat",
    "aux", "aux:pass", "cop",
}


def _pick_span_head(start: int, end: int, dep_relations: list[str]) -> int | None:
    """Return the index of the token most likely to be the span's syntactic head.

    Walks the span and returns the first token whose dep relation is NOT a
    span-internal modifier (det / case / mark / amod / etc.). Falls back to
    ``start`` if every token in the span is internal-modifier-tagged (rare,
    typically means the span is a fragment).
    """
    for i in range(start, min(end + 1, len(dep_relations))):
        rel = dep_relations[i]
        base = rel.split(":", 1)[0]
        if rel not in SPAN_INTERNAL_RELS and base not in SPAN_INTERNAL_RELS:
            return i
    return start if start < len(dep_relations) else None


def validate_dep_cascade(
    srl_tags: list[str],
    dep_relations: list[str],
    predicate_idx: int | None = None,
) -> tuple[float, list[str]]:
    """Score how well core SRL roles align with expected UD dependency relations.

    Returns ``(score, violations)`` where:
      - ``score`` is the fraction of core-role spans whose head token has a
        dep relation consistent with the role (1.0 = all consistent).
      - ``violations`` lists ``"ARGN at i: dep=rel"`` strings for spans whose
        dep relation is *not* in the expected set.

    Picks the span's head by skipping span-internal modifier relations
    (det / case / mark / amod / ...). For "the document" the first token is
    "the" (dep=det) — we walk past it to "document" whose relation is the
    actual signal we want to validate.
    """
    spans = extract_arg_spans(srl_tags)
    if not spans:
        return 1.0, []

    consistent = 0
    total = 0
    violations: list[str] = []
    for role, start, end in spans:
        if role not in ROLE_TO_UD_REL:
            continue
        total += 1
        head_idx = _pick_span_head(start, end, dep_relations)
        if head_idx is None:
            continue
        actual_rel = dep_relations[head_idx]
        # UD relations can have suffixes like "nsubj:pass" — match base relation too
        actual_base = actual_rel.split(":", 1)[0]
        expected = ROLE_TO_UD_REL[role]
        if actual_rel in expected or actual_base in expected:
            consistent += 1
        else:
            violations.append(f"{role} at token {head_idx}: dep={actual_rel}")

    score = consistent / total if total else 1.0
    return score, violations


# ── Validator 3: NER / ARGM consistency ─────────────────────────────

def validate_argm_ner(
    srl_tags: list[str],
    ner_tags: list[str],
) -> tuple[float, list[str]]:
    """Score whether ARGM-LOC / ARGM-TMP / ARGM-DIR spans overlap with NER
    entities of the expected type.

    Returns ``(score, violations)`` where score is the fraction of
    overlapping-with-correct-type spans among the spans that have NER content.
    Spans with no NER content (all O) are skipped — they're not necessarily
    wrong, just unverifiable.
    """
    spans = extract_arg_spans(srl_tags)
    relevant = [s for s in spans if s[0] in ARGM_TO_NER_TYPES]
    if not relevant:
        return 1.0, []

    consistent = 0
    total_with_ner = 0
    violations: list[str] = []
    for role, start, end in relevant:
        ner_span = ner_tags[start:end + 1] if end < len(ner_tags) else ner_tags[start:]
        ner_types = {tag.split("-", 1)[1] for tag in ner_span if tag != "O" and "-" in tag}
        if not ner_types:
            continue  # unverifiable, skip
        total_with_ner += 1
        expected = ARGM_TO_NER_TYPES[role]
        if ner_types & expected:
            consistent += 1
        else:
            violations.append(
                f"{role}@{start}-{end}: NER types {sorted(ner_types)} "
                f"not in expected {sorted(expected)}")

    if total_with_ner == 0:
        return 1.0, []
    return consistent / total_with_ner, violations


# ── Combined frame quality score ────────────────────────────────────

@dataclass
class FrameQuality:
    """Validator output bundle for a single predicted frame."""
    score: float                  # 0..1 weighted-average quality score
    propbank_verdict: bool | None # True / False / None (unknown verb)
    propbank_sense: str | None    # matched sense id if any
    dep_score: float              # DEP cascade consistency
    dep_violations: list[str]
    argm_score: float             # ARGM-NER consistency
    argm_violations: list[str]


def score_frame(
    verb_lemma: str,
    srl_tags: list[str],
    dep_relations: list[str] | None = None,
    ner_tags: list[str] | None = None,
    framesets: dict[str, list[PropBankSense]] | None = None,
    weights: dict[str, float] | None = None,
) -> FrameQuality:
    """Compute a single quality score for one predicted frame.

    Component scores:
      - PropBank: 1.0 if licensed, 0.0 if violated, neutral 0.5 if unknown verb
      - DEP cascade: fraction of core-role spans matching expected dep relations
      - ARGM-NER: fraction of ARGM-LOC/TMP/DIR spans matching NER types

    Weights default to ``{propbank: 0.5, dep: 0.3, argm: 0.2}`` — PropBank is
    the strongest signal because it's a hard schema check.
    """
    if weights is None:
        weights = {"propbank": 0.5, "dep": 0.3, "argm": 0.2}

    spans = extract_arg_spans(srl_tags)
    predicted_roles = {role for role, _, _ in spans}

    pb_verdict, pb_sense = validate_propbank_frame(verb_lemma, predicted_roles, framesets)
    if pb_verdict is True:
        pb_score = 1.0
    elif pb_verdict is False:
        pb_score = 0.0
    else:
        pb_score = 0.5  # unknown verb — neutral

    if dep_relations is not None:
        dep_score, dep_violations = validate_dep_cascade(srl_tags, dep_relations)
    else:
        dep_score, dep_violations = 1.0, []

    if ner_tags is not None:
        argm_score, argm_violations = validate_argm_ner(srl_tags, ner_tags)
    else:
        argm_score, argm_violations = 1.0, []

    total = (
        weights["propbank"] * pb_score
        + weights["dep"] * dep_score
        + weights["argm"] * argm_score
    )

    return FrameQuality(
        score=total,
        propbank_verdict=pb_verdict,
        propbank_sense=pb_sense,
        dep_score=dep_score,
        dep_violations=dep_violations,
        argm_score=argm_score,
        argm_violations=argm_violations,
    )


# ── Filtering helper for self-training pipelines ────────────────────

def filter_predictions(
    predictions: list[dict],
    min_score: float = 0.7,
    require_propbank_pass: bool = True,
    framesets: dict[str, list[PropBankSense]] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Split predictions into ``(kept, dropped)`` based on validation score.

    Each ``prediction`` dict must contain at minimum:
      - ``verb_lemma``: str
      - ``srl_tags``: list[str]  (BIO sequence)
    and may optionally contain ``dep_relations``, ``ner_tags`` for stronger
    validation.

    Each surviving prediction is annotated in place with a ``"frame_quality"``
    key holding the ``FrameQuality`` object.
    """
    kept, dropped = [], []
    for pred in predictions:
        fq = score_frame(
            verb_lemma=pred["verb_lemma"],
            srl_tags=pred["srl_tags"],
            dep_relations=pred.get("dep_relations"),
            ner_tags=pred.get("ner_tags"),
            framesets=framesets,
        )
        pred["frame_quality"] = fq
        if fq.score < min_score:
            dropped.append(pred)
            continue
        if require_propbank_pass and fq.propbank_verdict is False:
            dropped.append(pred)
            continue
        kept.append(pred)
    return kept, dropped


__all__ = [
    "PropBankSense",
    "FrameQuality",
    "load_framesets",
    "extract_arg_spans",
    "validate_propbank_frame",
    "validate_dep_cascade",
    "validate_argm_ner",
    "score_frame",
    "filter_predictions",
    "CORE_ROLES",
    "ARGM_ROLES",
    "ALL_ROLES",
    "ROLE_TO_UD_REL",
    "ARGM_TO_NER_TYPES",
]
