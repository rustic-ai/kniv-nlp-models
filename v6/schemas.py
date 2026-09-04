"""JSON schemas for structured annotator output, one per layer.

Design rule: **the annotator never re-tokenizes.** It receives an indexed
token list and must return exactly one entry per index. Length and range are
validated in :mod:`v6.annotate.base`; a mismatch is recorded as a hard
failure and never padded or coerced to a default label. Silent coercion is
how a corpus quietly fills with wrong labels that look clean.
"""
from __future__ import annotations

UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
    "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]

# UD English EWT deprel inventory (same list the v5 biaffine head uses).
DEPRELS = [
    "root", "acl", "acl:relcl", "advcl", "advcl:relcl", "advmod", "amod",
    "appos", "aux", "aux:pass", "case", "cc", "cc:preconj", "ccomp",
    "compound", "compound:prt", "conj", "cop", "csubj", "csubj:outer",
    "csubj:pass", "dep", "det", "det:predet", "discourse", "dislocated",
    "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod",
    "nmod:desc", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj",
    "nsubj:outer", "nsubj:pass", "nummod", "obj", "obl", "obl:agent",
    "obl:npmod", "obl:tmod", "orphan", "parataxis", "punct", "reparandum",
    "vocative", "xcomp",
]

SRL_ROLES = [
    "ARG0", "ARG1", "ARG2", "ARG3", "ARG4",
    "ARGM-TMP", "ARGM-LOC", "ARGM-MNR", "ARGM-CAU", "ARGM-PRP", "ARGM-NEG",
    "ARGM-ADV", "ARGM-DIR", "ARGM-DIS", "ARGM-EXT", "ARGM-MOD", "ARGM-PRD",
    "ARGM-GOL", "ARGM-COM", "ARGM-REC",
]
SRL_TAGS = ["O", "V"] + [f"{p}-{r}" for r in SRL_ROLES for p in ("B", "I")]

# OntoNotes 5.0 18-type inventory — the schema v5's NER head was trained on.
NER_TYPES = [
    "PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY",
    "QUANTITY", "ORDINAL", "CARDINAL",
]
NER_LABELS = ["O"] + [f"{p}-{t}" for t in NER_TYPES for p in ("B", "I")]


def _array_of(items: dict, description: str) -> dict:
    return {"type": "array", "description": description, "items": items}


def _envelope(name: str, payload: dict) -> dict:
    """Wrap a payload in the strict-mode envelope the APIs require."""
    return {
        "name": name,
        "strict": True,
        "schema": {
            "type": "object",
            "properties": payload,
            "required": list(payload),
            "additionalProperties": False,
        },
    }


POS_SCHEMA = _envelope("pos_tags", {
    "tags": _array_of(
        {"type": "string", "enum": UPOS_TAGS},
        "One UPOS tag per input token, in order. Length must equal the "
        "number of input tokens.",
    ),
})

LEMMA_SCHEMA = _envelope("lemmas", {
    "lemmas": _array_of(
        {"type": "string"},
        "One lemma per input token, in order. Preserve the token's own "
        "casing conventions for proper nouns. Length must equal the number "
        "of input tokens.",
    ),
})

MORPH_SCHEMA = _envelope("morph_features", {
    "feats": _array_of(
        {"type": "string"},
        "One UD FEATS string per input token, in order, e.g. "
        "'Number=Sing|Person=3|Tense=Past'. Use '_' when the token has no "
        "features. Length must equal the number of input tokens.",
    ),
})

DEP_SCHEMA = _envelope("dependency_tree", {
    "arcs": _array_of(
        {
            "type": "object",
            "properties": {
                "head": {
                    "type": "integer",
                    "description": "1-indexed head token id; 0 for the "
                                   "sentence root.",
                },
                "rel": {"type": "string", "enum": DEPRELS},
            },
            "required": ["head", "rel"],
            "additionalProperties": False,
        },
        "One arc per input token, in order. Exactly one token must have "
        "head=0 and rel='root'. Length must equal the number of input tokens.",
    ),
})

SRL_SCHEMA = _envelope("srl_tags", {
    "tags": _array_of(
        {"type": "string", "enum": SRL_TAGS},
        "One BIO tag per input token, in order, for the single indicated "
        "predicate. The predicate token itself is tagged 'V'. Length must "
        "equal the number of input tokens.",
    ),
})

NER_SCHEMA = _envelope("ner_tags", {
    "tags": _array_of(
        {"type": "string", "enum": NER_LABELS},
        "One BIO tag per input token, in order, using the OntoNotes 5.0 "
        "entity types. Length must equal the number of input tokens.",
    ),
})

COREF_SCHEMA = _envelope("coref_clusters", {
    "clusters": _array_of(
        {
            "type": "array",
            "description": "One coreference chain: two or more mentions that "
                           "refer to the same entity.",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer",
                              "description": "1-indexed first token of the mention"},
                    "end": {"type": "integer",
                            "description": "1-indexed last token, inclusive"},
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            },
        },
        "Coreference chains. Omit singletons — a chain needs at least two "
        "mentions. Mentions are contiguous token spans.",
    ),
})

SCHEMAS = {
    "ner": NER_SCHEMA,
    "coref": COREF_SCHEMA,
    "pos": POS_SCHEMA,
    "lemma": LEMMA_SCHEMA,
    "morph": MORPH_SCHEMA,
    "dep": DEP_SCHEMA,
    "srl": SRL_SCHEMA,
}

# Which key inside the returned object holds the per-token array.
PAYLOAD_KEY = {
    "coref": "clusters",
    "ner": "tags",
    "pos": "tags", "lemma": "lemmas", "morph": "feats",
    "dep": "arcs", "srl": "tags",
}
