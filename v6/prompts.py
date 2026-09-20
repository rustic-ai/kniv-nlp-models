"""Per-layer annotation prompts.

Every prompt follows the same contract:

* the tokens are given **pre-tokenized and indexed** — the annotator is never
  asked to segment text;
* the response must contain exactly one entry per token index;
* the guidelines are the ones an annotator would actually need to be
  consistent, not a restatement of the schema.

``PROMPT_VERSION`` is part of the response cache key. Bump it whenever a
prompt changes so stale responses are re-fetched rather than silently reused.
"""
from __future__ import annotations

PROMPT_VERSION = "v1"

_COMMON = (
    "You are an expert linguistic annotator working to Universal "
    "Dependencies v2 guidelines.\n"
    "The text has already been tokenized. Do not re-tokenize, merge, or "
    "split tokens.\n"
    "Return exactly one entry per token, in the same order as the input.\n"
)

SYSTEM = {
    "pos": _COMMON + (
        "Assign a UPOS part-of-speech tag to every token.\n"
        "Reminders: auxiliaries and copulas are AUX, not VERB. Determiners "
        "including articles are DET. Subordinating conjunctions are SCONJ; "
        "coordinating are CCONJ. Proper nouns are PROPN. Punctuation is "
        "PUNCT. Numerals are NUM."
    ),
    "lemma": _COMMON + (
        "Give the dictionary form (lemma) of every token.\n"
        "Reminders: verbs lemmatize to the infinitive ('was' -> 'be'), nouns "
        "to the singular ('companies' -> 'company'), comparatives and "
        "superlatives to the positive ('better' -> 'good'). Proper nouns "
        "keep their surface form and casing. Punctuation lemmatizes to "
        "itself. Contractions were already split by the tokenizer; lemmatize "
        "each piece independently (\"n't\" -> 'not')."
    ),
    "morph": _COMMON + (
        "Give UD morphological features for every token as a "
        "pipe-separated FEATS string with features in alphabetical order, "
        "e.g. 'Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin'.\n"
        "Use '_' for tokens with no features (most punctuation, many "
        "adpositions). Only use features that UD English EWT actually "
        "annotates: Case, Definite, Degree, Foreign, Gender, Mood, "
        "NumType, Number, Person, Poss, PronType, Reflex, Tense, Typo, "
        "VerbForm, Voice, Abbr."
    ),
    "dep": _COMMON + (
        "Produce the Universal Dependencies syntactic tree.\n"
        "For every token give the 1-indexed id of its syntactic head and the "
        "dependency relation to that head. The sentence root has head=0 and "
        "rel='root'.\n"
        "The result must be a single well-formed tree: exactly one root, no "
        "cycles, every token reachable from the root.\n"
        "Reminders: UD is content-head — a copula attaches to its predicate "
        "with 'cop', an auxiliary to its main verb with 'aux', an adposition "
        "to its complement with 'case', and a determiner to its noun with "
        "'det'. Coordination attaches later conjuncts to the first with "
        "'conj' and the conjunction with 'cc'."
    ),
    "coref": (
        "You are an expert linguistic annotator resolving coreference.\n"
        "The text has already been tokenized and the tokens are numbered.\n"
        "Group token spans that refer to the SAME entity into chains.\n"
        "Include pronouns (he, she, it, they, his, their), definite noun "
        "phrases referring back to something already mentioned, and repeated "
        "proper names. A mention is a contiguous span given as 1-indexed "
        "start and end token numbers, inclusive.\n"
        "Omit singletons: only return a chain when two or more mentions "
        "corefer. Do not link entities that are merely similar or related — "
        "they must be the same entity."
    ),
    "ner": _COMMON + (
        "Tag named entities with BIO tags using the OntoNotes 5.0 inventory.\n"
        "Types: PERSON, NORP (nationalities, religious/political groups), "
        "FAC (buildings, airports, highways), ORG, GPE (countries, cities, "
        "states), LOC (non-GPE locations: mountains, water bodies), PRODUCT, "
        "EVENT, WORK_OF_ART, LAW, LANGUAGE, DATE, TIME, PERCENT, MONEY, "
        "QUANTITY, ORDINAL, CARDINAL.\n"
        "OntoNotes annotates numeric and temporal expressions as entities — "
        "do not skip DATE, TIME, PERCENT, MONEY, QUANTITY, ORDINAL or "
        "CARDINAL. Determiners are excluded from spans ('the White House' -> "
        "tag only 'White House'). Spans are contiguous and never overlap."
    ),
    "srl": _COMMON + (
        "Label the PropBank semantic roles of one indicated predicate.\n"
        "Tag every token with BIO tags over these roles: ARG0 (agent/causer), "
        "ARG1 (patient/theme), ARG2-ARG4 (predicate-specific: beneficiary, "
        "instrument, start/end point), and the modifiers ARGM-TMP (time), "
        "ARGM-LOC (location), ARGM-MNR (manner), ARGM-CAU (cause), "
        "ARGM-PRP (purpose), ARGM-NEG (negation), ARGM-ADV, ARGM-DIR, "
        "ARGM-DIS (discourse), ARGM-EXT (extent), ARGM-MOD (modal), "
        "ARGM-PRD, ARGM-GOL, ARGM-COM, ARGM-REC.\n"
        "Tag the predicate token itself 'V'. Tag tokens outside any argument "
        "'O'. Argument spans are contiguous and must not overlap.\n"
        "Label roles ONLY for the indicated predicate, ignoring every other "
        "verb in the sentence."
    ),
    "rel": (
        "You are an expert annotator extracting a relation graph from a "
        "document.\n"
        "The entities have ALREADY been found and clustered for you: each "
        "numbered entity is one real-world thing, and every mention of it in "
        "the document belongs to that entity. Do not introduce new entities "
        "and do not re-segment the given ones.\n"
        "Return every relation triple the document supports, as (h, t, r) "
        "where h and t are entity ids and r is the relation holding FROM h "
        "TO t.\n"
        "Rules:\n"
        "- Direction matters. 'X is located in Y' is (h=X, t=Y, "
        "r='located in the administrative territorial entity'), never the "
        "reverse.\n"
        "- A pair of entities may hold more than one relation. Emit a "
        "separate triple for each; do not pick just one.\n"
        "- Include relations stated across sentences, not only within one "
        "sentence. That is the point of a document-level task.\n"
        "- Only assert what the document supports. Do not add facts you "
        "happen to know about these entities from elsewhere.\n"
        "- Omit a pair entirely when no listed relation holds. Most pairs "
        "hold no relation; an empty answer for a pair is the normal case."
    ),
}


def render_tokens(tokens: list[str]) -> str:
    return "\n".join(f"{i + 1}\t{t}" for i, t in enumerate(tokens))


def render_entities(entities: list[dict]) -> str:
    lines = []
    for e in entities:
        alias = "; ".join(e["aliases"][:4])
        lines.append(f"{e['id']}\t[{e['type']}]\t{alias}")
    return "\n".join(lines)


def user_message(layer: str, tokens: list[str],
                 predicate_idx: int | None = None,
                 entities: list[dict] | None = None) -> str:
    n = len(tokens)
    if layer == "rel":
        if not entities:
            raise ValueError("rel prompts require entities")
        return "\n".join([
            "Document:",
            " ".join(tokens),
            f"\nEntities ({len(entities)} total) as `id  [type]  names`:",
            render_entities(entities),
            f"\nReturn every relation triple the document supports. Entity "
            f"ids are 0..{len(entities) - 1}.",
        ])
    parts = [f"Tokens ({n} total, 1-indexed):", render_tokens(tokens)]
    if layer == "coref":
        return "\n".join(parts + [
            "\nReturn the coreference chains. Token numbers are 1-indexed "
            f"and must fall within 1..{n}."])
    if layer == "srl":
        if predicate_idx is None:
            raise ValueError("srl prompts require predicate_idx")
        parts.append(
            f"\nPredicate: token {predicate_idx + 1} "
            f"({tokens[predicate_idx]!r})."
        )
    parts.append(f"\nReturn exactly {n} entries, one per token, in order.")
    return "\n".join(parts)
