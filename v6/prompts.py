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
    "cls": (
        "You label the communicative function of ONE sentence, shown in the "
        "context of the surrounding message.\n"
        "The six labels are ISO 24617-2 general-purpose communicative "
        "functions, flattened across dimensions:\n"
        "- Question: the speaker seeks information they do not have. "
        "Rhetorical and checking questions count.\n"
        "- Inform: propositional content is asserted. ANSWERS are Inform. "
        "Agreement and disagreement about content are Inform, not Feedback.\n"
        "- Directive: the speaker tries to get the ADDRESSEE to act — "
        "request, instruct, suggest. Includes polite interrogative forms "
        "like 'Can you send the report?'.\n"
        "- Commissive: the SPEAKER commits to act — offer, promise, or "
        "accepting/refusing a request. A bare 'Sure.' after a request is "
        "Commissive.\n"
        "- Feedback: the speaker signals their own uptake of what was said "
        "— 'mm-hm', 'ok', 'sorry what?'.\n"
        "- Social: a social obligation is discharged — greeting, goodbye, "
        "thanking, apology, congratulation.\n"
        "\n"
        "Multi-label: return EVERY function the sentence performs. A tag "
        "question over an assertion is Question AND Inform. 'Got it, "
        "thanks.' is Feedback AND Social.\n"
        "Return an EMPTY list when none applies — fillers and fragments "
        "like 'um, so, yeah' have no general-purpose function. Do not "
        "invent a label to avoid an empty answer.\n"
        "\n"
        "The distinction that matters most: Feedback is about the "
        "COMMUNICATION, Inform (agreement) is about the CONTENT. 'Right.' "
        "as a backchannel is Feedback; 'Right, it shipped Tuesday' is "
        "Inform."
    ),
    "sentiment": (
        "You label the sentiment ONE sentence expresses, shown in the "
        "context of the surrounding message.\n"
        "positive, negative, or neutral. Judge the sentiment the speaker "
        "conveys, not the subject matter: a calm factual report of bad news "
        "is neutral, and sarcasm is negative however positive its wording."
    ),
    "keywords": (
        "You extract the salient terms of a passage.\n"
        "Between three and ten, ordered most to least salient. Use the "
        "passage's OWN wording — do not translate, generalise, or introduce "
        "terms it does not contain. Prefer multi-word terms where the "
        "passage uses them. Skip function words and boilerplate."
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


SYSTEM["discover"] = (
    "You are building an inventory of relation types for a general-purpose "
    "information-extraction model that reads conversations, news, technical "
    "documentation, business documents, fiction and encyclopedia articles.\n\n"
    "Read the passage and list the TYPES of relation it expresses between "
    "entities -- people, organizations, places, products, software, works, "
    "events, concepts, times and quantities. A relation type is general and "
    "reusable across documents: name it as a short lowercase phrase read from "
    "head to tail ('works for', 'located in', 'depends on', 'prefers', "
    "'parent of'), never with the passage's own names in it. Give a one-"
    "sentence general definition, the kinds of entity at each end, and the "
    "instance from this passage as 'head -> tail'.\n\n"
    "List only relations the passage actually states or clearly implies. "
    "Prefer the most specific type that is still general ('employed by' over "
    "'related to'). Skip relations that are just grammar ('subject of'). "
    "Return at most 15, or an empty list if the passage expresses none."
)


SYSTEM["bankmerge"] = (
    "You are consolidating a relation inventory for a general-purpose "
    "information-extraction model. You will receive relation names that a "
    "reader proposed while reading passages from six domains (conversation, "
    "news, technical, business, fiction, encyclopedia), each with how often "
    "it was proposed, in which domains, and one proposed definition.\n\n"
    "Produce a consolidated inventory of roughly 150 to 250 relations:\n"
    "- Merge synonyms and near-synonyms into one relation; list the merged "
    "names as aliases. Keep relations distinct when a user would want to "
    "tell them apart (employed by vs founded).\n"
    "- Keep relations general and reusable across documents; drop ones that "
    "are one-off, too vague ('related to', 'associated with'), purely "
    "grammatical, or that describe the text rather than the world.\n"
    "- Prefer relations proposed often or in several domains, but keep a rare "
    "one if it is clearly general and useful (kinship, ownership, dependency).\n"
    "- Name each as a short lowercase phrase read from head to tail. Write a "
    "one-sentence general definition that would let a reader decide whether "
    "the relation holds. Give typical head and tail entity kinds, a category, "
    "and its inverse's name when the inverse is also in the inventory.\n"
    "- Cover every domain: people and their relationships, preferences and "
    "opinions, organizations, places, creative works, events, software and "
    "technical systems, business and finance, causes and effects."
)


SYSTEM["kg"] = (
    "You extract knowledge from text for a general-purpose information-"
    "extraction model. You are given a passage, split into numbered "
    "sentences, and a schema: the entity types and relations to extract, "
    "each with a definition. Extract only what the schema lists.\n\n"
    "ENTITIES. Find every entity of a listed type. One entity per real-world "
    "referent: merge all the mentions of the same thing. For each mention give "
    "its sentence number and the text exactly as it is written there. Include "
    "names and descriptive phrases ('the company', 'my sister'); do NOT list "
    "pronouns as mentions. If participants are listed, they are already "
    "entities 0..k-1: do not repeat them; number your entities from k.\n\n"
    "FACTS. Extract every listed relation the text states or clearly implies "
    "between two entities, head to tail as the relation's definition reads. "
    "Be exhaustive within the schema, but never add what the text does not "
    "support. A participant may be a head or tail ('I love jazz' -> the "
    "speaker likes jazz). For each fact:\n"
    "- polarity: negative if the text says the relation does NOT hold "
    "('I don't eat meat'), else positive.\n"
    "- modality: asserted (stated as true), hypothetical (if/would/might), "
    "planned (intended or scheduled), desired (wished or wanted), questioned "
    "(asked about), reported (someone else is said to claim it).\n"
    "- holder: the entity whose claim or belief it is -- the speaker in a "
    "conversation, the quoted source in reported speech -- or -1 for the "
    "author or narrator.\n"
    "- time: the id of the time entity the fact is anchored to, or -1.\n"
    "- evidence: the sentence numbers that support it."
)


def kg_user_message(sentences: list[str], speakers: list[str | None] | None,
                    participants: list[str], types: list[dict], relations: list[dict]) -> str:
    """One window for the ``kg`` layer: numbered sentences (with speakers in a
    conversation), the participants, and the schema with definitions."""
    lines = ["Entity types:"]
    lines += [f"- {t['name']}: {t['description']}" for t in types]
    lines += ["", "Relations (head -> tail):"]
    lines += [f"- {r['name']}: {r['description']}" for r in relations]
    if participants:
        lines += ["", "Participants (already entities):"]
        lines += [f"{i}\t{p}" for i, p in enumerate(participants)]
    lines += ["", "Passage:"]
    for j, sent in enumerate(sentences):
        who = f"[{speakers[j]}] " if speakers and speakers[j] else ""
        lines.append(f"{j + 1}\t{who}{sent}")
    lines += ["", f"Entity ids you add start at {len(participants)}."]
    return "\n".join(lines)


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
                 entities: list[dict] | None = None,
                 context: str | None = None,
                 target: str | None = None) -> str:
    n = len(tokens)
    if layer in ("cls", "sentiment"):
        # One sentence at a time, with its window as context. The v5 CLS head
        # reads 0.951 in-domain and 0.613 in the wild partly because it saw
        # one utterance plus at most one predecessor; the label is still per
        # sentence, but the evidence is the whole message.
        parts = []
        if context:
            parts.append(f"Context (the surrounding message):\n{context}\n")
        parts.append(f"Sentence to label:\n{target or ' '.join(tokens)}")
        return "\n".join(parts)
    if layer == "keywords":
        return f"Passage:\n{' '.join(tokens)}\n\nReturn its salient terms."
    if layer == "bankmerge":
        return f"Proposed relations (name | count | domains | definition):\n{target}"
    if layer == "discover":
        head = f"Domain: {context}\n\n" if context else ""
        return f"{head}Passage:\n{target or ' '.join(tokens)}\n\nList the relation types it expresses."
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
