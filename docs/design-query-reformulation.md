# Design: Query Reformulation via Cascade Understanding

## 1. Problem Statement

Query reformulation transforms a user's input into a more effective form
for downstream systems — search engines, RAG pipelines, dialog agents,
or databases. The goal is to preserve intent while improving clarity,
completeness, or structure.

```
User:     "who made that phone company in cupertino"
Reform:   "Who founded Apple Inc.?"
          ├── coreference: "that phone company" → "Apple Inc."
          ├── lexical: "made" → "founded"
          └── entity grounding: "in cupertino" → context-resolved

User:     "I want to cancel my flight to paris next tuesday"
Reform:   {intent: cancel, entity: flight, dest: Paris, date: 2026-05-05}
          └── structured extraction for API call
```

**Why this matters**: LLMs are powerful reformulators but expensive and
opaque. A cascade-based approach provides **deterministic, interpretable
reformulation** at encoder-only cost, with LLM integration as an optional
enhancement layer.

---

## 2. Reformulation Taxonomy

Not all reformulation is the same. Our cascade supports different levels:

| Level | What Changes | Example | Required Heads |
|-------|-------------|---------|----------------|
| **Structural** | Syntax only | Active→passive, question→statement | POS, DEP, SRL |
| **Lexical** | Word choice | "big"→"large", "made"→"founded" | POS, embeddings |
| **Resolution** | Resolve references | "he"→"Steve Jobs", "there"→"Paris" | NER, Coref (future) |
| **Expansion** | Add implicit info | "flights tomorrow" → "flights departing on 2026-04-26" | NER, CLS, domain knowledge |
| **Decomposition** | Split complex query | "capital of country where Apple was founded" → 2 sub-queries | SRL, DEP, QA |
| **Extraction** | Query → structured form | Free text → intent + slots | NER, SRL, CLS |

Our cascade can handle **structural, extraction, and decomposition** today.
**Resolution** needs Coref (planned). **Lexical** and **expansion** benefit
from LLM integration.

---

## 3. Architecture Overview

Query reformulation is NOT a single model head — it's an **orchestration
layer** that composes outputs from multiple cascade heads.

```
                    ┌─────────────────────────────┐
   User query ───► │      kniv-cascade encoder     │
                    │  POS + NER + SRL + DEP + CLS  │
                    └──────────────┬──────────────┘
                                   │ structured analysis
                                   ▼
                    ┌─────────────────────────────┐
                    │   Reformulation Orchestrator  │
                    │                               │
                    │  ┌─────────┐ ┌────────────┐  │
                    │  │ Template │ │ Structured │  │
                    │  │ Engine   │ │ Extractor  │  │
                    │  └─────────┘ └────────────┘  │
                    │  ┌─────────┐ ┌────────────┐  │
                    │  │  Query  │ │    LLM     │  │
                    │  │Decompose│ │ Enhancer   │  │
                    │  └─────────┘ └────────────┘  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    Reformulated query / structured output
```

There are four reformulation engines, each suited to different use cases.
They can be used independently or composed.

---

## 4. Engine 1: SRL-Based Structural Reformulation

Uses SRL predicate-argument frames to restructure sentences deterministically.

### 4.1 SRL Frame Extraction

```
Input:  "The contract was signed by both parties in December."
SRL:    signed(V)
          ARG1  = "The contract"       (patient — what was signed)
          ARG0  = "both parties"       (agent — who signed)
          ARGM-TMP = "in December"     (temporal)

Frame:  {
    predicate: "signed",
    ARG0: "both parties",
    ARG1: "The contract",
    ARGM-TMP: "in December"
}
```

### 4.2 Template-Based Transforms

```python
REFORM_TEMPLATES = {
    # Passive → Active
    "passive_to_active": "{ARG0} {predicate} {ARG1} {ARGM-TMP}",
    # → "Both parties signed the contract in December."

    # Active → Passive
    "active_to_passive": "{ARG1} was {predicate} by {ARG0} {ARGM-TMP}",

    # Statement → Question (who)
    "who_question": "Who {predicate} {ARG1} {ARGM-TMP}?",
    # → "Who signed the contract in December?"

    # Statement → Question (what)
    "what_question": "What did {ARG0} {predicate_base} {ARGM-TMP}?",
    # → "What did both parties sign in December?"

    # Statement → Question (when)
    "when_question": "When did {ARG0} {predicate_base} {ARG1}?",
    # → "When did both parties sign the contract?"

    # Nominalization
    "nominalize": "The {predicate_noun} of {ARG1} by {ARG0} {ARGM-TMP}",
    # → "The signing of the contract by both parties in December"
}
```

### 4.3 Implementation

```python
class SRLReformulator:
    """Deterministic reformulation using SRL frames."""

    def __init__(self, cascade_model):
        self.model = cascade_model

    def extract_frames(self, text):
        """Run cascade and extract SRL frames."""
        outputs = self.model.predict(text)
        frames = []
        for pred_idx, srl_tags in outputs["srl"]:
            frame = {"predicate": outputs["tokens"][pred_idx]}
            current_arg, current_tokens = None, []
            for i, tag in enumerate(srl_tags):
                if tag.startswith("B-"):
                    if current_arg:
                        frame[current_arg] = " ".join(current_tokens)
                    current_arg = tag[2:]
                    current_tokens = [outputs["tokens"][i]]
                elif tag.startswith("I-") and current_arg:
                    current_tokens.append(outputs["tokens"][i])
                else:
                    if current_arg:
                        frame[current_arg] = " ".join(current_tokens)
                    current_arg, current_tokens = None, []
            if current_arg:
                frame[current_arg] = " ".join(current_tokens)
            frames.append(frame)
        return frames

    def reformulate(self, text, target="active"):
        """Reformulate text using SRL frames and templates."""
        frames = self.extract_frames(text)
        if not frames:
            return text  # no predicates found

        frame = frames[0]  # primary predicate
        template = REFORM_TEMPLATES.get(target, "{ARG0} {predicate} {ARG1}")

        # Fill template, skip missing arguments
        result = template
        for arg in ["ARG0", "ARG1", "ARG2", "ARGM-TMP", "ARGM-LOC",
                     "ARGM-MNR", "ARGM-CAU", "predicate", "predicate_base",
                     "predicate_noun"]:
            value = frame.get(arg, "")
            result = result.replace(f"{{{arg}}}", value)

        # Clean up empty slots and extra whitespace
        result = " ".join(result.split())
        return result
```

### 4.4 Supported Transforms

| Transform | Input | Output | Uses |
|-----------|-------|--------|------|
| Passive → Active | "The ball was kicked by John" | "John kicked the ball" | SRL (ARG0, ARG1, V) |
| Active → Passive | "John kicked the ball" | "The ball was kicked by John" | SRL + POS (verb form) |
| Statement → Who-Q | "John kicked the ball" | "Who kicked the ball?" | SRL (drop ARG0, add "Who") |
| Statement → What-Q | "John kicked the ball" | "What did John kick?" | SRL (drop ARG1, add "What") |
| Statement → When-Q | "John left in May" | "When did John leave?" | SRL (drop ARGM-TMP) |
| Nominalization | "John signed the contract" | "John's signing of the contract" | SRL + POS (verb→noun) |
| Simplification | "John, who is a teacher, left" | "John is a teacher. John left." | DEP (relative clause detection) |

---

## 5. Engine 2: Structured Query Extraction

Converts free-text queries into structured intent + slot representations
for API calls, search, or dialog systems.

### 5.1 Extraction Pipeline

```
Input: "I need to book a flight from San Francisco to Tokyo next Friday"

Step 1 — CLS:  intent = "request"
Step 2 — NER:  "San Francisco" = GPE, "Tokyo" = GPE, "next Friday" = DATE
Step 3 — SRL:  book(ARG0="I", ARG1="a flight", ARG-DIR="from SF to Tokyo",
                     ARGM-TMP="next Friday")
Step 4 — DEP:  "from San Francisco" ← obl(book), "to Tokyo" ← obl(book)

Output: {
    "intent": "request",
    "action": "book",
    "slots": {
        "object": "flight",
        "origin": {"text": "San Francisco", "type": "GPE"},
        "destination": {"text": "Tokyo", "type": "GPE"},
        "date": {"text": "next Friday", "type": "DATE"},
    }
}
```

### 5.2 Slot Extraction Rules

Slots are derived by combining NER entities with their SRL/DEP roles:

```python
SLOT_MAPPING = {
    # SRL role → DEP relation → slot name
    ("ARG1", None):           "object",
    ("ARGM-TMP", None):       "date",
    ("ARGM-LOC", None):       "location",
    (None, "obl:tmod"):       "date",
    (None, "obl"):            "location",
}

NER_SLOT_HINTS = {
    # NER type → likely slot type (tiebreaker)
    "DATE":     "date",
    "TIME":     "time",
    "GPE":      "location",
    "MONEY":    "amount",
    "PERSON":   "person",
    "ORG":      "organization",
    "CARDINAL": "quantity",
}

def extract_slots(cascade_output):
    """Combine NER + SRL + DEP to extract structured slots."""
    slots = {}
    tokens = cascade_output["tokens"]

    # Get named entities
    entities = extract_entities(cascade_output["ner"])

    # Get SRL arguments
    for frame in cascade_output["srl_frames"]:
        for role, span_text in frame.items():
            if role in ("V", "predicate"):
                continue
            # Check if span overlaps with a named entity
            matching_entity = find_overlapping_entity(span_text, entities)
            if matching_entity:
                slot_name = NER_SLOT_HINTS.get(matching_entity["type"],
                            SLOT_MAPPING.get((role, None), role))
                slots[slot_name] = {
                    "text": span_text,
                    "type": matching_entity["type"],
                    "srl_role": role,
                }
            else:
                slot_name = SLOT_MAPPING.get((role, None), role)
                slots[slot_name] = {"text": span_text, "srl_role": role}

    return slots
```

### 5.3 Domain-Specific Slot Schemas

For production use, define expected slot schemas per intent:

```python
SLOT_SCHEMAS = {
    "book_flight": {
        "required": ["origin", "destination", "date"],
        "optional": ["time", "class", "passengers"],
    },
    "cancel_reservation": {
        "required": ["reservation_id"],
        "optional": ["reason"],
    },
    "search": {
        "required": ["query"],
        "optional": ["filters", "date_range", "location"],
    },
}

def validate_extraction(intent, slots, schema):
    """Check extracted slots against schema, identify missing."""
    expected = SLOT_SCHEMAS.get(intent, {})
    missing = [s for s in expected.get("required", []) if s not in slots]
    return {"slots": slots, "missing": missing, "complete": len(missing) == 0}
```

---

## 6. Engine 3: Query Decomposition

Breaks complex multi-hop queries into sequential sub-queries using
DEP and SRL analysis.

### 6.1 Complexity Detection

```python
def is_complex_query(cascade_output):
    """Detect queries that need decomposition."""
    dep = cascade_output["dep"]

    # Multiple predicates (multiple verb clauses)
    n_predicates = sum(1 for tag in cascade_output["pos"] if tag == "VERB")
    if n_predicates >= 2:
        return True

    # Relative clauses (acl:relcl in DEP)
    if any("acl:relcl" in str(d) for d in dep["deprels"]):
        return True

    # Nested entities referencing other entities
    # "the capital of the country where X was Y"
    if any(d == "nmod" for d in dep["deprels"]):
        ner_types = set(cascade_output["ner"])
        if len(ner_types - {"O"}) >= 2:
            return True

    return False
```

### 6.2 Decomposition Strategies

**Strategy A: Clause splitting** (DEP-based)

```
Input:  "Find restaurants that serve Italian food and are near downtown"
DEP:    "that serve" ← acl:relcl("restaurants")
        "are near"   ← conj("serve")

Decompose:
  Q1: "Find restaurants that serve Italian food"
  Q2: "Filter for restaurants near downtown"
```

**Strategy B: SRL frame chaining**

```
Input:  "Who directed the movie that won the Oscar in 2024?"

SRL Frame 1: won(ARG1="the movie", ARG2="the Oscar", ARGM-TMP="in 2024")
SRL Frame 2: directed(ARG0=?, ARG1="the movie")

Decompose:
  Q1: "Which movie won the Oscar in 2024?"  → answer: "Oppenheimer"
  Q2: "Who directed Oppenheimer?"           → answer: "Christopher Nolan"
```

**Strategy C: Entity-bridged decomposition**

```
Input:  "What is the population of the country where Toyota is headquartered?"

NER:    "Toyota" = ORG
SRL:    headquartered(ARG1="Toyota", ARGM-LOC=?)
DEP:    "country" ← nmod:poss ← "population"

Decompose:
  Q1: "Where is Toyota headquartered?" → "Japan"
  Q2: "What is the population of Japan?" → "125 million"
```

### 6.3 Implementation

```python
class QueryDecomposer:
    """Break complex queries into sub-queries using cascade analysis."""

    def __init__(self, cascade_model):
        self.model = cascade_model

    def decompose(self, query):
        """Returns list of sub-queries in execution order."""
        output = self.model.predict(query)

        if not is_complex_query(output):
            return [query]  # simple query, no decomposition

        sub_queries = []

        # Strategy B: SRL frame chaining
        frames = output["srl_frames"]
        if len(frames) >= 2:
            # Order frames by dependency (inner clause first)
            for frame in reversed(frames):
                # Find the missing argument (the "?" in the frame)
                missing = self._find_missing_arg(frame, output)
                if missing:
                    sub_q = self._frame_to_question(frame, missing)
                    sub_queries.append({
                        "query": sub_q,
                        "expected_type": self._expected_answer_type(missing),
                        "bridges_to": missing,  # plug answer into next query
                    })
            return sub_queries

        # Strategy A: Clause splitting via DEP
        clauses = self._split_clauses(output)
        return [{"query": c, "expected_type": None} for c in clauses]

    def _find_missing_arg(self, frame, output):
        """Identify which argument is being asked about."""
        question_words = {"who", "what", "where", "when", "which", "how"}
        tokens_lower = [t.lower() for t in output["tokens"]]
        for qw in question_words:
            if qw in tokens_lower:
                return {"who": "ARG0", "what": "ARG1", "where": "ARGM-LOC",
                        "when": "ARGM-TMP", "which": "ARG1",
                        "how": "ARGM-MNR"}.get(qw, "ARG1")
        return None

    def _frame_to_question(self, frame, target_role):
        """Convert SRL frame to a question targeting a specific role."""
        wh_word = {"ARG0": "Who", "ARG1": "What", "ARGM-LOC": "Where",
                   "ARGM-TMP": "When", "ARGM-MNR": "How"}.get(target_role, "What")
        predicate = frame.get("predicate", "")
        # Build question from remaining arguments
        args = " ".join(v for k, v in frame.items()
                       if k not in ("predicate", "V", target_role) and v)
        return f"{wh_word} {predicate} {args}?".strip()
```

---

## 7. Engine 4: LLM-Enhanced Reformulation

For cases requiring world knowledge, lexical creativity, or fluency
beyond what templates provide, the cascade feeds structured analysis
to an LLM.

### 7.1 Cascade-Guided Prompting

Instead of asking an LLM to reformulate raw text (expensive, unreliable),
we provide the cascade analysis as structured context:

```python
def cascade_guided_reformulation(query, cascade_output, llm_client):
    """Use cascade analysis to guide LLM reformulation."""

    prompt = f"""Reformulate this query for a search engine.

Query: {query}

Analysis (from NLP pipeline):
- Intent: {cascade_output['cls']}
- Entities: {format_entities(cascade_output['ner'])}
- Predicate: {cascade_output['srl_frames'][0]['predicate']}
- Arguments: {format_args(cascade_output['srl_frames'][0])}
- POS pattern: {' '.join(cascade_output['pos'])}

Rules:
1. Preserve all named entities exactly
2. Resolve pronouns if possible from context
3. Expand abbreviations
4. Keep it concise (under 15 words)

Reformulated query:"""

    return llm_client.generate(prompt, max_tokens=50)
```

### 7.2 When to Use LLM vs Templates

```python
def select_reformulation_engine(cascade_output):
    """Route to the right engine based on query complexity."""

    # Template: structural transforms (voice, question type)
    if needs_structural_transform(cascade_output):
        return "srl_template"

    # Extraction: intent + slot queries
    if cascade_output["cls"] in ("request", "question"):
        if has_clear_entities(cascade_output):
            return "structured_extraction"

    # Decomposition: multi-hop queries
    if is_complex_query(cascade_output):
        return "decomposition"

    # LLM: everything else (ambiguous, needs world knowledge)
    return "llm_enhanced"
```

### 7.3 Cost Comparison

| Engine | Latency | Cost | Deterministic | Needs LLM |
|--------|---------|------|---------------|-----------|
| SRL Template | ~50ms | Cascade inference only | Yes | No |
| Structured Extraction | ~50ms | Cascade inference only | Yes | No |
| Query Decomposition | ~50ms | Cascade inference only | Yes | No |
| LLM Enhanced | ~500ms | Cascade + LLM call | No | Yes |

The cascade handles 70-80% of reformulation cases deterministically.
LLM is the fallback for the remaining 20-30%.

---

## 8. Search Query Reformulation (RAG Use Case)

The most immediate production use case: improving queries for
retrieval-augmented generation (RAG) pipelines.

### 8.1 Problem

RAG retrieval quality depends heavily on query quality. User queries are
often vague, conversational, or poorly structured for vector search:

```
User:  "what did they decide about the budget thing"
Issue: "they" = unresolved, "thing" = vague, conversational tone
```

### 8.2 Cascade-Powered RAG Query Pipeline

```
User query
    │
    ▼
┌─────────────────────────────────┐
│  1. Cascade Analysis             │
│     CLS: question                │
│     NER: (none detected)         │
│     SRL: decide(ARG0="they",     │
│          ARG1="about the budget  │
│          thing")                 │
│     DEP: "budget" ← compound     │
│          ("thing")               │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  2. Reformulation                │
│     a. Entity resolution         │
│        (needs conversation       │
│         context for "they")      │
│     b. Keyword extraction        │
│        → "decide", "budget"      │
│     c. SRL-guided expansion      │
│        → "decision about budget" │
└──────────────┬──────────────────┘
               │
               ▼
┌─────────────────────────────────┐
│  3. Multi-Query Retrieval        │
│     q1: "budget decision"        │
│     q2: "decide about budget"    │
│     q3: original query           │
│     → retrieve with all 3,       │
│       reciprocal rank fusion     │
└──────────────┬──────────────────┘
               │
               ▼
         Retrieved passages
```

### 8.3 Implementation

```python
class RAGQueryReformulator:
    """Reformulate queries for better RAG retrieval."""

    def __init__(self, cascade_model):
        self.model = cascade_model

    def reformulate_for_retrieval(self, query, conversation_context=None):
        """Generate multiple retrieval queries from user input."""
        output = self.model.predict(query)
        queries = [query]  # always include original

        # Extract key terms (NER entities + SRL predicates + head nouns)
        key_terms = self._extract_key_terms(output)
        if key_terms:
            queries.append(" ".join(key_terms))

        # SRL-based reformulation (if has predicate)
        if output["srl_frames"]:
            frame = output["srl_frames"][0]
            # Declarative form: "ARG0 predicate ARG1"
            decl = self._frame_to_declarative(frame)
            if decl and decl != query:
                queries.append(decl)

        # Question → statement (for embedding similarity)
        if output["cls"] == "question":
            statement = self._question_to_statement(output)
            if statement:
                queries.append(statement)

        return queries

    def _extract_key_terms(self, output):
        """Extract the most important terms for retrieval."""
        terms = []
        # Named entities
        for entity in extract_entities(output["ner"]):
            terms.append(entity["text"])
        # Main verb (predicate)
        for frame in output["srl_frames"]:
            terms.append(frame["predicate"])
        # Head nouns from DEP (nsubj, obj, obl)
        for i, rel in enumerate(output["dep"]["deprels"]):
            if rel in ("nsubj", "obj", "obl", "nsubj:pass"):
                terms.append(output["tokens"][i])
        return list(dict.fromkeys(terms))  # deduplicate, preserve order
```

---

## 9. Evaluation

### 9.1 Intrinsic Evaluation

| Metric | Measures | How |
|--------|----------|-----|
| **Structural correctness** | Template output is grammatical | Manual review of 200 examples |
| **Slot extraction F1** | Extracted slots match gold | Annotate 500 queries with gold slots |
| **Decomposition accuracy** | Sub-queries are valid + complete | Manual review of 100 complex queries |

### 9.2 Extrinsic Evaluation (RAG)

| Metric | Measures | How |
|--------|----------|-----|
| **Retrieval Recall@k** | Does reformulated query retrieve the right passage? | Compare original vs reformulated on NQ/SQuAD contexts |
| **End-to-end QA accuracy** | Does better retrieval → better answers? | RAG pipeline with/without reformulation |
| **Query diversity** | Do multi-query reformulations cover more aspects? | Measure unique passages retrieved |

### 9.3 Benchmarks

| Dataset | Task | License | Notes |
|---------|------|---------|-------|
| **MS MARCO** | Query reformulation for search | Research only | 1M queries + reformulations |
| **Natural Questions** | Question + answer passages | CC-BY-SA-3.0 | Can measure retrieval improvement |
| **CANARD** | Conversational query reformulation | CC-BY-SA-4.0 | Resolves conversational references |
| **QReCC** | Query rewriting in context | MIT | 14K conversations with rewrites |

CANARD (CC-BY-SA-4.0) and QReCC (MIT) are commercially usable and directly
relevant — they contain conversational queries with gold reformulations.

---

## 10. Implementation Plan

### Phase 1: SRL Templates (Available Now)

No new model training needed. Implement `SRLReformulator` using existing
cascade heads:

```
Effort:    3-5 days of engineering
Requires:  Working cascade model (POS + NER + SRL + DEP + CLS)
Covers:    Structural transforms, voice changes, question generation
```

### Phase 2: Structured Extraction (Available Now)

No new training needed. Implement `StructuredExtractor` combining
NER + SRL + CLS outputs:

```
Effort:    3-5 days of engineering
Requires:  Working cascade model + domain slot schemas
Covers:    Intent + slot extraction for dialog/search
```

### Phase 3: Query Decomposition (Available Now)

No new training needed. Implement `QueryDecomposer` using DEP + SRL:

```
Effort:    5-7 days of engineering (more complex logic)
Requires:  Working cascade model + QA head (for answering sub-queries)
Covers:    Multi-hop question decomposition
```

### Phase 4: RAG Query Reformulation

Combines all engines + evaluation on retrieval benchmarks:

```
Effort:    1-2 weeks
Requires:  Phases 1-3 + retrieval pipeline for evaluation
Covers:    Multi-query generation, keyword extraction, query expansion
Data:      CANARD (CC-BY-SA-4.0), QReCC (MIT) for training/eval
```

### Phase 5: LLM Integration (Optional Enhancement)

Add LLM fallback for cases templates can't handle:

```
Effort:    3-5 days
Requires:  LLM API access
Covers:    Lexical reformulation, world knowledge, fluency
```

---

## 11. Architecture Integration

Query reformulation sits **on top of** the cascade — it consumes head
outputs but doesn't add new model weights (except optionally in Phase 4):

```
                        model.pt (unchanged)
                    ┌──────────────────────┐
                    │  Encoder + 5 heads    │
                    │  POS NER SRL DEP CLS  │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  reformulation/       │  ← new Python package
                    │    templates.py       │     (no model weights)
                    │    extraction.py      │
                    │    decomposition.py   │
                    │    rag.py             │
                    │    router.py          │
                    └──────────────────────┘
```

The reformulation engines are **pure Python** — they take cascade outputs
as dicts and return reformulated strings or structured data. No GPU needed
beyond the initial cascade inference.

### API Surface

```python
from kniv import CascadeModel, Reformulator

model = CascadeModel.load("models/kniv-v4")
reform = Reformulator(model)

# Structural reformulation
reform.to_active("The ball was kicked by John")
# → "John kicked the ball"

reform.to_question("John kicked the ball", target="who")
# → "Who kicked the ball?"

# Structured extraction
reform.extract_slots("Book a flight from SF to Tokyo next Friday")
# → {"intent": "request", "action": "book", "object": "flight",
#     "origin": "SF", "destination": "Tokyo", "date": "next Friday"}

# Query decomposition
reform.decompose("Who directed the movie that won the Oscar in 2024?")
# → [{"query": "Which movie won the Oscar in 2024?", "expected_type": "WORK_OF_ART"},
#    {"query": "Who directed {answer}?", "expected_type": "PERSON"}]

# RAG multi-query
reform.for_retrieval("what did they decide about the budget")
# → ["what did they decide about the budget",
#    "budget decision",
#    "decide about budget"]
```

---

## 12. Relation to QA Head

Query reformulation and extractive QA are complementary:

```
                  ┌──────────────────────┐
                  │   Query Understanding │
                  │   (Reformulation)     │
                  │                       │
  User query ───►│  "What is the capital │
                  │   of France?"         │
                  │                       │
                  │  CLS: question         │
                  │  NER: France=GPE       │
                  │  SRL: ?(ARG1=capital,  │
                  │       ARGM=France)     │
                  └──────────┬────────────┘
                             │ structured query
                             ▼
                  ┌──────────────────────┐
                  │   Retrieval           │
                  │   (RAG or DB lookup)  │
                  └──────────┬────────────┘
                             │ context passages
                             ▼
                  ┌──────────────────────┐
                  │   Answer Extraction   │
                  │   (QA Head)           │
                  │                       │
                  │  "Paris"              │
                  │  type: GPE            │
                  │  role: ARG1           │
                  └──────────┬────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Answer Reformulation│
                  │   (SRL Templates)     │
                  │                       │
                  │  "The capital of      │
                  │   France is Paris."   │
                  └──────────────────────┘
```

The full pipeline: **understand → retrieve → extract → reformulate**.
All four stages use the same cascade model.

---

## 13. Open Questions

1. **Coreference for resolution**: "they decided about the budget" requires
   knowing who "they" refers to. This needs the Coref head (planned).
   Until then, conversation context must be provided explicitly.

2. **Temporal normalization**: "next Tuesday" → "2026-04-29" requires
   a date resolver. Not an ML task — rule-based with `dateutil` or
   similar. Should be part of the extraction pipeline.

3. **Lexical paraphrase without LLM**: Can we build a lightweight
   synonym/paraphrase head on the encoder? E.g., predict that "founded"
   and "established" are interchangeable in context. This would reduce
   LLM dependency for lexical reformulation.

4. **Evaluation gap**: There's no standard benchmark for cascade-powered
   reformulation specifically. We'll need to build our own eval set or
   adapt CANARD/QReCC for our structured output format.

5. **Multi-lingual**: The reformulation templates are English-specific.
   A multilingual version would need language-specific templates or
   a cross-lingual approach. SRL frames are language-agnostic in
   principle but templates are not.
