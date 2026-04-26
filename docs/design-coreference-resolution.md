# Design: Coreference Resolution for kniv-cascade (Rust Runtime)

## 1. Problem Statement

Coreference resolution identifies expressions in text that refer to the
same real-world entity. It operates at the **document level**, linking
mentions across sentences.

```
"John went to the store. He bought milk there."
  ^^^^                    ^^              ^^^^^
  │                       │               │
  └───── coref cluster 1 ─┘               │
                                           │
          the store ────── coref cluster 2 ─┘
```

**Constraints**:
- Runtime is Rust (no Python dependencies in production)
- Must work with our ONNX-exported cascade model
- Document-level, not sentence-level
- Should leverage existing cascade outputs (POS, NER, DEP, Morph)

---

## 2. Architecture: Three-Tier Approach

Rather than a single solution, we implement three tiers that can be
deployed independently or composed:

```
Tier 1: Rule-based resolver (Rust, no model)
        ↓ handles 60-70% of cases
Tier 2: Lightweight neural mention-pair scorer (ONNX)
        ↓ handles 85-90% of cases
Tier 3: Full span-ranking coref head on cascade encoder (ONNX)
        ↓ handles 95%+ of cases

Production: Tier 1 + Tier 2 (fast, accurate, Rust-native)
Research:   Tier 3 (best quality, higher latency)
```

---

## 3. Tier 1: Rule-Based Coreference (Rust Native)

A deterministic resolver that uses cascade outputs to link pronouns
and definite descriptions to their antecedents. No neural model needed.

### 3.1 Input: Cascade Outputs

The rule-based resolver consumes structured output from our cascade:

```rust
struct Token {
    text: String,
    pos: PosTag,          // PRON, PROPN, NOUN, etc.
    ner: NerTag,          // PERSON, ORG, GPE, O, etc.
    dep_rel: DepRel,      // nsubj, obj, obl, etc.
    dep_head: usize,      // head token index
    morph: MorphFeatures,  // Gender, Number, Person
    lemma: String,
}

struct Sentence {
    tokens: Vec<Token>,
    srl_frames: Vec<SrlFrame>,
}

struct Document {
    sentences: Vec<Sentence>,
}
```

### 3.2 Mention Detection

Identify candidate mentions from cascade outputs:

```rust
enum MentionType {
    Pronoun,         // he, she, it, they, his, her, their, ...
    ProperNoun,      // Steve Jobs, Apple, Paris (NER != O)
    NominalPhrase,   // the company, the president, a man
    Demonstrative,   // this, that, these, those
    Reflexive,       // himself, herself, themselves
    Relative,        // who, which, that (in relative clauses)
}

fn detect_mentions(sentence: &Sentence) -> Vec<Mention> {
    let mut mentions = vec![];
    for (i, token) in sentence.tokens.iter().enumerate() {
        match token.pos {
            // Pronouns — always a mention
            POS::PRON => mentions.push(Mention {
                span: (i, i),
                mention_type: classify_pronoun(token),
                head_token: i,
                features: extract_features(token),
            }),

            // Named entities — span from B- to last I-
            _ if token.ner.starts_with("B-") => {
                let end = find_entity_end(sentence, i);
                mentions.push(Mention {
                    span: (i, end),
                    mention_type: MentionType::ProperNoun,
                    head_token: find_head_in_span(sentence, i, end),
                    features: extract_ner_features(sentence, i, end),
                });
            },

            // Nominal phrases — nouns with determiner
            POS::NOUN if has_determiner(sentence, i) => {
                let span = get_np_span(sentence, i);  // use DEP subtree
                mentions.push(Mention {
                    span,
                    mention_type: MentionType::NominalPhrase,
                    head_token: i,
                    features: extract_features(token),
                });
            },
            _ => {}
        }
    }
    mentions
}
```

### 3.3 Resolution Rules

Ordered by priority (highest first):

**Rule 1: Reflexive pronouns** → bind to subject of same clause
```
"John hurt himself"  →  himself = John
Logic: PRON(reflexive) → find nsubj in same DEP subtree
```

**Rule 2: Relative pronouns** → bind to NP being modified
```
"The man who left"  →  who = the man
Logic: PRON(relative) → find head of acl:relcl
```

**Rule 3: Personal pronouns** → nearest compatible antecedent
```
"Mary said she would go"  →  she = Mary
Logic: PRON(3rd, fem, sing) → nearest PERSON or NOUN matching gender/number
```

**Rule 4: Demonstrative/definite NPs** → string match or hypernym
```
"Apple released a phone. The company reported..."  →  the company = Apple
Logic: "the company" + NER(ORG) nearby → match ORG entity
```

**Rule 5: Possessive pronouns** → same as personal pronouns
```
"John lost his wallet"  →  his = John
Logic: PRON(poss, masc, sing) → nearest compatible entity
```

### 3.4 Feature Matching

```rust
struct MentionFeatures {
    gender: Option<Gender>,      // from Morph head (Masc, Fem, Neut)
    number: Option<Number>,      // from Morph head (Sing, Plur)
    person: Option<Person>,      // from Morph head (1, 2, 3)
    ner_type: Option<NerType>,   // from NER head
    animacy: Animacy,            // derived: PERSON→animate, ORG→inanimate
    dep_role: DepRel,            // from DEP head (nsubj, obj, etc.)
}

fn features_compatible(mention: &MentionFeatures, antecedent: &MentionFeatures) -> bool {
    // Gender must match (if both known)
    if let (Some(g1), Some(g2)) = (mention.gender, antecedent.gender) {
        if g1 != g2 { return false; }
    }
    // Number must match
    if let (Some(n1), Some(n2)) = (mention.number, antecedent.number) {
        if n1 != n2 { return false; }
    }
    // Animacy must match (he/she → animate, it → inanimate)
    if mention.animacy != Animacy::Unknown && antecedent.animacy != Animacy::Unknown {
        if mention.animacy != antecedent.animacy { return false; }
    }
    // NER type compatibility
    // "he" can refer to PERSON, not to ORG
    // "it" can refer to ORG, PRODUCT, not to PERSON
    if let Some(ner) = &antecedent.ner_type {
        if mention.animacy == Animacy::Animate && !ner.is_animate() { return false; }
        if mention.animacy == Animacy::Inanimate && ner.is_animate() { return false; }
    }
    true
}
```

### 3.5 Resolution Algorithm

```rust
fn resolve_document(doc: &Document) -> Vec<CorefCluster> {
    let mut mentions: Vec<Mention> = vec![];
    let mut clusters: Vec<CorefCluster> = vec![];

    // 1. Collect all mentions across all sentences
    for (sent_idx, sentence) in doc.sentences.iter().enumerate() {
        for mut mention in detect_mentions(sentence) {
            mention.sentence_idx = sent_idx;
            mentions.push(mention);
        }
    }

    // 2. Process mentions left-to-right (document order)
    for i in 0..mentions.len() {
        let mention = &mentions[i];

        // Skip non-anaphoric mentions (proper nouns start their own cluster)
        if mention.mention_type == MentionType::ProperNoun {
            // Check if this proper noun matches an existing cluster
            if let Some(cluster_id) = find_matching_proper_noun(&clusters, mention) {
                clusters[cluster_id].add(mention.clone());
                continue;
            }
            // Start new cluster
            clusters.push(CorefCluster::new(mention.clone()));
            continue;
        }

        // 3. Find best antecedent (look backwards, max window = 5 sentences)
        let max_lookback = 5;
        let mut best_antecedent: Option<usize> = None;
        let mut best_score = 0.0;

        for j in (0..i).rev() {
            let candidate = &mentions[j];

            // Skip if too far back
            if mention.sentence_idx - candidate.sentence_idx > max_lookback {
                break;
            }

            // Apply rules
            if let Some(score) = score_antecedent(mention, candidate) {
                if score > best_score {
                    best_score = score;
                    best_antecedent = Some(j);
                }
            }
        }

        // 4. Link or start new cluster
        if let Some(ant_idx) = best_antecedent {
            let ant_cluster = find_cluster_of(&clusters, &mentions[ant_idx]);
            clusters[ant_cluster].add(mention.clone());
        } else {
            clusters.push(CorefCluster::new(mention.clone()));
        }
    }

    clusters
}

fn score_antecedent(mention: &Mention, candidate: &Mention) -> Option<f32> {
    // Check feature compatibility first
    if !features_compatible(&mention.features, &candidate.features) {
        return None;
    }

    let mut score = 0.0;

    // Distance penalty (prefer closer antecedents)
    let sent_dist = mention.sentence_idx - candidate.sentence_idx;
    score -= sent_dist as f32 * 0.5;

    // Same sentence bonus
    if sent_dist == 0 { score += 1.0; }

    // Syntactic role bonus (subjects are preferred antecedents)
    if candidate.features.dep_role == DepRel::Nsubj { score += 1.5; }
    if candidate.features.dep_role == DepRel::Obj { score += 0.5; }

    // NER type match bonus
    if candidate.features.ner_type.is_some() { score += 1.0; }

    // Exact string match (for nominal phrases)
    if mention.head_lemma() == candidate.head_lemma() { score += 3.0; }

    // Proper noun antecedent preferred over nominal
    if candidate.mention_type == MentionType::ProperNoun { score += 1.0; }

    Some(score)
}
```

### 3.6 Expected Coverage

| Phenomenon | Example | Handled? | Rule |
|-----------|---------|----------|------|
| Personal pronouns | "John... He..." | Yes | Rule 3 (gender/number match) |
| Possessive pronouns | "John... his..." | Yes | Rule 5 |
| Reflexive pronouns | "John hurt himself" | Yes | Rule 1 (same clause subject) |
| Relative pronouns | "the man who left" | Yes | Rule 2 (DEP: acl:relcl) |
| Definite NPs | "Apple... the company..." | Partial | Rule 4 (NER type + string match) |
| Proper noun repetition | "Apple... Apple..." | Yes | String match |
| Cataphora | "Before he left, John..." | No | Would need backward resolution |
| Event coref | "the attack... it happened..." | No | Needs event detection |
| Bridging | "the car... the engine..." | No | Needs world knowledge |
| Split antecedents | "John and Mary... they..." | Partial | Plural matching |

**Estimated F1**: ~55-60 on OntoNotes (rule-based systems typically score here).
This covers the **high-confidence, high-frequency** cases that matter most
in production (pronoun resolution is ~70% of all coreference).

---

## 4. Tier 2: Lightweight Neural Mention-Pair Scorer

A small neural model that scores (mention, candidate) pairs, exported to
ONNX and run in Rust via `ort`.

### 4.1 Architecture

```
mention_repr  = [head_token_emb; span_width_emb; NER_type; gender; number]
candidate_repr = [head_token_emb; span_width_emb; NER_type; gender; number]
pair_features = [distance; same_sentence; same_speaker; string_match]

score = MLP(concat(mention_repr, candidate_repr, mention * candidate, pair_features))
        → sigmoid → coref probability
```

This is NOT a full encoder model — it takes pre-computed features from our
cascade encoder and scores pairs with a lightweight MLP (~500K params).

### 4.2 Feature Extraction

Features come from the cascade encoder's hidden states (already computed
during the main inference pass):

```
Per mention:
  - Head token embedding from ScalarMix(top layers): [1024]
  - Span width embedding: [64]  (learned, index by width 1-30)
  - NER type one-hot: [18]
  - POS of head: [17]
  - Gender/Number from Morph: [8]
  Total: ~1131 dims

Per pair:
  - Sentence distance: [1]
  - Token distance (binned): [10]
  - String match (exact/partial/none): [3]
  - Same NER type: [1]
  - Same dep role: [1]
  Total: ~16 dims
```

### 4.3 Scorer Network

```python
class MentionPairScorer(nn.Module):
    def __init__(self, mention_dim=1131, pair_dim=16):
        super().__init__()
        combined = mention_dim * 2 + mention_dim + pair_dim  # m1, m2, m1*m2, pair
        self.scorer = nn.Sequential(
            nn.Linear(combined, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, mention1, mention2, pair_features):
        combined = torch.cat([mention1, mention2, mention1 * mention2, pair_features], dim=-1)
        return self.scorer(combined)
```

### 4.4 Training Data

| Source | License | Size | Quality |
|--------|---------|------|---------|
| fastcoref silver on our corpus | MIT (tool) | ~50K docs | ~78 F1 teacher |
| LitBank gold | CC-BY-4.0 | 100 docs, 210K tokens | Gold annotations |

**Pipeline**:
1. Run fastcoref on our corpus → silver coref clusters
2. Extract mention pairs (positive = same cluster, negative = different)
3. Compute features using our cascade
4. Train the MLP scorer
5. Export to ONNX

### 4.5 ONNX Export

```python
# Export mention-pair scorer to ONNX
dummy_m1 = torch.randn(1, 1131)
dummy_m2 = torch.randn(1, 1131)
dummy_pair = torch.randn(1, 16)
torch.onnx.export(scorer, (dummy_m1, dummy_m2, dummy_pair),
                  "coref_scorer.onnx", input_names=["mention1", "mention2", "pair_features"])
```

Size: ~2MB. Inference: <1ms per pair on CPU.

### 4.6 Rust Integration

```rust
use ort::{Session, Value};

struct CorefScorer {
    session: Session,
}

impl CorefScorer {
    fn score_pair(&self, mention1: &[f32], mention2: &[f32], pair_feats: &[f32]) -> f32 {
        let inputs = vec![
            Value::from_array(mention1),
            Value::from_array(mention2),
            Value::from_array(pair_feats),
        ];
        let outputs = self.session.run(inputs).unwrap();
        let score: f32 = outputs[0].extract_scalar().unwrap();
        sigmoid(score)
    }
}

fn resolve_with_scorer(
    mentions: &[Mention],
    cascade_embeddings: &[Vec<f32>],
    scorer: &CorefScorer,
) -> Vec<CorefCluster> {
    // 1. Rule-based pre-filtering (Tier 1 rules for feature compatibility)
    // 2. Score remaining pairs with neural scorer
    // 3. Greedy left-to-right antecedent selection (threshold > 0.5)
    // 4. Build clusters via transitive closure
}
```

### 4.7 Expected Performance

| Component | F1 (est.) | Latency | Size |
|-----------|-----------|---------|------|
| Tier 1 rules only | ~55-60 | <1ms/doc | 0 (code) |
| Tier 1 + Tier 2 scorer | ~72-76 | ~5ms/doc | 2MB ONNX |
| fastcoref (Python, reference) | 78-81 | ~10ms/doc | 500MB |

---

## 5. Tier 3: Full Span-Ranking Coref Head (Future)

A native coreference head on our cascade encoder, handling mention detection
and antecedent ranking end-to-end.

### 5.1 Architecture

```
Document (multi-sentence, up to 4096 tokens)
    │
    ▼
DeBERTa encoder (single pass, max_length=4096 or sliding window)
    │
    ├──► Cascade heads (POS, NER, DEP, SRL, CLS) — per-sentence
    │
    ▼
Mention detector (BIO tagger on encoder hidden states)
    │
    ▼
Mention representations (span pooling: start + end + attention)
    │
    ▼
Pairwise antecedent scorer (bilinear + MLP)
    │
    ▼
Greedy clustering → coreference chains
```

### 5.2 Key Challenges

**Context length**: Our encoder uses max_length=128. Coref needs 512-4096.
Options:
- Increase to 512 during coref training only (DeBERTa supports it)
- Sliding window with overlap and mention merging
- Encode sentences separately, cross-attend mention representations

**Training data**: LitBank (100 docs, CC-BY-4.0) is small. Supplement with
fastcoref silver (MIT) on our corpus.

**ONNX export**: The mention detection + pairwise scoring involves dynamic
shapes (variable number of mentions per document). Need careful ONNX
graph construction with padding.

### 5.3 Timeline

Tier 3 is a significant engineering effort (4-6 weeks). Recommend building
Tier 1 + Tier 2 first, which covers production needs, then pursue Tier 3
when quality requirements demand it.

---

## 6. Cascade Features for Coreference

Our cascade provides uniquely rich features that most coref systems don't have:

| Feature | Source | How It Helps Coref |
|---------|--------|-------------------|
| **NER types** | NER head | "he" → must be PERSON; "it" → must be ORG/PRODUCT/etc. |
| **Gender/Number** | Morph head (future) | "she" → Fem,Sing; "they" → Plur. Hard constraint. |
| **Syntactic role** | DEP head | Subjects are preferred antecedents (centering theory) |
| **SRL roles** | SRL head | ARG0 (agent) of same verb → likely coreferent |
| **Entity spans** | NER head | Proper entity boundaries for mention detection |
| **POS tags** | POS head | Distinguish pronouns, proper nouns, common nouns |
| **DEP subtree** | DEP head | NP span = det + amod + noun (DEP subtree of head noun) |

**The cascade advantage**: Standard coref models (fastcoref, LingMess) must
learn all of this implicitly from raw text. Our Tier 1 rules + Tier 2 scorer
get these features explicitly, making the model much smaller and faster
while approaching similar quality.

---

## 7. API Design (Rust)

```rust
use kniv::{CascadeModel, CorefResolver};

// Load cascade model + coref resolver
let model = CascadeModel::load("kniv-v4.onnx")?;
let coref = CorefResolver::new(
    CorefConfig {
        tier: CorefTier::RulePlusNeural,  // Tier 1 + 2
        scorer_path: Some("coref_scorer.onnx"),
        max_sentence_lookback: 5,
        threshold: 0.5,
    }
)?;

// Process document
let text = "John went to the store. He bought milk there. \
            Later, John called Mary. She was busy.";

// Step 1: Run cascade on each sentence
let sentences = split_sentences(text);
let cascade_outputs: Vec<CascadeOutput> = sentences.iter()
    .map(|s| model.predict(s))
    .collect();

// Step 2: Resolve coreference across sentences
let coref_result = coref.resolve(&cascade_outputs);

// Step 3: Access results
for cluster in &coref_result.clusters {
    println!("Cluster: {:?}", cluster.mentions.iter()
        .map(|m| m.text.as_str())
        .collect::<Vec<_>>());
}
// Output:
//   Cluster: ["John", "He", "John"]
//   Cluster: ["the store", "there"]
//   Cluster: ["Mary", "She"]

// Step 4: Get resolved text
let resolved = coref_result.resolve_pronouns(&cascade_outputs);
// "John went to the store. John bought milk at the store.
//  Later, John called Mary. Mary was busy."
```

### 7.1 Integration with Other Features

Coreference feeds into downstream applications:

```rust
// Entity consolidation: merge NER entities across sentences via coref
let entities = coref_result.consolidated_entities();
// [Entity("John", PERSON, mentions=3), Entity("Mary", PERSON, mentions=2),
//  Entity("the store", FAC, mentions=2)]

// SRL with resolved arguments
let resolved_srl = coref_result.resolve_srl_args(&cascade_outputs);
// bought(ARG0="John", ARG1="milk", ARGM-LOC="the store")
// instead of: bought(ARG0="He", ARG1="milk", ARGM-LOC="there")

// Information extraction with coref
let facts = extract_facts(&cascade_outputs, &coref_result);
// [(John, went_to, the store), (John, bought, milk), (John, called, Mary)]
```

---

## 8. Implementation Plan

### Phase 1: Tier 1 — Rule-Based (Rust, 1-2 weeks)

```
1. Implement mention detection from cascade outputs     (2 days)
   - Pronouns, NER entities, NPs with determiners
2. Implement feature extraction (gender, number, animacy) (1 day)
   - From POS, NER, Morph cascade outputs
3. Implement resolution rules (5 rules)                  (2 days)
4. Implement clustering + output API                     (1 day)
5. Evaluate on hand-annotated test set (50 docs)         (1 day)
```

Deliverable: `kniv::CorefResolver` with `CorefTier::RuleOnly`

### Phase 2: Tier 2 — Neural Scorer (Python train + Rust inference, 2-3 weeks)

```
1. Run fastcoref on corpus → silver coref data           (2 days)
2. Extract mention-pair features from cascade             (2 days)
3. Train MLP scorer in Python                             (1 day)
4. Export to ONNX                                         (1 day)
5. Integrate in Rust via ort                              (2 days)
6. Evaluate: Tier 1 + 2 combined                          (1 day)
```

Deliverable: `coref_scorer.onnx` (2MB) + updated `CorefResolver`

### Phase 3: Tier 3 — Full Head (Future, 4-6 weeks)

```
1. Extend cascade for long-context (sliding window)       (1 week)
2. Implement mention detection head                       (1 week)
3. Implement span-ranking architecture                    (1 week)
4. Train on fastcoref silver + LitBank gold               (1 week)
5. ONNX export + Rust integration                         (1 week)
```

---

## 9. Evaluation

### Metrics

| Metric | Description |
|--------|-------------|
| **MUC** | Link-based — counts correct/missing coreference links |
| **B-cubed** | Entity-based — precision/recall per mention |
| **CEAF-e** | Entity-based — best alignment between predicted/gold clusters |
| **CoNLL F1** | Average of MUC + B-cubed + CEAF-e (standard) |
| **LEA** | Link-based entity-aware (newer, more discriminative) |

### Evaluation Data

| Dataset | License | Size | Notes |
|---------|---------|------|-------|
| LitBank | CC-BY-4.0 | 100 docs | Gold, commercially OK |
| Hand-annotated from corpus | Ours | 50 docs | Create manually for our domain |
| CoNLL-2012 | LDC | 2802 docs | Eval only (with LDC access) |

### Expected Results

| Configuration | CoNLL F1 (est.) | Latency/doc | Model Size |
|--------------|----------------|-------------|------------|
| Tier 1 (rules only) | 55-60 | <1ms | 0 |
| Tier 1 + Tier 2 | 72-76 | ~5ms | 2MB |
| Tier 3 (full head) | 78-82 | ~50ms | ~15MB |
| fastcoref (Python ref) | 78-81 | ~10ms | 500MB |

---

## 10. Open Questions

1. **Morph head dependency**: Tier 1 rules rely heavily on gender/number
   from morphological features. The Morph head is planned but not yet
   trained. Until then, use heuristic gender detection (name lists +
   pronoun forms) as a stopgap.

2. **Speaker detection**: In conversational text, "I" and "you" switch
   referents between speakers. Need speaker detection (from CLS dialog
   act head or formatting cues) to handle this correctly.

3. **Singleton mentions**: Should we output mentions that don't corefer
   with anything? Useful for entity tracking but adds noise. Default: no.

4. **Cross-document coreference**: Same entity mentioned across different
   documents. Out of scope for now — requires entity linking to a knowledge
   base, not just within-document resolution.

5. **Evaluation without CoNLL-2012**: OntoNotes (LDC) is the standard
   benchmark but requires paid access. LitBank (CC-BY-4.0, 100 docs) is
   our primary eval set. Consider creating a small gold-annotated test
   set from our corpus domains for domain-specific evaluation.
