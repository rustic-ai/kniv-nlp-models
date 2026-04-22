"""Prepare SRL training data from four sources.

1. PropBank EWT: gold annotations aligned to UD EWT (CC-BY-SA-4.0)
2. MASC PropBank: gold dependency-based SRL, converted to BIO (unrestricted)
3. QA-SRL Bank 2.0: crowdsourced, converted to BIO (MIT)
4. Few-NERD + silver: model-generated via cross-lingual-srl-v3 (MIT)

Produces: srl_train.json, srl_dev.json, srl_test.json

Usage:
    python models/kniv-deberta-cascade-large-nlp-en/prepare_srl.py
"""

from __future__ import annotations

import gzip
import glob
import json
import os
import random
import re
from pathlib import Path

import conllu


# ── SRL label set (42 BIO tags) ─────────────────────────────────

SRL_CORE = ["ARG0", "ARG1", "ARG2", "ARG3", "ARG4"]
SRL_MODIFIERS = [
    "ARGM-TMP", "ARGM-LOC", "ARGM-MNR", "ARGM-CAU", "ARGM-PRP",
    "ARGM-NEG", "ARGM-ADV", "ARGM-DIR", "ARGM-DIS", "ARGM-EXT",
    "ARGM-MOD", "ARGM-PRD", "ARGM-GOL", "ARGM-COM", "ARGM-REC",
]
SRL_TAGS = ["O", "V"] + [f"{p}-{r}" for r in SRL_CORE + SRL_MODIFIERS for p in ["B", "I"]]

DATA_DIR = Path(__file__).parent.parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "prepared" / "kniv-deberta-cascade"
PROPBANK_DIR = DATA_DIR / "propbank-release"
UD_DIR = DATA_DIR / "ud-english-ewt"
MASC_DIR = DATA_DIR / "masc-propbank" / "masc-conll"
QASRL_DIR = DATA_DIR / "qasrl-bank" / "qasrl-v2"


# ── Source 1: PropBank EWT (gold) ────────────────────────────────

def _parse_gold_skel_bracket(col_values: list[str]) -> list[str]:
    """Convert PropBank bracket notation to BIO tags for one predicate column."""
    bio = ["O"] * len(col_values)
    current_role = None

    for i, col in enumerate(col_values):
        if col == "*":
            if current_role:
                bio[i] = f"I-{current_role}"
        elif col.startswith("(") and col.endswith("*)"):
            # Single-token span: (ARG0*) or (V*)
            role = col[1:-2]
            if role == "V":
                bio[i] = "V"
            else:
                bio[i] = f"B-{role}"
        elif col.startswith("(") and col.endswith("*"):
            # Start of multi-token span: (ARG0*
            role = col[1:-1]
            if role == "V":
                bio[i] = "V"
            else:
                bio[i] = f"B-{role}"
                current_role = role
        elif col == "*)":
            if current_role:
                bio[i] = f"I-{current_role}"
                current_role = None
        elif col.startswith("(") and not col.endswith("*"):
            # Nested or complex bracket — try to extract role
            role = col.lstrip("(").rstrip(")")
            if role == "V":
                bio[i] = "V"
            elif role:
                bio[i] = f"B-{role}"
                current_role = role

    return bio


def _parse_gold_skel_file(filepath: str) -> list[dict]:
    """Parse a single .gold_skel file into per-predicate annotations."""
    with open(filepath) as f:
        lines = f.readlines()

    # Split into sentences (blank line separated)
    sentences = []
    current = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                sentences.append(current)
                current = []
        else:
            current.append(stripped)
    if current:
        sentences.append(current)

    results = []
    for sent_lines in sentences:
        tok_data = []
        for tok_line in sent_lines:
            parts = tok_line.split()
            if len(parts) < 8:
                continue
            tok_data.append({
                "doc_id": parts[0],
                "sent_idx": int(parts[1]),
                "tok_idx": int(parts[2]),
                "pos": parts[4],
                "lemma": parts[6],
                "roleset": parts[7],
                "srl_cols": parts[8:],
            })

        if not tok_data:
            continue

        doc_id = tok_data[0]["doc_id"]
        sent_idx = tok_data[0]["sent_idx"]
        num_preds = len(tok_data[0]["srl_cols"])

        for pred_col in range(num_preds):
            col_values = [t["srl_cols"][pred_col] for t in tok_data]
            bio_tags = _parse_gold_skel_bracket(col_values)

            # Find predicate token
            pred_tok = None
            roleset = None
            for t in tok_data:
                if t["srl_cols"][pred_col] in ("(V*)", "(V*"):
                    pred_tok = t["tok_idx"]
                    roleset = t["roleset"]
                    break

            results.append({
                "doc_id": doc_id,
                "sent_idx": sent_idx,
                "pred_token_idx": pred_tok,
                "roleset": roleset,
                "bio_tags": bio_tags,
                "num_tokens": len(tok_data),
            })

    return results


def _build_propbank_index(propbank_dir: Path) -> dict:
    """Build index: (ud_doc_prefix, sent_idx) → list of predicate annotations."""
    ewt_dir = propbank_dir / "data" / "google" / "ewt"
    if not ewt_dir.exists():
        raise FileNotFoundError(f"PropBank EWT not found at {ewt_dir}")

    skel_files = glob.glob(str(ewt_dir / "**" / "*.gold_skel"), recursive=True)
    print(f"  Found {len(skel_files)} .gold_skel files")

    index = {}
    total_preds = 0
    for filepath in skel_files:
        preds = _parse_gold_skel_file(filepath)
        total_preds += len(preds)

        for pred in preds:
            # Convert PropBank doc_id to UD prefix
            # PropBank: google/ewt/weblog/00/name.xml → UD: weblog-name
            doc_id = pred["doc_id"]
            # Extract genre and doc name
            parts = doc_id.replace("google/ewt/", "").split("/")
            genre = parts[0]
            doc_name = parts[-1].replace(".xml", "")
            ud_prefix = f"{genre}-{doc_name}"

            key = (ud_prefix, pred["sent_idx"])
            if key not in index:
                index[key] = []
            index[key].append(pred)

    print(f"  Total predicates: {total_preds}")
    print(f"  Unique (doc, sent) pairs: {len(index)}")
    return index


def load_propbank_ewt(
    ud_splits: dict[str, list[dict]],
    propbank_dir: Path = PROPBANK_DIR,
) -> dict[str, list[dict]]:
    """Align PropBank gold SRL to UD EWT, filter to root-verb only.

    Args:
        ud_splits: dict with "train"/"validation"/"test" UD examples
                   (each has 'words', 'heads', 'deprels', 'pos_tags')
        propbank_dir: path to cloned propbank-release repo

    Returns:
        dict with "train"/"validation"/"test" SRL examples
    """
    print("Loading PropBank EWT gold annotations...")
    pb_index = _build_propbank_index(propbank_dir)

    # Load UD CoNLL-U to get sent_ids
    ud_files = {
        "train": UD_DIR / "en_ewt-ud-train.conllu",
        "validation": UD_DIR / "en_ewt-ud-dev.conllu",
        "test": UD_DIR / "en_ewt-ud-test.conllu",
    }

    results = {}
    for split_name, ud_examples in ud_splits.items():
        conllu_path = ud_files[split_name]
        with open(conllu_path) as f:
            conllu_sents = conllu.parse(f.read())

        srl_examples = []
        matched = 0
        root_verb_found = 0

        for ud_ex, conllu_sent in zip(ud_examples, conllu_sents):
            sent_id = conllu_sent.metadata.get("sent_id", "")
            # Extract UD prefix and sentence number
            # sent_id: weblog-name-0001 → prefix=weblog-name, num=0001
            last_dash = sent_id.rfind("-")
            if last_dash == -1:
                continue
            ud_prefix = sent_id[:last_dash]
            sent_num = int(sent_id[last_dash + 1:])
            pb_sent_idx = sent_num - 1  # UD is 1-based, PropBank is 0-based

            key = (ud_prefix, pb_sent_idx)
            preds = pb_index.get(key, [])
            if not preds:
                continue
            matched += 1

            # Find root token in UD
            root_idx = None
            for i, head in enumerate(ud_ex["heads"]):
                if head == -1:  # root
                    root_idx = i
                    break
            if root_idx is None:
                continue

            # Find PropBank predicate that matches the root
            root_pred = None
            for pred in preds:
                if pred["pred_token_idx"] == root_idx:
                    root_pred = pred
                    break

            if root_pred is None:
                continue
            root_verb_found += 1

            # Verify token count matches
            if root_pred["num_tokens"] != len(ud_ex["words"]):
                continue

            # Normalize BIO tags to our label set
            bio_tags = []
            for tag in root_pred["bio_tags"]:
                if tag == "O" or tag == "V":
                    bio_tags.append(tag)
                elif tag.startswith("B-") or tag.startswith("I-"):
                    prefix, role = tag[:2], tag[2:]
                    # Normalize role names (some PropBank uses C- or R- for continuations)
                    if role.startswith("C-") or role.startswith("R-"):
                        role = role[2:]
                    full_tag = f"{prefix}{role}"
                    if full_tag in SRL_TAGS:
                        bio_tags.append(full_tag)
                    else:
                        bio_tags.append("O")  # unknown role
                else:
                    bio_tags.append("O")

            srl_examples.append({
                "words": ud_ex["words"],
                "text": ud_ex["text"],
                "srl_tags": bio_tags,
                "pos_tags": ud_ex["pos_tags"],
                "predicate_idx": root_idx,
            })

        results[split_name] = srl_examples
        print(f"  {split_name}: {matched} matched, {root_verb_found} root-verb, "
              f"{len(srl_examples)} valid")

    return results


# ── Source 2: MASC PropBank (unrestricted) ─────────────────────────

# MASC CoNLL 2008 uses short labels (A0, A1, AM-TMP) → normalize to PropBank
_MASC_LABEL_MAP = {
    "A0": "ARG0", "A1": "ARG1", "A2": "ARG2", "A3": "ARG3", "A4": "ARG4",
    "A5": "ARG4",  # rare, collapse into ARG4
    "AM-TMP": "ARGM-TMP", "AM-LOC": "ARGM-LOC", "AM-MNR": "ARGM-MNR",
    "AM-CAU": "ARGM-CAU", "AM-PRP": "ARGM-PRP", "AM-NEG": "ARGM-NEG",
    "AM-ADV": "ARGM-ADV", "AM-DIR": "ARGM-DIR", "AM-DIS": "ARGM-DIS",
    "AM-EXT": "ARGM-EXT", "AM-MOD": "ARGM-MOD", "AM-PRD": "ARGM-PRD",
    "AM-GOL": "ARGM-GOL", "AM-COM": "ARGM-COM", "AM-REC": "ARGM-REC",
}


def _get_subtree_span(token_id: int, heads: list[int]) -> tuple[int, int]:
    """Get contiguous span [start, end) for the dependency subtree rooted at token_id.

    Collects all descendants via BFS, returns min..max+1 indices.
    """
    # Build children map
    children = {i: [] for i in range(len(heads))}
    for i, h in enumerate(heads):
        if h >= 0 and h < len(heads) and h != i:
            children[h].append(i)

    # BFS from token_id
    visited = {token_id}
    queue = [token_id]
    while queue:
        node = queue.pop(0)
        for child in children.get(node, []):
            if child not in visited:
                visited.add(child)
                queue.append(child)

    return min(visited), max(visited) + 1


def load_masc_propbank(masc_dir: Path = MASC_DIR) -> list[dict]:
    """Load MASC PropBank CoNLL 2008 data, convert dep-based SRL to BIO spans.

    Filters to root-verb predicates only.
    """
    data_dir = masc_dir / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"MASC data not found at {data_dir}")

    conll_files = sorted(glob.glob(str(data_dir / "**" / "*.conll"), recursive=True))
    print(f"  Found {len(conll_files)} MASC CoNLL files")

    all_examples = []
    total_sents = 0
    root_verb_found = 0

    for filepath in conll_files:
        with open(filepath) as f:
            lines = f.readlines()

        # Split into sentences (blank lines)
        sentences = []
        current = []
        for line in lines:
            stripped = line.rstrip("\n")
            if not stripped:
                if current:
                    sentences.append(current)
                    current = []
            else:
                current.append(stripped)
        if current:
            sentences.append(current)

        for sent_lines in sentences:
            total_sents += 1
            tokens = []
            for line in sent_lines:
                parts = line.split("\t")
                if len(parts) < 11:
                    continue
                tokens.append({
                    "id": int(parts[0]) - 1,  # 0-based
                    "form": parts[1],
                    "gpos": parts[3],
                    "head": int(parts[8]) - 1,  # 0-based (-1 = root)
                    "deprel": parts[9],
                    "pred": parts[10],
                    "args": parts[11:] if len(parts) > 11 else [],
                })

            if not tokens:
                continue

            # Find ROOT token (head == -1 means HEAD was 0)
            root_idx = None
            for t in tokens:
                if t["head"] == -1:
                    root_idx = t["id"]
                    break
            if root_idx is None:
                continue

            # Find which predicate column corresponds to the root verb
            # If ROOT itself is not a predicate, check its VC (verbal complement) child
            pred_indices = [t["id"] for t in tokens if t["pred"] != "_"]
            if not pred_indices:
                continue

            # Try ROOT first, then VC child
            target_pred_idx = None
            if root_idx in pred_indices:
                target_pred_idx = root_idx
            else:
                # Look for verbal complement child of ROOT
                for t in tokens:
                    if t["head"] == root_idx and t["deprel"] in ("VC", "IM") and t["id"] in pred_indices:
                        target_pred_idx = t["id"]
                        break
            if target_pred_idx is None:
                continue

            root_verb_found += 1

            # Find which arg column index this predicate corresponds to
            # Predicates are in textual order: first PRED != "_" → col 0, second → col 1, etc.
            pred_col_idx = pred_indices.index(target_pred_idx)

            # Build dependency heads for subtree computation
            heads = [t["head"] for t in tokens]
            words = [t["form"] for t in tokens]

            # Extract argument labels and convert to BIO
            bio_tags = ["O"] * len(tokens)
            bio_tags[target_pred_idx] = "V"

            for t in tokens:
                if pred_col_idx >= len(t["args"]):
                    continue
                arg_label = t["args"][pred_col_idx]
                if arg_label == "_" or arg_label == "V":
                    continue

                # Normalize: strip C- and R- prefixes
                clean = arg_label
                if clean.startswith("C-") or clean.startswith("R-"):
                    clean = clean[2:]

                role = _MASC_LABEL_MAP.get(clean)
                if role is None:
                    continue  # skip SU (support), AA, etc.

                # Get span from dependency subtree
                start, end = _get_subtree_span(t["id"], heads)

                # Don't overwrite the V tag or existing labels
                for k in range(start, end):
                    if bio_tags[k] != "O":
                        continue
                    if k == start:
                        bio_tags[k] = f"B-{role}"
                    else:
                        bio_tags[k] = f"I-{role}"

            # Verify all BIO tags are in our label set
            bio_tags = [t if t in _SRL_TAG_SET else "O" for t in bio_tags]

            # Only keep if we got at least one argument
            non_o = sum(1 for t in bio_tags if t not in ("O", "V"))
            if non_o == 0:
                continue

            all_examples.append({
                "words": words,
                "text": " ".join(words),
                "srl_tags": bio_tags,
                "predicate_idx": target_pred_idx,
            })

    print(f"  MASC: {total_sents} sentences, {root_verb_found} root-verb, "
          f"{len(all_examples)} valid")
    return all_examples


# ── Source 3: QA-SRL Bank 2.0 (MIT) ───────────────────────────────

# Structured question slot mapping → PropBank roles
QASRL_WH_ROLE = {
    "where": "ARGM-LOC",
    "when": "ARGM-TMP",
    "why": "ARGM-CAU",
    "how": "ARGM-MNR",
    "how much": "ARGM-EXT",
    "how long": "ARGM-TMP",
}


def _qasrl_slots_to_role(slots: dict, is_passive: bool) -> str | None:
    """Map QA-SRL structured question slots to a PropBank role.

    Uses the decomposed question structure (wh, subj, obj, prep, isPassive)
    for more reliable role assignment than regex on the question string.
    """
    wh = slots.get("wh", "").lower()
    subj = slots.get("subj", "")
    obj = slots.get("obj", "")
    prep = slots.get("prep", "")
    obj2 = slots.get("obj2", "")

    # Modifier WH-words
    if wh in QASRL_WH_ROLE:
        return QASRL_WH_ROLE[wh]

    # Core roles: who/what — use slot structure for disambiguation
    if wh in ("who", "what"):
        if subj == "_":
            # WH-word IS the subject
            if is_passive:
                return "ARG1"   # passive subject = patient
            return "ARG0"       # active subject = agent
        if obj == "_":
            # WH-word IS the direct object
            if is_passive:
                return "ARG0"   # passive "by whom" object = agent
            return "ARG1"       # active object = patient
        if prep != "_":
            # Prepositional argument — typically ARG2 (benefactive, instrument)
            if prep.lower() in ("to", "for"):
                return "ARG2"
            if prep.lower() in ("with", "from"):
                return "ARG2"
            if prep.lower() == "by":
                return "ARG0"  # by-phrase = agent
            return "ARG2"
        if obj2 != "_":
            # Second object slot (rare) — ARG2
            return "ARG2"
        # Fallback: if subj is filled and obj is filled, wh is something else
        return "ARG1"

    return None


def load_qasrl_bank(
    qasrl_dir: Path = QASRL_DIR,
    subsample: int = 25000,
    seed: int = 42,
) -> list[dict]:
    """Load QA-SRL Bank 2.0 from local JSONL.gz files, convert to BIO SRL format.

    Uses spaCy to identify root verbs, then converts QA pairs for the
    root verb's annotations to BIO tags using structured question slots.
    """
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "textcat"])

    orig_dir = qasrl_dir / "orig"
    if not orig_dir.exists():
        raise FileNotFoundError(f"QA-SRL orig data not found at {orig_dir}")

    all_examples = []
    skipped = 0
    no_root_match = 0

    for split_file in ["train.jsonl.gz"]:
        filepath = orig_dir / split_file
        print(f"  Loading {filepath.name}...")

        with gzip.open(filepath, "rt") as f:
            for line in f:
                sent = json.loads(line)
                words = sent["sentenceTokens"]
                verb_entries = sent.get("verbEntries", {})

                if len(words) < 3 or not verb_entries:
                    skipped += 1
                    continue

                # Use spaCy to find root verb
                text = " ".join(words)
                doc = nlp(text)
                root_token = None
                for tok in doc:
                    if tok.dep_ == "ROOT":
                        root_token = tok
                        break
                if root_token is None:
                    skipped += 1
                    continue

                # Match spaCy root to QA-SRL verb index (by token position)
                # spaCy tokenization may differ — find closest verb entry
                root_verb_idx = None
                best_dist = float("inf")
                for _vidx_str, ventry in verb_entries.items():
                    vidx = ventry["verbIndex"]
                    dist = abs(vidx - root_token.i)
                    if dist < best_dist:
                        best_dist = dist
                        root_verb_idx = _vidx_str

                if root_verb_idx is None or best_dist > 2:
                    no_root_match += 1
                    skipped += 1
                    continue

                ventry = verb_entries[root_verb_idx]
                predicate_idx = ventry["verbIndex"]
                q_labels = ventry.get("questionLabels", {})
                if not q_labels:
                    skipped += 1
                    continue

                # Convert each QA pair to BIO
                bio_tags = ["O"] * len(words)
                bio_tags[predicate_idx] = "V"

                for _q_str, qlabel in q_labels.items():
                    slots = qlabel.get("questionSlots", {})
                    is_passive = qlabel.get("isPassive", False)

                    role = _qasrl_slots_to_role(slots, is_passive)
                    if role is None:
                        continue

                    # Use majority-voted valid spans
                    judgments = qlabel.get("answerJudgments", [])
                    valid_spans = []
                    for j in judgments:
                        if j.get("isValid", False):
                            for span in j.get("spans", []):
                                valid_spans.append(tuple(span))

                    if not valid_spans:
                        continue

                    # Take the most common span (majority vote)
                    from collections import Counter
                    span_counts = Counter(valid_spans)
                    best_span = span_counts.most_common(1)[0][0]
                    start, end = best_span  # [start_inclusive, end_exclusive]

                    # Only assign if all tokens in range are currently O
                    if start < len(words) and end <= len(words):
                        if all(bio_tags[t] == "O" for t in range(start, end)):
                            bio_tags[start] = f"B-{role}"
                            for t in range(start + 1, end):
                                bio_tags[t] = f"I-{role}"

                # Verify all BIO tags are in our label set
                bio_tags = [t if t in _SRL_TAG_SET else "O" for t in bio_tags]

                # Only keep if we got at least one argument
                non_o = sum(1 for t in bio_tags if t not in ("O", "V"))
                if non_o == 0:
                    skipped += 1
                    continue

                all_examples.append({
                    "words": words,
                    "text": text,
                    "srl_tags": bio_tags,
                    "predicate_idx": predicate_idx,
                })

    print(f"  QA-SRL: {len(all_examples)} converted, {skipped} skipped "
          f"({no_root_match} no root match)")

    # Subsample
    rng = random.Random(seed)
    if len(all_examples) > subsample:
        all_examples = rng.sample(all_examples, subsample)
        print(f"  Subsampled to {subsample}")

    return all_examples


# ── Source 3: Silver SRL from Few-NERD ───────────────────────────

_SRL_TAG_SET = set(SRL_TAGS)


def _map_srl_label(label: str) -> str:
    """Map model's 85 labels to our 42-tag set."""
    if label == "O":
        return "O"

    # B-V / I-V → V (predicate marker)
    if label in ("B-V", "I-V"):
        return "V"

    # Direct match (B-ARG0, I-ARGM-TMP, etc.)
    if label in _SRL_TAG_SET:
        return label

    # Continuations: C-ARG0 → ARG0, C-ARGM-TMP → ARGM-TMP
    # References: R-ARG0 → ARG0, R-ARGM-LOC → ARGM-LOC
    if label.startswith(("B-C-", "I-C-", "B-R-", "I-R-")):
        prefix = label[:2]  # B- or I-
        role = label[4:]     # strip C- or R-
        mapped = f"{prefix}{role}"
        if mapped in _SRL_TAG_SET:
            return mapped

    # ARG5 → ARG4, ARG1-DSP → ARG1
    if "ARG5" in label:
        return label.replace("ARG5", "ARG4")
    if "ARG1-DSP" in label:
        return label.replace("ARG1-DSP", "ARG1")

    # Drop unknown modifier types (ARGM-ADJ, ARGM-CXN, ARGM-LVB, ARGM-PRR)
    return "O"


def generate_silver_srl(
    few_nerd_file: Path = OUTPUT_DIR / "fewnerd_train.json",
    subsample: int = 45000,
    batch_size: int = 32,
    seed: int = 42,
) -> list[dict]:
    """Generate silver SRL labels on Few-NERD using liaad/srl-en_xlmr-large.

    Runs inference, filters to root-verb frame, converts to our BIO format.
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    import torch

    print("  Loading Few-NERD for silver SRL generation...")
    with open(few_nerd_file) as f:
        few_nerd = json.load(f)

    # Subsample first (running model on all 150K is too slow)
    rng = random.Random(seed)
    if len(few_nerd) > subsample:
        few_nerd = rng.sample(few_nerd, subsample)

    print(f"  Loading cheralathan-m/cross-lingual-srl-v3 (MIT, F1=0.872)...")
    model_name = "cheralathan-m/cross-lingual-srl-v3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # Get label list from model config
    id2label = model.config.id2label

    all_examples = []
    for i in range(0, len(few_nerd), batch_size):
        batch = few_nerd[i:i + batch_size]
        texts = [ex["text"] for ex in batch]

        encoding = tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt",
        )
        # Keep BatchEncoding for word_ids(), send tensors to device separately
        model_input = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**model_input)
        preds = outputs.logits.argmax(dim=-1).cpu()

        for j, ex in enumerate(batch):
            words = ex["words"]
            pred_labels = [id2label[p.item()] for p in preds[j]]

            # Align subword predictions back to words
            word_ids = encoding.word_ids(j)
            bio_tags = ["O"] * len(words)
            prev_word = None
            for k, wid in enumerate(word_ids):
                if wid is not None and wid != prev_word and wid < len(words):
                    label = pred_labels[k]
                    # Map model labels to our 42-tag set
                    mapped = _map_srl_label(label)
                    bio_tags[wid] = mapped
                prev_word = wid

            # Find predicate (V tag)
            pred_idx = None
            for k, tag in enumerate(bio_tags):
                if tag == "V":
                    pred_idx = k
                    break

            # Only keep if we have a predicate and at least one argument
            non_o = sum(1 for t in bio_tags if t not in ("O", "V"))
            if pred_idx is not None and non_o > 0:
                all_examples.append({
                    "words": words,
                    "text": ex["text"],
                    "srl_tags": bio_tags,
                    "predicate_idx": pred_idx,
                })

        if (i // batch_size) % 100 == 0:
            print(f"    Processed {i + len(batch)}/{len(few_nerd)}", flush=True)

    print(f"  Silver SRL: {len(all_examples)} examples generated")
    return all_examples


# ── Main ─────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prepare SRL training data")
    parser.add_argument("--skip-silver", action="store_true",
                        help="Skip silver SRL generation (slow, requires GPU)")
    parser.add_argument("--qasrl-subsample", type=int, default=25000,
                        help="Max QA-SRL examples to include")
    parser.add_argument("--silver-subsample", type=int, default=70000,
                        help="Max Few-NERD examples for silver labeling")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load UD EWT for PropBank alignment
    from prepare_data import load_ud_data
    ud_train = load_ud_data(UD_DIR / "en_ewt-ud-train.conllu")
    ud_dev = load_ud_data(UD_DIR / "en_ewt-ud-dev.conllu")
    ud_test = load_ud_data(UD_DIR / "en_ewt-ud-test.conllu")

    ud_splits = {"train": ud_train, "validation": ud_dev, "test": ud_test}

    # Source 1: PropBank EWT gold
    propbank = load_propbank_ewt(ud_splits)
    print(f"  PropBank EWT: train={len(propbank['train'])}, "
          f"dev={len(propbank['validation'])}, test={len(propbank['test'])}")

    # Source 2: MASC PropBank gold
    print("\nLoading MASC PropBank...")
    masc = load_masc_propbank()
    print(f"  MASC total: {len(masc)}")

    # Source 3: QA-SRL Bank 2.0
    print("\nLoading QA-SRL Bank 2.0...")
    qasrl = load_qasrl_bank(subsample=args.qasrl_subsample)
    print(f"  QA-SRL total: {len(qasrl)}")

    # Source 4: Silver SRL from Few-NERD
    silver = []
    if not args.skip_silver:
        print("\nGenerating silver SRL from Few-NERD...")
        silver = generate_silver_srl(subsample=args.silver_subsample)
        print(f"  Silver total: {len(silver)}")
    else:
        # Load existing silver if available
        silver_path = OUTPUT_DIR / "srl_silver_cache.json"
        if silver_path.exists():
            with open(silver_path) as f:
                silver = json.load(f)
            print(f"\n  Loaded cached silver SRL: {len(silver)}")

    # Merge: train = PropBank gold + MASC gold + QA-SRL + silver
    # Dev/test = PropBank EWT gold only (reliable evaluation)
    srl_train = propbank["train"] + masc + qasrl + silver
    random.shuffle(srl_train)

    srl_dev = propbank["validation"]
    srl_test = propbank["test"]

    print(f"\nFinal SRL dataset:")
    print(f"  Train: {len(srl_train)} "
          f"(PropBank {len(propbank['train'])} + MASC {len(masc)} + "
          f"QA-SRL {len(qasrl)} + Silver {len(silver)})")
    print(f"  Dev: {len(srl_dev)} (PropBank gold)")
    print(f"  Test: {len(srl_test)} (PropBank gold)")

    # Save
    for name, data in [("srl_full_train", srl_train), ("srl_dev", srl_dev), ("srl_test", srl_test)]:
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(data, f)
        print(f"  Saved {name}.json ({len(data)} examples)")

    # Update label_vocabs.json
    vocabs_path = OUTPUT_DIR / "label_vocabs.json"
    with open(vocabs_path) as f:
        vocabs = json.load(f)
    vocabs["srl_labels"] = SRL_TAGS
    with open(vocabs_path, "w") as f:
        json.dump(vocabs, f, indent=2)
    print(f"  Updated label_vocabs.json with {len(SRL_TAGS)} SRL labels")

    # Show SRL tag distribution
    from collections import Counter
    tag_counts = Counter()
    for ex in srl_train[:5000]:
        for tag in ex["srl_tags"]:
            if tag != "O":
                tag_counts[tag] += 1
    print(f"\nSRL tag distribution (5K sample):")
    for tag, count in tag_counts.most_common(15):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
