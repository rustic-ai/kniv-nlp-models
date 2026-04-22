"""Prepare SRL training data from three sources.

1. PropBank EWT: gold annotations aligned to UD EWT (CC-BY-SA-4.0)
2. QA-SRL Bank 2.0: crowdsourced, converted to BIO (MIT)
3. Few-NERD + silver: model-generated via liaad/srl-en_xlmr-large (Apache 2.0)

Produces: srl_train.json, srl_dev.json, srl_test.json

Usage:
    python models/kniv-deberta-cascade-large-nlp-en/prepare_srl.py
"""

from __future__ import annotations

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


# ── Source 2: QA-SRL Bank 2.0 (MIT) ─────────────────────────────

# QA question patterns → PropBank roles
QASRL_WH_ROLE = {
    "where": "ARGM-LOC",
    "when": "ARGM-TMP",
    "why": "ARGM-CAU",
    "how": "ARGM-MNR",
    "how much": "ARGM-EXT",
    "how long": "ARGM-TMP",
}


def _qasrl_question_to_role(question: str, is_passive: bool) -> str | None:
    """Map a QA-SRL question to a PropBank role."""
    q_lower = question.lower().strip()

    # Check modifier WH-words first
    for wh, role in QASRL_WH_ROLE.items():
        if q_lower.startswith(wh):
            return role

    # Core roles: who/what — depends on position and voice
    if q_lower.startswith("who") or q_lower.startswith("what"):
        # If the question asks about the subject position
        # "Who killed someone?" → ARG0 (active) or ARG1 (passive)
        # "What was killed?" → ARG1
        if "was " in q_lower or "were " in q_lower or "been " in q_lower:
            return "ARG1"
        if is_passive:
            # "Who was [verb]ed by?" → ARG0
            if " by " in q_lower:
                return "ARG0"
            return "ARG1"
        # Active voice: subject = ARG0, object = ARG1
        # Heuristic: if question has "someone/something" as object → ARG0
        if "someone" in q_lower or "something" in q_lower:
            return "ARG0"
        return "ARG1"

    return None


def load_qasrl_bank(
    subsample: int = 25000,
    seed: int = 42,
    data_dir: Path | None = None,
) -> list[dict]:
    """Load QA-SRL Bank 2.0, convert to BIO SRL format.

    Downloads from HuggingFace if not found locally.
    Filters to root-verb predicates and subsamples.
    """
    from datasets import load_dataset

    print("  Loading QA-SRL Bank 2.0...")
    ds = load_dataset("biu-nlp/qa_srl2018", trust_remote_code=True)

    all_examples = []
    skipped = 0

    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for row in ds[split]:
            words = row["words"]
            if len(words) < 3:
                continue

            # Find root verb (simple heuristic: first verb)
            # QA-SRL annotates specific predicates — use their index
            predicate_idx = row.get("predicate_idx") or row.get("verb_idx")
            if predicate_idx is None:
                skipped += 1
                continue

            # Get QA pairs for this predicate
            questions = row.get("questions", [])
            answers = row.get("answers", [])
            if not questions:
                skipped += 1
                continue

            # Check if passive
            is_passive = row.get("is_passive", False)

            # Convert each QA pair to a span + role
            bio_tags = ["O"] * len(words)
            bio_tags[predicate_idx] = "V"
            valid = True

            for q, a in zip(questions, answers):
                role = _qasrl_question_to_role(q, is_passive)
                if role is None:
                    continue

                # Answer span
                spans = a if isinstance(a, list) else [a]
                for span in spans:
                    if isinstance(span, dict):
                        start = span.get("start", span.get("s", 0))
                        end = span.get("end", span.get("e", start + 1))
                    elif isinstance(span, (list, tuple)) and len(span) >= 2:
                        start, end = span[0], span[1]
                    else:
                        continue

                    # Only assign if all tokens in range are currently O
                    if start < len(words) and end <= len(words):
                        if all(bio_tags[t] == "O" for t in range(start, end)):
                            bio_tags[start] = f"B-{role}"
                            for t in range(start + 1, end):
                                bio_tags[t] = f"I-{role}"

            # Only keep if we got at least one argument
            non_o = sum(1 for t in bio_tags if t not in ("O", "V"))
            if non_o == 0:
                skipped += 1
                continue

            all_examples.append({
                "words": words,
                "text": " ".join(words),
                "srl_tags": bio_tags,
                "predicate_idx": predicate_idx,
            })

    print(f"  QA-SRL: {len(all_examples)} converted, {skipped} skipped")

    # Subsample
    rng = random.Random(seed)
    if len(all_examples) > subsample:
        all_examples = rng.sample(all_examples, subsample)
        print(f"  Subsampled to {subsample}")

    return all_examples


# ── Source 3: Silver SRL from Few-NERD ───────────────────────────

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

    print(f"  Loading liaad/srl-en_xlmr-large...")
    model_name = "liaad/srl-en_xlmr-large"
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
        encoding = {k: v.to(device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = model(**encoding)
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
                    if label in SRL_TAGS:
                        bio_tags[wid] = label
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load UD EWT for PropBank alignment
    from prepare_data import load_ud_data
    ud_train = load_ud_data(UD_DIR / "en_ewt-ud-train.conllu")
    ud_dev = load_ud_data(UD_DIR / "en_ewt-ud-dev.conllu")
    ud_test = load_ud_data(UD_DIR / "en_ewt-ud-test.conllu")

    ud_splits = {"train": ud_train, "validation": ud_dev, "test": ud_test}

    # Source 1: PropBank gold
    propbank = load_propbank_ewt(ud_splits)
    print(f"  PropBank total: train={len(propbank['train'])}, "
          f"dev={len(propbank['validation'])}, test={len(propbank['test'])}")

    # Source 2: Silver SRL from Few-NERD
    # (QA-SRL Bank 2.0 skipped — HF loading script deprecated, data URL broken)
    print("\nGenerating silver SRL from Few-NERD...")
    silver = generate_silver_srl(subsample=70000)
    print(f"  Silver total: {len(silver)}")

    # Merge: train = PropBank gold + silver
    # Dev/test = PropBank gold only (reliable evaluation)
    srl_train = propbank["train"] + silver
    random.shuffle(srl_train)

    srl_dev = propbank["validation"]
    srl_test = propbank["test"]

    print(f"\nFinal SRL dataset:")
    print(f"  Train: {len(srl_train)} (PropBank {len(propbank['train'])} + "
          f"Silver {len(silver)})")
    print(f"  Dev: {len(srl_dev)} (PropBank gold)")
    print(f"  Test: {len(srl_test)} (PropBank gold)")

    # Save
    for name, data in [("srl_train", srl_train), ("srl_dev", srl_dev), ("srl_test", srl_test)]:
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
