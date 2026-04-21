"""Validate dep2label round-trip: encode → decode must reconstruct the original tree.

RUN THIS BEFORE TRAINING.  If any sentence fails, dep2label.py has a bug
and training on corrupted labels would be wasted.

Usage:
    python models/distilroberta-nlp-en/validate_dep2label.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import conllu

from dep2label import encode_sentence, decode_sentence


DATA_DIR = Path(__file__).parent.parent.parent / "data"
UD_DIR = DATA_DIR / "ud-english-ewt"


def validate_file(conllu_path: Path) -> tuple[int, int, list[str]]:
    """Round-trip every sentence in a CoNLL-U file.

    Returns (passed, total, error_messages).
    """
    with open(conllu_path) as f:
        sentences = conllu.parse(f.read())

    passed = 0
    total = 0
    errors = []

    for sent in sentences:
        tokens = [t for t in sent if isinstance(t["id"], int)]
        if not tokens:
            continue

        words = [t["form"] for t in tokens]
        upos = [t["upos"] for t in tokens]
        heads = [t["head"] - 1 if t["head"] > 0 else -1 for t in tokens]
        deprels = [t["deprel"] for t in tokens]

        total += 1
        sent_id = sent.metadata.get("sent_id", f"#{total}")

        # Encode
        try:
            labels = encode_sentence(words, heads, deprels, upos)
        except Exception as e:
            errors.append(f"{sent_id}: encode failed: {e}")
            continue

        if len(labels) != len(words):
            errors.append(f"{sent_id}: label count {len(labels)} != word count {len(words)}")
            continue

        # Decode
        try:
            pred_heads, pred_deprels = decode_sentence(labels, upos)
        except Exception as e:
            errors.append(f"{sent_id}: decode failed: {e}")
            continue

        # Compare
        head_match = pred_heads == heads
        rel_match = pred_deprels == deprels

        if head_match and rel_match:
            passed += 1
        else:
            mismatches = []
            for i in range(len(words)):
                if pred_heads[i] != heads[i]:
                    mismatches.append(
                        f"  token {i} '{words[i]}': head {heads[i]} → {pred_heads[i]}"
                    )
                elif pred_deprels[i] != deprels[i]:
                    mismatches.append(
                        f"  token {i} '{words[i]}': rel '{deprels[i]}' → '{pred_deprels[i]}'"
                    )
            detail = "\n".join(mismatches[:5])
            errors.append(f"{sent_id}: round-trip mismatch ({len(mismatches)} tokens)\n{detail}")

    return passed, total, errors


def main():
    splits = {
        "train": UD_DIR / "en_ewt-ud-train.conllu",
        "dev": UD_DIR / "en_ewt-ud-dev.conllu",
        "test": UD_DIR / "en_ewt-ud-test.conllu",
    }

    all_passed = True

    for name, path in splits.items():
        if not path.exists():
            print(f"⚠ {name}: file not found ({path}). Run data/download_ud.sh first.")
            all_passed = False
            continue

        passed, total, errors = validate_file(path)
        pct = 100 * passed / total if total > 0 else 0

        if passed == total:
            print(f"✓ {name}: {passed}/{total} sentences round-trip correctly (100%)")
        else:
            print(f"✗ {name}: {passed}/{total} ({pct:.1f}%) — {total - passed} FAILURES")
            all_passed = False
            for err in errors[:10]:
                print(f"  {err}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more")

    print()
    if all_passed:
        print("✓ All dep2label round-trip validations PASSED. Safe to train.")
        sys.exit(0)
    else:
        print("✗ dep2label round-trip validation FAILED. Fix dep2label.py before training.")
        sys.exit(1)


if __name__ == "__main__":
    main()
