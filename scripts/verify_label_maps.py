"""Assert the v5 student label sets are internally consistent.

Two checks, both guarding the same failure mode: a label file that disagrees
with the weights decodes to the wrong class, silently and far from the
mismatch.

1. ``models/label_maps.json`` matches ``models/student_loader.py``. The Python
   module is the source of truth; the JSON is the cross-language artifact
   (Rust uniko, the ONNX inference path, any future JS client).

2. No ``label_vocabs.json`` sits in a student model directory. That filename
   belongs to the dep2label generation (``deberta-v3-*``,
   ``kniv-deberta-cascade-*``), whose DEP head is a linear classifier over
   ~1,411 ``{offset}@{deprel}@{head_UPOS}`` composites and whose CLS head has
   9 units. The student family is biaffine over 53 plain deprels with 8 CLS
   units. One such file was found beside the v5 large checkpoint: decoding
   with it read CLS index 1 as ``correction`` where the model means
   ``request``, and only four of the nine names overlapped at all.

Run this as part of CI, or manually after touching either file:

    uv run python scripts/verify_label_maps.py

Exits non-zero on any mismatch with a diff-style report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))

from student_loader import (  # noqa: E402
    POS_LABELS, NER_LABELS, SRL_TAGS, CLS_LABELS, DEPREL_LIST,
)

EXPECTED = {
    "pos": POS_LABELS,
    "ner": NER_LABELS,
    "srl": SRL_TAGS,
    "cls": CLS_LABELS,
    "deprel": DEPREL_LIST,
}


# Student directories use label_maps.json; label_vocabs.json means dep2label.
STUDENT_GLOB = "kniv-deberta-nlp-base-en-*"


def check_no_stale_vocabs() -> int:
    """Fail if a dep2label-era vocabulary file sits in a student directory."""
    offenders = sorted((REPO / "models").glob(f"{STUDENT_GLOB}/label_vocabs.json"))
    for p in offenders:
        rel = p.relative_to(REPO)
        print(f"ERROR: {rel} should not exist", file=sys.stderr)
        print("  label_vocabs.json is the dep2label generation's file "
              "(9 CLS units, ~1,411 composite DEP tags).", file=sys.stderr)
        print("  This model family is 8 CLS units and 53 plain deprels; its "
              "label file is label_maps.json.", file=sys.stderr)
        print("  Decoding student output with it mislabels every CLS "
              "prediction. Delete it.", file=sys.stderr)
    return len(offenders)


def main() -> int:
    path = REPO / "models" / "label_maps.json"
    if not path.exists():
        print(f"ERROR: {path} does not exist", file=sys.stderr)
        return 1

    data = json.loads(path.read_text())
    failures = 0
    for key, expected in EXPECTED.items():
        actual = data.get(key)
        if actual is None:
            print(f"ERROR: '{key}' missing from {path.name}", file=sys.stderr)
            failures += 1
            continue
        if actual != expected:
            print(f"ERROR: '{key}' mismatch", file=sys.stderr)
            print(f"  Python ({len(expected)}): {expected[:5]} ... {expected[-3:]}",
                  file=sys.stderr)
            print(f"  JSON   ({len(actual)}): {actual[:5]} ... {actual[-3:]}",
                  file=sys.stderr)
            # First differing index
            for i in range(min(len(expected), len(actual))):
                if expected[i] != actual[i]:
                    print(f"  first diff at i={i}: "
                          f"py={expected[i]!r} json={actual[i]!r}",
                          file=sys.stderr)
                    break
            failures += 1
        else:
            print(f"  ✓ {key}: {len(actual)} labels")

    stale = check_no_stale_vocabs()
    if not stale:
        print(f"  ✓ no label_vocabs.json in {STUDENT_GLOB} directories")

    if failures or stale:
        if failures:
            print(f"\n{failures} mismatch(es). "
                  "Update one side to match the other.", file=sys.stderr)
        if stale:
            print(f"{stale} stale vocabulary file(s).", file=sys.stderr)
        return 1
    print(f"\nAll 5 label sets in {path.name} match student_loader.py, "
          "and no stale vocabulary files are present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
