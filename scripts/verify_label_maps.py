"""Assert that ``models/label_maps.json`` matches ``models/student_loader.py``.

The Python module is the source of truth; the JSON is the cross-language
artifact (Rust uniko, the ONNX inference path, any future JS client). If
the two ever drift the resulting bugs are silent and far from the
mismatch — labels just decode to the wrong class.

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

    if failures:
        print(f"\n{failures} mismatch(es). Update one side to match the other.",
              file=sys.stderr)
        return 1
    print(f"\nAll 5 label sets in {path.name} match student_loader.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
