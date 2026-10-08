"""Rebuild OntoNotes 5.0 NER gold from tner's authoritative label map.

Why this exists: ``benchmarks/ontonotes5_test.json`` as shipped in the
kniv-corpus-en dataset repo carries a broken tag-id -> tag-name mapping.
2,205 tokens (1.44%) are mislabelled — ``Europe`` tagged ``I-TIME``, ``1/4``
tagged ``B-TIME`` — which drives six entity types to exactly 0.000 F1 and
understates any model scored against it by ~10 F1 points.

Rather than trust a checked-in conversion, this rebuilds the gold directly
from ``tner/ontonotes5``'s own ``dataset/label.json`` and verifies the result
before writing.

Usage:
    uv run python -m v6.gold.build_ner_gold
    uv run python -m v6.gold.build_ner_gold --check   # report drift, write nothing
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from ..config import DATA_DIR

BENCH_DIR = DATA_DIR / "benchmarks"
REPO = "tner/ontonotes5"

# Sentinel facts about the authoritative mapping. If tner ever renumbers,
# these fail loudly rather than silently producing a differently-wrong file.
EXPECTED_N_LABELS = 37
SPOT_CHECKS = {0: "O", 21: "B-TIME", 23: "B-LOC", 29: "I-TIME", 36: "I-LANGUAGE"}


def fetch_authoritative() -> tuple[dict[int, str], list[dict]]:
    from huggingface_hub import hf_hub_download

    lbl = json.loads(Path(hf_hub_download(
        REPO, "dataset/label.json", repo_type="dataset")).read_text())
    id2 = {v: k for k, v in lbl.items()}
    if len(id2) != EXPECTED_N_LABELS:
        raise RuntimeError(
            f"{REPO} label map has {len(id2)} entries, expected "
            f"{EXPECTED_N_LABELS}. The upstream tagset changed — re-verify "
            f"before regenerating gold.")
    for idx, name in SPOT_CHECKS.items():
        if id2.get(idx) != name:
            raise RuntimeError(
                f"{REPO} label {idx} is {id2.get(idx)!r}, expected {name!r}.")

    rows = [json.loads(line) for line in Path(hf_hub_download(
        REPO, "dataset/test.json", repo_type="dataset")).read_text().splitlines()
        if line.strip()]
    return id2, rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true",
                    help="Report drift against the existing file; write nothing")
    args = ap.parse_args()

    id2, rows = fetch_authoritative()
    fixed = [{"tokens": r["tokens"], "tags": [id2[t] for t in r["tags"]]}
             for r in rows]
    print(f"Rebuilt {len(fixed):,} sentences from {REPO} (label map verified)")

    dst = BENCH_DIR / "ontonotes5_test.json"
    if dst.exists():
        existing = json.loads(dst.read_text())
        bad = collections.Counter()
        ntok = 0
        for old, new in zip(existing, fixed):
            for a, b in zip(old.get("tags", []), new["tags"]):
                ntok += 1
                if a != b:
                    bad[(a, b)] += 1
        n_bad = sum(bad.values())
        if n_bad:
            print(f"  existing file disagrees on {n_bad:,}/{ntok:,} tokens "
                  f"({n_bad / ntok:.2%})")
            for (a, b), n in bad.most_common(6):
                print(f"    {a:<14} should be {b:<14} {n:>5}")
        else:
            print("  existing file already matches — nothing to do")
            return 0

    if args.check:
        print("  --check: no files written")
        return 0

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        shipped = BENCH_DIR / "ontonotes5_shipped_test.json"
        if not shipped.exists():
            shipped.write_text(dst.read_text())
            print(f"  preserved original as {shipped.name}")
    dst.write_text(json.dumps(fixed))
    print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
