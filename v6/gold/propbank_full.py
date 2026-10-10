"""PropBank EWT with every predicate of every sentence, aligned to UD EWT.

    uv run python -m v6.gold.propbank_full          # builds, verifies, writes

v5's ``prepare_srl.py`` kept only each sentence's dependency-root predicate
("filter to root-verb only"), so the prepared SRL files hold about one
predicate per sentence. That cannot teach predicate *identification*: a
predicate left out would be taught as "not a predicate". The PropBank release
annotates every predicate -- verbal, and the nominal and adjectival ones of
PropBank's later expansion -- so this rebuilds the EWT portion from the
release's ``.gold_skel`` files with all of them.

Alignment, role normalisation and the token-count check are ported from v5
(v6 does not import v5 code), so the root-predicate subset of the result must
reproduce ``srl_test.json`` frame for frame; ``build`` refuses to write if it
does not.

Needs ``data/propbank-release`` (github.com/propbank/propbank-release,
CC-BY-SA-4.0; annotations only, the text comes from UD EWT).

Output, one file per split under ``data/prepared/propbank-full/``::

    {"words": [...], "upos": [...], "sent_id": "...",
     "frames": [{"predicate_idx": int, "roleset": "launch.01", "srl_tags": [...]}, ...]}
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path

import conllu

from ..config import DATA_DIR, PREPARED_DIR, UD_DIR
from ..schemas import SRL_TAGS

RELEASE = DATA_DIR / "propbank-release"
OUT = DATA_DIR / "prepared" / "propbank-full"
SPLITS = {"train": "train", "dev": "dev", "test": "test"}


def _bio(col_values: list[str]) -> list[str]:
    """One predicate column's bracket notation -> BIO (v5's parser)."""
    bio = ["O"] * len(col_values)
    current = None
    for i, col in enumerate(col_values):
        if col == "*":
            if current:
                bio[i] = f"I-{current}"
        elif col.startswith("(") and col.endswith("*)"):
            role = col[1:-2]
            bio[i] = "V" if role == "V" else f"B-{role}"
        elif col.startswith("(") and col.endswith("*"):
            role = col[1:-1]
            if role == "V":
                bio[i] = "V"
            else:
                bio[i] = f"B-{role}"
                current = role
        elif col == "*)":
            if current:
                bio[i] = f"I-{current}"
                current = None
        elif col.startswith("(") and not col.endswith("*"):
            role = col.lstrip("(").rstrip(")")
            if role == "V":
                bio[i] = "V"
            elif role:
                bio[i] = f"B-{role}"
                current = role
    return bio


def _normalise(tags: list[str], stats: Counter) -> list[str]:
    """v5's normalisation: C-/R- prefixes dropped, roles outside the 42 -> O."""
    out = []
    for tag in tags:
        if tag in ("O", "V"):
            out.append(tag)
        elif tag[:2] in ("B-", "I-"):
            prefix, role = tag[:2], tag[2:]
            if role.startswith(("C-", "R-")):
                role = role[2:]
            full = f"{prefix}{role}"
            if full in SRL_TAGS:
                out.append(full)
            else:
                stats[f"role outside inventory: {role}"] += 1
                out.append("O")
        else:
            out.append("O")
    return out


def _index(release: Path) -> dict[tuple, list]:
    """``(ud_doc_prefix, sentence index) -> [predicate annotations]``."""
    files = glob.glob(str(release / "data" / "google" / "ewt" / "**" / "*.gold_skel"),
                      recursive=True)
    if not files:
        raise SystemExit(f"no .gold_skel under {release}; clone propbank-release first")
    index: dict[tuple, list] = {}
    for path in files:
        sents, cur = [], []
        for line in Path(path).read_text().splitlines():
            if line.strip():
                cur.append(line.split())
            elif cur:
                sents.append(cur); cur = []
        if cur:
            sents.append(cur)
        for toks in sents:
            toks = [t for t in toks if len(t) >= 8]
            if not toks:
                continue
            doc, sidx = toks[0][0], int(toks[0][1])
            genre_parts = doc.replace("google/ewt/", "").split("/")
            prefix = f"{genre_parts[0]}-{genre_parts[-1].replace('.xml', '')}"
            for col in range(len(toks[0]) - 8):
                values = [t[8 + col] for t in toks]
                pred = next((t for t in toks if t[8 + col] in ("(V*)", "(V*")), None)
                index.setdefault((prefix, sidx), []).append({
                    "predicate_idx": int(pred[2]) if pred else None,
                    "roleset": pred[7] if pred else None,
                    "bio": _bio(values), "num_tokens": len(toks)})
    return index


def build(release: Path = RELEASE) -> tuple[dict, Counter]:
    index = _index(release)
    stats = Counter()
    data = {}
    for split, ud in SPLITS.items():
        rows = []
        for sent in conllu.parse((UD_DIR / f"en_ewt-ud-{ud}.conllu").read_text(encoding="utf-8")):
            toks = [t for t in sent if isinstance(t["id"], int)]
            sid = sent.metadata.get("sent_id", "")
            if "-" not in sid:
                continue
            prefix, num = sid[:sid.rfind("-")], int(sid[sid.rfind("-") + 1:])
            preds = index.get((prefix, num - 1), [])
            if not preds:
                stats[f"{split}: sentences without PropBank"] += 1
                continue
            if preds[0]["num_tokens"] != len(toks):
                stats[f"{split}: token count differs from UD (skipped)"] += 1
                continue
            frames = []
            for p in preds:
                if p["predicate_idx"] is None:
                    stats["predicate column without a V"] += 1
                    continue
                frames.append({"predicate_idx": p["predicate_idx"], "roleset": p["roleset"],
                               "srl_tags": _normalise(p["bio"], stats)})
            rows.append({"sent_id": sid, "words": [t["form"] for t in toks],
                         "upos": [t["upostag"] for t in toks],
                         "root": next((i for i, t in enumerate(toks) if t["head"] == 0), None),
                         "frames": frames})
        data[split] = rows
    return data, stats


def verify_against_v5(data: dict) -> tuple[int, int]:
    """The root-predicate subset must reproduce v5's ``srl_test.json``."""
    v5 = json.loads((PREPARED_DIR / "srl_test.json").read_text())
    # Keyed by text, so identical sentences (stock phrases recur in EWT) keep
    # every one of their root frames rather than overwriting each other.
    ours: dict[tuple, list] = {}
    for r in data["test"]:
        for f in r["frames"]:
            if f["predicate_idx"] == r["root"]:
                ours.setdefault((tuple(r["words"]), f["predicate_idx"]), []).append(f["srl_tags"])
    same = sum(e["srl_tags"] in ours.get((tuple(e["words"]), e["predicate_idx"]), [])
               for e in v5)
    return same, len(v5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--release", type=Path, default=RELEASE)
    args = ap.parse_args()
    data, stats = build(args.release)
    same, total = verify_against_v5(data)
    print(f"root-predicate subset vs v5's srl_test.json: {same}/{total} frames identical")
    if same != total:
        raise SystemExit("alignment does not reproduce v5's test file; not writing")
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in data.items():
        (OUT / f"{split}.json").write_text(json.dumps(rows))
        frames = [f for r in rows for f in r["frames"]]
        pos = Counter(r["upos"][f["predicate_idx"]] for r in rows for f in r["frames"])
        n = sum(pos.values())
        print(f"{split:5s}: {len(rows):6,} sentences, {len(frames):6,} frames "
              f"({len(frames) / max(len(rows), 1):.2f}/sentence); predicate POS "
              + ", ".join(f"{k} {v / n:.1%}" for k, v in pos.most_common(5)))
    for k, v in sorted(stats.items()):
        print(f"  {k}: {v:,}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
