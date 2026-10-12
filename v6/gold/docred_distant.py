"""DocRED's distantly supervised training split (MIT): 101,873 Wikipedia
introductions whose relations come from aligning Wikidata, not from people.

Its labels are noisy both ways -- a Wikidata fact the text does not state is
labelled, one it states but Wikidata lacks is not -- so it is training
material for the schema-conditioned relation head, treated as partial, and
never an evaluation set. Documents whose titles are in Re-DocRED dev or test
are excluded, so nothing evaluated there is seen in training.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..config import DATA_DIR

DIR = DATA_DIR / "docred"
HF_REPO = "thunlp/docred"


def download() -> Path:
    dst = DIR / "train_distant.parquet"
    if not dst.exists():
        import shutil
        from huggingface_hub import hf_hub_download
        DIR.mkdir(parents=True, exist_ok=True)
        p = hf_hub_download(HF_REPO, "default/train_distant/0000.parquet",
                            repo_type="dataset", revision="refs/convert/parquet")
        shutil.copy(p, dst)
    return dst


def _held_titles() -> set[str]:
    from .redocred import download as rd_download
    return {d["title"] for sp in ("dev", "test") for d in json.loads(rd_download(sp).read_text())}


def load(n: int | None = 40000, seed: str = "distant") -> list[dict]:
    """A deterministic subset of ``n`` documents (hash order), as
    ``{"id", "tokens", "sentence_spans", "entities": [{"type", "mentions":
    [(start, end_exclusive)]}], "triples": [(h, t, Wikidata code)]}``."""
    import pyarrow.parquet as pq
    held = _held_titles()
    rows = pq.read_table(download()).to_pylist()
    rows = [r for r in rows if r["title"] not in held]
    rows.sort(key=lambda r: hashlib.sha1(f"{seed}:{r['title']}".encode()).hexdigest())
    out = []
    for r in rows[:n] if n else rows:
        tokens, spans = [], []
        for s in r["sents"]:
            spans.append([len(tokens), len(tokens) + len(s)])
            tokens.extend(s)
        ents = [{"type": chain[0]["type"],
                 "mentions": sorted({(spans[m["sent_id"]][0] + m["pos"][0],
                                      spans[m["sent_id"]][0] + m["pos"][1]) for m in chain})}
                for chain in r["vertexSet"]]
        lab = r["labels"]
        triples = sorted({(h, t, c) for h, t, c in zip(lab["head"], lab["tail"], lab["relation_id"])})
        out.append({"id": r["title"], "tokens": tokens, "sentence_spans": spans,
                    "entities": ents, "triples": triples})
    return out
