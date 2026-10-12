"""Few-NERD, supervised setting (CC BY-SA 4.0): 188k Wikipedia sentences,
66 fine-grained entity types under 8 coarse ones, human-annotated.

Used for the schema-conditioned entity head: training on its train split,
zero-shot and supervised evaluation on its test split. Tags are IO, not
BIO -- a run of tokens with the same fine type is one span -- so two
adjacent same-type entities merge, as in the dataset's own evaluation.
"""
from __future__ import annotations

import json
import urllib.request

from ..config import DATA_DIR

FEWNERD_DIR = DATA_DIR / "fewnerd"
HF = "https://huggingface.co/datasets/DFKI-SLT/few-nerd/resolve/main/supervised"
SPLITS = {"train": "train", "dev": "validation", "test": "test"}


def _parquet(split: str):
    dst = FEWNERD_DIR / f"supervised-{SPLITS[split]}.parquet"
    if not dst.exists():
        FEWNERD_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{HF}/{SPLITS[split]}-00000-of-00001.parquet",
                                    timeout=300) as r:
            dst.write_bytes(r.read())
    return dst


def fine_types() -> list[str]:
    """The 66 fine types (index 0, 'O', excluded), as ``coarse-fine``."""
    import pyarrow.parquet as pq
    md = json.loads(pq.read_schema(_parquet("dev")).metadata[b"huggingface"])
    return md["info"]["features"]["fine_ner_tags"]["feature"]["names"][1:]


def type_text(code: str) -> tuple[str, str]:
    """``("film", "a film, a work of art")`` from ``art-film``."""
    coarse, fine = code.split("-", 1)
    coarse_name = {"art": "work of art", "building": "building", "event": "event",
                   "location": "location", "organization": "organization",
                   "other": "other named thing", "person": "person",
                   "product": "product"}[coarse]
    fine_name = fine.replace("/", " or ").replace("broadcastprogram", "broadcast program") \
        .replace("writtenart", "written work").replace("sportsfacility", "sports facility") \
        .replace("theater", "theatre").replace("sportsevent", "sports event") \
        .replace("attack or battle or war or militaryconflict", "attack, battle or war") \
        .replace("politicalparty", "political party").replace("sportsleague", "sports league") \
        .replace("sportsteam", "sports team").replace("showorganization", "show organization") \
        .replace("religion", "religion").replace("bodiesofwater", "body of water") \
        .replace("road or railway or highway or transit", "road, railway or transit line") \
        .replace("educationaldegree", "educational degree").replace("livingthing", "living thing") \
        .replace("chemicalthing", "chemical").replace("biologything", "biological thing") \
        .replace("astronomything", "astronomical object").replace("medical", "medical thing") \
        .replace("currency", "currency").replace("god", "god").replace("law", "law") \
        .replace("disease", "disease").replace("language", "language") \
        .replace("software", "software").replace("weapon", "weapon").replace("ship", "ship") \
        .replace("airplane", "airplane").replace("car", "car").replace("train", "train") \
        .replace("game", "game").replace("food", "food").replace("award", "award")
    if fine == "other":
        return f"other {coarse_name}", f"a {coarse_name} of a kind not listed elsewhere"
    return fine_name, f"a {fine_name}, a kind of {coarse_name}"


def load(split: str, limit: int | None = None) -> list[dict]:
    """``{"tokens", "spans": [(type_code, start, end_exclusive)]}`` per sentence."""
    import pyarrow.parquet as pq
    names = ["O"] + fine_types()
    out = []
    for r in pq.read_table(_parquet(split), columns=["tokens", "fine_ner_tags"]).to_pylist():
        spans, cur, start = [], 0, 0
        tags = r["fine_ner_tags"] + [0]
        for i, t in enumerate(tags):
            if t != cur:
                if cur:
                    spans.append((names[cur], start, i))
                cur, start = t, i
        out.append({"tokens": r["tokens"], "spans": spans})
        if limit and len(out) >= limit:
            break
    return out
