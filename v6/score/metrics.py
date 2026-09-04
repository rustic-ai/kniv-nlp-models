"""Per-layer scoring against gold.

Every score is reported alongside **coverage** — the fraction of items the
annotator returned a usable answer for. Accuracy is computed over covered
items only, so a model that fails half the corpus cannot post a flattering
number; the pair (score, coverage) is the honest summary.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class LayerScore:
    layer: str
    annotator: str
    primary: float                  # accuracy | UAS | span-F1
    primary_name: str
    secondary: float | None = None  # LAS for dep, exact-match for morph
    secondary_name: str | None = None
    coverage: float = 0.0
    n_items: int = 0
    n_scored: int = 0
    n_units: int = 0                # tokens or spans compared
    well_formed_rate: float | None = None
    precision: float | None = None
    recall: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def bio_spans(tags: list[str]) -> set[tuple[str, int, int]]:
    """Extract ``(role, start, end_exclusive)`` spans from BIO tags.

    ``V`` (the predicate itself) is excluded — it is given, not predicted,
    so scoring it inflates F1. This matches the v5 evaluation convention.
    """
    spans: set[tuple[str, int, int]] = set()
    role: str | None = None
    start = 0
    for i, tag in enumerate(list(tags) + ["O"]):
        if tag.startswith("B-") or tag in ("O", "V") or not tag.startswith("I-"):
            if role is not None:
                spans.add((role, start, i))
                role = None
            if tag.startswith("B-"):
                role, start = tag[2:], i
        elif tag.startswith("I-"):
            if role is None:                 # I- without B-: treat as start
                role, start = tag[2:], i
            elif tag[2:] != role:
                spans.add((role, start, i))
                role, start = tag[2:], i
    return spans


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _feats_set(feats: str) -> set[str]:
    if not feats or feats == "_":
        return set()
    return {kv for kv in feats.split("|") if "=" in kv}


def score_layer(layer: str, annotator: str, gold_items: list,
                preds: dict[str, object],
                well_formed: dict[str, bool] | None = None) -> LayerScore:
    """Score one annotator on one layer.

    ``preds`` maps item id -> validated payload (absent id = failed item).
    """
    n_items = len(gold_items)
    scored = [it for it in gold_items if it.id in preds]
    n_scored = len(scored)
    coverage = n_scored / n_items if n_items else 0.0

    wf_rate = None
    if layer == "dep" and well_formed:
        vals = [well_formed[it.id] for it in scored if it.id in well_formed]
        wf_rate = sum(vals) / len(vals) if vals else None

    if not scored:
        return LayerScore(layer, annotator, 0.0, _PRIMARY[layer],
                          coverage=coverage, n_items=n_items,
                          well_formed_rate=wf_rate)

    if layer in ("pos", "lemma"):
        correct = total = 0
        for it in scored:
            gold = it.layers[layer]
            pred = preds[it.id]
            for g, p in zip(gold, pred):
                total += 1
                correct += (g == p)
        return LayerScore(layer, annotator, correct / total if total else 0.0,
                          "accuracy", coverage=coverage, n_items=n_items,
                          n_scored=n_scored, n_units=total)

    if layer == "morph":
        tp = n_pred = n_gold = 0
        exact = total = 0
        for it in scored:
            for g, p in zip(it.layers["morph"], preds[it.id]):
                gs, ps = _feats_set(g), _feats_set(p)
                tp += len(gs & ps)
                n_pred += len(ps)
                n_gold += len(gs)
                total += 1
                exact += (g == p)
        prec, rec, f1 = _prf(tp, n_pred, n_gold)
        return LayerScore(layer, annotator, f1, "feat-F1",
                          secondary=exact / total if total else 0.0,
                          secondary_name="exact-match",
                          coverage=coverage, n_items=n_items,
                          n_scored=n_scored, n_units=total,
                          precision=prec, recall=rec)

    if layer == "coref":
        from .coref import conll_f1
        agg = {"muc_f1": 0.0, "b3_f1": 0.0, "ceafe_f1": 0.0, "conll_f1": 0.0}
        n_clusters = 0
        for it in scored:
            r = conll_f1(it.layers["coref"], preds[it.id])
            for k in agg:
                agg[k] += r[k]
            n_clusters += len(it.layers["coref"])
        k = len(scored)
        return LayerScore(layer, annotator, agg["conll_f1"] / k, "CoNLL-F1",
                          secondary=agg["b3_f1"] / k, secondary_name="B3-F1",
                          coverage=coverage, n_items=n_items,
                          n_scored=n_scored, n_units=n_clusters)

    if layer == "dep":
        uas = las = total = 0
        for it in scored:
            g = it.layers["dep"]
            p = preds[it.id]
            for gh, gr, ph, pr in zip(g["heads"], g["rels"],
                                      p["heads"], p["rels"]):
                total += 1
                if gh == ph:
                    uas += 1
                    las += (gr == pr)
        return LayerScore(layer, annotator, uas / total if total else 0.0, "UAS",
                          secondary=las / total if total else 0.0,
                          secondary_name="LAS", coverage=coverage,
                          n_items=n_items, n_scored=n_scored, n_units=total,
                          well_formed_rate=wf_rate)

    if layer in ("srl", "ner"):
        tp = n_pred = n_gold = 0
        for it in scored:
            gs = bio_spans(it.layers[layer])
            ps = bio_spans(preds[it.id])
            tp += len(gs & ps)
            n_pred += len(ps)
            n_gold += len(gs)
        prec, rec, f1 = _prf(tp, n_pred, n_gold)
        return LayerScore(layer, annotator, f1, "span-F1", coverage=coverage,
                          n_items=n_items, n_scored=n_scored,
                          n_units=n_gold, precision=prec, recall=rec)

    raise ValueError(f"no scorer for layer {layer!r}")


_PRIMARY = {"pos": "accuracy", "lemma": "accuracy", "morph": "feat-F1",
            "dep": "UAS", "srl": "span-F1", "ner": "span-F1",
            "coref": "CoNLL-F1"}


def pairwise_agreement(layer: str, items: list,
                       per_annotator: dict[str, dict]) -> dict[str, float]:
    """Token-level agreement between every pair of annotators.

    This is the empirical answer to "are these annotators actually
    independent?". Consensus only decorrelates errors across genuinely
    different lineages; two variants of one base model will agree with each
    other far above their agreement with an outside family, and that shows up
    here as a number rather than an assumption. High mutual agreement paired
    with a low score against gold means the ensemble is confidently wrong.
    """
    names = sorted(per_annotator)
    out: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pa, pb = per_annotator[a], per_annotator[b]
            same = total = 0
            for it in items:
                if it.id not in pa or it.id not in pb:
                    continue
                if layer == "dep":
                    for h1, r1, h2, r2 in zip(pa[it.id]["heads"], pa[it.id]["rels"],
                                              pb[it.id]["heads"], pb[it.id]["rels"]):
                        total += 1
                        same += (h1 == h2 and r1 == r2)
                else:
                    for x, y in zip(pa[it.id], pb[it.id]):
                        total += 1
                        same += (x == y)
            if total:
                out[f"{a}|{b}"] = same / total
    return out
