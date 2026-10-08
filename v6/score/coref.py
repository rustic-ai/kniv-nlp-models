"""Standard coreference metrics: MUC, B-cubed, CEAF-e, and their CoNLL mean.

Implemented directly so the numbers are comparable to published coreference
results rather than to a bespoke link-counting score. CoNLL F1 — the mean of
the three — is the figure the literature reports.
"""
from __future__ import annotations

Span = tuple[int, int]
Cluster = list[Span]


def _as_sets(clusters) -> list[set[Span]]:
    return [{(int(s), int(e)) for s, e in c} for c in clusters if len(c) > 0]


def muc(gold, pred) -> tuple[float, float, float]:
    """Link-based: how many merge operations are shared."""
    def score(a, b):
        num = den = 0
        for c in a:
            den += len(c) - 1
            # partitions of c induced by b (unmatched mentions count alone)
            parts = 0
            covered = set()
            for d in b:
                inter = c & d
                if inter:
                    parts += 1
                    covered |= inter
            parts += len(c - covered)
            num += len(c) - parts
        return num / den if den else 0.0
    g, p = _as_sets(gold), _as_sets(pred)
    r, pr = score(g, p), score(p, g)
    return pr, r, (2 * pr * r / (pr + r) if pr + r else 0.0)


def b_cubed(gold, pred) -> tuple[float, float, float]:
    """Mention-based: per-mention overlap of its gold and predicted cluster."""
    g, p = _as_sets(gold), _as_sets(pred)
    def score(a, b):
        total = n = 0.0
        for c in a:
            for m in c:
                d = next((x for x in b if m in x), None)
                n += 1
                if d:
                    total += len(c & d) / len(c)
        return total / n if n else 0.0
    r, pr = score(g, p), score(p, g)
    return pr, r, (2 * pr * r / (pr + r) if pr + r else 0.0)


def ceafe(gold, pred) -> tuple[float, float, float]:
    """Entity-based: optimal 1:1 cluster alignment under the phi-4 similarity."""
    g, p = _as_sets(gold), _as_sets(pred)
    if not g or not p:
        return 0.0, 0.0, 0.0
    sim = [[2 * len(a & b) / (len(a) + len(b)) for b in p] for a in g]
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
        rows, cols = linear_sum_assignment(-np.array(sim))
        total = sum(sim[i][j] for i, j in zip(rows, cols))
    except ImportError:                        # greedy fallback
        used, total = set(), 0.0
        for i in sorted(range(len(g)), key=lambda i: -max(sim[i])):
            j = max((j for j in range(len(p)) if j not in used),
                    key=lambda j: sim[i][j], default=None)
            if j is not None:
                total += sim[i][j]
                used.add(j)
    pr = total / len(p)
    r = total / len(g)
    return pr, r, (2 * pr * r / (pr + r) if pr + r else 0.0)


def conll_f1(gold, pred) -> dict:
    mp, mr, mf = muc(gold, pred)
    bp, br, bf = b_cubed(gold, pred)
    cp, cr, cf = ceafe(gold, pred)
    return {"muc_f1": mf, "b3_f1": bf, "ceafe_f1": cf,
            "conll_f1": (mf + bf + cf) / 3,
            "b3_p": bp, "b3_r": br}
