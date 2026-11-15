"""The held-out news set (plan r4 V3, step 1): a verifier's score against
d123's margin on the same judged candidates, under GPT-6, DeepSeek and
agreement.

    uv run python -m v6.eval.v3_news labels                       # DeepSeek's c3k labels as an M1 label file, and agreement
    uv run python -m v6.train.verifier score-m1 --run ultrabert-380m-v8p3-d123-ver1c \\
        --name ultrabert-380m-v8p3-d123-news-eval --config base --tag v1c
    uv run python -m v6.eval.v3_news read --scored ultrabert-380m-v8p3-d123-news-eval-v-v1c-base

The candidates are d123's end to end at floor 3 (down to 3 under the bar,
the best two relations per pair), each judged by both judges. ``read``:
AUC of each score on expressed / not; precision of the top ``k`` facts by
each score, ``k`` the number d123 returns at its deployed bar (margin above
0), overall with a paired bootstrap over texts and per relation. Texts a
judge did not read (content filter, a failed call) are left out of that
judge's reading.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

import numpy as np

from .m1_precision import OUT, POSITIVE, labels_path

RUN = "ultrabert-380m-v8p3-d123-news-eval"


def labels(run: str = RUN, s6_name: str = "news-eval") -> int:
    from .s5_judges import merge
    from .s6_second import _path
    A = json.loads(labels_path(run, "astra", "c3").read_text())
    D = {t["id"]: t for t in json.loads(_path(s6_name).read_text())}
    texts = []
    for t in A["texts"]:
        d = D.get(t["id"])
        ok = d is not None and d["judged"] and len(d["facts"]) == len(t["facts"])
        facts = [{**f, "verdict": (g["verdict"] if ok else None)}
                 for f, g in zip(t["facts"], d["facts"] if ok else [{}] * len(t["facts"]))]
        texts.append({"id": t["id"], "judged": ok, "facts": facts})
    labels_path(run, "deepseek", "c3").write_text(json.dumps({"configs": A["configs"], "texts": texts}))
    print(f"deepseek labels: {sum(t['judged'] for t in texts)} of {len(texts)} texts judged")
    return merge(run, "astra", "deepseek", "c3")


def _auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    o = s.argsort().argsort() + 1
    n1, n0 = y.sum(), (~y).sum()
    return float((o[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))


def read(scored: str, run: str = RUN, base: str = "base", boot: int = 2000) -> int:
    P = pickle.loads((OUT / f"preds-{scored}.pkl").read_bytes())
    ver = f"{base}+v"
    for judge in ("astra", "deepseek", "both"):
        L = json.loads(labels_path(run, judge, "c3").read_text())
        rows = []                                    # (text, relation, margin, verifier score, expressed)
        for i, t in enumerate(L["texts"]):
            if not t["judged"]:
                continue
            for f in t["facts"]:
                if f["verdict"] is None:
                    continue
                k = next((k for k in P["margins"][base][i] if (sorted(k[0]), k[1], sorted(k[2]))
                          == (sorted(f["key"][0]), f["key"][1], sorted(f["key"][2]))), None)
                if k is None:
                    continue
                rows.append((i, k[1], P["margins"][base][i][k], P["margins"][ver][i][k], f["verdict"] in POSITIVE))
        texts = sorted({r[0] for r in rows})
        y = np.array([r[4] for r in rows])
        m = np.array([r[2] for r in rows])
        v = np.array([r[3] for r in rows])
        unread = int((v <= -1e2).sum())
        K = int((m > 0).sum())
        top_m, top_v = np.argsort(-m)[:K], np.argsort(-v)[:K]
        print(f"\n## {judge}: {len(texts)} texts, {len(rows)} judged candidates ({y.mean():.3f} expressed), "
              f"{unread} the verifier could not read (ranked last)")
        print(f"AUC: d123 margin {_auc(m, y):.3f}, verifier {_auc(v, y):.3f}")
        # paired bootstrap over texts at equal size: precision of the top K by each score
        sel_m, sel_v = np.zeros(len(rows), bool), np.zeros(len(rows), bool)
        sel_m[top_m] = True; sel_v[top_v] = True
        ti = np.array([texts.index(r[0]) for r in rows])
        A = np.zeros((len(texts), 4))
        np.add.at(A, ti, np.stack([y & sel_m, sel_m, y & sel_v, sel_v], 1).astype(float))
        rng = np.random.default_rng(13)
        d = []
        for _ in range(boot):
            s = A[rng.integers(0, len(texts), len(texts))].sum(0)
            d.append(s[2] / max(s[3], 1) - s[0] / max(s[1], 1))
        tot = A.sum(0)
        print(f"top {K} (d123's deployed count): d123 {tot[0] / tot[1]:.3f}, verifier {tot[2] / tot[3]:.3f}, "
              f"difference {100 * (tot[2] / tot[3] - tot[0] / tot[1]):+.1f} "
              f"[{100 * np.percentile(d, 2.5):+.1f}, {100 * np.percentile(d, 97.5):+.1f}]")
        by = defaultdict(list)
        for j, r in enumerate(rows):
            by[r[1]].append(j)
        print("| relation | candidates | expressed | d123 deployed k | P top k: d123 | P top k: verifier | AUC d123 | AUC verifier |")
        print("|---|---|---|---|---|---|---|---|")
        for rel, js in sorted(by.items(), key=lambda kv: -len(kv[1])):
            js = np.array(js)
            k = int((m[js] > 0).sum())
            if len(js) < 20:
                continue
            pm = y[js[np.argsort(-m[js])[:k]]].mean() if k else float("nan")
            pv = y[js[np.argsort(-v[js])[:k]]].mean() if k else float("nan")
            am = _auc(m[js], y[js]) if 0 < y[js].sum() < len(js) else float("nan")
            av = _auc(v[js], y[js]) if 0 < y[js].sum() < len(js) else float("nan")
            print(f"| {rel} | {len(js)} | {int(y[js].sum())} | {k} | {pm:.3f} | {pv:.3f} | {am:.3f} | {av:.3f} |")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["labels", "read"])
    ap.add_argument("--scored")
    a = ap.parse_args()
    return labels() if a.cmd == "labels" else read(a.scored)


if __name__ == "__main__":
    raise SystemExit(main())
