"""Per-relation reading of a judged M1-format file (plan r4 V2 gates).

    uv run python -m v6.eval.v2_read rel --run c4v1c-conll04 --configs d123,v1c [--judges astra,deepseek,both]
    uv run python -m v6.eval.v2_read merge --ref m1v1c-n500 --preds FILE --label v2 --out m1v2-n500

``merge``: a new config's predictions (an ``m1_precision predict`` file,
config ``ver``) added to a fixed reference file (its items, configs and
three label files, unchanged): each new fact is matched to any judged fact
of the reference by mention sets and relation; facts no judged fact matches
are counted not expressed, and their number is printed. No judge calls.

Per relation and config, under each judge: the stored predictions (each
config's judged set: the deployed bar, or p 0.05 for a verifier), how many
are judged expressed, their precision; and the precision of the config's top ``k`` facts of that
relation by margin, ``k`` the first config's deployed count (equal size,
so a config that adds predictions is not read only at its own bar). A
prediction no judged fact matches counts as not expressed.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict

from .m1_precision import OUT, POSITIVE, _same, labels_path


def rel(run: str, configs: list[str], judges: list[str]) -> int:
    P = pickle.loads((OUT / f"preds-{run}.pkl").read_bytes())
    for judge in judges:
        L = json.loads(labels_path(run, judge, "c3").read_text())
        rows = defaultdict(list)                     # (config, relation) -> [(margin, expressed)]
        for i, t in enumerate(L["texts"]):
            if not t["judged"]:
                continue
            for c in configs:
                for k, m in P["margins"][c][i].items():
                    f = next((f for f in t["facts"] if _same((set(f["key"][0]), f["key"][1], set(f["key"][2])), k)),
                             None)
                    rows[(c, k[1])].append((m, f is not None and f["verdict"] in POSITIVE))
        rels = sorted({r for _, r in rows})
        print(f"\n## {run}, {judge}")
        print("| relation | " + " | ".join(f"{c}: predicted / expressed / P" for c in configs)
              + " | " + " | ".join(f"{c}: P of top k" for c in configs) + " |")
        print("|---|" + "---|" * (2 * len(configs)))
        for r in rels:
            dep = {}
            for c in configs:
                xs = [ok for m, ok in rows[(c, r)]]
                dep[c] = (len(xs), sum(xs))
            k = dep[configs[0]][0]
            topk = []
            for c in configs:
                xs = sorted(rows[(c, r)], key=lambda z: -z[0])[:k]
                topk.append(f"{sum(ok for _, ok in xs) / max(len(xs), 1):.3f} (k {len(xs)})")
            print(f"| {r} | " + " | ".join(f"{n} / {e} / {e / max(n, 1):.3f}" for n, e in dep.values())
                  + " | " + " | ".join(topk) + " |")
    return 0


def merge(ref: str, preds: str, label: str, out: str) -> int:
    from pathlib import Path
    R = pickle.loads((OUT / f"preds-{ref}.pkl").read_bytes())
    S = pickle.loads(Path(preds).read_bytes())
    ix = {it["id"]: n for n, it in enumerate(S["items"])}
    J = [ix[it["id"]] for it in R["items"]]
    R["preds"][label] = [S["preds"]["ver"][j] for j in J]
    R["margins"][label] = [S["margins"]["ver"][j] for j in J]
    R["run"] = out
    (OUT / f"preds-{out}.pkl").write_bytes(pickle.dumps(R))
    for judge in ("astra", "deepseek", "both"):
        L = json.loads(labels_path(ref, judge, "c3").read_text())
        un = n = 0
        for i, t in enumerate(L["texts"]):
            for k in R["margins"][label][i]:
                n += 1
                f = next((f for f in t["facts"] if _same((set(f["key"][0]), f["key"][1], set(f["key"][2])), k)), None)
                if f is None:
                    un += 1
                elif label not in f["src"]:
                    f["src"] = f["src"] + [label]
        L["configs"] = list(L["configs"]) + [label]
        labels_path(out, judge, "c3").write_text(json.dumps(L))
        print(f"{judge}: {n} {label} facts, {un} matched to no judged fact (counted not expressed)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=["rel", "merge"])
    ap.add_argument("--run")
    ap.add_argument("--configs")
    ap.add_argument("--ref")
    ap.add_argument("--preds")
    ap.add_argument("--label", default="v2")
    ap.add_argument("--out")
    ap.add_argument("--judges", default="astra,deepseek,both")
    a = ap.parse_args()
    if a.cmd == "merge":
        return merge(a.ref, a.preds, a.label, a.out)
    return rel(a.run, a.configs.split(","), a.judges.split(","))


if __name__ == "__main__":
    raise SystemExit(main())
