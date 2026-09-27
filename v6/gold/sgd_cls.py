"""SGD gold dialogue acts as an external CLS benchmark.

    uv run python -m v6.gold.sgd_cls --split validation --limit 2000

The CLS layer has **no gold at all**: it is produced by an LLM and evaluated
against nothing, which means "CLS quality" has so far been a claim rather than
a number. Schema-Guided Dialogue carries human dialogue-act annotation over
task-oriented conversation, licensed CC-BY-SA-4.0, and its 18 acts map onto
the six ISO functions of CLS_TAXONOMY.md. That makes it the one external
reference available for this layer without commissioning annotation.

**What this measures and what it does not.** SGD is task-oriented
human<->assistant dialogue: it is dense in Directive, Commissive and Social,
and it has no argumentative or narrative text at all. A score here is
evidence about CLS on task-oriented dialogue, which is a large part of our
conversation domain and none of the other four. It is a benchmark, not the
ceiling, and it does not replace the adjudicated gold set of sequencing
step 7 -- an LLM and a human can agree with each other and both differ from
SGD's annotation conventions.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

# ClassLabel order from the dataset's own arrow metadata, not from the paper:
# the integer in `act` indexes this list.
SGD_ACTS = ["AFFIRM", "AFFIRM_INTENT", "CONFIRM", "GOODBYE", "INFORM",
            "INFORM_COUNT", "INFORM_INTENT", "NEGATE", "NEGATE_INTENT",
            "NOTIFY_FAILURE", "NOTIFY_SUCCESS", "OFFER", "OFFER_INTENT",
            "REQUEST", "REQUEST_ALTS", "REQ_MORE", "SELECT", "THANK_YOU"]

# SGD act -> the six ISO functions. Justified per line against the table in
# CLS_TAXONOMY.md "The labels" and its edge cases.
ACT_TO_CLS: dict[str, tuple[str, ...]] = {
    # Read from actual utterances per act, not from the act's NAME. The names
    # mislead: SGD's OFFER is not the taxonomy's Commissive "offer".
    #
    # OFFER presents an ENTITY -- "I found a good restaurant in Milpitas" --
    # which asserts propositional content. The taxonomy's Commissive "offer"
    # is the speaker offering to ACT, which is SGD's OFFER_INTENT. Mapping
    # OFFER to Commissive (38,270 acts, the third most common) was the single
    # largest error in the first pass and it alone produced an apparent
    # Commissive recall of 0.075.
    "OFFER": ("Inform",),
    "INFORM": ("Inform",),
    "INFORM_COUNT": ("Inform",),
    "NOTIFY_SUCCESS": ("Inform",),
    "NOTIFY_FAILURE": ("Inform",),
    # AFFIRM is "That's correct.", NEGATE is "No, that doesn't work for me":
    # agreement and disagreement with content, which the taxonomy places under
    # Inform ("No, it was Wednesday." -> Inform), not Commissive.
    "AFFIRM": ("Inform",),
    "NEGATE": ("Inform",),
    # The speaker commits to act, or accepts/declines a proposed action.
    "OFFER_INTENT": ("Commissive",),     # "shall i reserve a table for you?"
    "AFFIRM_INTENT": ("Commissive",),    # "Yes, please do so."
    "NEGATE_INTENT": ("Commissive",),
    "SELECT": ("Commissive",),           # "Yes, that works for me."
    # Stating one's own goal both provides information and pushes the
    # addressee to act on it.
    "INFORM_INTENT": ("Inform", "Directive"),
    # Seeking information the speaker does not have.
    "REQUEST": ("Question",),            # "What city?"
    # Function over form, as the taxonomy rules for "Can you send the
    # report?": "Ok, Find me another restaurant?" asks the addressee to act.
    "REQUEST_ALTS": ("Directive",),
    # "Can I help you with anything else?" -- an offer to act in question
    # form, and genuinely also a question about what is needed.
    "REQ_MORE": ("Question", "Commissive"),
    # A check question over content the speaker already believes; the taxonomy
    # gives "It ships Tuesday, right?" both labels.
    "CONFIRM": ("Question", "Inform"),
    # Social Obligations Management.
    "GOODBYE": ("Social",),
    "THANK_YOU": ("Social",),
}

SNAP = ("~/.cache/huggingface/hub/datasets--google-research-datasets--"
        "schema_guided_dstc8/snapshots/*/dialogues/%s/*.parquet")


def load_gold(split: str = "validation", limit: int | None = None) -> list[dict]:
    """``[{dialogue_id, turn_idx, speaker, text, acts, cls}]`` from the cache.

    Read from the local HuggingFace cache rather than re-downloaded, so this
    runs offline and against exactly the copy the corpus was collected from.
    """
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(Path(SNAP % split).expanduser())))
    if not files:
        raise SystemExit(f"no cached SGD parquet for split {split!r}")
    out = []
    for f in files:
        for d in pq.ParquetFile(f).read().to_pylist():
            turns = d["turns"]
            for i, utt in enumerate(turns["utterance"]):
                acts: set[str] = set()
                for fr in (turns["frames"][i] or {}).get("actions") or []:
                    for a in (fr.get("act") or []):
                        acts.add(SGD_ACTS[a])
                cls: set[str] = set()
                for a in acts:
                    cls.update(ACT_TO_CLS.get(a, ()))
                out.append({
                    "dialogue_id": d["dialogue_id"], "turn_idx": i,
                    "speaker": "user" if turns["speaker"][i] == 0 else "assistant",
                    "text": utt, "acts": sorted(acts), "cls": sorted(cls)})
                if limit and len(out) >= limit:
                    return out
    return out


# Feedback has NO gold here: no SGD act maps to it, because task-oriented
# dialogue annotated for slot-filling does not mark backchannels. Predicting
# Feedback is therefore penalised as a false positive no matter how correct it
# is, so it is excluded from the aggregate and reported separately.
UNSCORABLE = ("Feedback",)


def evaluate(pairs: list[tuple[set, set]]) -> dict:
    """Per-label and micro scores over ``(predicted, gold)`` label sets."""
    labels = [l for l in ("Question", "Inform", "Directive", "Commissive",
                          "Feedback", "Social")]
    tp, fp, fn = Counter(), Counter(), Counter()
    exact = jac = 0.0
    for pred, g in pairs:
        exact += pred == g
        jac += len(pred & g) / max(len(pred | g), 1)
        for l in labels:
            if l in pred and l in g:
                tp[l] += 1
            elif l in pred:
                fp[l] += 1
            elif l in g:
                fn[l] += 1
    per = {}
    for l in labels:
        P = tp[l] / max(tp[l] + fp[l], 1)
        R = tp[l] / max(tp[l] + fn[l], 1)
        per[l] = {"p": P, "r": R, "f1": 2 * P * R / max(P + R, 1e-9),
                  "gold": tp[l] + fn[l], "pred": tp[l] + fp[l]}
    scored = [l for l in labels if l not in UNSCORABLE]
    TP = sum(tp[l] for l in scored)
    mp = TP / max(TP + sum(fp[l] for l in scored), 1)
    mr = TP / max(TP + sum(fn[l] for l in scored), 1)
    return {"n": len(pairs), "exact": exact / max(len(pairs), 1),
            "jaccard": jac / max(len(pairs), 1), "per_label": per,
            "micro": {"p": mp, "r": mr,
                      "f1": 2 * mp * mr / max(mp + mr, 1e-9)}}


def _norm(t: str) -> str:
    """Whitespace- and case-insensitive key for matching a turn to a sentence.

    Our tokenizer re-spaces punctuation, so the corpus sentence and the SGD
    utterance differ by whitespace even when the text is identical.
    """
    return re.sub(r"\s+", "", t.lower())


def score(annotator: str, gold_file: Path) -> dict:
    """Score one annotator's cached CLS against the gold turns.

    Matches on exact text so nothing is scored against a turn it is not: a
    sentence that does not correspond to a whole SGD turn is skipped rather
    than aligned approximately.
    """
    from ..annotate.base import CacheStore
    from ..config import RUNS_DIR
    from ..build_windows_corpus import (CACHE_VERSION, _get, item_key,
                                        load_windows)
    gold: dict[str, set] = {}
    with gold_file.open() as f:
        for line in f:
            g = json.loads(line)
            gold.setdefault(_norm(g["text"]), set(g["cls"]))

    cache = CacheStore(RUNS_DIR / "_cache", CACHE_VERSION)
    pairs, matched = [], 0
    for w in load_windows():
        if w["source"] != "sgd":
            continue
        for si, (a, b) in enumerate(w["sentence_spans"]):
            toks = w["tokens"][a:b]
            g = gold.get(_norm(" ".join(toks)))
            if g is None:
                continue
            matched += 1
            p = _get(cache, annotator, "cls",
                     item_key(w["window_id"], si, toks), b - a)
            if p is not None:
                pairs.append((set(p), g))
    out = evaluate(pairs)
    out["matched_turns"] = matched
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--score", metavar="ANNOTATOR",
                    help="score this annotator's cached cls against the gold")
    ap.add_argument("--split", default="validation",
                    choices=["train", "validation", "test"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", type=Path,
                    default=Path("data/v6-corpus/gold/cls/sgd_gold.jsonl"))
    args = ap.parse_args()

    if args.score:
        r = score(args.score, args.out)
        print(f"{args.score} vs {args.out.name}: n={r['n']} "
              f"(of {r['matched_turns']} matched turns)")
        print(f"  exact set match {r['exact']:.1%}  "
              f"mean Jaccard {r['jaccard']:.3f}")
        print(f"  {'label':12s} {'P':>6s} {'R':>6s} {'F1':>6s} "
              f"{'gold':>7s} {'pred':>7s}")
        for l, v in r["per_label"].items():
            tag = "  unscorable: no gold in SGD" if l in UNSCORABLE else ""
            print(f"  {l:12s} {v['p']:6.3f} {v['r']:6.3f} {v['f1']:6.3f} "
                  f"{v['gold']:7d} {v['pred']:7d}{tag}")
        m = r["micro"]
        print(f"  micro (excluding {', '.join(UNSCORABLE)}): "
              f"P {m['p']:.3f} R {m['r']:.3f} F1 {m['f1']:.3f}")
        return 0

    gold = load_gold(args.split, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for g in gold:
            f.write(json.dumps(g) + "\n")

    acts = Counter(a for g in gold for a in g["acts"])
    labs = Counter(c for g in gold for c in g["cls"])
    n = len(gold)
    print(f"{n} turns -> {args.out}")
    print(f"  turns with no act: {sum(1 for g in gold if not g['acts'])} "
          f"({sum(1 for g in gold if not g['acts'])/max(n,1):.1%})")
    print(f"  multi-label: {sum(1 for g in gold if len(g['cls'])>1)/max(n,1):.1%}")
    print("  acts: " + " ".join(f"{k}={v}" for k, v in acts.most_common(6)))
    print("  cls:  " + " ".join(
        f"{k} {v/max(sum(labs.values()),1):.1%}" for k, v in labs.most_common()))
    unmapped = sorted(set(acts) - set(ACT_TO_CLS))
    if unmapped:
        print(f"  UNMAPPED ACTS: {unmapped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
