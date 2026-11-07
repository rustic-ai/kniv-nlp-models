"""Memory probe: does kniv's extraction capture the facts a memory system
would need? LongMemEval's oracle split (MIT) or LoCoMo (CC BY-NC 4.0:
internal measurement only, never in public material), no LLM in the loop.

    uv run python -m v6.eval.memory_probe --questions 150
    uv run python -m v6.eval.memory_probe --dataset locomo --questions 200

Each question's evidence turns (flagged ``has_answer``) are run through
``v6.infer.Extractor``, each with the turn before it, as a two-speaker
conversation, against one fixed generic memory schema. Then, for questions
whose answer is short enough to match as text (<= 5 words; abstention
questions excluded):

- ``answer_in_entities``: some extracted entity's text matches the answer
  (word-overlap F1 >= 0.5);
- ``answer_in_facts``: the answer is the head or tail of an extracted fact;
- ``facts_about_user``: the evidence produced a fact with the user as an
  argument.

By question type. A recall-style probe: it shows whether the information
reaches the store, not whether a downstream reader answers correctly.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict

from ..config import DATA_DIR, RUNS_DIR

ORACLE = DATA_DIR / "longmemeval" / "longmemeval_oracle.json"
LOCOMO = DATA_DIR / "locomo" / "locomo10.json"
# LoCoMo's QA categories as its evaluation code numbers them; 5 (adversarial,
# unanswerable) is left out.
LOCOMO_CATS = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}

RELATIONS = [
    ("residence", "the place where the person is or has been, resident"),
    ("employer", "person or organization for which the subject works or worked"),
    ("occupation", "occupation of a person"),
    ("owns", "The head holds ownership rights over the tail asset."),
    ("likes", "The head has a favorable attitude toward or takes pleasure in the tail"),
    ("dislikes", "The head feels aversion or an unfavorable attitude toward the tail."),
    ("prefers", "The head favors the tail option or characteristic over relevant alternatives."),
    ("interested in", "The head has curiosity, attention, or an inclination toward the tail subject"),
    ("friend of", "The head and tail have a personal relationship of friendship."),
    ("spouse", "the subject has the object as their spouse (husband, wife, partner, etc.)"),
    ("sibling", "the subject and the object have at least one common parent (brother, sister)"),
    ("member of", "organization, club or musical group to which the subject belongs"),
    ("educated at", "educational institution attended by subject"),
    ("uses", "The head employs the tail tool, resource, method, or component"),
    ("has goal", "The head pursues or is intended to achieve the tail objective or outcome"),
    ("recommends", "The head advises choosing, using, or undertaking the tail."),
    ("scheduled for", "The head event or activity is planned to take place at the tail date or time"),
    ("occurred at", "The head event happened at the tail specified date or time."),
    ("has price", "The head product, service, or booking is offered or charged at the tail amount"),
    ("has count", "The head collection or group contains the tail number of members or items"),
    ("participant in", "event in which a person, organization or creative work was/is a participant"),
    ("located in", "The head is physically situated within the tail place or container."),
    ("created by", "The head work, artifact, or entity was brought into existence by the tail"),
    # our own wording: labels no training run saw
    ("visited", "the head person went to or stayed at the tail place"),
    ("purchased", "the head person bought or acquired the tail item"),
    ("has pet", "the head person owns or keeps the tail animal as a pet"),
    ("plans to", "the head person intends to do or obtain the tail in the future"),
    ("attended", "the head person was present at the tail event, class or appointment"),
    ("had problem with", "the head person experienced an issue or fault with the tail thing"),
    ("completed", "the head person finished the tail task, course, book or activity"),
]
TYPES = ["person", "organization", "location", "product", "event", "date", "time", "number",
         "money", "animal", "food", "activity", "vehicle", "device", "work of art"]


# A fact whose head or tail is only a pronoun ("you" from the assistant's turn,
# "I") names no one a memory store can key on.
PRONOUNS = {"i", "me", "my", "mine", "myself", "you", "your", "yours", "yourself", "we", "us", "our"}


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(s).lower())


def overlap_f1(a: str, b: str) -> float:
    x, y = _words(a), _words(b)
    if not x or not y:
        return 0.0
    common = sum((Counter(x) & Counter(y)).values())
    if not common:
        return 0.0
    p, r = common / len(y), common / len(x)
    return 2 * p * r / (p + r)


def longmemeval_questions(max_words: int) -> list[dict]:
    """Each question's evidence turns (``has_answer``), each with the turn
    before it, as a user/assistant conversation."""
    out = []
    for q in json.loads(ORACLE.read_text()):
        if q["question_id"].endswith("_abs") or len(_words(q["answer"])) > max_words:
            continue
        ev = []
        for s in q["haystack_sessions"]:
            for i, turn in enumerate(s):
                if turn.get("has_answer"):
                    ev.append(([(s[i - 1]["role"], s[i - 1]["content"])] if i > 0 else [])
                              + [(turn["role"], turn["content"])])
        out.append({"type": q["question_type"], "answer": q["answer"], "evidence": ev,
                    "speakers": {"user"}})
    return out


def locomo_questions(max_words: int) -> list[dict]:
    """LoCoMo QA (categories 1-4): each evidence dialogue turn with the turn
    before it, between the conversation's two named speakers."""
    out = []
    for conv in json.loads(LOCOMO.read_text()):
        c = conv["conversation"]
        turns, order = {}, []
        for k in sorted((k for k in c if k.startswith("session_") and not k.endswith("date_time")),
                        key=lambda k: int(k.split("_")[1])):
            for t in c[k]:
                turns[t["dia_id"]] = len(order)
                order.append((t["speaker"], t["text"]))
        for q in conv["qa"]:
            cat = q.get("category")
            if cat not in LOCOMO_CATS or "answer" not in q or len(_words(q["answer"])) > max_words:
                continue
            ev = []
            for d in q.get("evidence", []):
                for d1 in str(d).replace(" ", "").split(";"):
                    if d1 in turns:
                        i = turns[d1]
                        ev.append(([order[i - 1]] if i > 0 else []) + [order[i]])
            if ev:
                out.append({"type": LOCOMO_CATS[cat], "answer": str(q["answer"]), "evidence": ev,
                            "speakers": {c["speaker_a"], c["speaker_b"]}})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="deberta-v3-large-p2-zs-v62-ftr")
    ap.add_argument("--questions", type=int, default=150)
    ap.add_argument("--unseen-offset", type=float,
                    help="override the run's calibrated new-relation offset")
    ap.add_argument("--max-answer-words", type=int, default=5)
    ap.add_argument("--dataset", choices=["longmemeval", "locomo"], default="longmemeval")
    ap.add_argument("--release", default=None,
                    help="a released model directory: run its ONNX runtime (INT8) instead of --run")
    ap.add_argument("--fp32", action="store_true", help="with --release: the fp32 graphs")
    args = ap.parse_args()
    qs = (locomo_questions if args.dataset == "locomo" else longmemeval_questions)(args.max_answer_words)
    by = defaultdict(list)
    for q in qs:
        by[q["type"]].append(q)
    per = max(1, args.questions // len(by))
    qs = [q for t in sorted(by) for q in by[t][:per]]
    if args.release:
        import sys
        sys.path.insert(0, args.release)
        from kniv_v7 import Kniv
        ex = Kniv(args.release, int8=not args.fp32)
    else:
        from ..infer import Extractor
        ex = Extractor(args.run)
        if args.unseen_offset is not None:
            ex.unseen_offset = args.unseen_offset
        import v6.train.span_eval as se
        se.DESCRIBE = True                    # the span head's trained text format
    rels = [{"name": n, "description": d} for n, d in RELATIONS]
    types = [{"name": t} for t in TYPES]
    stats = defaultdict(Counter)
    rel_use = Counter()
    t0 = time.time()
    examples = []
    for qi, q in enumerate(qs):
        ents_txt, fact_args, user_facts, n_facts, n_pron = [], [], 0, 0, 0
        for turns in q["evidence"]:
            if True:
                for w in ex.extract(turns=turns, relations=rels, entity_types=types):
                    E = w["entities"]
                    for e in E:
                        ents_txt += [e["text"]] + list(e.get("mentions", []))
                    for f in w["facts"]:
                        n_facts += 1
                        rel_use[f["relation"]] += 1
                        h, t = E[f["head"]], E[f["tail"]]
                        fact_args += [h["text"], t["text"]] + h.get("mentions", []) + t.get("mentions", [])
                        user_facts += int(h.get("text") in q["speakers"] or t.get("text") in q["speakers"])
                        n_pron += int(h.get("text", "").lower() in PRONOUNS or t.get("text", "").lower() in PRONOUNS)
                        if len(examples) < 25 and qi % 5 == 0:
                            examples.append(f"[{q['type']}] {h['text']} --{f['relation']}--> "
                                            f"{t['text']} ({f['margin']})")
        a = q["answer"]
        c = stats[q["type"]]
        c["questions"] += 1
        c["answer_in_entities"] += int(any(overlap_f1(a, x) >= 0.5 for x in ents_txt))
        c["answer_in_facts"] += int(any(overlap_f1(a, x) >= 0.5 for x in fact_args))
        c["facts_about_user"] += int(user_facts > 0)
        c["facts"] += n_facts
        c["pronoun_facts"] += n_pron
        if (qi + 1) % 10 == 0:
            print(f"{qi + 1}/{len(qs)} questions, {time.time() - t0:.0f}s", flush=True)
    tot = Counter()
    for c in stats.values():
        tot.update(c)
    stats["ALL"] = tot
    rows = {t: {"questions": c["questions"],
                "answer_in_entities": c["answer_in_entities"] / max(c["questions"], 1),
                "answer_in_facts": c["answer_in_facts"] / max(c["questions"], 1),
                "facts_about_user": c["facts_about_user"] / max(c["questions"], 1),
                "facts_per_question": c["facts"] / max(c["questions"], 1),
                "pronoun_fact_share": c["pronoun_facts"] / max(c["facts"], 1)} for t, c in stats.items()}
    print(f"{'type':26s} {'n':>4s} {'ans-in-ents':>11s} {'ans-in-facts':>12s} {'user-facts':>10s} {'facts/q':>8s} {'pron-args':>9s}")
    for t, r in sorted(rows.items(), key=lambda x: (x[0] == "ALL", x[0])):
        print(f"{t:26s} {r['questions']:4d} {r['answer_in_entities']:11.3f} {r['answer_in_facts']:12.3f} "
              f"{r['facts_about_user']:10.3f} {r['facts_per_question']:8.1f} {r['pronoun_fact_share']:9.3f}")
    print("most used relations:", rel_use.most_common(12))
    print("\n".join(examples[:15]))
    out = RUNS_DIR / "memory-probe" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"run": args.release or args.run, "dataset": args.dataset,
                                                  "rows": rows,
                                                  "relations": dict(rel_use), "examples": examples},
                                                 indent=1))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
