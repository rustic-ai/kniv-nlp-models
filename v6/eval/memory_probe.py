"""Memory probe: does kniv's extraction capture the facts a memory system
would need? LongMemEval's oracle split (MIT), no LLM in the loop.

    uv run python -m v6.eval.memory_probe --questions 150

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


def main() -> int:
    from ..infer import Extractor
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="deberta-v3-large-p2-zs-v62-ftr")
    ap.add_argument("--questions", type=int, default=150)
    ap.add_argument("--max-answer-words", type=int, default=5)
    args = ap.parse_args()
    data = json.loads(ORACLE.read_text())
    qs = [q for q in data if not q["question_id"].endswith("_abs")
          and len(_words(q["answer"])) <= args.max_answer_words]
    by = defaultdict(list)
    for q in qs:
        by[q["question_type"]].append(q)
    per = max(1, args.questions // len(by))
    qs = [q for t in sorted(by) for q in by[t][:per]]
    ex = Extractor(args.run)
    rels = [{"name": n, "description": d} for n, d in RELATIONS]
    types = [{"name": t} for t in TYPES]
    import v6.train.span_eval as se
    se.DESCRIBE = True                        # the span head's trained text format
    stats = defaultdict(Counter)
    rel_use = Counter()
    t0 = time.time()
    examples = []
    for qi, q in enumerate(qs):
        ents_txt, fact_args, user_facts, n_facts = [], [], 0, 0
        for s in q["haystack_sessions"]:
            for i, turn in enumerate(s):
                if not turn.get("has_answer"):
                    continue
                turns = ([(s[i - 1]["role"], s[i - 1]["content"])] if i > 0 else []) + \
                    [(turn["role"], turn["content"])]
                for w in ex.extract(turns=turns, relations=rels, entity_types=types):
                    E = w["entities"]
                    for e in E:
                        ents_txt += [e["text"]] + list(e.get("mentions", []))
                    for f in w["facts"]:
                        n_facts += 1
                        rel_use[f["relation"]] += 1
                        h, t = E[f["head"]], E[f["tail"]]
                        fact_args += [h["text"], t["text"]] + h.get("mentions", []) + t.get("mentions", [])
                        user_facts += int(h.get("text") == "user" or t.get("text") == "user")
                        if len(examples) < 25 and qi % 5 == 0:
                            examples.append(f"[{q['question_type']}] {h['text']} --{f['relation']}--> "
                                            f"{t['text']} ({f['margin']})")
        a = q["answer"]
        c = stats[q["question_type"]]
        c["questions"] += 1
        c["answer_in_entities"] += int(any(overlap_f1(a, x) >= 0.5 for x in ents_txt))
        c["answer_in_facts"] += int(any(overlap_f1(a, x) >= 0.5 for x in fact_args))
        c["facts_about_user"] += int(user_facts > 0)
        c["facts"] += n_facts
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
                "facts_per_question": c["facts"] / max(c["questions"], 1)} for t, c in stats.items()}
    print(f"{'type':26s} {'n':>4s} {'ans-in-ents':>11s} {'ans-in-facts':>12s} {'user-facts':>10s} {'facts/q':>8s}")
    for t, r in sorted(rows.items(), key=lambda x: (x[0] == "ALL", x[0])):
        print(f"{t:26s} {r['questions']:4d} {r['answer_in_entities']:11.3f} {r['answer_in_facts']:12.3f} "
              f"{r['facts_about_user']:10.3f} {r['facts_per_question']:8.1f}")
    print("most used relations:", rel_use.most_common(12))
    print("\n".join(examples[:15]))
    out = RUNS_DIR / "memory-probe" / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"run": args.run, "rows": rows,
                                                  "relations": dict(rel_use), "examples": examples},
                                                 indent=1))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
