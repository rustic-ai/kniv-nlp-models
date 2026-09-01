"""Measure how much SRL error the programmatic validators detect.

Loads a kniv student model, runs inference on PropBank EWT test, then for
each example asks `shared.srl_validators.score_frame()` whether the
predicted frame looks consistent. Reports recall on errors and false
positive rate on correct predictions across score thresholds.

This is the measurement step before we wire validators into a training
loop or quality gate — if recall is too low or FPR too high here, the
signal isn't usable downstream.

Usage:
    uv run python scripts/run_srl_validators.py \\
        --model-dir models/kniv-deberta-nlp-base-en-small \\
        --max-examples 500
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "models"))

from student_loader import (  # noqa: E402
    load_student, viterbi_decode,
    NER_LABELS, SRL_TAGS, DEPREL_LIST, N_DEP,
)
from student_benchmark_standard import SrlDataset, collate_basic  # noqa: E402
from shared.srl_validators import (  # noqa: E402
    load_framesets, extract_arg_spans, score_frame,
)


def smoke_test(framesets):
    """Quick sanity check that validators behave as expected end-to-end."""
    print("=" * 60)
    print("Smoke test")
    print("=" * 60)
    # give: ditransitive, ARG0/ARG1/ARG2 should pass
    # SRL tag format matches our pipeline: bare "V" for predicate, B-/I- for args
    srl = ["B-ARG0", "V", "B-ARG2", "B-ARG1"]
    dep = ["nsubj", "root", "iobj", "obj"]
    fq = score_frame("give", srl, dep_relations=dep, framesets=framesets)
    print(f"  give/ditransitive: score={fq.score:.2f} pb={fq.propbank_verdict} "
          f"sense={fq.propbank_sense} dep={fq.dep_score:.2f}")
    assert fq.propbank_verdict is True, f"expected True, got {fq.propbank_verdict}"

    # eat: transitive only — ARG2 should be flagged invalid
    srl_bad = ["B-ARG0", "V", "B-ARG2"]
    dep_bad = ["nsubj", "root", "obl"]
    fq_bad = score_frame("eat", srl_bad, dep_relations=dep_bad, framesets=framesets)
    print(f"  eat/+ARG2:         score={fq_bad.score:.2f} pb={fq_bad.propbank_verdict}")
    assert fq_bad.propbank_verdict is False, (
        f"expected False (eat has no ARG2 in PropBank), got {fq_bad.propbank_verdict}"
    )
    print("  OK\n")


def decode_ner_words(ner_logits, word_ids):
    """Decode NER logits to word-level BIO tags via Viterbi."""
    # Reduce to first-subtoken-per-word logits
    keep_idx, prev = [], None
    for k, wid in enumerate(word_ids):
        if wid is None or wid == prev:
            continue
        keep_idx.append(k)
        prev = wid
    if not keep_idx:
        return []
    word_logits = ner_logits[keep_idx]
    path = viterbi_decode(word_logits, NER_LABELS)
    return [NER_LABELS[p] for p in path]


def decode_dep_words(arc_scores, label_scores, word_ids):
    """Decode predicted DEP relations (argmax at predicted head) per word."""
    keep_idx, prev = [], None
    for k, wid in enumerate(word_ids):
        if wid is None or wid == prev:
            continue
        keep_idx.append(k)
        prev = wid
    if not keep_idx:
        return []
    rels = []
    for k in keep_idx:
        h = int(arc_scores[k].argmax())
        rel_id = int(label_scores[k, h].argmax())
        rels.append(DEPREL_LIST[rel_id])
    return rels


def decode_srl_words(srl_logits, valid_idx):
    """Decode SRL via Viterbi on the valid-token positions (matches benchmark)."""
    path = viterbi_decode(srl_logits[valid_idx], SRL_TAGS)
    tags = [SRL_TAGS[p] if p < len(SRL_TAGS) else "O" for p in path]
    # Treat "V" as "O" for argument-span comparison (predicate marker, not arg)
    return [t if t != "V" else "O" for t in tags]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", default="model.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--max-examples", type=int, default=500,
                    help="0 = all examples")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--bench-path", default=str(REPO / "data" / "benchmarks" / "propbank_srl_test.json"))
    ap.add_argument("--json", default=None, help="Optional path to dump per-example records")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    framesets = load_framesets()
    print(f"Loaded {len(framesets)} PropBank verbs from frameset XML")
    smoke_test(framesets)
    if args.smoke_only:
        return

    import spacy
    from spacy.tokens import Doc
    print("Loading spaCy en_core_web_sm for lemmatization")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    print(f"Loading model from {args.model_dir}")
    model, tokenizer, info = load_student(
        args.model_dir, checkpoint=args.checkpoint, device=args.device,
    )
    device = info["device"]
    print(f"  encoder={info['encoder']} device={device}")

    examples = json.loads(Path(args.bench_path).read_text())
    if args.max_examples and args.max_examples < len(examples):
        examples = examples[:args.max_examples]
    print(f"Evaluating on {len(examples)} examples")

    # Pre-compute predicate lemmas using spaCy on pre-tokenized words.
    # Doc(vocab, words=...) preserves our tokenization; nlp.pipe runs the
    # tagger + attribute_ruler + lemmatizer in batches.
    print("Lemmatizing predicates")
    pre_docs = (Doc(nlp.vocab, words=ex["words"]) for ex in examples)
    verb_lemmas = [
        doc[ex["predicate_idx"]].lemma_.lower()
        for ex, doc in zip(examples, nlp.pipe(pre_docs, batch_size=64))
    ]

    dataset = SrlDataset(examples, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_basic)

    records = []
    unknown_verbs = Counter()

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, desc="inference")):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pidx = batch["predicate_idx"].to(device)
            pos_l, ner_l, arc_s, lab_s, srl_l, _ = model(ids, mask, pidx)

            for j in range(ids.size(0)):
                ex = examples[bi * args.batch_size + j]
                # Re-encode this example to recover word_ids (collate dropped them)
                enc = tokenizer(
                    ex["words"], is_split_into_words=True, max_length=128,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                word_ids = enc.word_ids()
                labs = batch["labels"][j]
                vi = (labs != -100).nonzero(as_tuple=True)[0]
                if not len(vi):
                    continue

                pred_srl = decode_srl_words(srl_l[j].cpu(), vi)
                gold_srl = [(SRL_TAGS[labs[k]] if SRL_TAGS[labs[k]] != "V" else "O")
                            for k in vi.tolist()]
                pred_ner = decode_ner_words(ner_l[j].cpu(), word_ids)
                pred_dep = decode_dep_words(arc_s[j].cpu(), lab_s[j].cpu(), word_ids)

                # Align predictions to gold word-level length (gold is per-word)
                n_words = len(ex["words"])
                pred_ner = pred_ner[:n_words] + ["O"] * max(0, n_words - len(pred_ner))
                pred_dep = pred_dep[:n_words] + ["dep"] * max(0, n_words - len(pred_dep))
                # pred_srl is already aligned to vi (which corresponds to word positions
                # whose tag != -100; the SrlDataset aligns one-to-one with ex["srl_tags"])
                if len(pred_srl) != len(gold_srl):
                    # truncated: skip this example
                    continue

                gold_spans = set(extract_arg_spans(gold_srl))
                pred_spans = set(extract_arg_spans(pred_srl))
                gold_correct = (gold_spans == pred_spans)

                verb_lemma = verb_lemmas[bi * args.batch_size + j]
                if verb_lemma not in framesets:
                    unknown_verbs[verb_lemma] += 1

                # score_frame expects pred_srl, pred_dep, pred_ner aligned at the
                # same indexing. pred_srl is a subset of words (the ones with tags
                # != -100), but for DEP/NER consistency we want them at the same
                # word positions. SrlDataset emits one tag per word in ex["words"],
                # so pred_srl is per-word, length == n_words.
                fq = score_frame(
                    verb_lemma=verb_lemma,
                    srl_tags=pred_srl,
                    dep_relations=pred_dep[:len(pred_srl)],
                    ner_tags=pred_ner[:len(pred_srl)],
                    framesets=framesets,
                )

                records.append({
                    "verb": verb_lemma,
                    "predicate_idx": ex["predicate_idx"],
                    "n_words": n_words,
                    "gold_correct": gold_correct,
                    "score": fq.score,
                    "propbank_verdict": fq.propbank_verdict,
                    "propbank_sense": fq.propbank_sense,
                    "dep_score": fq.dep_score,
                    "argm_score": fq.argm_score,
                    "n_dep_violations": len(fq.dep_violations),
                    "dep_violations": fq.dep_violations,
                    "n_argm_violations": len(fq.argm_violations),
                })

    # ── Aggregate ─────────────────────────────────────────────────
    total = len(records)
    correct = [r for r in records if r["gold_correct"]]
    incorrect = [r for r in records if not r["gold_correct"]]
    pb_known = sum(1 for r in records if r["propbank_verdict"] is not None)

    print()
    print("=" * 60)
    print(f"Coverage")
    print("=" * 60)
    print(f"  total examples evaluated:    {total}")
    print(f"  correct (gold==pred spans):  {len(correct)} ({len(correct)/total:.1%})")
    print(f"  incorrect:                   {len(incorrect)} ({len(incorrect)/total:.1%})")
    print(f"  verbs found in framesets:    {pb_known} ({pb_known/total:.1%})")
    print(f"  unique unknown verbs:        {len(unknown_verbs)}")
    if unknown_verbs:
        top_unk = unknown_verbs.most_common(5)
        print(f"  top unknown:                 {top_unk}")

    print()
    print("=" * 60)
    print(f"Detection rates (recall on errors / FPR on correct)")
    print("=" * 60)
    print(f"  {'threshold':>10}  {'recall':>8}  {'FPR':>8}  {'flagged':>8}")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        flagged_err = sum(1 for r in incorrect if r["score"] < thr)
        flagged_ok = sum(1 for r in correct if r["score"] < thr)
        recall = flagged_err / len(incorrect) if incorrect else 0.0
        fpr = flagged_ok / len(correct) if correct else 0.0
        flagged = flagged_err + flagged_ok
        print(f"  {thr:>10.2f}  {recall:>8.2%}  {fpr:>8.2%}  {flagged:>8d}")

    # Per-validator contribution: among incorrect frames, what fraction
    # would *each* validator alone flag?
    print()
    print("=" * 60)
    print(f"Per-validator recall on errors")
    print("=" * 60)
    pb_flagged = sum(1 for r in incorrect if r["propbank_verdict"] is False)
    dep_flagged = sum(1 for r in incorrect if r["dep_score"] < 1.0)
    argm_flagged = sum(1 for r in incorrect if r["argm_score"] < 1.0)
    print(f"  PropBank says invalid:       {pb_flagged/len(incorrect):.2%}" if incorrect else "  -")
    print(f"  DEP cascade has violations:  {dep_flagged/len(incorrect):.2%}" if incorrect else "  -")
    print(f"  ARGM-NER has violations:     {argm_flagged/len(incorrect):.2%}" if incorrect else "  -")

    print()
    print(f"Per-validator FPR on correct frames")
    pb_flagged_ok = sum(1 for r in correct if r["propbank_verdict"] is False)
    dep_flagged_ok = sum(1 for r in correct if r["dep_score"] < 1.0)
    argm_flagged_ok = sum(1 for r in correct if r["argm_score"] < 1.0)
    print(f"  PropBank says invalid:       {pb_flagged_ok/len(correct):.2%}" if correct else "  -")
    print(f"  DEP cascade has violations:  {dep_flagged_ok/len(correct):.2%}" if correct else "  -")
    print(f"  ARGM-NER has violations:     {argm_flagged_ok/len(correct):.2%}" if correct else "  -")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "model_dir": str(args.model_dir),
            "encoder": info["encoder"],
            "n_total": total,
            "n_correct": len(correct),
            "n_incorrect": len(incorrect),
            "records": records,
        }, indent=2))
        print(f"\nWrote per-example records to {args.json}")


if __name__ == "__main__":
    main()
