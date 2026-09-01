"""Generate v2 distillation parquet shards from the teacher cascade.

Local-reproducible port of the data-generation notebook
``docs/kniv-student-srl-distill.ipynb``. Loads the published teacher
checkpoint (custom split state-dict at
``models/kniv-deberta-nlp-base-en-large/model.pt``), runs per-sentence
and per-verb inference over an input corpus, and writes parquet shards
that the v2 student trainer consumes.

Output schema (one row per sentence):

    words, n_words,
    pos_logits, ner_logits, cls_logits, dep_arc_scores, dep_label_top5,
    pos_hard, ner_hard,
    hidden_l12, hidden_l18, hidden_l24,                # PKD targets
    per_verb_count, per_verb_indices, per_verb,        # per-verb metadata
    per_verb_srl_logits, per_verb_pred_hidden          # per-verb tensors

All float tensors are stored as fp16 byte-blobs (numpy ``.tobytes()``).
Per-verb byte columns are concatenations of ``per_verb_count`` arrays;
the dataset reader slices them by offset.

The teacher's custom state-dict format predates StudentCascadeV5 and
does not load via ``load_student``. This script reconstructs the
teacher in pieces using building blocks from ``models/student_loader.py``.

Usage:

    uv run python scripts/generate_distillation_shards.py \\
        --teacher-dir models/kniv-deberta-nlp-base-en-large \\
        --corpus corpus/output/annotated \\
        --output data/distillation \\
        --max-sentences 200000

If ``--output`` already contains shard_*.parquet files, generation
resumes from where it left off (existing shards counted, new shards
appended).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))

from student_loader import (  # noqa: E402
    ScalarMix, AttentionPool, BiaffineDEPHead,
    POS_LABELS, NER_LABELS, SRL_TAGS, CLS_LABELS, DEPREL_LIST,
    N_POS, N_NER, N_SRL, N_DEP, N_CLS, H_TEACHER,
)

TEACHER_LAYERS_TO_STORE = (12, 18, 24)
MAX_LENGTH = 128


# ── Teacher loading (custom split state-dict) ──────────────────────

class TeacherCascade:
    """Holds the teacher's encoder + heads as separate ``nn.Module`` parts.

    The published teacher's state-dict has a flat dict structure with
    keys for each component (``deberta``, ``pred_embedding``,
    ``pos_scalar_mix``, ...). We don't reconstruct a unified module —
    we keep the parts and call them in the inference pipeline. This
    matches the notebook exactly so any drift is visible.
    """

    def __init__(self, teacher_dir: Path, device: torch.device):
        self.device = device
        meta = json.loads((teacher_dir / "metadata.json").read_text())
        if meta.get("encoder") != "DeBERTa-v3-large":
            raise ValueError(f"Expected DeBERTa-v3-large teacher, got {meta.get('encoder')!r}")

        self.tokenizer = AutoTokenizer.from_pretrained(teacher_dir)
        self.encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large").float()
        H = self.encoder.config.hidden_size
        NL = self.encoder.config.num_hidden_layers + 1
        assert H == H_TEACHER, f"Teacher hidden dim {H} != H_TEACHER {H_TEACHER}"
        self.H = H
        self.NL = NL

        state = torch.load(teacher_dir / "model.pt", map_location="cpu", weights_only=True)

        self.encoder.load_state_dict(state["deberta"])
        self.pred_embedding = nn.Embedding(2, H)
        self.pred_embedding.load_state_dict(state["pred_embedding"])

        self.pos_sm = ScalarMix(NL); self.pos_sm.load_state_dict(state["pos_scalar_mix"])
        self.pos_head = nn.Linear(H, N_POS); self.pos_head.load_state_dict(state["pos_head"])

        self.ner_sm = ScalarMix(NL); self.ner_sm.load_state_dict(state["ner_scalar_mix"])
        self.ner_lstm = nn.LSTM(H, 256, bidirectional=True, batch_first=True)
        self.ner_lstm.load_state_dict(state["ner_lstm"])
        self.ner_proj = nn.Linear(512, H); self.ner_proj.load_state_dict(state["ner_proj"])
        self.ner_head = nn.Sequential(
            nn.LayerNorm(H + N_POS), nn.Linear(H + N_POS, H),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(H, N_NER),
        )
        self.ner_head.load_state_dict(state["ner_head"])

        self.dep_sm = ScalarMix(NL); self.dep_sm.load_state_dict(state["dep_scalar_mix"])
        self.dep_proj = nn.Sequential(
            nn.LayerNorm(H + N_POS + N_NER), nn.Linear(H + N_POS + N_NER, H), nn.GELU(),
        )
        self.dep_proj.load_state_dict(state["dep_proj"])
        self.dep_biaff = BiaffineDEPHead(H, 512, 128, N_DEP)
        self.dep_biaff.load_state_dict(state["dep_biaffine"])

        self.cls_sm = ScalarMix(NL); self.cls_sm.load_state_dict(state["cls_scalar_mix"])
        self.cls_pool = AttentionPool(H); self.cls_pool.load_state_dict(state["cls_pool"])
        self.cls_head = nn.Sequential(
            nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H // 2, N_CLS),
        )
        self.cls_head.load_state_dict(state["cls_head"])

        # SRL head — keyed as "classifier" in the teacher checkpoint
        self.srl_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H, N_SRL),
        )
        self.srl_head.load_state_dict(state["classifier"])

        for m in self.all_modules():
            m.float().to(device).eval()
        del state
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def all_modules(self):
        return [
            self.encoder, self.pred_embedding,
            self.pos_sm, self.pos_head,
            self.ner_sm, self.ner_lstm, self.ner_proj, self.ner_head,
            self.dep_sm, self.dep_proj, self.dep_biaff,
            self.cls_sm, self.cls_pool, self.cls_head,
            self.srl_head,
        ]


# ── Per-batch inference ─────────────────────────────────────────────

@torch.no_grad()
def predict_sentence_batch(t: TeacherCascade, words_batch: list[list[str]]
                           ) -> tuple[list[dict], list[dict[int, int]],
                                      torch.Tensor, torch.Tensor]:
    """Run one batched teacher pass without predicate embedding.

    Returns:
        sent_records: per-sentence dicts (without per-verb columns yet)
        w2t_list: per-sentence word→first-subtoken-index map
        ids: input_ids tensor [B, S] (kept for per-verb passes)
        mask: attention_mask [B, S]
    """
    encs = t.tokenizer(
        words_batch, is_split_into_words=True, return_tensors="pt",
        padding=True, truncation=True, max_length=MAX_LENGTH,
    )
    ids = encs["input_ids"].to(t.device)
    mask = encs["attention_mask"].to(t.device)
    B = ids.size(0)

    emb = t.encoder.embeddings(ids)
    enc_out = t.encoder.encoder(emb, mask, output_hidden_states=True)
    layers = list(enc_out.hidden_states)

    pos_logits = t.pos_head(t.pos_sm(layers))
    pos_probs = torch.softmax(pos_logits, -1)

    ner_h = t.ner_sm(layers)
    lo, _ = t.ner_lstm(ner_h)
    adapted = t.ner_proj(lo) + ner_h
    ner_logits = t.ner_head(torch.cat([adapted, pos_probs], -1))
    ner_probs = torch.softmax(ner_logits, -1)

    dep_h = t.dep_sm(layers)
    dep_features = t.dep_proj(torch.cat([dep_h, pos_probs, ner_probs], -1))
    arc_scores, label_scores = t.dep_biaff(dep_features)

    cls_h = t.cls_sm(layers)
    cls_logits = t.cls_head(t.cls_pool(cls_h, mask))

    pos_logits_cpu = pos_logits.cpu()
    ner_logits_cpu = ner_logits.cpu()
    arc_scores_cpu = arc_scores.cpu()
    label_scores_cpu = label_scores.cpu()
    cls_logits_cpu = cls_logits.cpu()
    teacher_hidden = {l: layers[l].cpu() for l in TEACHER_LAYERS_TO_STORE}

    sent_records: list[dict] = []
    w2t_list: list[dict[int, int]] = []
    for b in range(B):
        words = words_batch[b]
        n = len(words)
        word_ids = encs.word_ids(b)
        w2t: dict[int, int] = {}
        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev and wid < n:
                w2t[wid] = k
            prev = wid

        w_pos = np.zeros((n, N_POS), dtype=np.float16)
        w_ner = np.zeros((n, N_NER), dtype=np.float16)
        for wid, tidx in w2t.items():
            w_pos[wid] = pos_logits_cpu[b, tidx].half().numpy()
            w_ner[wid] = ner_logits_cpu[b, tidx].half().numpy()

        # Word-level arc-score matrix and top-5 labels
        w_arc = np.zeros((n, n), dtype=np.float16)
        for wi in range(n):
            if wi not in w2t:
                continue
            for wj in range(n):
                if wj in w2t:
                    w_arc[wi, wj] = arc_scores_cpu[b, w2t[wi], w2t[wj]].half().item()
        dep_top5: list[list[tuple[int, float]]] = []
        arc_preds = w_arc.argmax(axis=-1)
        for wi in range(n):
            hi = int(arc_preds[wi])
            if wi in w2t and hi in w2t:
                scores = label_scores_cpu[b, w2t[wi], w2t[hi]].half().numpy()
                top5 = scores.argsort()[-5:][::-1]
                dep_top5.append([(int(idx), float(scores[idx])) for idx in top5])
            else:
                dep_top5.append([(0, 1.0)])

        pos_hard = [POS_LABELS[w_pos[i].argmax()] for i in range(n)]
        ner_hard = [NER_LABELS[w_ner[i].argmax()] for i in range(n)]

        word_hidden: dict[int, np.ndarray] = {}
        for layer_idx in TEACHER_LAYERS_TO_STORE:
            wh = np.zeros((n, t.H), dtype=np.float16)
            for wid, tidx in w2t.items():
                wh[wid] = teacher_hidden[layer_idx][b, tidx].half().numpy()
            word_hidden[layer_idx] = wh

        sent_records.append({
            "words": words, "n_words": n,
            "pos_logits": w_pos.tobytes(),
            "ner_logits": w_ner.tobytes(),
            "dep_arc_scores": w_arc.tobytes(),
            "dep_label_top5": json.dumps(dep_top5),
            "cls_logits": cls_logits_cpu[b].half().numpy().tobytes(),
            "pos_hard": pos_hard, "ner_hard": ner_hard,
            **{f"hidden_l{l}": word_hidden[l].tobytes() for l in TEACHER_LAYERS_TO_STORE},
        })
        w2t_list.append(w2t)

    return sent_records, w2t_list, ids, mask


@torch.no_grad()
def predict_per_verb(t: TeacherCascade, n: int, ids_b: torch.Tensor,
                     mask_b: torch.Tensor, w2t: dict[int, int],
                     verb_indices: list[int]) -> list[dict]:
    """For each verb, run a predicate-aware forward pass and emit SRL artifacts."""
    if not verb_indices:
        return []
    S = ids_b.size(1)
    out: list[dict] = []
    for vi in verb_indices:
        if vi not in w2t:
            continue
        indicator = torch.zeros(1, S, dtype=torch.long, device=t.device)
        indicator[0, w2t[vi]] = 1
        emb = t.encoder.embeddings(ids_b) + t.pred_embedding(indicator)
        enc_out = t.encoder.encoder(emb, mask_b, output_hidden_states=True)
        last = enc_out.last_hidden_state  # [1, S, H_TEACHER]
        srl_logits = t.srl_head(last)

        last_cpu = last.cpu()
        srl_cpu = srl_logits.cpu()

        w_srl = np.zeros((n, N_SRL), dtype=np.float16)
        w_pred_hidden = np.zeros((n, t.H), dtype=np.float16)
        for wid, tidx in w2t.items():
            w_srl[wid] = srl_cpu[0, tidx].half().numpy()
            w_pred_hidden[wid] = last_cpu[0, tidx].half().numpy()
        srl_hard = [SRL_TAGS[w_srl[i].argmax()] for i in range(n)]

        out.append({
            "verb_idx": vi,
            "srl_hard": srl_hard,
            "srl_logits_bytes": w_srl.tobytes(),
            "pred_hidden_bytes": w_pred_hidden.tobytes(),
        })
    return out


def process_batch(t: TeacherCascade, words_batch: list[list[str]]) -> list[dict]:
    """Full per-sentence + per-verb pipeline for one mini-batch."""
    sent_records, w2t_list, ids, mask = predict_sentence_batch(t, words_batch)
    out: list[dict] = []
    for b, record in enumerate(sent_records):
        w2t = w2t_list[b]
        n = record["n_words"]
        verb_indices = [i for i, p in enumerate(record["pos_hard"]) if p == "VERB"]
        per_verb = predict_per_verb(
            t, n, ids[b:b + 1], mask[b:b + 1], w2t, verb_indices,
        )
        record["per_verb_count"] = len(per_verb)
        record["per_verb_indices"] = json.dumps([pv["verb_idx"] for pv in per_verb])
        record["per_verb"] = json.dumps([
            {"verb_idx": pv["verb_idx"], "srl_hard": pv["srl_hard"]}
            for pv in per_verb
        ])
        record["per_verb_srl_logits"] = b"".join(
            pv["srl_logits_bytes"] for pv in per_verb
        )
        record["per_verb_pred_hidden"] = b"".join(
            pv["pred_hidden_bytes"] for pv in per_verb
        )
        out.append(record)
    return out


# ── Corpus loading ─────────────────────────────────────────────────

def load_corpus(corpus_dir: Path, prepared_dir: Path | None,
                max_sentences: int, min_words: int = 3, max_words: int = 80,
                seed: int = 42) -> list[list[str]]:
    """Mirror the notebook's corpus assembly: annotated jsonl + UD/NER train fallback."""
    sentences: list[list[str]] = []

    if corpus_dir.is_dir():
        for domain in sorted(corpus_dir.iterdir(), key=lambda d: d.name):
            if not domain.is_dir():
                continue
            jsonl = domain / "annotated.jsonl"
            if not jsonl.exists():
                continue
            count = 0
            with jsonl.open() as f:
                for line in f:
                    if len(sentences) >= max_sentences:
                        break
                    obj = json.loads(line)
                    words = [tok["form"] for tok in obj.get("tokens", [])]
                    if min_words <= len(words) <= max_words:
                        sentences.append(words)
                        count += 1
            print(f"  corpus/{domain.name}: +{count:,} (running total {len(sentences):,})")
            if len(sentences) >= max_sentences:
                break

    if prepared_dir is not None:
        for fname in ("ud_train.json", "ner_train.json"):
            path = prepared_dir / fname
            if path.exists() and len(sentences) < max_sentences:
                added = 0
                for ex in json.loads(path.read_text()):
                    if len(sentences) >= max_sentences:
                        break
                    words = ex.get("words", [])
                    if len(words) >= min_words:
                        sentences.append(words)
                        added += 1
                print(f"  prepared/{fname}: +{added:,} (running total {len(sentences):,})")

    rng = random.Random(seed)
    rng.shuffle(sentences)
    return sentences


# ── Sharded write loop ─────────────────────────────────────────────

def existing_shard_count(out_dir: Path) -> tuple[int, int]:
    """Returns (n_shards, n_sentences_already_written) for resume support."""
    import pyarrow.parquet as pq
    shards = sorted(out_dir.glob("shard_*.parquet"))
    if not shards:
        return 0, 0
    n_sentences = sum(pq.read_metadata(p).num_rows for p in shards)
    return len(shards), n_sentences


def write_shard(rows: list[dict], path: Path) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path.stat().st_size


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--teacher-dir", type=Path,
                    default=REPO / "models" / "kniv-deberta-nlp-base-en-large")
    ap.add_argument("--corpus", type=Path,
                    default=REPO / "corpus" / "output" / "annotated",
                    help="Directory of <domain>/annotated.jsonl files")
    ap.add_argument("--prepared", type=Path,
                    default=REPO / "data" / "prepared" / "kniv-deberta-cascade",
                    help="Optional fallback dir with ud_train.json / ner_train.json")
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination directory for shard_*.parquet")
    ap.add_argument("--max-sentences", type=int, default=200_000)
    ap.add_argument("--shard-size", type=int, default=50_000)
    ap.add_argument("--batch-size", type=int, default=64,
                    help="Teacher inference mini-batch (sweet spot per notebook)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("V2 DISTILLATION DATA GENERATION")
    print("=" * 60)
    print(f"  teacher: {args.teacher_dir}")
    print(f"  corpus:  {args.corpus}")
    print(f"  output:  {args.output}")
    print(f"  device:  {device}")

    print("\nLoading teacher...")
    t = TeacherCascade(args.teacher_dir, device)
    n_params = sum(p.numel() for m in t.all_modules() for p in m.parameters())
    print(f"  loaded: {n_params / 1e6:.0f}M params")

    print("\nLoading corpus...")
    sentences = load_corpus(
        args.corpus,
        args.prepared if args.prepared.is_dir() else None,
        max_sentences=args.max_sentences, seed=args.seed,
    )
    print(f"  total: {len(sentences):,} sentences "
          f"(avg length {np.mean([len(s) for s in sentences]):.1f})")

    n_shards, n_done = existing_shard_count(args.output)
    if n_done:
        print(f"\nResuming: {n_shards} shards / {n_done:,} sentences already written")
    sentences_to_process = sentences[n_done:]
    shard_idx = n_shards

    print(f"\nProcessing {len(sentences_to_process):,} sentences "
          f"(shard size {args.shard_size:,}, batch size {args.batch_size})...")

    rows: list[dict] = []
    total = n_done
    start = time.time()

    for i in tqdm(range(0, len(sentences_to_process), args.batch_size),
                  desc="Distilling"):
        batch = sentences_to_process[i:i + args.batch_size]
        rows.extend(process_batch(t, batch))
        total += len(batch)

        if len(rows) >= args.shard_size:
            path = args.output / f"shard_{shard_idx:03d}.parquet"
            size = write_shard(rows, path)
            elapsed = time.time() - start
            rate = (total - n_done) / max(elapsed, 1e-9)
            eta_min = (len(sentences) - total) / max(rate, 1e-9) / 60
            print(f"\n  ✓ {path.name}  {len(rows):,} rows  "
                  f"{size / 1e9:.2f} GB  |  {rate:.0f} sent/s  ETA {eta_min:.0f} min")
            rows = []
            shard_idx += 1

    if rows:
        path = args.output / f"shard_{shard_idx:03d}.parquet"
        size = write_shard(rows, path)
        print(f"\n  ✓ {path.name}  {len(rows):,} rows  {size / 1e9:.2f} GB  (final partial)")
        shard_idx += 1

    elapsed = time.time() - start
    total_gb = sum(p.stat().st_size for p in args.output.glob("shard_*.parquet")) / 1e9
    meta = {
        "total_sentences": total,
        "total_shards": shard_idx,
        "total_size_gb": round(total_gb, 2),
        "shard_size": args.shard_size,
        "teacher": "kniv-deberta-nlp-base-en-large",
        "teacher_layers_stored": list(TEACHER_LAYERS_TO_STORE),
        "max_length": MAX_LENGTH,
        "version": "v2",
        "schema": {
            "sentence": [
                "words", "n_words",
                "pos_logits", "ner_logits", "cls_logits",
                "dep_arc_scores", "dep_label_top5",
                "pos_hard", "ner_hard",
                *[f"hidden_l{l}" for l in TEACHER_LAYERS_TO_STORE],
            ],
            "per_verb": [
                "per_verb_count", "per_verb_indices", "per_verb",
                "per_verb_srl_logits", "per_verb_pred_hidden",
            ],
        },
    }
    (args.output / "metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  total sentences: {total:,}")
    print(f"  total shards:    {shard_idx}")
    print(f"  total size:      {total_gb:.1f} GB")
    print(f"  total time:      {elapsed / 60:.0f} min")
    print(f"  output:          {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
