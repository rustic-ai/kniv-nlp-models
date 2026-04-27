"""Generate distillation dataset from the teacher cascade model.

Runs the teacher on a corpus and saves word-level soft predictions
for all 5 heads. Output is stored as parquet shards for fast I/O.
Any student tokenizer can align to these word-level predictions.

Usage:
    .venv/bin/python models/kniv-deberta-nlp-base-en-large/generate_distillation_data.py
    .venv/bin/python models/kniv-deberta-nlp-base-en-large/generate_distillation_data.py \
        --model models/kniv-deberta-nlp-base-en-large \
        --corpus corpus/output/annotated \
        --output data/distillation \
        --max-sentences 500000
"""
from __future__ import annotations
import argparse, json, os, re, struct
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# ── Label sets ────────────────────────────────────────────────

POS_LABELS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
NER_LABELS = [
    "O", "B-PERSON", "I-PERSON", "B-NORP", "I-NORP", "B-FAC", "I-FAC",
    "B-ORG", "I-ORG", "B-GPE", "I-GPE", "B-LOC", "I-LOC",
    "B-PRODUCT", "I-PRODUCT", "B-EVENT", "I-EVENT",
    "B-WORK_OF_ART", "I-WORK_OF_ART", "B-LAW", "I-LAW",
    "B-LANGUAGE", "I-LANGUAGE", "B-DATE", "I-DATE",
    "B-TIME", "I-TIME", "B-PERCENT", "I-PERCENT",
    "B-MONEY", "I-MONEY", "B-QUANTITY", "I-QUANTITY",
    "B-ORDINAL", "I-ORDINAL", "B-CARDINAL", "I-CARDINAL",
]
SRL_TAGS = [
    "O", "V",
    "B-ARG0", "I-ARG0", "B-ARG1", "I-ARG1", "B-ARG2", "I-ARG2",
    "B-ARG3", "I-ARG3", "B-ARG4", "I-ARG4",
    "B-ARGM-TMP", "I-ARGM-TMP", "B-ARGM-LOC", "I-ARGM-LOC",
    "B-ARGM-MNR", "I-ARGM-MNR", "B-ARGM-CAU", "I-ARGM-CAU",
    "B-ARGM-PRP", "I-ARGM-PRP", "B-ARGM-NEG", "I-ARGM-NEG",
    "B-ARGM-ADV", "I-ARGM-ADV", "B-ARGM-DIR", "I-ARGM-DIR",
    "B-ARGM-DIS", "I-ARGM-DIS", "B-ARGM-EXT", "I-ARGM-EXT",
    "B-ARGM-MOD", "I-ARGM-MOD", "B-ARGM-PRD", "I-ARGM-PRD",
    "B-ARGM-GOL", "I-ARGM-GOL", "B-ARGM-COM", "I-ARGM-COM",
    "B-ARGM-REC", "I-ARGM-REC",
]
CLS_LABELS = ["inform", "request", "question", "confirm", "reject", "offer", "social", "status"]
DEPREL_LIST = [
    "root", "acl", "acl:relcl", "advcl", "advcl:relcl", "advmod", "amod", "appos",
    "aux", "aux:pass", "case", "cc", "cc:preconj", "ccomp", "compound", "compound:prt",
    "conj", "cop", "csubj", "csubj:outer", "csubj:pass", "dep", "det", "det:predet",
    "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark",
    "nmod", "nmod:desc", "nmod:npmod", "nmod:poss", "nmod:tmod", "nsubj", "nsubj:outer",
    "nsubj:pass", "nummod", "obj", "obl", "obl:agent", "obl:npmod", "obl:tmod",
    "orphan", "parataxis", "punct", "reparandum", "vocative", "xcomp",
]

# ── Model components ──────────────────────────────────────────

class ScalarMix(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n))
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, layers):
        w = torch.softmax(self.weights, dim=0)
        return self.scale * sum(wi * li for wi, li in zip(w, layers))

class AttentionPool(nn.Module):
    def __init__(self, H):
        super().__init__()
        self.attn = nn.Linear(H, 1)
    def forward(self, hidden, mask):
        scores = self.attn(hidden).squeeze(-1).masked_fill(~mask.bool(), -1e9)
        return (hidden * torch.softmax(scores, -1).unsqueeze(-1)).sum(1)

class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim + 1, in_dim + 1))
    def forward(self, h_dep, h_head):
        B, S, D = h_dep.size()
        h_dep = torch.cat([h_dep, torch.ones(B, S, 1, device=h_dep.device)], -1)
        h_head = torch.cat([h_head, torch.ones(B, S, 1, device=h_head.device)], -1)
        scores = torch.einsum("bxi,oij,byj->boxy", h_dep, self.weight, h_head)
        return scores.squeeze(1).contiguous() if scores.size(1) == 1 else scores.permute(0, 2, 3, 1).contiguous()

class BiaffineDEPHead(nn.Module):
    def __init__(self, H, arc_dim=512, label_dim=128, num_labels=53, dropout=0.1):
        super().__init__()
        self.arc_dep = nn.Sequential(nn.Linear(H, arc_dim), nn.LayerNorm(arc_dim), nn.GELU(), nn.Dropout(dropout))
        self.arc_head = nn.Sequential(nn.Linear(H, arc_dim), nn.LayerNorm(arc_dim), nn.GELU(), nn.Dropout(dropout))
        self.label_dep = nn.Sequential(nn.Linear(H, label_dim), nn.LayerNorm(label_dim), nn.GELU(), nn.Dropout(dropout))
        self.label_head = nn.Sequential(nn.Linear(H, label_dim), nn.LayerNorm(label_dim), nn.GELU(), nn.Dropout(dropout))
        self.biaffine_arc = Biaffine(arc_dim, 1)
        self.biaffine_label = Biaffine(label_dim, num_labels)
    def forward(self, hidden):
        return (self.biaffine_arc(self.arc_dep(hidden), self.arc_head(hidden)),
                self.biaffine_label(self.label_dep(hidden), self.label_head(hidden)))


def tokenize_words(text):
    return re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)


# ── Load teacher ──────────────────────────────────────────────

def load_teacher(model_dir, device):
    """Load teacher model. Returns dict of modules."""
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large").float()
    H = encoder.config.hidden_size
    NL = encoder.config.num_hidden_layers + 1

    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["deberta"])

    pred_embedding = nn.Embedding(2, H)
    pred_embedding.load_state_dict(state["pred_embedding"])

    pos_sm = ScalarMix(NL); pos_sm.load_state_dict(state["pos_scalar_mix"])
    pos_head = nn.Linear(H, len(POS_LABELS)); pos_head.load_state_dict(state["pos_head"])

    ner_sm = ScalarMix(NL); ner_sm.load_state_dict(state["ner_scalar_mix"])
    ner_lstm = nn.LSTM(H, 256, bidirectional=True, batch_first=True); ner_lstm.load_state_dict(state["ner_lstm"])
    ner_proj = nn.Linear(512, H); ner_proj.load_state_dict(state["ner_proj"])
    ner_head = nn.Sequential(nn.LayerNorm(H + len(POS_LABELS)), nn.Linear(H + len(POS_LABELS), H),
                             nn.GELU(), nn.Dropout(0.1), nn.Linear(H, len(NER_LABELS)))
    ner_head.load_state_dict(state["ner_head"])

    srl_cls = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
                            nn.Dropout(0.1), nn.Linear(H, len(SRL_TAGS)))
    srl_cls.load_state_dict(state["classifier"])

    dep_sm = ScalarMix(NL); dep_sm.load_state_dict(state["dep_scalar_mix"])
    in_dim = H + len(POS_LABELS) + len(NER_LABELS)
    dep_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU())
    dep_proj.load_state_dict(state["dep_proj"])
    dep_biaff = BiaffineDEPHead(H, num_labels=len(DEPREL_LIST))
    dep_biaff.load_state_dict(state["dep_biaffine"])

    cls_sm = ScalarMix(NL); cls_sm.load_state_dict(state["cls_scalar_mix"])
    cls_pool = AttentionPool(H); cls_pool.load_state_dict(state["cls_pool"])
    cls_head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(),
                             nn.Dropout(0.1), nn.Linear(H // 2, len(CLS_LABELS)))
    cls_head.load_state_dict(state["cls_head"])

    m = dict(encoder=encoder, tokenizer=tokenizer, pred_embedding=pred_embedding,
             pos_sm=pos_sm, pos_head=pos_head,
             ner_sm=ner_sm, ner_lstm=ner_lstm, ner_proj=ner_proj, ner_head=ner_head,
             srl_cls=srl_cls,
             dep_sm=dep_sm, dep_proj=dep_proj, dep_biaff=dep_biaff,
             cls_sm=cls_sm, cls_pool=cls_pool, cls_head=cls_head)
    for v in m.values():
        if isinstance(v, nn.Module):
            v.float().to(device).eval()
    return m


# ── Generate predictions ──────────────────────────────────────

@torch.no_grad()
def predict_batch(words_batch, m, device, max_length=128):
    """Run teacher on a batch of word lists. Returns word-level predictions."""
    tok = m["tokenizer"]
    enc = m["encoder"]

    # Tokenize batch
    encs = tok(words_batch, is_split_into_words=True, return_tensors="pt",
               padding=True, truncation=True, max_length=max_length)
    ids = encs["input_ids"].to(device)
    mask = encs["attention_mask"].to(device)
    B, S = ids.size()

    # Encoder pass (without predicate embedding — for POS/NER/DEP/CLS)
    emb = enc.embeddings(ids)
    enc_out = enc.encoder(emb, mask, output_hidden_states=True)
    hidden_states = list(enc_out.hidden_states)

    # POS
    pos_h = m["pos_sm"](hidden_states)
    pos_logits = m["pos_head"](pos_h)
    pos_probs = torch.softmax(pos_logits, dim=-1)

    # NER
    ner_h = m["ner_sm"](hidden_states)
    lo, _ = m["ner_lstm"](ner_h)
    adapted = m["ner_proj"](lo) + ner_h
    ner_logits = m["ner_head"](torch.cat([adapted, pos_probs], dim=-1))

    # DEP
    dep_h = m["dep_sm"](hidden_states)
    ner_probs = torch.softmax(ner_logits, dim=-1)
    proj_h = m["dep_proj"](torch.cat([dep_h, pos_probs, ner_probs], dim=-1))
    arc_scores, label_scores = m["dep_biaff"](proj_h)

    # CLS
    cls_h = m["cls_sm"](hidden_states)
    cls_pooled = m["cls_pool"](cls_h, mask)
    cls_logits = m["cls_head"](cls_pooled)

    # Move to CPU
    pos_logits = pos_logits.cpu()
    ner_logits = ner_logits.cpu()
    arc_scores = arc_scores.cpu()
    label_scores = label_scores.cpu()
    cls_logits = cls_logits.cpu()

    # Extract word-level predictions per example
    results = []
    for b in range(B):
        words = words_batch[b]
        word_ids = encs.word_ids(b)
        n_words = len(words)

        # Map word → first subword token
        w2t = {}
        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                w2t[wid] = k
            prev = wid

        # Word-level POS/NER logits
        w_pos = np.zeros((n_words, len(POS_LABELS)), dtype=np.float16)
        w_ner = np.zeros((n_words, len(NER_LABELS)), dtype=np.float16)
        for wid, tidx in w2t.items():
            if wid < n_words:
                w_pos[wid] = pos_logits[b, tidx].half().numpy()
                w_ner[wid] = ner_logits[b, tidx].half().numpy()

        # Word-level DEP arc probs (softmax over valid word positions)
        valid_tidxs = [w2t[w] for w in range(n_words) if w in w2t]
        w_arc = np.zeros((n_words, n_words), dtype=np.float16)
        for wi in range(n_words):
            if wi in w2t:
                ti = w2t[wi]
                for wj in range(n_words):
                    if wj in w2t:
                        tj = w2t[wj]
                        w_arc[wi, wj] = arc_scores[b, ti, tj].half().item()

        # DEP label: top-5 per (word, head) pair — store sparse
        arc_preds = w_arc.argmax(axis=-1)
        w_dep_labels = []
        for wi in range(n_words):
            hi = int(arc_preds[wi])
            if wi in w2t and hi in w2t:
                ti, th = w2t[wi], w2t[hi]
                scores = label_scores[b, ti, th].half().numpy()
                top5_idx = scores.argsort()[-5:][::-1]
                top5 = [(int(idx), float(scores[idx])) for idx in top5_idx]
            else:
                top5 = [(0, 1.0)]
            w_dep_labels.append(top5)

        # CLS logits (sentence-level)
        w_cls = cls_logits[b].half().numpy()

        # Hard predictions
        pos_hard = [POS_LABELS[w_pos[i].argmax()] for i in range(n_words)]
        ner_hard = [NER_LABELS[w_ner[i].argmax()] for i in range(n_words)]

        # Find first verb for SRL
        verb_idx = next((i for i, p in enumerate(pos_hard) if p == "VERB"), -1)

        results.append({
            "words": words,
            "n_words": n_words,
            "pos_logits": w_pos,
            "ner_logits": w_ner,
            "dep_arc_scores": w_arc,
            "dep_label_top5": w_dep_labels,
            "cls_logits": w_cls,
            "pos_hard": pos_hard,
            "ner_hard": ner_hard,
            "verb_idx": verb_idx,
        })

    # SRL pass — one per example for the first verb
    for b in range(B):
        r = results[b]
        vi = r["verb_idx"]
        n = r["n_words"]
        if vi < 0 or vi not in w2t:
            r["srl_logits"] = np.zeros((n, len(SRL_TAGS)), dtype=np.float16)
            r["predicate_word_idx"] = 0
            continue

        tidx = w2t[vi]
        indicator = torch.zeros(1, S, dtype=torch.long, device=device)
        indicator[0, tidx] = 1
        srl_emb = emb[b:b+1] + m["pred_embedding"](indicator)
        srl_out = enc.encoder(srl_emb, mask[b:b+1])
        srl_logits = m["srl_cls"](srl_out.last_hidden_state).cpu()

        w_srl = np.zeros((n, len(SRL_TAGS)), dtype=np.float16)
        word_ids = encs.word_ids(b)
        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev and wid < n:
                w_srl[wid] = srl_logits[0, k].half().numpy()
            prev = wid

        r["srl_logits"] = w_srl
        r["predicate_word_idx"] = vi

    return results


# ── Corpus loading ────────────────────────────────────────────

def load_corpus(corpus_dir, max_sentences):
    """Load sentences from annotated corpus."""
    corpus_dir = Path(corpus_dir)
    sentences = []
    for domain_dir in sorted(corpus_dir.iterdir()):
        jsonl = domain_dir / "annotated.jsonl"
        if not jsonl.exists():
            continue
        count = 0
        with open(jsonl) as f:
            for line in f:
                if len(sentences) >= max_sentences:
                    break
                obj = json.loads(line)
                tokens = obj.get("tokens", [])
                words = [t["form"] for t in tokens]
                if len(words) < 3 or len(words) > 80:
                    continue
                sentences.append(words)
                count += 1
        print(f"  {domain_dir.name}: {count:,}")
        if len(sentences) >= max_sentences:
            break
    return sentences


def load_prepared_data(data_dir, max_sentences):
    """Load from prepared JSON files as additional data."""
    data_dir = Path(data_dir)
    sentences = []

    for fname in ["ud_train.json", "ner_train.json", "srl_train.json"]:
        path = data_dir / fname
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for ex in data:
            if len(sentences) >= max_sentences:
                break
            words = ex.get("words", [])
            if len(words) >= 3:
                sentences.append(words)
        print(f"  {fname}: {len(data):,} → {len(sentences):,} total")
        if len(sentences) >= max_sentences:
            break

    return sentences


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/kniv-deberta-nlp-base-en-large")
    parser.add_argument("--corpus", default="corpus/output/annotated")
    parser.add_argument("--prepared-data", default="data/prepared/kniv-deberta-cascade")
    parser.add_argument("--output", default="data/distillation")
    parser.add_argument("--max-sentences", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=50000)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)

    # Load corpus
    print("Loading corpus...")
    sentences = []
    if os.path.isdir(args.corpus):
        sentences = load_corpus(args.corpus, args.max_sentences)
    print(f"  Corpus: {len(sentences):,}")

    # Add prepared data
    remaining = args.max_sentences - len(sentences)
    if remaining > 0 and os.path.isdir(args.prepared_data):
        print("Loading prepared data...")
        extra = load_prepared_data(args.prepared_data, remaining)
        sentences.extend(extra)
    print(f"  Total: {len(sentences):,} sentences")

    # Load teacher
    print("\nLoading teacher model...")
    m = load_teacher(args.model, device)
    print(f"Teacher loaded on {device}.")

    # Generate predictions in batches
    shard_idx = 0
    shard_rows = []
    total_processed = 0

    for batch_start in tqdm(range(0, len(sentences), args.batch_size), desc="Generating"):
        batch_words = sentences[batch_start:batch_start + args.batch_size]
        results = predict_batch(batch_words, m, device, args.max_length)

        for r in results:
            shard_rows.append({
                "words": r["words"],
                "n_words": r["n_words"],
                "pos_logits": r["pos_logits"].tobytes(),
                "ner_logits": r["ner_logits"].tobytes(),
                "srl_logits": r["srl_logits"].tobytes(),
                "dep_arc_scores": r["dep_arc_scores"].tobytes(),
                "dep_label_top5": json.dumps(r["dep_label_top5"]),
                "cls_logits": r["cls_logits"].tobytes(),
                "pos_hard": r["pos_hard"],
                "ner_hard": r["ner_hard"],
                "predicate_word_idx": r["predicate_word_idx"],
            })
            total_processed += 1

        # Write shard
        if len(shard_rows) >= args.shard_size:
            shard_path = os.path.join(args.output, f"shard_{shard_idx:03d}.parquet")
            table = pa.Table.from_pylist(shard_rows)
            pq.write_table(table, shard_path)
            print(f"\n  Saved {shard_path} ({len(shard_rows):,} rows, "
                  f"{os.path.getsize(shard_path)/1e6:.0f} MB)")
            shard_rows = []
            shard_idx += 1

    # Write remaining
    if shard_rows:
        shard_path = os.path.join(args.output, f"shard_{shard_idx:03d}.parquet")
        table = pa.Table.from_pylist(shard_rows)
        pq.write_table(table, shard_path)
        print(f"\n  Saved {shard_path} ({len(shard_rows):,} rows, "
              f"{os.path.getsize(shard_path)/1e6:.0f} MB)")
        shard_idx += 1

    # Save metadata
    meta = {
        "teacher": args.model,
        "total_sentences": total_processed,
        "num_shards": shard_idx,
        "max_length": args.max_length,
        "heads": {
            "pos": {"n_labels": len(POS_LABELS), "dtype": "float16", "shape": "[n_words, 17]"},
            "ner": {"n_labels": len(NER_LABELS), "dtype": "float16", "shape": "[n_words, 37]"},
            "srl": {"n_labels": len(SRL_TAGS), "dtype": "float16", "shape": "[n_words, 42]"},
            "dep_arc": {"dtype": "float16", "shape": "[n_words, n_words]"},
            "dep_label": {"dtype": "json", "shape": "[n_words, top5]"},
            "cls": {"n_labels": len(CLS_LABELS), "dtype": "float16", "shape": "[8]"},
        },
        "storage": "logits stored as raw bytes (float16), dep labels as JSON top-5",
        "alignment": "word-level (first subword per word). Students align at training time.",
    }
    meta_path = os.path.join(args.output, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Summary
    total_size = sum(os.path.getsize(os.path.join(args.output, f))
                     for f in os.listdir(args.output) if f.endswith(".parquet"))
    print(f"\n{'=' * 60}")
    print(f"DISTILLATION DATA GENERATED")
    print(f"{'=' * 60}")
    print(f"  Sentences: {total_processed:,}")
    print(f"  Shards: {shard_idx}")
    print(f"  Total size: {total_size / 1e9:.1f} GB")
    print(f"  Output: {args.output}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
