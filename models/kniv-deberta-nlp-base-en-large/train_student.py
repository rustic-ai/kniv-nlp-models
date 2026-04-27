"""Train a student cascade model from teacher distillation data.

All 5 heads train simultaneously in a single session using KL divergence
on teacher soft labels + optional CE on hard labels.

Usage:
    .venv/bin/python models/kniv-deberta-nlp-base-en-large/train_student.py \
        --student microsoft/deberta-v3-base \
        --distillation-data data/distillation \
        --output models/kniv-deberta-nlp-base-en-base

    .venv/bin/python models/kniv-deberta-nlp-base-en-large/train_student.py \
        --student answerdotai/ModernBERT-base \
        --distillation-data data/distillation \
        --output models/kniv-modernbert-nlp-base-en-base

Supported students:
    microsoft/deberta-v3-base       (86M, 12 layers, 768 hidden)
    microsoft/deberta-v3-xsmall     (22M, 6 layers, 384 hidden)
    answerdotai/ModernBERT-base     (150M, 22 layers, 768 hidden)
    any HuggingFace encoder model
"""
from __future__ import annotations
import argparse, json, os, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import pyarrow.parquet as pq
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

N_POS, N_NER, N_SRL, N_DEP, N_CLS = len(POS_LABELS), len(NER_LABELS), len(SRL_TAGS), len(DEPREL_LIST), len(CLS_LABELS)
pos_map = {l: i for i, l in enumerate(POS_LABELS)}
ner_map = {l: i for i, l in enumerate(NER_LABELS)}

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
        nn.init.xavier_uniform_(self.weight)
    def forward(self, h_dep, h_head):
        B, S, D = h_dep.size()
        h_dep = torch.cat([h_dep, torch.ones(B, S, 1, device=h_dep.device)], -1)
        h_head = torch.cat([h_head, torch.ones(B, S, 1, device=h_head.device)], -1)
        scores = torch.einsum("bxi,oij,byj->boxy", h_dep, self.weight, h_head)
        return scores.squeeze(1).contiguous() if scores.size(1) == 1 else scores.permute(0, 2, 3, 1).contiguous()

class BiaffineDEPHead(nn.Module):
    def __init__(self, H, arc_dim, label_dim, num_labels, dropout=0.1):
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


class StudentCascade(nn.Module):
    """Student cascade model. All heads, single encoder pass."""

    def __init__(self, encoder, H, num_layers):
        super().__init__()
        self.encoder = encoder
        self.pred_embedding = nn.Embedding(2, H)
        nn.init.zeros_(self.pred_embedding.weight)

        NL = num_layers
        self.pos_sm = ScalarMix(NL)
        self.pos_head = nn.Linear(H, N_POS)

        self.ner_sm = ScalarMix(NL)
        lstm_h = max(H // 4, 64)
        self.ner_lstm = nn.LSTM(H, lstm_h, bidirectional=True, batch_first=True)
        self.ner_proj = nn.Linear(lstm_h * 2, H)
        self.ner_head = nn.Sequential(
            nn.LayerNorm(H + N_POS), nn.Linear(H + N_POS, H),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(H, N_NER))

        self.dep_sm = ScalarMix(NL)
        in_dim = H + N_POS + N_NER
        self.dep_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU())
        arc_dim, label_dim = max(H // 2, 64), max(H // 8, 32)
        self.dep_biaff = BiaffineDEPHead(H, arc_dim, label_dim, N_DEP)

        self.srl_head = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H, N_SRL))

        self.cls_sm = ScalarMix(NL)
        self.cls_pool = AttentionPool(H)
        self.cls_head = nn.Sequential(
            nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H // 2, N_CLS))

    def forward(self, input_ids, attention_mask, predicate_idx):
        B, S = input_ids.size()

        # Encoder with predicate embedding
        if hasattr(self.encoder, 'embeddings'):
            emb = self.encoder.embeddings(input_ids)
            indicator = torch.zeros(B, S, dtype=torch.long, device=input_ids.device)
            indicator.scatter_(1, predicate_idx.unsqueeze(1), 1)
            emb = emb + self.pred_embedding(indicator)
            enc_out = self.encoder.encoder(emb, attention_mask, output_hidden_states=True)
        else:
            # For models without separate embeddings/encoder (e.g., ModernBERT)
            enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                   output_hidden_states=True)

        layers = list(enc_out.hidden_states)
        last = enc_out.last_hidden_state

        # POS
        pos_logits = self.pos_head(self.pos_sm(layers))
        pos_probs = torch.softmax(pos_logits, dim=-1)

        # NER (BiLSTM + POS cascade)
        ner_h = self.ner_sm(layers)
        lo, _ = self.ner_lstm(ner_h)
        adapted = self.ner_proj(lo) + ner_h
        ner_logits = self.ner_head(torch.cat([adapted, pos_probs.detach()], dim=-1))
        ner_probs = torch.softmax(ner_logits, dim=-1)

        # DEP (Biaffine + POS/NER cascade)
        dep_h = self.dep_sm(layers)
        proj = self.dep_proj(torch.cat([dep_h, pos_probs.detach(), ner_probs.detach()], dim=-1))
        arc_scores, label_scores = self.dep_biaff(proj)

        # SRL (from last hidden — encoder already has predicate embedding)
        srl_logits = self.srl_head(last)

        # CLS
        cls_logits = self.cls_head(self.cls_pool(self.cls_sm(layers), attention_mask))

        return pos_logits, ner_logits, arc_scores, label_scores, srl_logits, cls_logits


# ── Dataset ───────────────────────────────────────────────────

class DistillationDataset(Dataset):
    """Loads teacher predictions and aligns to student tokenizer."""

    def __init__(self, parquet_paths, student_tokenizer, max_length=128):
        self.tokenizer = student_tokenizer
        self.max_length = max_length
        # Load all shards
        tables = [pq.read_table(p) for p in parquet_paths]
        self.data = pa.concat_tables(tables).to_pandas()
        print(f"  Loaded {len(self.data):,} examples from {len(parquet_paths)} shards")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        words = row["words"]
        n_words = row["n_words"]

        # Tokenize with student's tokenizer
        enc = self.tokenizer(words, is_split_into_words=True,
                             max_length=self.max_length, padding="max_length",
                             truncation=True, return_tensors="pt")
        word_ids = enc.word_ids()
        S = self.max_length

        # Decode teacher logits from bytes
        t_pos = np.frombuffer(row["pos_logits"], dtype=np.float16).reshape(n_words, N_POS)
        t_ner = np.frombuffer(row["ner_logits"], dtype=np.float16).reshape(n_words, N_NER)
        t_srl = np.frombuffer(row["srl_logits"], dtype=np.float16).reshape(n_words, N_SRL)
        t_cls = np.frombuffer(row["cls_logits"], dtype=np.float16).reshape(N_CLS)
        t_arc = np.frombuffer(row["dep_arc_scores"], dtype=np.float16).reshape(n_words, n_words)

        # Align teacher word-level logits → student subword tokens
        pos_teacher = torch.zeros(S, N_POS)
        ner_teacher = torch.zeros(S, N_NER)
        srl_teacher = torch.zeros(S, N_SRL)
        arc_teacher = torch.zeros(S, S)
        valid_mask = torch.zeros(S, dtype=torch.bool)

        # Hard labels
        pos_hard = torch.full((S,), -100, dtype=torch.long)
        ner_hard = torch.full((S,), -100, dtype=torch.long)

        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev and wid < n_words:
                pos_teacher[k] = torch.from_numpy(t_pos[wid].astype(np.float32))
                ner_teacher[k] = torch.from_numpy(t_ner[wid].astype(np.float32))
                srl_teacher[k] = torch.from_numpy(t_srl[wid].astype(np.float32))
                valid_mask[k] = True

                # Arc scores: map word-level to token-level
                for wj in range(n_words):
                    # Find token for word wj
                    for k2, wid2 in enumerate(word_ids):
                        if wid2 == wj:
                            arc_teacher[k, k2] = t_arc[wid, wj]
                            break

                # Hard labels
                pos_tag = row["pos_hard"][wid] if wid < len(row["pos_hard"]) else "X"
                ner_tag = row["ner_hard"][wid] if wid < len(row["ner_hard"]) else "O"
                pos_hard[k] = pos_map.get(pos_tag, 0)
                ner_hard[k] = ner_map.get(ner_tag, 0)
            prev = wid

        cls_teacher = torch.from_numpy(t_cls.astype(np.float32))

        # Predicate index (word → token)
        pred_wid = row["predicate_word_idx"]
        pred_tidx = 0
        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev and wid == pred_wid:
                pred_tidx = k
                break
            prev = wid

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "predicate_idx": torch.tensor(pred_tidx, dtype=torch.long),
            "pos_teacher": pos_teacher,
            "ner_teacher": ner_teacher,
            "srl_teacher": srl_teacher,
            "arc_teacher": arc_teacher,
            "cls_teacher": cls_teacher,
            "pos_hard": pos_hard,
            "ner_hard": ner_hard,
            "valid_mask": valid_mask,
        }


def collate(feats):
    batch = {}
    for k in feats[0]:
        vals = [f[k] for f in feats]
        if isinstance(vals[0], torch.Tensor):
            batch[k] = torch.stack(vals)
        else:
            batch[k] = vals
    return batch


# ── Loss ──────────────────────────────────────────────────────

def distill_loss(student_logits, teacher_logits, valid_mask, temperature=3.0):
    """Token-level KL divergence on softened predictions."""
    # Only compute on valid (non-padding, first-subword) positions
    if valid_mask.dim() == 1:
        sl = student_logits[valid_mask]
        tl = teacher_logits[valid_mask]
    else:
        sl = student_logits
        tl = teacher_logits

    if sl.numel() == 0:
        return torch.tensor(0.0, device=student_logits.device)

    soft_s = F.log_softmax(sl / temperature, dim=-1)
    soft_t = F.softmax(tl / temperature, dim=-1)
    return F.kl_div(soft_s, soft_t, reduction="batchmean") * (temperature ** 2)


def hard_loss(student_logits, labels, valid_mask):
    """Standard CE on hard labels."""
    sl = student_logits.view(-1, student_logits.size(-1))
    ll = labels.view(-1)
    return F.cross_entropy(sl, ll, ignore_index=-100)


# ── Evaluation ────────────────────────────────────────────────

@torch.no_grad()
def quick_eval(model, eval_data, tokenizer, device, max_n=500):
    """Quick POS accuracy on UD EWT dev."""
    model.eval()
    correct, total = 0, 0
    for ex in eval_data[:max_n]:
        words = ex["words"]
        enc = tokenizer(words, is_split_into_words=True, return_tensors="pt",
                        padding=True, truncation=True, max_length=128)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        pred_idx = torch.zeros(1, dtype=torch.long, device=device)

        pos_logits = model(ids, mask, pred_idx)[0]

        prev = None
        for k, wid in enumerate(enc.word_ids()):
            if wid is not None and wid != prev and wid < len(ex["pos_tags"]):
                pred = POS_LABELS[pos_logits[0, k].argmax().item()]
                if pred == ex["pos_tags"][wid]:
                    correct += 1
                total += 1
            prev = wid
    model.train()
    return correct / total if total > 0 else 0


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", required=True, help="HuggingFace encoder name (e.g., microsoft/deberta-v3-base)")
    parser.add_argument("--distillation-data", default="data/distillation")
    parser.add_argument("--output", required=True, help="Output directory for student model")
    parser.add_argument("--eval-data", default="data/prepared/kniv-deberta-cascade/ud_dev.json")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-encoder", type=float, default=2e-5)
    parser.add_argument("--lr-heads", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for distillation vs hard labels")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device(args.device)

    # Load student encoder
    print(f"Loading student encoder: {args.student}")
    student_tokenizer = AutoTokenizer.from_pretrained(args.student)
    student_encoder = AutoModel.from_pretrained(args.student).float()
    H = student_encoder.config.hidden_size
    NL = student_encoder.config.num_hidden_layers + 1
    total_enc = sum(p.numel() for p in student_encoder.parameters())
    print(f"  Hidden: {H}, Layers: {NL - 1}, Params: {total_enc / 1e6:.0f}M")

    # Build student cascade
    model = StudentCascade(student_encoder, H, NL).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    head_params = total_params - total_enc
    print(f"  Head params: {head_params / 1e6:.1f}M, Total: {total_params / 1e6:.0f}M")

    # Load distillation data
    print(f"\nLoading distillation data from {args.distillation_data}...")
    shard_paths = sorted(Path(args.distillation_data).glob("shard_*.parquet"))
    if not shard_paths:
        print("ERROR: No parquet shards found. Run generate_distillation_data.py first.")
        return
    dataset = DistillationDataset(shard_paths, student_tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=2, pin_memory=True)

    # Load eval data
    eval_data = []
    if os.path.exists(args.eval_data):
        with open(args.eval_data) as f:
            eval_data = json.load(f)
        print(f"  Eval: {len(eval_data)} sentences (POS accuracy)")

    # Optimizer
    enc_params = list(model.encoder.parameters()) + list(model.pred_embedding.parameters())
    head_param_list = [p for n, p in model.named_parameters()
                       if not n.startswith("encoder.") and not n.startswith("pred_embedding.")]
    optimizer = torch.optim.AdamW([
        {"params": enc_params, "lr": args.lr_encoder},
        {"params": head_param_list, "lr": args.lr_heads},
    ], weight_decay=0.01)

    total_steps = args.epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * 0.1), total_steps)

    T = args.temperature
    alpha = args.alpha
    print(f"\nTraining: {args.epochs} epochs, {len(loader)} batches/epoch")
    print(f"  Temperature: {T}, Alpha: {alpha} (distill/hard)")
    print(f"  Encoder LR: {args.lr_encoder}, Head LR: {args.lr_heads}")
    print()

    # Training loop
    best_pos = 0.0
    state = {}

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(loader, desc=f"E{epoch + 1}/{args.epochs}"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            pred_idx = batch["predicate_idx"].to(device)
            valid = batch["valid_mask"].to(device)

            # Teacher targets
            t_pos = batch["pos_teacher"].to(device)
            t_ner = batch["ner_teacher"].to(device)
            t_srl = batch["srl_teacher"].to(device)
            t_arc = batch["arc_teacher"].to(device)
            t_cls = batch["cls_teacher"].to(device)
            h_pos = batch["pos_hard"].to(device)
            h_ner = batch["ner_hard"].to(device)

            # Student forward
            s_pos, s_ner, s_arc, s_lab, s_srl, s_cls = model(ids, mask, pred_idx)

            B, S = ids.size()
            flat_valid = valid.view(B * S)

            # Distillation losses
            loss_pos = distill_loss(s_pos.view(B * S, -1), t_pos.view(B * S, -1), flat_valid, T)
            loss_ner = distill_loss(s_ner.view(B * S, -1), t_ner.view(B * S, -1), flat_valid, T)
            loss_srl = distill_loss(s_srl.view(B * S, -1), t_srl.view(B * S, -1), flat_valid, T)
            loss_cls = distill_loss(s_cls, t_cls, torch.ones(B, dtype=torch.bool, device=device), T)

            # DEP arc distillation (token-level, valid positions only)
            loss_arc = distill_loss(s_arc.view(B * S, -1), t_arc.view(B * S, -1), flat_valid, T)

            # Hard label losses (where available)
            loss_pos_hard = hard_loss(s_pos, h_pos, valid)
            loss_ner_hard = hard_loss(s_ner, h_ner, valid)

            # Combined
            distill = (1.0 * loss_pos + 1.5 * loss_ner + 1.5 * loss_arc +
                       2.0 * loss_srl + 1.0 * loss_cls)
            hard = 1.0 * loss_pos_hard + 1.5 * loss_ner_hard
            loss = alpha * distill + (1 - alpha) * hard

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Eval
        pos_acc = quick_eval(model, eval_data, student_tokenizer, device) if eval_data else 0

        print(f"  E{epoch + 1} — Loss: {avg_loss:.4f}, POS acc: {pos_acc:.4f}")

        if pos_acc > best_pos or not eval_data:
            best_pos = pos_acc
            # Save state
            state = {
                "encoder": {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()},
                "pred_embedding": {k: v.cpu().clone() for k, v in model.pred_embedding.state_dict().items()},
                "pos_sm": {k: v.cpu().clone() for k, v in model.pos_sm.state_dict().items()},
                "pos_head": {k: v.cpu().clone() for k, v in model.pos_head.state_dict().items()},
                "ner_sm": {k: v.cpu().clone() for k, v in model.ner_sm.state_dict().items()},
                "ner_lstm": {k: v.cpu().clone() for k, v in model.ner_lstm.state_dict().items()},
                "ner_proj": {k: v.cpu().clone() for k, v in model.ner_proj.state_dict().items()},
                "ner_head": {k: v.cpu().clone() for k, v in model.ner_head.state_dict().items()},
                "dep_sm": {k: v.cpu().clone() for k, v in model.dep_sm.state_dict().items()},
                "dep_proj": {k: v.cpu().clone() for k, v in model.dep_proj.state_dict().items()},
                "dep_biaff": {k: v.cpu().clone() for k, v in model.dep_biaff.state_dict().items()},
                "srl_head": {k: v.cpu().clone() for k, v in model.srl_head.state_dict().items()},
                "cls_sm": {k: v.cpu().clone() for k, v in model.cls_sm.state_dict().items()},
                "cls_pool": {k: v.cpu().clone() for k, v in model.cls_pool.state_dict().items()},
                "cls_head": {k: v.cpu().clone() for k, v in model.cls_head.state_dict().items()},
            }
            torch.save(state, os.path.join(args.output, "model.pt"))
            print(f"    Saved (POS: {best_pos:.4f})")

    # Save tokenizer + metadata
    student_tokenizer.save_pretrained(args.output)
    meta = {
        "version": "v5-student",
        "teacher": "dragonscale-ai/kniv-deberta-nlp-base-en-large",
        "student_encoder": args.student,
        "hidden_size": H,
        "num_layers": NL - 1,
        "total_params": total_params,
        "head_params": head_params,
        "training": {
            "epochs": args.epochs,
            "temperature": T,
            "alpha": alpha,
            "lr_encoder": args.lr_encoder,
            "lr_heads": args.lr_heads,
        },
        "best_pos_accuracy": round(best_pos, 4),
    }
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"STUDENT TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Student: {args.student}")
    print(f"  Params: {total_params / 1e6:.0f}M ({head_params / 1e6:.1f}M heads)")
    print(f"  Best POS: {best_pos:.4f}")
    print(f"  Saved to: {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
