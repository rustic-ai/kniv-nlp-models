"""Evaluate kniv-deberta-nlp-base-en-large v5 on standard test sets.

Runs all 5 heads (POS, NER, SRL, DEP, CLS) on test splits and reports
benchmark numbers.

Usage:
    uv run python models/kniv-deberta-nlp-base-en-large/evaluate_v5.py
    uv run python models/kniv-deberta-nlp-base-en-large/evaluate_v5.py --model-dir models/kniv-deberta-nlp-base-en-large
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "prepared" / "kniv-deberta-cascade"

# ── Label sets ────────────────────────────────────────────────
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

# ── Modules ───────────────────────────────────────────────────
class ScalarMix(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n))
        self.scale = nn.Parameter(torch.ones(1))
    def forward(self, layers):
        w = torch.softmax(self.weights, dim=0)
        return self.scale * sum(wi * li for wi, li in zip(w, layers))

class AttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)
    def forward(self, hidden, mask):
        scores = self.attn(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = torch.softmax(scores, dim=-1)
        return (hidden * weights.unsqueeze(-1)).sum(dim=1)

class Biaffine(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim + 1, in_dim + 1))
    def forward(self, h_dep, h_head):
        B, S, D = h_dep.size()
        h_dep = torch.cat([h_dep, torch.ones(B, S, 1, device=h_dep.device)], dim=-1)
        h_head = torch.cat([h_head, torch.ones(B, S, 1, device=h_head.device)], dim=-1)
        scores = torch.einsum("bxi,oij,byj->boxy", h_dep, self.weight, h_head)
        if scores.size(1) == 1:
            return scores.squeeze(1).contiguous()
        return scores.permute(0, 2, 3, 1).contiguous()

class BiaffineDEPHead(nn.Module):
    def __init__(self, hidden_size, arc_dim=512, label_dim=128, num_labels=53, dropout=0.1):
        super().__init__()
        self.arc_dep = nn.Sequential(nn.Linear(hidden_size, arc_dim), nn.LayerNorm(arc_dim), nn.GELU(), nn.Dropout(dropout))
        self.arc_head = nn.Sequential(nn.Linear(hidden_size, arc_dim), nn.LayerNorm(arc_dim), nn.GELU(), nn.Dropout(dropout))
        self.label_dep = nn.Sequential(nn.Linear(hidden_size, label_dim), nn.LayerNorm(label_dim), nn.GELU(), nn.Dropout(dropout))
        self.label_head = nn.Sequential(nn.Linear(hidden_size, label_dim), nn.LayerNorm(label_dim), nn.GELU(), nn.Dropout(dropout))
        self.biaffine_arc = Biaffine(arc_dim, out_dim=1)
        self.biaffine_label = Biaffine(label_dim, out_dim=num_labels)
    def forward(self, hidden):
        return (self.biaffine_arc(self.arc_dep(hidden), self.arc_head(hidden)),
                self.biaffine_label(self.label_dep(hidden), self.label_head(hidden)))

# ── Viterbi ───────────────────────────────────────────────────
def viterbi_decode(logits, tag2id):
    num_tags = logits.size(-1)
    seq_len = logits.size(0)
    log_probs = torch.log_softmax(logits, dim=-1)
    allowed = torch.ones(num_tags, num_tags, dtype=torch.bool)
    tag_names = {v: k for k, v in tag2id.items()}
    for j in range(num_tags):
        j_name = tag_names.get(j, "O")
        if j_name.startswith("I-"):
            role = j_name[2:]
            b_tag = tag2id.get(f"B-{role}", -1)
            for i in range(num_tags):
                if i != b_tag and i != j:
                    allowed[i][j] = False
    NEG_INF = -1e9
    viterbi = torch.full((seq_len, num_tags), NEG_INF)
    backptr = torch.zeros(seq_len, num_tags, dtype=torch.long)
    viterbi[0] = log_probs[0]
    for t in range(1, seq_len):
        for j in range(num_tags):
            scores = viterbi[t - 1].clone()
            scores[~allowed[:, j]] = NEG_INF
            best = scores.argmax()
            viterbi[t, j] = scores[best] + log_probs[t, j]
            backptr[t, j] = best
    path = [0] * seq_len
    path[-1] = viterbi[-1].argmax().item()
    for t in range(seq_len - 2, -1, -1):
        path[t] = backptr[t + 1, path[t + 1]].item()
    return path

# ── Datasets ──────────────────────────────────────────────────
def collate(feats):
    return {k: torch.stack([f[k] for f in feats]) for k in feats[0]}

class TokenDataset(Dataset):
    def __init__(self, examples, tokenizer, label_map, label_key, max_length=128):
        self.examples, self.tokenizer, self.label_map = examples, tokenizer, label_map
        self.label_key, self.max_length = label_key, max_length
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        words = ex["words"]
        tags = ex.get(self.label_key, [])
        enc = self.tokenizer(words, is_split_into_words=True, max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        aligned, prev = [], None
        for wid in enc.word_ids():
            if wid is None: aligned.append(-100)
            elif wid != prev: aligned.append(self.label_map.get(tags[wid] if wid < len(tags) else "O", 0))
            else: aligned.append(-100)
            prev = wid
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(aligned, dtype=torch.long)}

class SRLDataset(Dataset):
    def __init__(self, examples, tokenizer, label_map, max_length=128):
        self.examples, self.tokenizer, self.label_map, self.max_length = examples, tokenizer, label_map, max_length
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        words, tags = ex["words"], ex["srl_tags"]
        pred_idx = ex["predicate_idx"]
        enc = self.tokenizer(words, is_split_into_words=True, max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        pred_token_idx, aligned, prev = 0, [], None
        for k, wid in enumerate(enc.word_ids()):
            if wid is None: aligned.append(-100)
            elif wid != prev:
                aligned.append(self.label_map.get(tags[wid] if wid < len(tags) else "O", 0))
                if wid == pred_idx: pred_token_idx = k
            else: aligned.append(-100)
            prev = wid
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(aligned, dtype=torch.long),
                "predicate_idx": torch.tensor(pred_token_idx, dtype=torch.long)}

class BiaffineDEPDataset(Dataset):
    def __init__(self, examples, tokenizer, deprel_map, max_length=128):
        self.examples, self.tokenizer, self.deprel_map, self.max_length = examples, tokenizer, deprel_map, max_length
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        words, heads, deprels = ex["words"], ex["heads"], ex["deprels"]
        enc = self.tokenizer(words, is_split_into_words=True, max_length=self.max_length,
                             padding="max_length", truncation=True, return_tensors="pt")
        word_ids = enc.word_ids()
        seq_len = enc["input_ids"].size(1)
        head_labels = torch.full((seq_len,), -1, dtype=torch.long)
        rel_labels = torch.full((seq_len,), -1, dtype=torch.long)
        word_to_token = {}
        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev: word_to_token[wid] = k
            prev = wid
        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev and wid < len(heads):
                hw = heads[wid]
                if hw == -1: head_labels[k] = k
                elif hw in word_to_token: head_labels[k] = word_to_token[hw]
                rel_labels[k] = self.deprel_map.get(deprels[wid] if wid < len(deprels) else "dep", 0)
            prev = wid
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "head_labels": head_labels, "rel_labels": rel_labels}

class CLSDataset(Dataset):
    def __init__(self, examples, tokenizer, label_map, max_length=128):
        self.examples, self.tokenizer, self.label_map, self.max_length = examples, tokenizer, label_map, max_length
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx):
        ex = self.examples[idx]
        text = ex.get("text", "")
        prev_text = ex.get("prev_text")
        if prev_text:
            enc = self.tokenizer(prev_text, text, max_length=self.max_length, padding="max_length",
                                 truncation=True, return_tensors="pt")
        else:
            enc = self.tokenizer(text, max_length=self.max_length, padding="max_length",
                                 truncation=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.label_map[ex["cls_label"]], dtype=torch.long)}

# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/kniv-deberta-nlp-base-en-large")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-ner", type=int, default=5000, help="Max NER test examples (full set is 84K)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")

    # Load vocabs
    with open(DATA_DIR / "label_vocabs.json") as f:
        vocabs = json.load(f)
    pos_labels = vocabs["pos_labels"]
    ner_labels = vocabs["ner_labels"]
    pos_map = {l: i for i, l in enumerate(pos_labels)}
    ner_map = {l: i for i, l in enumerate(ner_labels)}
    srl_map = {t: i for i, t in enumerate(SRL_TAGS)}
    cls_map = {l: i for i, l in enumerate(CLS_LABELS)}
    deprel_map = {r: i for i, r in enumerate(DEPREL_LIST)}
    num_deprels = len(DEPREL_LIST)

    # Load tokenizer + encoder
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large")
    H = encoder.config.hidden_size
    num_layers = encoder.config.num_hidden_layers + 1

    # Load checkpoint
    state = torch.load(f"{args.model_dir}/model.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["deberta"])

    # Build heads matching checkpoint keys exactly
    pred_embedding = nn.Embedding(2, H)
    pred_embedding.load_state_dict(state["pred_embedding"])

    pos_scalar_mix = ScalarMix(num_layers)
    pos_scalar_mix.load_state_dict(state["pos_scalar_mix"])
    pos_head = nn.Linear(H, len(pos_labels))
    pos_head.load_state_dict(state["pos_head"])

    ner_scalar_mix = ScalarMix(num_layers)
    ner_scalar_mix.load_state_dict(state["ner_scalar_mix"])
    ner_lstm = nn.LSTM(H, 256, num_layers=1, bidirectional=True, batch_first=True)
    ner_lstm.load_state_dict(state["ner_lstm"])
    ner_proj = nn.Linear(512, H)
    ner_proj.load_state_dict(state["ner_proj"])
    ner_head = nn.Sequential(nn.LayerNorm(H + len(pos_labels)), nn.Linear(H + len(pos_labels), H),
                             nn.GELU(), nn.Dropout(0.1), nn.Linear(H, len(ner_labels)))
    ner_head.load_state_dict(state["ner_head"])

    # SRL — NO BiLSTM in this checkpoint, just MLP classifier
    srl_classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(), nn.Dropout(0.1),
                                   nn.Linear(H, len(SRL_TAGS)))
    srl_classifier.load_state_dict(state["classifier"])

    dep_scalar_mix = ScalarMix(num_layers)
    dep_scalar_mix.load_state_dict(state["dep_scalar_mix"])
    in_dim = H + len(pos_labels) + len(ner_labels)
    dep_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU())
    dep_proj.load_state_dict(state["dep_proj"])
    dep_biaffine = BiaffineDEPHead(H, arc_dim=512, label_dim=128, num_labels=num_deprels)
    dep_biaffine.load_state_dict(state["dep_biaffine"])

    cls_scalar_mix = ScalarMix(num_layers)
    cls_scalar_mix.load_state_dict(state["cls_scalar_mix"])
    cls_pool = AttentionPool(H)
    cls_pool.load_state_dict(state["cls_pool"])
    cls_head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(0.1),
                             nn.Linear(H // 2, len(CLS_LABELS)))
    cls_head.load_state_dict(state["cls_head"])

    # Move all to device, float, eval
    modules = [encoder, pred_embedding, pos_scalar_mix, pos_head,
               ner_scalar_mix, ner_lstm, ner_proj, ner_head,
               srl_classifier,
               dep_scalar_mix, dep_proj, dep_biaffine,
               cls_scalar_mix, cls_pool, cls_head]
    for m in modules:
        m.float().to(device).eval()

    # ── Load TEST data ────────────────────────────────────
    with open(DATA_DIR / "ud_test.json") as f:
        ud_test = json.load(f)
    with open(DATA_DIR / "ner_test.json") as f:
        ner_test_full = json.load(f)
    ner_test = ner_test_full[:args.max_ner]
    with open(DATA_DIR / "srl_test.json") as f:
        srl_test = json.load(f)
    with open(DATA_DIR / "cls_sgd_mwoz_train.json") as f:
        cls_data = json.load(f)
    random.seed(42)
    random.shuffle(cls_data)
    cls_dev = cls_data[:4000]

    print(f"\nTest sets: POS/DEP={len(ud_test)}, NER={len(ner_test)} (of {len(ner_test_full)}), "
          f"SRL={len(srl_test)}, CLS={len(cls_dev)}")
    print("=" * 60)

    BS = args.batch_size

    # ── POS ───────────────────────────────────────────────
    print("\n[POS] Evaluating on UD EWT test...")
    pos_loader = DataLoader(TokenDataset(ud_test, tokenizer, pos_map, "pos_tags"), batch_size=BS, collate_fn=collate)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(pos_loader, desc="POS"):
            out = encoder(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device), output_hidden_states=True)
            preds = pos_head(pos_scalar_mix(out.hidden_states)).argmax(dim=-1).cpu()
            mask = batch["labels"] != -100
            correct += (preds[mask] == batch["labels"][mask]).sum().item()
            total += mask.sum().item()
    pos_acc = correct / total
    print(f"  POS Accuracy: {pos_acc:.4f} (UD EWT test, {len(ud_test)} sentences)")

    # ── NER ───────────────────────────────────────────────
    print("\n[NER] Evaluating on SpanMarker silver test...")
    from seqeval.metrics import f1_score as seq_f1
    ner_loader = DataLoader(TokenDataset(ner_test, tokenizer, ner_map, "ner_tags"), batch_size=BS, collate_fn=collate)
    all_g, all_p = [], []
    with torch.no_grad():
        for batch in tqdm(ner_loader, desc="NER"):
            out = encoder(input_ids=batch["input_ids"].to(device),
                          attention_mask=batch["attention_mask"].to(device), output_hidden_states=True)
            ner_h = ner_scalar_mix(out.hidden_states)
            lstm_out, _ = ner_lstm(ner_h)
            adapted = ner_proj(lstm_out) + ner_h
            pos_h = pos_scalar_mix(out.hidden_states)
            pos_p = torch.softmax(pos_head(pos_h), dim=-1)
            logits = ner_head(torch.cat([adapted, pos_p], dim=-1)).cpu()
            labs = batch["labels"]
            for j in range(labs.size(0)):
                vi = (labs[j] != -100).nonzero(as_tuple=True)[0]
                if len(vi) == 0: continue
                vl = logits[j, vi]
                vd = viterbi_decode(vl, ner_map)
                all_g.append([ner_labels[labs[j, k]] for k in vi.tolist()])
                all_p.append([ner_labels[vd[ki]] if vd[ki] < len(ner_labels) else "O" for ki in range(len(vi))])
    ner_f1 = seq_f1(all_g, all_p)
    print(f"  NER F1: {ner_f1:.4f} (SpanMarker silver test, {len(ner_test)} sentences)")
    print(f"  Note: evaluated on silver labels, not CoNLL-2003 gold")

    # ── SRL ───────────────────────────────────────────────
    print("\n[SRL] Evaluating on PropBank EWT test...")
    srl_loader = DataLoader(SRLDataset(srl_test, tokenizer, srl_map), batch_size=BS, collate_fn=collate)
    all_g, all_p = [], []
    with torch.no_grad():
        for batch in tqdm(srl_loader, desc="SRL"):
            B, S = batch["input_ids"].size()
            embed = encoder.embeddings(batch["input_ids"].to(device))
            indicator = torch.zeros(B, S, dtype=torch.long, device=device)
            indicator[torch.arange(B, device=device), batch["predicate_idx"].to(device)] = 1
            embed = embed + pred_embedding(indicator)
            enc_out = encoder.encoder(embed, batch["attention_mask"].to(device))
            h = enc_out.last_hidden_state
            logits = srl_classifier(h).cpu()
            labs = batch["labels"]
            for j in range(labs.size(0)):
                vi = (labs[j] != -100).nonzero(as_tuple=True)[0]
                if len(vi) == 0: continue
                vl = logits[j, vi]
                vd = viterbi_decode(vl, srl_map)
                gl = [SRL_TAGS[labs[j, k]] for k in vi.tolist()]
                pl = [SRL_TAGS[vd[ki]] if vd[ki] < len(SRL_TAGS) else "O" for ki in range(len(vi))]
                all_g.append([g if g != "V" else "O" for g in gl])
                all_p.append([p if p != "V" else "O" for p in pl])
    srl_f1 = seq_f1(all_g, all_p)
    print(f"  SRL F1: {srl_f1:.4f} (PropBank EWT test, {len(srl_test)} examples)")

    # ── DEP ───────────────────────────────────────────────
    print("\n[DEP] Evaluating on UD EWT test...")
    dep_loader = DataLoader(BiaffineDEPDataset(ud_test, tokenizer, deprel_map), batch_size=BS, collate_fn=collate)
    arc_correct, rel_correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(dep_loader, desc="DEP"):
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            out = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            dep_h = dep_scalar_mix(out.hidden_states)
            pos_h = pos_scalar_mix(out.hidden_states)
            pos_p = torch.softmax(pos_head(pos_h), dim=-1)
            ner_h = ner_scalar_mix(out.hidden_states)
            lstm_out, _ = ner_lstm(ner_h)
            adapted = ner_proj(lstm_out) + ner_h
            ner_p = torch.softmax(ner_head(torch.cat([adapted, pos_p], dim=-1)), dim=-1)
            arc_scores, label_scores = dep_biaffine(dep_proj(torch.cat([dep_h, pos_p, ner_p], dim=-1)))
            head_labels, rel_labels = batch["head_labels"], batch["rel_labels"]
            pred_heads = arc_scores.argmax(dim=-1).cpu()
            vmask = head_labels >= 0
            arc_correct += (pred_heads[vmask] == head_labels[vmask]).sum().item()
            B, S = head_labels.size()
            ph = pred_heads.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, num_deprels)
            pred_rels = label_scores.cpu().gather(2, ph).squeeze(2).argmax(dim=-1)
            both = (pred_heads[vmask] == head_labels[vmask]) & (pred_rels[vmask] == rel_labels[vmask])
            rel_correct += both.sum().item()
            total += vmask.sum().item()
    uas = arc_correct / total
    las = rel_correct / total
    print(f"  DEP UAS: {uas:.4f}, LAS: {las:.4f} (UD EWT test, {len(ud_test)} sentences)")

    # ── CLS ───────────────────────────────────────────────
    print("\n[CLS] Evaluating on SGD+GPT dev...")
    cls_loader = DataLoader(CLSDataset(cls_dev, tokenizer, cls_map), batch_size=BS, collate_fn=collate)
    all_g, all_p = [], []
    with torch.no_grad():
        for batch in tqdm(cls_loader, desc="CLS"):
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            out = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            h = cls_scalar_mix(out.hidden_states)
            pooled = cls_pool(h, mask)
            logits = cls_head(pooled)
            all_p.extend(logits.argmax(dim=-1).cpu().tolist())
            all_g.extend(batch["labels"].tolist())
    from sklearn.metrics import f1_score as clf_f1, classification_report
    cls_mf1 = clf_f1(all_g, all_p, average="macro")
    print(f"  CLS Macro F1: {cls_mf1:.4f} (SGD+GPT dev, {len(cls_dev)} examples)")
    print(f"\n{classification_report(all_g, all_p, target_names=CLS_LABELS, digits=3)}")

    # ── Summary ───────────────────────────────────────────
    results = {
        "pos": {"score": round(pos_acc, 4), "metric": "accuracy", "benchmark": "UD EWT test", "n": len(ud_test)},
        "ner": {"score": round(ner_f1, 4), "metric": "F1", "benchmark": "SpanMarker silver test", "n": len(ner_test)},
        "srl": {"score": round(srl_f1, 4), "metric": "F1", "benchmark": "PropBank EWT test", "n": len(srl_test)},
        "dep": {"score": round(uas, 4), "metric": "UAS", "las": round(las, 4), "benchmark": "UD EWT test", "n": len(ud_test)},
        "cls": {"score": round(cls_mf1, 4), "metric": "macro_f1", "benchmark": "SGD+GPT dev", "n": len(cls_dev)},
    }

    print("\n" + "=" * 60)
    print("v5 BENCHMARK RESULTS")
    print("=" * 60)
    for h in ["pos", "ner", "srl", "dep", "cls"]:
        r = results[h]
        extra = f", LAS={r['las']}" if "las" in r else ""
        print(f"  {h.upper():>4}: {r['score']:.4f} ({r['metric']}{extra}) — {r['benchmark']}")
    print("=" * 60)

    # Save results
    out_path = Path(args.model_dir) / "eval_results_v5.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Compare with metadata
    meta_path = Path(args.model_dir) / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print("\nDelta vs metadata (dev scores):")
        for h in ["pos", "ner", "srl", "dep", "cls"]:
            if h in meta["heads"]:
                dev = meta["heads"][h]["score"]
                test = results[h]["score"]
                delta = test - dev
                print(f"  {h.upper():>4}: {dev:.4f} (dev) → {test:.4f} (test)  ({delta:+.4f})")


if __name__ == "__main__":
    main()
