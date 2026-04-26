"""Benchmark kniv-deberta-nlp-base-en-large against standard public datasets.

Evaluates on:
  - POS: UD English EWT test (standard)
  - NER: CoNLL-2003 test (standard, with entity type mapping)
  - DEP: UD English EWT test (standard)
  - SRL: PropBank EWT test (gold)
  - CLS: DailyDialog test (with label mapping)

Usage:
    .venv/bin/python models/kniv-deberta-nlp-base-en-large/benchmark_standard.py
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from seqeval.metrics import f1_score as seq_f1, classification_report as seq_report
from sklearn.metrics import f1_score as clf_f1, classification_report, accuracy_score

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

# ── OntoNotes → CoNLL-2003 entity mapping ─────────────────────
ONTO_TO_CONLL = {
    "PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC",
    "NORP": "MISC", "FAC": "LOC", "PRODUCT": "MISC", "EVENT": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
    # Numeric types — no CoNLL equivalent, map to O
    "DATE": None, "TIME": None, "PERCENT": None, "MONEY": None,
    "QUANTITY": None, "ORDINAL": None, "CARDINAL": None,
}

def map_ner_tag_to_conll(tag: str) -> str:
    """Map OntoNotes BIO tag to CoNLL-2003 BIO tag."""
    if tag == "O":
        return "O"
    prefix, ent_type = tag.split("-", 1)
    conll_type = ONTO_TO_CONLL.get(ent_type)
    if conll_type is None:
        return "O"
    return f"{prefix}-{conll_type}"

# ── CLS → DailyDialog label mapping ──────────────────────────
# DailyDialog: 1=inform, 2=question, 3=directive, 4=commissive (0=dummy)
# Our model: inform, request, question, confirm, reject, offer, social, status
OUR_CLS_TO_DD = {
    "inform": 1,      # inform → inform
    "request": 3,     # request → directive
    "question": 2,    # question → question
    "confirm": 4,     # confirm → commissive
    "reject": 3,      # reject → directive (closest)
    "offer": 4,       # offer → commissive
    "social": 1,      # social → inform (closest)
    "status": 1,      # status → inform (closest)
}

# ── Modules (same as evaluate_v5.py) ─────────────────────────
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
        scores = self.attn(hidden).squeeze(-1).masked_fill(~mask.bool(), -1e9)
        return (hidden * torch.softmax(scores, dim=-1).unsqueeze(-1)).sum(dim=1)

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

def viterbi_decode(logits, tag2id):
    num_tags, seq_len = logits.size(-1), logits.size(0)
    log_probs = torch.log_softmax(logits, dim=-1)
    allowed = torch.ones(num_tags, num_tags, dtype=torch.bool)
    tag_names = {v: k for k, v in tag2id.items()}
    for j in range(num_tags):
        if tag_names.get(j, "O").startswith("I-"):
            role = tag_names[j][2:]
            b_tag = tag2id.get(f"B-{role}", -1)
            for i in range(num_tags):
                if i != b_tag and i != j: allowed[i][j] = False
    NEG_INF = -1e9
    viterbi = torch.full((seq_len, num_tags), NEG_INF)
    backptr = torch.zeros(seq_len, num_tags, dtype=torch.long)
    viterbi[0] = log_probs[0]
    for t in range(1, seq_len):
        for j in range(num_tags):
            scores = viterbi[t-1].clone(); scores[~allowed[:, j]] = NEG_INF
            best = scores.argmax()
            viterbi[t, j] = scores[best] + log_probs[t, j]; backptr[t, j] = best
    path = [0] * seq_len
    path[-1] = viterbi[-1].argmax().item()
    for t in range(seq_len - 2, -1, -1): path[t] = backptr[t+1, path[t+1]].item()
    return path

def collate(feats):
    return {k: torch.stack([f[k] for f in feats]) for k in feats[0]}

# ── Main ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="models/kniv-deberta-nlp-base-en-large")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}\n")

    # Load vocabs
    with open(DATA_DIR / "label_vocabs.json") as f: vocabs = json.load(f)
    pos_labels = vocabs["pos_labels"]
    ner_labels = vocabs["ner_labels"]
    pos_map = {l: i for i, l in enumerate(pos_labels)}
    ner_map = {l: i for i, l in enumerate(ner_labels)}
    srl_map = {t: i for i, t in enumerate(SRL_TAGS)}
    deprel_map = {r: i for i, r in enumerate(DEPREL_LIST)}
    num_deprels = len(DEPREL_LIST)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large")
    H = encoder.config.hidden_size
    num_layers = encoder.config.num_hidden_layers + 1
    state = torch.load(f"{args.model_dir}/model.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["deberta"])

    pred_embedding = nn.Embedding(2, H); pred_embedding.load_state_dict(state["pred_embedding"])
    pos_scalar_mix = ScalarMix(num_layers); pos_scalar_mix.load_state_dict(state["pos_scalar_mix"])
    pos_head = nn.Linear(H, len(pos_labels)); pos_head.load_state_dict(state["pos_head"])
    ner_scalar_mix = ScalarMix(num_layers); ner_scalar_mix.load_state_dict(state["ner_scalar_mix"])
    ner_lstm = nn.LSTM(H, 256, bidirectional=True, batch_first=True); ner_lstm.load_state_dict(state["ner_lstm"])
    ner_proj = nn.Linear(512, H); ner_proj.load_state_dict(state["ner_proj"])
    ner_head = nn.Sequential(nn.LayerNorm(H + len(pos_labels)), nn.Linear(H + len(pos_labels), H),
                             nn.GELU(), nn.Dropout(0.1), nn.Linear(H, len(ner_labels)))
    ner_head.load_state_dict(state["ner_head"])
    srl_classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(), nn.Dropout(0.1), nn.Linear(H, len(SRL_TAGS)))
    srl_classifier.load_state_dict(state["classifier"])
    dep_scalar_mix = ScalarMix(num_layers); dep_scalar_mix.load_state_dict(state["dep_scalar_mix"])
    in_dim = H + len(pos_labels) + len(ner_labels)
    dep_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU()); dep_proj.load_state_dict(state["dep_proj"])
    dep_biaffine = BiaffineDEPHead(H, num_labels=num_deprels); dep_biaffine.load_state_dict(state["dep_biaffine"])
    cls_scalar_mix = ScalarMix(num_layers); cls_scalar_mix.load_state_dict(state["cls_scalar_mix"])
    cls_pool = AttentionPool(H); cls_pool.load_state_dict(state["cls_pool"])
    cls_head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(H // 2, len(CLS_LABELS)))
    cls_head.load_state_dict(state["cls_head"])

    modules = [encoder, pred_embedding, pos_scalar_mix, pos_head, ner_scalar_mix, ner_lstm, ner_proj, ner_head,
               srl_classifier, dep_scalar_mix, dep_proj, dep_biaffine, cls_scalar_mix, cls_pool, cls_head]
    for m in modules: m.float().to(device).eval()

    results = {}
    BS = args.batch_size

    # ══════════════════════════════════════════════════════
    # 1. POS — UD English EWT test
    # ══════════════════════════════════════════════════════
    print("=" * 70)
    print("BENCHMARK 1: POS — UD English EWT test (standard)")
    print("=" * 70)
    with open(DATA_DIR / "ud_test.json") as f: ud_test = json.load(f)

    class PosDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx):
            ex = self.examples[idx]
            enc = tokenizer(ex["words"], is_split_into_words=True, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            aligned, prev = [], None
            for wid in enc.word_ids():
                if wid is None: aligned.append(-100)
                elif wid != prev: aligned.append(pos_map.get(ex["pos_tags"][wid], 0))
                else: aligned.append(-100)
                prev = wid
            return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(aligned, dtype=torch.long)}

    loader = DataLoader(PosDataset(ud_test), batch_size=BS, collate_fn=collate)
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="POS"):
            out = encoder(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), output_hidden_states=True)
            preds = pos_head(pos_scalar_mix(out.hidden_states)).argmax(-1).cpu()
            mask = batch["labels"] != -100
            correct += (preds[mask] == batch["labels"][mask]).sum().item(); total += mask.sum().item()
    pos_acc = correct / total
    results["pos"] = {"score": round(pos_acc, 4), "metric": "accuracy", "benchmark": "UD English EWT test",
                      "n": len(ud_test), "standard": True}
    print(f"  POS Accuracy: {pos_acc:.4f}\n")

    # ══════════════════════════════════════════════════════
    # 2. NER — CoNLL-2003 test (with entity type mapping)
    # ══════════════════════════════════════════════════════
    print("=" * 70)
    print("BENCHMARK 2: NER — CoNLL-2003 test (standard, mapped from OntoNotes)")
    print("=" * 70)
    from datasets import load_dataset
    conll = load_dataset("eriktks/conll2003")
    conll_test = conll["test"]
    conll_ner_names = conll_test.features["ner_tags"].feature.names
    print(f"  CoNLL-2003 test: {len(conll_test):,} sentences")
    print(f"  CoNLL tags: {conll_ner_names}")
    print(f"  Mapping: {dict((k, v) for k, v in ONTO_TO_CONLL.items() if v)}")

    class CoNLLNERDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
        def __len__(self): return len(self.dataset)
        def __getitem__(self, idx):
            ex = self.dataset[idx]
            words = ex["tokens"]
            gold_tags = [conll_ner_names[t] for t in ex["ner_tags"]]
            enc = tokenizer(words, is_split_into_words=True, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            # We don't need training labels — just gold for comparison
            gold_aligned, prev = [], None
            for wid in enc.word_ids():
                if wid is None: gold_aligned.append("PAD")
                elif wid != prev: gold_aligned.append(gold_tags[wid] if wid < len(gold_tags) else "O")
                else: gold_aligned.append("PAD")
                prev = wid
            return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                    "gold_tags": gold_aligned, "n_tokens": len(words)}

    def collate_conll(feats):
        batch = {k: torch.stack([f[k] for f in feats]) for k in ["input_ids", "attention_mask"]}
        batch["gold_tags"] = [f["gold_tags"] for f in feats]
        batch["n_tokens"] = [f["n_tokens"] for f in feats]
        return batch

    loader = DataLoader(CoNLLNERDataset(conll_test), batch_size=BS, collate_fn=collate_conll)
    all_gold, all_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="NER (CoNLL-2003)"):
            out = encoder(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), output_hidden_states=True)
            ner_h = ner_scalar_mix(out.hidden_states)
            lstm_out, _ = ner_lstm(ner_h); adapted = ner_proj(lstm_out) + ner_h
            pos_p = torch.softmax(pos_head(pos_scalar_mix(out.hidden_states)), dim=-1)
            logits = ner_head(torch.cat([adapted, pos_p], dim=-1)).cpu()

            for j in range(logits.size(0)):
                gold_tags = batch["gold_tags"][j]
                # Get non-PAD positions
                sent_gold, sent_pred = [], []
                for k, gt in enumerate(gold_tags):
                    if gt == "PAD": continue
                    # Our model's prediction → map to CoNLL
                    pred_idx = logits[j, k].argmax().item()
                    pred_onto = ner_labels[pred_idx] if pred_idx < len(ner_labels) else "O"
                    pred_conll = map_ner_tag_to_conll(pred_onto)
                    sent_gold.append(gt)
                    sent_pred.append(pred_conll)
                if sent_gold:
                    all_gold.append(sent_gold)
                    all_pred.append(sent_pred)

    # Use Viterbi on mapped tags? No — Viterbi operates on our 37-tag space.
    # Instead, run Viterbi on our tags first, then map. Let me redo this properly.
    all_gold, all_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="NER (CoNLL-2003, Viterbi)"):
            out = encoder(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), output_hidden_states=True)
            ner_h = ner_scalar_mix(out.hidden_states)
            lstm_out, _ = ner_lstm(ner_h); adapted = ner_proj(lstm_out) + ner_h
            pos_p = torch.softmax(pos_head(pos_scalar_mix(out.hidden_states)), dim=-1)
            logits = ner_head(torch.cat([adapted, pos_p], dim=-1)).cpu()

            for j in range(logits.size(0)):
                gold_tags = batch["gold_tags"][j]
                # Get valid positions (non-PAD)
                valid_indices = [k for k, gt in enumerate(gold_tags) if gt != "PAD"]
                if not valid_indices: continue

                # Viterbi decode on valid positions
                vl = logits[j, valid_indices]
                vd = viterbi_decode(vl, ner_map)

                sent_gold, sent_pred = [], []
                for ki, k in enumerate(valid_indices):
                    gt = gold_tags[k]
                    pred_onto = ner_labels[vd[ki]] if vd[ki] < len(ner_labels) else "O"
                    pred_conll = map_ner_tag_to_conll(pred_onto)
                    sent_gold.append(gt)
                    sent_pred.append(pred_conll)
                all_gold.append(sent_gold)
                all_pred.append(sent_pred)

    ner_f1 = seq_f1(all_gold, all_pred)
    print(f"  NER F1 (CoNLL-2003 mapped): {ner_f1:.4f}")
    print(f"  Mapping: OntoNotes 18 types → CoNLL-2003 4 types (PER, ORG, LOC, MISC)")
    print(f"  Note: numeric entities (DATE, CARDINAL, etc.) have no CoNLL equivalent → mapped to O")
    print(f"\n{seq_report(all_gold, all_pred, digits=3)}")
    results["ner_conll03"] = {"score": round(ner_f1, 4), "metric": "F1", "benchmark": "CoNLL-2003 test",
                              "n": len(conll_test), "standard": True,
                              "note": "OntoNotes→CoNLL type mapping (18→4 types, numeric entities dropped)"}

    # ══════════════════════════════════════════════════════
    # 3. DEP — UD English EWT test
    # ══════════════════════════════════════════════════════
    print("=" * 70)
    print("BENCHMARK 3: DEP — UD English EWT test (standard)")
    print("=" * 70)

    class DepDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx):
            ex = self.examples[idx]
            enc = tokenizer(ex["words"], is_split_into_words=True, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            word_ids = enc.word_ids(); seq_len = enc["input_ids"].size(1)
            head_labels = torch.full((seq_len,), -1, dtype=torch.long)
            rel_labels = torch.full((seq_len,), -1, dtype=torch.long)
            w2t = {}; prev = None
            for k, wid in enumerate(word_ids):
                if wid is not None and wid != prev: w2t[wid] = k
                prev = wid
            prev = None
            for k, wid in enumerate(word_ids):
                if wid is not None and wid != prev and wid < len(ex["heads"]):
                    hw = ex["heads"][wid]
                    head_labels[k] = k if hw == -1 else w2t.get(hw, -1)
                    rel_labels[k] = deprel_map.get(ex["deprels"][wid], 0)
                prev = wid
            return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                    "head_labels": head_labels, "rel_labels": rel_labels}

    loader = DataLoader(DepDataset(ud_test), batch_size=BS, collate_fn=collate)
    arc_correct, rel_correct, total = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="DEP"):
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            out = encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            dep_h = dep_scalar_mix(out.hidden_states)
            pos_p = torch.softmax(pos_head(pos_scalar_mix(out.hidden_states)), -1)
            ner_h = ner_scalar_mix(out.hidden_states)
            lo, _ = ner_lstm(ner_h); ad = ner_proj(lo) + ner_h
            ner_p = torch.softmax(ner_head(torch.cat([ad, pos_p], -1)), -1)
            arc_s, lab_s = dep_biaffine(dep_proj(torch.cat([dep_h, pos_p, ner_p], -1)))
            hl, rl = batch["head_labels"], batch["rel_labels"]
            ph = arc_s.argmax(-1).cpu(); vm = hl >= 0
            arc_correct += (ph[vm] == hl[vm]).sum().item()
            B, S = hl.size()
            phx = ph.clamp(0).unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, num_deprels)
            pr = lab_s.cpu().gather(2, phx).squeeze(2).argmax(-1)
            rel_correct += ((ph[vm] == hl[vm]) & (pr[vm] == rl[vm])).sum().item()
            total += vm.sum().item()
    uas, las = arc_correct / total, rel_correct / total
    results["dep"] = {"score": round(uas, 4), "las": round(las, 4), "metric": "UAS",
                      "benchmark": "UD English EWT test", "n": len(ud_test), "standard": True}
    print(f"  DEP UAS: {uas:.4f}, LAS: {las:.4f}\n")

    # ══════════════════════════════════════════════════════
    # 4. SRL — PropBank EWT test (gold)
    # ══════════════════════════════════════════════════════
    print("=" * 70)
    print("BENCHMARK 4: SRL — PropBank EWT test (gold)")
    print("=" * 70)
    with open(DATA_DIR / "srl_test.json") as f: srl_test = json.load(f)

    class SrlDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx):
            ex = self.examples[idx]
            enc = tokenizer(ex["words"], is_split_into_words=True, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            pred_tok, aligned, prev = 0, [], None
            for k, wid in enumerate(enc.word_ids()):
                if wid is None: aligned.append(-100)
                elif wid != prev:
                    aligned.append(srl_map.get(ex["srl_tags"][wid] if wid < len(ex["srl_tags"]) else "O", 0))
                    if wid == ex["predicate_idx"]: pred_tok = k
                else: aligned.append(-100)
                prev = wid
            return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(aligned, dtype=torch.long), "predicate_idx": torch.tensor(pred_tok, dtype=torch.long)}

    loader = DataLoader(SrlDataset(srl_test), batch_size=BS, collate_fn=collate)
    all_g, all_p = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="SRL"):
            B, S = batch["input_ids"].size()
            emb = encoder.embeddings(batch["input_ids"].to(device))
            ind = torch.zeros(B, S, dtype=torch.long, device=device)
            ind[torch.arange(B, device=device), batch["predicate_idx"].to(device)] = 1
            emb = emb + pred_embedding(ind)
            h = encoder.encoder(emb, batch["attention_mask"].to(device)).last_hidden_state
            logits = srl_classifier(h).cpu(); labs = batch["labels"]
            for j in range(labs.size(0)):
                vi = (labs[j] != -100).nonzero(as_tuple=True)[0]
                if not len(vi): continue
                vd = viterbi_decode(logits[j, vi], srl_map)
                gl = [SRL_TAGS[labs[j, k]] for k in vi.tolist()]
                pl = [SRL_TAGS[vd[ki]] if vd[ki] < len(SRL_TAGS) else "O" for ki in range(len(vi))]
                all_g.append([g if g != "V" else "O" for g in gl])
                all_p.append([p if p != "V" else "O" for p in pl])
    srl_f1 = seq_f1(all_g, all_p)
    results["srl"] = {"score": round(srl_f1, 4), "metric": "F1", "benchmark": "PropBank EWT test",
                      "n": len(srl_test), "standard": True}
    print(f"  SRL F1: {srl_f1:.4f}\n")

    # ══════════════════════════════════════════════════════
    # 5. CLS — DailyDialog test (with label mapping)
    # ══════════════════════════════════════════════════════
    print("=" * 70)
    print("BENCHMARK 5: CLS — DailyDialog test (mapped from 8→4 labels)")
    print("=" * 70)
    dd = load_dataset("daily_dialog")
    dd_test = dd["test"]
    dd_label_names = {0: "dummy", 1: "inform", 2: "question", 3: "directive", 4: "commissive"}
    print(f"  DailyDialog test: {len(dd_test):,} conversations")

    # Flatten to utterances
    dd_examples = []
    for conv in dd_test:
        utterances = conv["dialog"]
        acts = conv["act"]
        for i, (utt, act) in enumerate(zip(utterances, acts)):
            if act == 0: continue  # skip dummy
            prev_text = utterances[i - 1] if i > 0 else None
            dd_examples.append({"text": utt, "prev_text": prev_text, "gold_act": act})
    print(f"  Flattened: {len(dd_examples):,} utterances")

    class DDDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx):
            ex = self.examples[idx]
            if ex["prev_text"]:
                enc = tokenizer(ex["prev_text"], ex["text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            else:
                enc = tokenizer(ex["text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                    "gold_act": ex["gold_act"]}

    def collate_dd(feats):
        batch = {k: torch.stack([f[k] for f in feats]) for k in ["input_ids", "attention_mask"]}
        batch["gold_act"] = [f["gold_act"] for f in feats]
        return batch

    loader = DataLoader(DDDataset(dd_examples), batch_size=BS, collate_fn=collate_dd)
    dd_gold, dd_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="CLS (DailyDialog)"):
            out = encoder(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), output_hidden_states=True)
            h = cls_scalar_mix(out.hidden_states)
            logits = cls_head(cls_pool(h, batch["attention_mask"].to(device)))
            pred_cls = logits.argmax(-1).cpu().tolist()
            for j, pc in enumerate(pred_cls):
                our_label = CLS_LABELS[pc]
                mapped = OUR_CLS_TO_DD.get(our_label, 1)
                dd_pred.append(mapped)
                dd_gold.append(batch["gold_act"][j])

    dd_acc = accuracy_score(dd_gold, dd_pred)
    dd_mf1 = clf_f1(dd_gold, dd_pred, average="macro")
    dd_names = ["inform", "question", "directive", "commissive"]
    print(f"  DailyDialog Accuracy: {dd_acc:.4f}")
    print(f"  DailyDialog Macro F1: {dd_mf1:.4f}")
    print(f"  Mapping: our 8 labels → DailyDialog 4 labels")
    print(f"\n{classification_report(dd_gold, dd_pred, target_names=dd_names, digits=3)}")
    results["cls_dailydialog"] = {"score": round(dd_acc, 4), "f1": round(dd_mf1, 4), "metric": "accuracy",
                                  "benchmark": "DailyDialog test", "n": len(dd_examples), "standard": True,
                                  "note": "8→4 label mapping"}

    # Also eval on our internal CLS dev
    with open(DATA_DIR / "cls_sgd_mwoz_train.json") as f: cls_data = json.load(f)
    random.seed(42); random.shuffle(cls_data); cls_dev = cls_data[:4000]
    cls_map_local = {l: i for i, l in enumerate(CLS_LABELS)}

    class ClsDataset(Dataset):
        def __init__(self, examples):
            self.examples = examples
        def __len__(self): return len(self.examples)
        def __getitem__(self, idx):
            ex = self.examples[idx]
            if ex.get("prev_text"):
                enc = tokenizer(ex["prev_text"], ex["text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            else:
                enc = tokenizer(ex["text"], max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            return {"input_ids": enc["input_ids"].squeeze(0), "attention_mask": enc["attention_mask"].squeeze(0),
                    "labels": torch.tensor(cls_map_local[ex["cls_label"]], dtype=torch.long)}

    loader = DataLoader(ClsDataset(cls_dev), batch_size=BS, collate_fn=collate)
    ag, ap = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="CLS (internal)"):
            out = encoder(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device), output_hidden_states=True)
            logits = cls_head(cls_pool(cls_scalar_mix(out.hidden_states), batch["attention_mask"].to(device)))
            ap.extend(logits.argmax(-1).cpu().tolist()); ag.extend(batch["labels"].tolist())
    cls_internal = clf_f1(ag, ap, average="macro")
    results["cls_internal"] = {"score": round(cls_internal, 4), "metric": "macro_f1",
                               "benchmark": "SGD+GPT dev (internal)", "n": len(cls_dev), "standard": False}
    print(f"\n  Internal CLS Macro F1: {cls_internal:.4f} (SGD+GPT dev)")

    # ══════════════════════════════════════════════════════
    # Summary
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("STANDARD BENCHMARK RESULTS — kniv-deberta-nlp-base-en-large v5")
    print("=" * 70)
    print(f"\n  {'Head':<6} {'Score':>7}  {'Metric':<12} {'Benchmark'}")
    print(f"  {'─'*6} {'─'*7}  {'─'*12} {'─'*40}")
    for key in ["pos", "ner_conll03", "dep", "srl", "cls_dailydialog", "cls_internal"]:
        r = results[key]
        extra = f" (LAS={r['las']})" if "las" in r else ""
        std = " *" if r.get("standard") else ""
        name = key.replace("_conll03", " (CoNLL-03)").replace("_dailydialog", " (DailyDlg)").replace("_internal", " (internal)").upper()
        print(f"  {name:<14} {r['score']:>7.4f}  {r['metric']:<12} {r['benchmark']}{std}")
    print(f"\n  * = standard public benchmark")
    print("=" * 70)

    out_path = Path(args.model_dir) / "benchmark_results_v5.json"
    with open(out_path, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
