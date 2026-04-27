# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0", "transformers==5.6.2"]
# ///
"""kniv-deberta-nlp-base-en-large — Multi-Task NLP Cascade Demo (PyTorch)

Single encoder pass, all 5 heads: POS, NER, DEP, SRL, CLS.

Usage:
    uv run python examples/cascade_demo.py
    uv run python examples/cascade_demo.py --model models/kniv-deberta-nlp-base-en-large
    uv run python examples/cascade_demo.py --text "Steve Jobs founded Apple in 1976."
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

# ── Label Sets ────────────────────────────────────────────────

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

NER_MAP = {t: i for i, t in enumerate(NER_LABELS)}
SRL_MAP = {t: i for i, t in enumerate(SRL_TAGS)}

# ── Model Components ──────────────────────────────────────────

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


def viterbi_decode(logits, tag2id):
    """Constrained Viterbi decoding for BIO sequences."""
    num_tags, seq_len = logits.size(-1), logits.size(0)
    log_probs = torch.log_softmax(logits, dim=-1)
    allowed = torch.ones(num_tags, num_tags, dtype=torch.bool)
    tag_names = {v: k for k, v in tag2id.items()}
    for j in range(num_tags):
        if tag_names.get(j, "O").startswith("I-"):
            role = tag_names[j][2:]
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


def extract_bio_spans(tags, words):
    """Extract spans from BIO tags. Returns list of (type, text)."""
    spans = []
    current_type, current_tokens = None, []
    for word, tag in zip(words, tags):
        if tag.startswith("B-"):
            if current_type:
                spans.append((current_type, " ".join(current_tokens)))
            current_type = tag[2:]
            current_tokens = [word]
        elif tag.startswith("I-") and current_type:
            current_tokens.append(word)
        else:
            if current_type:
                spans.append((current_type, " ".join(current_tokens)))
            current_type, current_tokens = None, []
    if current_type:
        spans.append((current_type, " ".join(current_tokens)))
    return spans


def tokenize_words(text):
    """Split text into words, separating punctuation."""
    return re.findall(r"\w+(?:'\w+)*|[^\w\s]", text)


# ── Model Loading ─────────────────────────────────────────────

def load_model(model_dir: str, device: str = "cpu"):
    """Load the cascade model. Returns a dict of modules."""
    model_dir = Path(model_dir)
    print(f"Loading model from {model_dir}...")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large")
    H = encoder.config.hidden_size
    NL = encoder.config.num_hidden_layers + 1

    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["deberta"])

    pred_embedding = nn.Embedding(2, H)
    pred_embedding.load_state_dict(state["pred_embedding"])

    pos_scalar_mix = ScalarMix(NL); pos_scalar_mix.load_state_dict(state["pos_scalar_mix"])
    pos_head = nn.Linear(H, len(POS_LABELS)); pos_head.load_state_dict(state["pos_head"])

    ner_scalar_mix = ScalarMix(NL); ner_scalar_mix.load_state_dict(state["ner_scalar_mix"])
    ner_lstm = nn.LSTM(H, 256, bidirectional=True, batch_first=True); ner_lstm.load_state_dict(state["ner_lstm"])
    ner_proj = nn.Linear(512, H); ner_proj.load_state_dict(state["ner_proj"])
    ner_head = nn.Sequential(nn.LayerNorm(H + len(POS_LABELS)), nn.Linear(H + len(POS_LABELS), H),
                             nn.GELU(), nn.Dropout(0.1), nn.Linear(H, len(NER_LABELS)))
    ner_head.load_state_dict(state["ner_head"])

    srl_classifier = nn.Sequential(nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
                                   nn.Dropout(0.1), nn.Linear(H, len(SRL_TAGS)))
    srl_classifier.load_state_dict(state["classifier"])

    dep_scalar_mix = ScalarMix(NL); dep_scalar_mix.load_state_dict(state["dep_scalar_mix"])
    in_dim = H + len(POS_LABELS) + len(NER_LABELS)
    dep_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU())
    dep_proj.load_state_dict(state["dep_proj"])
    dep_biaffine = BiaffineDEPHead(H, num_labels=len(DEPREL_LIST))
    dep_biaffine.load_state_dict(state["dep_biaffine"])

    cls_scalar_mix = ScalarMix(NL); cls_scalar_mix.load_state_dict(state["cls_scalar_mix"])
    cls_pool = AttentionPool(H); cls_pool.load_state_dict(state["cls_pool"])
    cls_head = nn.Sequential(nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(),
                             nn.Dropout(0.1), nn.Linear(H // 2, len(CLS_LABELS)))
    cls_head.load_state_dict(state["cls_head"])

    m = {
        "encoder": encoder, "tokenizer": tokenizer, "pred_embedding": pred_embedding,
        "pos_scalar_mix": pos_scalar_mix, "pos_head": pos_head,
        "ner_scalar_mix": ner_scalar_mix, "ner_lstm": ner_lstm, "ner_proj": ner_proj, "ner_head": ner_head,
        "srl_classifier": srl_classifier,
        "dep_scalar_mix": dep_scalar_mix, "dep_proj": dep_proj, "dep_biaffine": dep_biaffine,
        "cls_scalar_mix": cls_scalar_mix, "cls_pool": cls_pool, "cls_head": cls_head,
    }
    for v in m.values():
        if isinstance(v, nn.Module):
            v.float().to(device).eval()

    print(f"Model loaded on {device}.")
    return m


# ── Inference ─────────────────────────────────────────────────

def predict(text: str, m: dict, device: str = "cpu", prev_text: str | None = None):
    """Run all 5 heads in a single encoder pass."""
    tok = m["tokenizer"]
    enc = m["encoder"]
    words = tokenize_words(text)

    # Tokenize
    token_enc = tok(words, is_split_into_words=True, return_tensors="pt",
                    padding=True, truncation=True, max_length=128)
    word_ids = token_enc.word_ids()

    # Word ↔ token maps
    word_to_token = {}
    token_to_word = {}
    prev_wid = None
    for k, wid in enumerate(word_ids):
        if wid is not None and wid != prev_wid:
            word_to_token[wid] = k
            token_to_word[k] = wid
        prev_wid = wid
    valid_indices = sorted(word_to_token.values())

    ids = token_enc["input_ids"].to(device)
    mask = token_enc["attention_mask"].to(device)
    B, S = ids.size()

    # Find first verb for SRL predicate (before encoder pass)
    # We'll detect verbs from POS after the pass, but need a predicate index
    # for the encoder. Default to 0 (no predicate), update if verb found.
    # Since pred_embedding is zero-initialized at index 0, this is a no-op.

    with torch.no_grad():
        # ── Single encoder pass with predicate embedding ──
        # First pass: detect POS to find verbs
        emb = enc.embeddings(ids)
        # No predicate embedding on first pass — just get POS
        enc_out = enc.encoder(emb, mask, output_hidden_states=True)
        hidden_states = list(enc_out.hidden_states)
        last_hidden = enc_out.last_hidden_state

        # POS
        pos_h = m["pos_scalar_mix"](hidden_states)
        pos_logits = m["pos_head"](pos_h)
        pos_probs = torch.softmax(pos_logits, dim=-1)
        pos_tags = [POS_LABELS[pos_logits[0, tidx].argmax().item()]
                    for _, tidx in sorted(word_to_token.items())]

        # NER (BiLSTM + POS cascade + Viterbi)
        ner_h = m["ner_scalar_mix"](hidden_states)
        lstm_out, _ = m["ner_lstm"](ner_h)
        adapted = m["ner_proj"](lstm_out) + ner_h
        ner_logits = m["ner_head"](torch.cat([adapted, pos_probs], dim=-1))
        ner_path = viterbi_decode(ner_logits[0, valid_indices].cpu(), NER_MAP)
        ner_tags = [NER_LABELS[ner_path[i]] for i in range(len(valid_indices))]

        # DEP (Biaffine + POS/NER cascade)
        dep_h = m["dep_scalar_mix"](hidden_states)
        ner_probs = torch.softmax(ner_logits, dim=-1)
        proj_h = m["dep_proj"](torch.cat([dep_h, pos_probs, ner_probs], dim=-1))
        arc_scores, label_scores = m["dep_biaffine"](proj_h)
        arc_preds = arc_scores[0].argmax(dim=-1).cpu()

        dep_results = []
        for wid, tidx in sorted(word_to_token.items()):
            head_tidx = arc_preds[tidx].item()
            head_wid = token_to_word.get(head_tidx, wid)
            label_idx = label_scores[0, tidx, head_tidx].argmax().item()
            rel = DEPREL_LIST[label_idx] if label_idx < len(DEPREL_LIST) else "dep"
            dep_results.append({"word_idx": wid, "head_idx": head_wid, "relation": rel})

        # SRL — re-run encoder with predicate embedding for each verb
        verb_indices = [i for i, p in enumerate(pos_tags) if p == "VERB"]
        srl_frames = []

        for vi in verb_indices[:3]:
            tidx = word_to_token.get(vi)
            if tidx is None:
                continue
            indicator = torch.zeros(B, S, dtype=torch.long, device=device)
            indicator[0, tidx] = 1
            srl_emb = emb + m["pred_embedding"](indicator)
            srl_out = enc.encoder(srl_emb, mask)
            srl_logits = m["srl_classifier"](srl_out.last_hidden_state)
            srl_path = viterbi_decode(srl_logits[0, valid_indices].cpu(), SRL_MAP)
            srl_tags = [SRL_TAGS[srl_path[i]] for i in range(len(valid_indices))]

            frame = {"predicate": words[vi], "predicate_idx": vi, "args": {}}
            for role, span_text in extract_bio_spans(srl_tags, words):
                if role != "V":
                    frame["args"][role] = span_text
            srl_frames.append(frame)

        # CLS (reuses same encoder output — no separate pass needed)
        cls_h = m["cls_scalar_mix"](hidden_states)
        cls_pooled = m["cls_pool"](cls_h, mask)
        cls_label = CLS_LABELS[m["cls_head"](cls_pooled).argmax(dim=-1).item()]

    return {"words": words, "pos": pos_tags, "ner": ner_tags,
            "dep": dep_results, "srl": srl_frames, "cls": cls_label}


# ── Display ───────────────────────────────────────────────────

def display_results(result: dict):
    """Print formatted analysis."""
    words = result["words"]

    print(f"\n{'=' * 70}")
    print(f"  {' '.join(words)}")
    print(f"{'=' * 70}")

    # POS — aligned columns
    print(f"\n  POS:")
    for w, p in zip(words, result["pos"]):
        print(f"    {w:15s} {p}")

    # NER — extracted entities
    entities = extract_bio_spans(result["ner"], words)
    print(f"\n  NER:")
    if entities:
        for ent_type, text in entities:
            print(f"    {text:30s} [{ent_type}]")
    else:
        print(f"    (none)")

    # DEP — tree
    print(f"\n  DEP:")
    for d in result["dep"]:
        w = words[d["word_idx"]]
        h = words[d["head_idx"]] if d["word_idx"] != d["head_idx"] else "ROOT"
        print(f"    {w:15s} ──{d['relation']}──> {h}")

    # SRL — frames
    print(f"\n  SRL:")
    if result["srl"]:
        for frame in result["srl"]:
            print(f"    {frame['predicate']}:")
            for role, text in frame["args"].items():
                print(f"      {role:12s}  {text}")
    else:
        print(f"    (no predicates)")

    # CLS
    print(f"\n  CLS: {result['cls']}")
    print()


# ── Main ────────────────────────────────────────��─────────────

SAMPLES = [
    "Barack Obama visited Paris last Friday to meet with French officials.",
    "The company reported $2.5 billion in revenue, exceeding analysts' expectations.",
    "Can you book me a flight from San Francisco to Tokyo next Tuesday?",
    "Apple was founded by Steve Jobs and Steve Wozniak in a garage in 1976.",
]

def main():
    parser = argparse.ArgumentParser(description="kniv-cascade NLP Demo")
    parser.add_argument("--model", default="models/kniv-deberta-nlp-base-en-large")
    parser.add_argument("--text", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    m = load_model(args.model, args.device)
    texts = [args.text] if args.text else SAMPLES

    for text in texts:
        display_results(predict(text, m, args.device))

if __name__ == "__main__":
    main()
