# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.0", "transformers==5.6.2", "huggingface-hub>=0.22"]
# ///
"""kniv-deberta-nlp-base-en-large — Multi-Task NLP Cascade Demo

Loads the model and runs all 5 heads (POS, NER, DEP, SRL, CLS) on sample text.

Usage:
    uv run python examples/cascade_demo.py
    uv run python examples/cascade_demo.py --model models/kniv-deberta-nlp-base-en-large
    uv run python examples/cascade_demo.py --text "Barack Obama visited Paris last Friday."
"""
from __future__ import annotations
import argparse, json
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


# ── Model Loading ─────────────────────────────────────────────

def load_model(model_dir: str, device: str = "cpu"):
    """Load the full cascade model from a local directory."""
    model_dir = Path(model_dir)
    print(f"Loading model from {model_dir}...")

    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    encoder = AutoModel.from_pretrained("microsoft/deberta-v3-large")
    H = encoder.config.hidden_size
    NL = encoder.config.num_hidden_layers + 1

    state = torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    encoder.load_state_dict(state["deberta"])

    pred_embedding = nn.Embedding(2, H)
    pred_embedding.load_state_dict(state["pred_embedding"])

    pos_scalar_mix = ScalarMix(NL)
    pos_scalar_mix.load_state_dict(state["pos_scalar_mix"])
    pos_head = nn.Linear(H, len(POS_LABELS))
    pos_head.load_state_dict(state["pos_head"])

    ner_scalar_mix = ScalarMix(NL)
    ner_scalar_mix.load_state_dict(state["ner_scalar_mix"])
    ner_lstm = nn.LSTM(H, 256, bidirectional=True, batch_first=True)
    ner_lstm.load_state_dict(state["ner_lstm"])
    ner_proj = nn.Linear(512, H)
    ner_proj.load_state_dict(state["ner_proj"])
    ner_head = nn.Sequential(
        nn.LayerNorm(H + len(POS_LABELS)), nn.Linear(H + len(POS_LABELS), H),
        nn.GELU(), nn.Dropout(0.1), nn.Linear(H, len(NER_LABELS)))
    ner_head.load_state_dict(state["ner_head"])

    srl_classifier = nn.Sequential(
        nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
        nn.Dropout(0.1), nn.Linear(H, len(SRL_TAGS)))
    srl_classifier.load_state_dict(state["classifier"])

    dep_scalar_mix = ScalarMix(NL)
    dep_scalar_mix.load_state_dict(state["dep_scalar_mix"])
    in_dim = H + len(POS_LABELS) + len(NER_LABELS)
    dep_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU())
    dep_proj.load_state_dict(state["dep_proj"])
    dep_biaffine = BiaffineDEPHead(H, num_labels=len(DEPREL_LIST))
    dep_biaffine.load_state_dict(state["dep_biaffine"])

    cls_scalar_mix = ScalarMix(NL)
    cls_scalar_mix.load_state_dict(state["cls_scalar_mix"])
    cls_pool = AttentionPool(H)
    cls_pool.load_state_dict(state["cls_pool"])
    cls_head = nn.Sequential(
        nn.LayerNorm(H), nn.Linear(H, H // 2), nn.GELU(),
        nn.Dropout(0.1), nn.Linear(H // 2, len(CLS_LABELS)))
    cls_head.load_state_dict(state["cls_head"])

    modules = {
        "encoder": encoder, "tokenizer": tokenizer, "pred_embedding": pred_embedding,
        "pos_scalar_mix": pos_scalar_mix, "pos_head": pos_head,
        "ner_scalar_mix": ner_scalar_mix, "ner_lstm": ner_lstm,
        "ner_proj": ner_proj, "ner_head": ner_head,
        "srl_classifier": srl_classifier,
        "dep_scalar_mix": dep_scalar_mix, "dep_proj": dep_proj,
        "dep_biaffine": dep_biaffine,
        "cls_scalar_mix": cls_scalar_mix, "cls_pool": cls_pool,
        "cls_head": cls_head,
    }
    for m in modules.values():
        if isinstance(m, nn.Module):
            m.float().to(device).eval()

    print(f"Model loaded on {device}.")
    return modules


# ── Inference ─────────────────────────────────────────────────

def predict(text: str, modules: dict, device: str = "cpu", prev_text: str | None = None):
    """Run all 5 heads on input text."""
    tok = modules["tokenizer"]
    enc = modules["encoder"]

    # Tokenize for token-level tasks
    words = text.split()
    token_enc = tok(words, is_split_into_words=True, return_tensors="pt",
                    padding=True, truncation=True, max_length=128)
    word_ids = token_enc.word_ids()

    # Map: first subword index for each word
    word_to_token = {}
    prev = None
    for k, wid in enumerate(word_ids):
        if wid is not None and wid != prev:
            word_to_token[wid] = k
        prev = wid

    ids = token_enc["input_ids"].to(device)
    mask = token_enc["attention_mask"].to(device)

    with torch.no_grad():
        # Shared encoder pass
        out = enc(input_ids=ids, attention_mask=mask, output_hidden_states=True)

        # ── POS ──
        pos_h = modules["pos_scalar_mix"](out.hidden_states)
        pos_logits = modules["pos_head"](pos_h)
        pos_tags = []
        for wid, tidx in sorted(word_to_token.items()):
            pos_tags.append(POS_LABELS[pos_logits[0, tidx].argmax().item()])

        # ── NER (BiLSTM + POS cascade + Viterbi) ──
        ner_h = modules["ner_scalar_mix"](out.hidden_states)
        lstm_out, _ = modules["ner_lstm"](ner_h)
        adapted = modules["ner_proj"](lstm_out) + ner_h
        pos_p = torch.softmax(pos_logits, dim=-1)
        ner_logits = modules["ner_head"](torch.cat([adapted, pos_p], dim=-1))

        ner_map = {t: i for i, t in enumerate(NER_LABELS)}
        valid_indices = sorted(word_to_token.values())
        ner_vl = ner_logits[0, valid_indices]
        ner_path = viterbi_decode(ner_vl.cpu(), ner_map)
        ner_tags = [NER_LABELS[ner_path[i]] for i in range(len(valid_indices))]

        # ── DEP (Biaffine + POS/NER cascade) ──
        dep_h = modules["dep_scalar_mix"](out.hidden_states)
        ner_p = torch.softmax(ner_logits, dim=-1)
        dep_input = torch.cat([dep_h, pos_p, ner_p], dim=-1)
        proj_h = modules["dep_proj"](dep_input)
        arc_scores, label_scores = modules["dep_biaffine"](proj_h)

        dep_heads_raw = arc_scores[0].argmax(dim=-1).cpu()
        dep_results = []
        for wid, tidx in sorted(word_to_token.items()):
            head_tidx = dep_heads_raw[tidx].item()
            # Find which word the head token belongs to
            head_wid = None
            for w, t in word_to_token.items():
                if t == head_tidx:
                    head_wid = w
                    break
            # Get relation label
            label_idx = label_scores[0, tidx, head_tidx].argmax().item()
            rel = DEPREL_LIST[label_idx] if label_idx < len(DEPREL_LIST) else "dep"
            dep_results.append({
                "word_idx": wid,
                "head_idx": head_wid if head_wid is not None else wid,
                "relation": rel,
            })

        # ── SRL (predicate embedding, one frame per verb) ──
        srl_map = {t: i for i, t in enumerate(SRL_TAGS)}
        verb_indices = [i for i, p in enumerate(pos_tags) if p == "VERB"]
        srl_frames = []

        for vi in verb_indices[:3]:  # max 3 predicates
            tidx = word_to_token.get(vi)
            if tidx is None:
                continue
            emb = enc.embeddings(ids)
            indicator = torch.zeros_like(ids)
            indicator[0, tidx] = 1
            emb = emb + modules["pred_embedding"](indicator)
            srl_out = enc.encoder(emb, mask)
            srl_logits = modules["srl_classifier"](srl_out.last_hidden_state)
            srl_vl = srl_logits[0, valid_indices]
            srl_path = viterbi_decode(srl_vl.cpu(), srl_map)
            srl_tags = [SRL_TAGS[srl_path[i]] for i in range(len(valid_indices))]

            # Extract frame
            frame = {"predicate": words[vi], "predicate_idx": vi, "args": {}}
            current_arg, current_tokens = None, []
            for i, tag in enumerate(srl_tags):
                if tag.startswith("B-"):
                    if current_arg:
                        frame["args"][current_arg] = " ".join(current_tokens)
                    current_arg = tag[2:]
                    current_tokens = [words[i]] if i < len(words) else []
                elif tag.startswith("I-") and current_arg:
                    if i < len(words):
                        current_tokens.append(words[i])
                else:
                    if current_arg:
                        frame["args"][current_arg] = " ".join(current_tokens)
                    current_arg, current_tokens = None, []
            if current_arg:
                frame["args"][current_arg] = " ".join(current_tokens)
            srl_frames.append(frame)

        # ── CLS (AttentionPool, with optional prev_text) ──
        if prev_text:
            cls_enc = tok(prev_text, text, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)
        else:
            cls_enc = tok(text, return_tensors="pt", padding=True,
                          truncation=True, max_length=128)
        cls_ids = cls_enc["input_ids"].to(device)
        cls_mask = cls_enc["attention_mask"].to(device)
        cls_out = enc(input_ids=cls_ids, attention_mask=cls_mask, output_hidden_states=True)
        cls_h = modules["cls_scalar_mix"](cls_out.hidden_states)
        cls_pooled = modules["cls_pool"](cls_h, cls_mask)
        cls_logits = modules["cls_head"](cls_pooled)
        cls_label = CLS_LABELS[cls_logits.argmax(dim=-1).item()]

    return {
        "words": words,
        "pos": pos_tags,
        "ner": ner_tags,
        "dep": dep_results,
        "srl": srl_frames,
        "cls": cls_label,
    }


# ── Display ───────────────────────────────────────────────────

def display_results(result: dict):
    """Print formatted analysis results."""
    words = result["words"]
    pos = result["pos"]
    ner = result["ner"]
    dep = result["dep"]
    srl = result["srl"]
    cls = result["cls"]

    print(f"\n{'=' * 70}")
    print(f"  Input: {' '.join(words)}")
    print(f"{'=' * 70}")

    # POS
    print(f"\n  POS Tags:")
    print(f"  {'  '.join(f'{w}' for w in words)}")
    print(f"  {'  '.join(f'{p}' for p in pos)}")

    # NER — extract entities
    print(f"\n  Named Entities:")
    entities = []
    current_type, current_tokens = None, []
    for word, tag in zip(words, ner):
        if tag.startswith("B-"):
            if current_type:
                entities.append((current_type, " ".join(current_tokens)))
            current_type = tag[2:]
            current_tokens = [word]
        elif tag.startswith("I-") and current_type:
            current_tokens.append(word)
        else:
            if current_type:
                entities.append((current_type, " ".join(current_tokens)))
            current_type, current_tokens = None, []
    if current_type:
        entities.append((current_type, " ".join(current_tokens)))

    if entities:
        for ent_type, text in entities:
            print(f"    {text:30s}  [{ent_type}]")
    else:
        print(f"    (no entities found)")

    # DEP — tree format
    print(f"\n  Dependency Tree:")
    for d in dep:
        wi = d["word_idx"]
        hi = d["head_idx"]
        rel = d["relation"]
        word = words[wi] if wi < len(words) else "?"
        head = words[hi] if hi < len(words) else "ROOT"
        arrow = f"{word} ──{rel}──> {head}" if wi != hi else f"{word} [root]"
        print(f"    {arrow}")

    # SRL
    print(f"\n  Semantic Role Labeling:")
    if srl:
        for frame in srl:
            print(f"    Predicate: {frame['predicate']}")
            for role, text in frame["args"].items():
                print(f"      {role:12s}: {text}")
    else:
        print(f"    (no verbal predicates found)")

    # CLS
    print(f"\n  Dialog Act: {cls}")
    print(f"{'=' * 70}")


# ── Main ──────────────────────────────────────────────────────

SAMPLE_TEXTS = [
    "Barack Obama visited Paris last Friday to meet with French officials.",
    "The company reported $2.5 billion in revenue, exceeding analysts' expectations.",
    "Can you book me a flight from San Francisco to Tokyo next Tuesday?",
    "Apple was founded by Steve Jobs and Steve Wozniak in a garage in 1976.",
]


def main():
    parser = argparse.ArgumentParser(description="kniv-cascade Multi-Task NLP Demo")
    parser.add_argument("--model", default="models/kniv-deberta-nlp-base-en-large",
                        help="Path to model directory")
    parser.add_argument("--text", default=None, help="Text to analyze (default: sample texts)")
    parser.add_argument("--prev-text", default=None, help="Previous utterance for CLS context")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    modules = load_model(args.model, args.device)

    if args.text:
        texts = [args.text]
    else:
        texts = SAMPLE_TEXTS
        print(f"\nAnalyzing {len(texts)} sample sentences...")

    for text in texts:
        result = predict(text, modules, args.device, prev_text=args.prev_text)
        display_results(result)


if __name__ == "__main__":
    main()
