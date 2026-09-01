"""Benchmark a kniv distilled student model against standard public datasets.

Same protocol as the teacher's benchmark_standard.py:
  - POS: UD English EWT test (standard)
  - NER: CoNLL-2003 test (standard, with 18→4 entity-type mapping from OntoNotes)
  - DEP: UD English EWT test (UAS / LAS)
  - SRL: PropBank EWT test (gold)
  - CLS: DailyDialog test (with 8→4 label mapping) + internal SGD+GPT dev

By default runs against the PyTorch checkpoint. Pass ``--backend all`` to also
benchmark the ONNX FP32 and ONNX INT8 exports (if they exist) and produce a
side-by-side quality comparison — useful for measuring quantization quality drop.

Usage:
    # PyTorch only (default)
    python models/student_benchmark_standard.py \
        --model-dir models/kniv-deberta-nlp-base-en-xsmall

    # All backends (PT + ONNX FP32 + ONNX INT8)
    python models/student_benchmark_standard.py \
        --model-dir models/kniv-deberta-nlp-base-en-xsmall \
        --backend all

    # Specific backend only
    python models/student_benchmark_standard.py --model-dir ... --backend onnx-int8

Results are written to ``<model-dir>/benchmark_results.json`` keyed by backend.
With ``--update-metadata`` the PT-backend scores are also written into
``<model-dir>/metadata.json``'s per-head ``score`` fields.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from seqeval.metrics import f1_score as seq_f1, classification_report as seq_report
from sklearn.metrics import f1_score as clf_f1, classification_report, accuracy_score

# Local import (this file lives next to student_loader.py)
sys.path.insert(0, str(Path(__file__).parent))
from student_loader import (
    load_student, viterbi_decode,
    POS_LABELS, NER_LABELS, SRL_TAGS, CLS_LABELS, DEPREL_LIST,
    N_POS, N_NER, N_SRL, N_DEP, N_CLS,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "prepared" / "kniv-deberta-cascade"
pos_map = {t: i for i, t in enumerate(POS_LABELS)}
ner_map = {t: i for i, t in enumerate(NER_LABELS)}
srl_map = {t: i for i, t in enumerate(SRL_TAGS)}
deprel_map = {r: i for i, r in enumerate(DEPREL_LIST)}
cls_map = {l: i for i, l in enumerate(CLS_LABELS)}

# OntoNotes (18 types) → CoNLL-2003 (4 types) mapping
ONTO_TO_CONLL = {
    "PERSON": "PER", "ORG": "ORG", "GPE": "LOC", "LOC": "LOC",
    "NORP": "MISC", "FAC": "LOC", "PRODUCT": "MISC", "EVENT": "MISC",
    "WORK_OF_ART": "MISC", "LAW": "MISC", "LANGUAGE": "MISC",
    "DATE": None, "TIME": None, "PERCENT": None, "MONEY": None,
    "QUANTITY": None, "ORDINAL": None, "CARDINAL": None,
}

OUR_CLS_TO_DD = {
    "inform": 1, "request": 3, "question": 2, "confirm": 4,
    "reject": 3, "offer": 4, "social": 1, "status": 1,
}


def map_ner_tag_to_conll(tag: str) -> str:
    if tag == "O":
        return "O"
    prefix, ent_type = tag.split("-", 1)
    conll_type = ONTO_TO_CONLL.get(ent_type)
    if conll_type is None:
        return "O"
    return f"{prefix}-{conll_type}"


def collate_basic(feats):
    return {k: torch.stack([f[k] for f in feats]) for k in feats[0]}


# ── Backend abstraction ─────────────────────────────────────────────
# Each backend exposes the same forward signature as StudentCascadeV5:
#   forward(input_ids, attention_mask, predicate_idx)
#       -> (pos_logits, ner_logits, arc_scores, label_scores, srl_logits, cls_logits)
# All outputs are torch tensors on CPU (so downstream eval logic is shared).

class PyTorchBackend:
    name = "pytorch"

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def forward(self, input_ids, attention_mask, predicate_idx):
        with torch.no_grad():
            outs = self.model(
                input_ids.to(self.device),
                attention_mask.to(self.device),
                predicate_idx.to(self.device),
            )
        return tuple(o.cpu() for o in outs)


class ONNXBackend:
    def __init__(self, session, name):
        self.session = session
        self.name = name

    def forward(self, input_ids, attention_mask, predicate_idx):
        outs = self.session.run(None, {
            "input_ids":      input_ids.cpu().numpy().astype(np.int64),
            "attention_mask": attention_mask.cpu().numpy().astype(np.int64),
            "predicate_idx":  predicate_idx.cpu().numpy().astype(np.int64),
        })
        return tuple(torch.from_numpy(o) for o in outs)


# ── Dataset adapters ────────────────────────────────────────────────
class PosDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(ex["words"], is_split_into_words=True, max_length=128,
                             padding="max_length", truncation=True, return_tensors="pt")
        aligned, prev = [], None
        for wid in enc.word_ids():
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                aligned.append(pos_map.get(ex["pos_tags"][wid], 0)
                               if wid < len(ex["pos_tags"]) else 0)
            else:
                aligned.append(-100)
            prev = wid
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned, dtype=torch.long),
        }


class CoNLLNERDataset(Dataset):
    def __init__(self, dataset, conll_tag_names, tokenizer):
        self.dataset = dataset
        self.conll_tag_names = conll_tag_names
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        words = ex["tokens"]
        gold_tags = [self.conll_tag_names[t] for t in ex["ner_tags"]]
        enc = self.tokenizer(words, is_split_into_words=True, max_length=128,
                             padding="max_length", truncation=True, return_tensors="pt")
        gold_aligned, prev = [], None
        for wid in enc.word_ids():
            if wid is None:
                gold_aligned.append("PAD")
            elif wid != prev:
                gold_aligned.append(gold_tags[wid] if wid < len(gold_tags) else "O")
            else:
                gold_aligned.append("PAD")
            prev = wid
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "gold_tags": gold_aligned,
        }


def collate_conll(feats):
    batch = {k: torch.stack([f[k] for f in feats]) for k in ["input_ids", "attention_mask"]}
    batch["gold_tags"] = [f["gold_tags"] for f in feats]
    return batch


class DepDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(ex["words"], is_split_into_words=True, max_length=128,
                             padding="max_length", truncation=True, return_tensors="pt")
        word_ids = enc.word_ids()
        seq_len = enc["input_ids"].size(1)
        head_labels = torch.full((seq_len,), -1, dtype=torch.long)
        rel_labels = torch.full((seq_len,), -1, dtype=torch.long)
        w2t, prev = {}, None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev:
                w2t[wid] = k
            prev = wid
        prev = None
        for k, wid in enumerate(word_ids):
            if wid is not None and wid != prev and wid < len(ex["heads"]):
                hw = ex["heads"][wid]
                head_labels[k] = k if hw == -1 else w2t.get(hw, -1)
                rel_labels[k] = deprel_map.get(ex["deprels"][wid], 0)
            prev = wid
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "head_labels": head_labels,
            "rel_labels": rel_labels,
        }


class SrlDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tokenizer(ex["words"], is_split_into_words=True, max_length=128,
                             padding="max_length", truncation=True, return_tensors="pt")
        pred_tok, aligned, prev = 0, [], None
        for k, wid in enumerate(enc.word_ids()):
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                aligned.append(srl_map.get(ex["srl_tags"][wid] if wid < len(ex["srl_tags"]) else "O", 0))
                if wid == ex["predicate_idx"]:
                    pred_tok = k
            else:
                aligned.append(-100)
            prev = wid
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned, dtype=torch.long),
            "predicate_idx": torch.tensor(pred_tok, dtype=torch.long),
        }


class DailyDialogDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if ex["prev_text"]:
            enc = self.tokenizer(ex["prev_text"], ex["text"], max_length=128,
                                 padding="max_length", truncation=True, return_tensors="pt")
        else:
            enc = self.tokenizer(ex["text"], max_length=128,
                                 padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "gold_act": ex["gold_act"],
        }


def collate_dd(feats):
    batch = {k: torch.stack([f[k] for f in feats]) for k in ["input_ids", "attention_mask"]}
    batch["gold_act"] = [f["gold_act"] for f in feats]
    return batch


class InternalCLSDataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        if ex.get("prev_text"):
            enc = self.tokenizer(ex["prev_text"], ex["text"], max_length=128,
                                 padding="max_length", truncation=True, return_tensors="pt")
        else:
            enc = self.tokenizer(ex["text"], max_length=128,
                                 padding="max_length", truncation=True, return_tensors="pt")
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(cls_map[ex["cls_label"]], dtype=torch.long),
        }


# ── Per-backend benchmark ───────────────────────────────────────────
def run_benchmarks(backend, tokenizer, conll_test, conll_tag_names, ud_test, srl_test,
                   dd_examples, internal_cls_dev, batch_size: int):
    """Run all 5 benchmarks against a single backend. Returns results dict."""
    results = {}
    BS = batch_size

    # ── 1. POS — UD English EWT test ──
    print("=" * 70)
    print(f"[{backend.name}] BENCHMARK 1: POS — UD English EWT test (standard)")
    print("=" * 70)
    loader = DataLoader(PosDataset(ud_test, tokenizer), batch_size=BS,
                        collate_fn=collate_basic)
    correct, total = 0, 0
    for batch in tqdm(loader, desc=f"[{backend.name}] POS"):
        ids = batch["input_ids"]; mask = batch["attention_mask"]
        B = ids.size(0)
        zero_pred = torch.zeros(B, dtype=torch.long)
        pos_logits, *_ = backend.forward(ids, mask, zero_pred)
        preds = pos_logits.argmax(-1)
        valid = batch["labels"] != -100
        correct += (preds[valid] == batch["labels"][valid]).sum().item()
        total += valid.sum().item()
    pos_acc = correct / total
    results["pos"] = {
        "score": round(pos_acc, 4), "metric": "accuracy",
        "benchmark": "UD English EWT test", "n": len(ud_test), "standard": True,
    }
    print(f"  POS Accuracy: {pos_acc:.4f}\n")

    # ── 2. NER — CoNLL-2003 (with mapping + Viterbi) ──
    print("=" * 70)
    print(f"[{backend.name}] BENCHMARK 2: NER — CoNLL-2003 test (mapped from OntoNotes)")
    print("=" * 70)
    loader = DataLoader(CoNLLNERDataset(conll_test, conll_tag_names, tokenizer),
                        batch_size=BS, collate_fn=collate_conll)
    all_gold, all_pred = [], []
    for batch in tqdm(loader, desc=f"[{backend.name}] NER"):
        ids = batch["input_ids"]; mask = batch["attention_mask"]
        B = ids.size(0)
        zero_pred = torch.zeros(B, dtype=torch.long)
        _, ner_logits, *_ = backend.forward(ids, mask, zero_pred)
        for j in range(B):
            gold_tags = batch["gold_tags"][j]
            valid_indices = [k for k, gt in enumerate(gold_tags) if gt != "PAD"]
            if not valid_indices:
                continue
            vl = ner_logits[j, valid_indices]
            vd = viterbi_decode(vl, NER_LABELS)
            sent_g, sent_p = [], []
            for ki, k in enumerate(valid_indices):
                pred_onto = NER_LABELS[vd[ki]] if vd[ki] < len(NER_LABELS) else "O"
                sent_g.append(gold_tags[k])
                sent_p.append(map_ner_tag_to_conll(pred_onto))
            all_gold.append(sent_g)
            all_pred.append(sent_p)
    ner_f1 = seq_f1(all_gold, all_pred)
    print(f"  NER F1 (CoNLL-03 mapped): {ner_f1:.4f}\n")
    results["ner_conll03"] = {
        "score": round(ner_f1, 4), "metric": "F1",
        "benchmark": "CoNLL-2003 test", "n": len(conll_test), "standard": True,
        "note": "OntoNotes→CoNLL type mapping (18→4 types, numeric entities dropped)",
    }

    # ── 3. DEP — UD English EWT test ──
    print("=" * 70)
    print(f"[{backend.name}] BENCHMARK 3: DEP — UD English EWT test (standard)")
    print("=" * 70)
    loader = DataLoader(DepDataset(ud_test, tokenizer), batch_size=BS,
                        collate_fn=collate_basic)
    arc_correct, rel_correct, total = 0, 0, 0
    for batch in tqdm(loader, desc=f"[{backend.name}] DEP"):
        ids = batch["input_ids"]; mask = batch["attention_mask"]
        B = ids.size(0)
        zero_pred = torch.zeros(B, dtype=torch.long)
        _, _, arc_s, lab_s, *_ = backend.forward(ids, mask, zero_pred)
        hl = batch["head_labels"]; rl = batch["rel_labels"]
        ph = arc_s.argmax(-1)
        vm = hl >= 0
        arc_correct += (ph[vm] == hl[vm]).sum().item()
        B2, S = hl.size()
        phx = ph.clamp(0).unsqueeze(-1).unsqueeze(-1).expand(B2, S, 1, N_DEP)
        pr = lab_s.gather(2, phx).squeeze(2).argmax(-1)
        rel_correct += ((ph[vm] == hl[vm]) & (pr[vm] == rl[vm])).sum().item()
        total += vm.sum().item()
    uas = arc_correct / total
    las = rel_correct / total
    results["dep"] = {
        "score": round(uas, 4), "las": round(las, 4), "metric": "UAS",
        "benchmark": "UD English EWT test", "n": len(ud_test), "standard": True,
    }
    print(f"  DEP UAS: {uas:.4f}, LAS: {las:.4f}\n")

    # ── 4. SRL — PropBank EWT test ──
    print("=" * 70)
    print(f"[{backend.name}] BENCHMARK 4: SRL — PropBank EWT test (gold)")
    print("=" * 70)
    loader = DataLoader(SrlDataset(srl_test, tokenizer), batch_size=BS,
                        collate_fn=collate_basic)
    all_g, all_p = [], []
    for batch in tqdm(loader, desc=f"[{backend.name}] SRL"):
        ids = batch["input_ids"]; mask = batch["attention_mask"]
        pred_idx = batch["predicate_idx"]
        *_, srl_logits, _ = backend.forward(ids, mask, pred_idx)
        labs = batch["labels"]
        for j in range(labs.size(0)):
            vi = (labs[j] != -100).nonzero(as_tuple=True)[0]
            if not len(vi):
                continue
            vd = viterbi_decode(srl_logits[j, vi], SRL_TAGS)
            gl = [SRL_TAGS[labs[j, k]] for k in vi.tolist()]
            pl = [SRL_TAGS[vd[ki]] if vd[ki] < len(SRL_TAGS) else "O"
                  for ki in range(len(vi))]
            all_g.append([g if g != "V" else "O" for g in gl])
            all_p.append([p if p != "V" else "O" for p in pl])
    srl_f1 = seq_f1(all_g, all_p)
    results["srl"] = {
        "score": round(srl_f1, 4), "metric": "F1",
        "benchmark": "PropBank EWT test", "n": len(srl_test), "standard": True,
    }
    print(f"  SRL F1: {srl_f1:.4f}\n")

    # ── 5. CLS — DailyDialog test + internal SGD+GPT dev ──
    print("=" * 70)
    print(f"[{backend.name}] BENCHMARK 5: CLS — DailyDialog (mapped 8→4) + internal dev")
    print("=" * 70)
    loader = DataLoader(DailyDialogDataset(dd_examples, tokenizer),
                        batch_size=BS, collate_fn=collate_dd)
    dd_gold, dd_pred = [], []
    for batch in tqdm(loader, desc=f"[{backend.name}] CLS-DD"):
        ids = batch["input_ids"]; mask = batch["attention_mask"]
        B = ids.size(0)
        zero_pred = torch.zeros(B, dtype=torch.long)
        *_, cls_logits = backend.forward(ids, mask, zero_pred)
        preds = cls_logits.argmax(-1).tolist()
        for j, pc in enumerate(preds):
            our_label = CLS_LABELS[pc]
            dd_pred.append(OUR_CLS_TO_DD.get(our_label, 1))
            dd_gold.append(batch["gold_act"][j])
    dd_acc = accuracy_score(dd_gold, dd_pred)
    dd_mf1 = clf_f1(dd_gold, dd_pred, average="macro")
    print(f"  DailyDialog Accuracy: {dd_acc:.4f}, Macro F1: {dd_mf1:.4f}")
    results["cls_dailydialog"] = {
        "score": round(dd_acc, 4), "f1": round(dd_mf1, 4),
        "metric": "accuracy", "benchmark": "DailyDialog test",
        "n": len(dd_examples), "standard": True, "note": "8→4 label mapping",
    }

    # Internal CLS dev
    if internal_cls_dev is not None:
        loader = DataLoader(InternalCLSDataset(internal_cls_dev, tokenizer),
                            batch_size=BS, collate_fn=collate_basic)
        ag, ap = [], []
        for batch in tqdm(loader, desc=f"[{backend.name}] CLS-internal"):
            ids = batch["input_ids"]; mask = batch["attention_mask"]
            B = ids.size(0)
            zero_pred = torch.zeros(B, dtype=torch.long)
            *_, cls_logits = backend.forward(ids, mask, zero_pred)
            ap.extend(cls_logits.argmax(-1).tolist())
            ag.extend(batch["labels"].tolist())
        cls_internal = clf_f1(ag, ap, average="macro")
        results["cls_internal"] = {
            "score": round(cls_internal, 4), "metric": "macro_f1",
            "benchmark": "SGD+GPT dev (internal)",
            "n": len(internal_cls_dev), "standard": False,
        }
        print(f"  Internal CLS Macro F1: {cls_internal:.4f}")

    return results


# ── Main ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True,
                        help="Path to kniv-deberta-nlp-base-en-{xsmall,small,base}")
    parser.add_argument("--checkpoint", default="model.pt",
                        help="PyTorch checkpoint filename within model-dir")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--backend", default="pytorch",
                        choices=["pytorch", "onnx-fp32", "onnx-int8", "all"],
                        help="Which backend(s) to evaluate (default: pytorch)")
    parser.add_argument("--onnx-providers", nargs="+", default=None,
                        help="ONNX Runtime execution providers. Default: CUDA if "
                             "available else CPU. Example: --onnx-providers "
                             "CUDAExecutionProvider CPUExecutionProvider")
    parser.add_argument("--update-metadata", action="store_true",
                        help="Write PT-backend benchmark scores into metadata.json")
    args = parser.parse_args()

    # Default ONNX providers: prefer CUDA if available
    if args.onnx_providers is None:
        try:
            import onnxruntime as _ort
            _avail = _ort.get_available_providers()
            if "CUDAExecutionProvider" in _avail:
                args.onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                args.onnx_providers = ["CPUExecutionProvider"]
        except ImportError:
            args.onnx_providers = ["CPUExecutionProvider"]

    model_dir = Path(args.model_dir)
    onnx_dir = model_dir / "onnx"
    print(f"Device: {args.device}")
    print(f"Model:  {model_dir}")
    print(f"Backends: {args.backend}\n")

    # ── Decide which backends to run ──
    requested = []
    if args.backend in ("pytorch", "all"):
        requested.append("pytorch")
    if args.backend in ("onnx-fp32", "all"):
        if (onnx_dir / "cascade.onnx").exists():
            requested.append("onnx-fp32")
        else:
            print(f"WARNING: {onnx_dir / 'cascade.onnx'} not found — "
                  f"run student_export_onnx.py first. Skipping onnx-fp32.")
    if args.backend in ("onnx-int8", "all"):
        if (onnx_dir / "cascade-int8.onnx").exists():
            requested.append("onnx-int8")
        else:
            print(f"WARNING: {onnx_dir / 'cascade-int8.onnx'} not found — "
                  f"run student_export_onnx.py first. Skipping onnx-int8.")
    if not requested:
        print("ERROR: no available backends matched the request")
        return

    # ── Load eval data once ──
    print("Loading eval data...")
    from datasets import load_dataset
    with open(DATA_DIR / "ud_test.json") as f:
        ud_test = json.load(f)
    with open(DATA_DIR / "srl_test.json") as f:
        srl_test = json.load(f)
    conll = load_dataset("eriktks/conll2003")
    conll_test = conll["test"]
    conll_tag_names = conll_test.features["ner_tags"].feature.names
    dd = load_dataset("daily_dialog")
    dd_test = dd["test"]
    dd_examples = []
    for conv in dd_test:
        utterances = conv["dialog"]
        acts = conv["act"]
        for i, (utt, act) in enumerate(zip(utterances, acts)):
            if act == 0:
                continue
            dd_examples.append({
                "text": utt,
                "prev_text": utterances[i - 1] if i > 0 else None,
                "gold_act": act,
            })
    cls_path = DATA_DIR / "cls_sgd_mwoz_train.json"
    internal_cls_dev = None
    if cls_path.exists():
        with open(cls_path) as f:
            cls_data = json.load(f)
        random.seed(42)
        random.shuffle(cls_data)
        internal_cls_dev = cls_data[:4000]

    # ── Load tokenizer once (shared across backends) ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # ── Run requested backends ──
    all_results = {}

    if "pytorch" in requested:
        print(f"\n{'#'*70}\n# PyTorch backend ({args.device})\n{'#'*70}")
        model, _, info = load_student(model_dir, checkpoint=args.checkpoint,
                                       device=args.device, strict=False)
        backend = PyTorchBackend(model, torch.device(args.device))
        backend.name = f"pytorch_{args.device}"
        all_results[backend.name] = run_benchmarks(
            backend, tokenizer, conll_test, conll_tag_names,
            ud_test, srl_test, dd_examples, internal_cls_dev, args.batch_size)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "onnx-fp32" in requested or "onnx-int8" in requested:
        try:
            import onnxruntime as ort
        except ImportError:
            print("WARNING: onnxruntime not installed; skipping ONNX backends")
            ort = None

        if ort is not None:
            for variant in ("onnx-fp32", "onnx-int8"):
                if variant not in requested:
                    continue
                fname = "cascade.onnx" if variant == "onnx-fp32" else "cascade-int8.onnx"
                path = onnx_dir / fname
                session = ort.InferenceSession(str(path),
                                                providers=args.onnx_providers)
                active = session.get_providers()[0]
                provider_short = (active.replace("ExecutionProvider", "")
                                        .lower())
                bname = f"{variant}_{provider_short}"
                print(f"\n{'#'*70}")
                print(f"# {bname} ({path})")
                print(f"# Active provider: {active}")
                print(f"{'#'*70}")
                backend = ONNXBackend(session, name=bname)
                all_results[bname] = run_benchmarks(
                    backend, tokenizer, conll_test, conll_tag_names,
                    ud_test, srl_test, dd_examples, internal_cls_dev, args.batch_size)
                del session

    # ── Side-by-side summary ──
    print(f"\n{'#'*80}")
    print(f"# QUALITY COMPARISON ACROSS BACKENDS — {model_dir.name}")
    print(f"{'#'*80}")

    head_keys = ["pos", "ner_conll03", "dep", "srl", "cls_dailydialog", "cls_internal"]
    head_display = {
        "pos": "POS Acc", "ner_conll03": "NER F1 (CoNLL-03)",
        "dep": "DEP UAS", "srl": "SRL F1",
        "cls_dailydialog": "CLS DailyDlg", "cls_internal": "CLS internal",
    }
    backends_run = list(all_results.keys())
    col_w = 14
    header = f"  {'Head':<22}" + "".join(f"{b:>{col_w}}" for b in backends_run)
    print(header)
    print(f"  {'─'*22}" + "".join(f" {'─'*(col_w-1)}" for _ in backends_run))
    for hk in head_keys:
        if not any(hk in all_results[b] for b in backends_run):
            continue
        row = f"  {head_display[hk]:<22}"
        for b in backends_run:
            r = all_results[b].get(hk)
            if r is None:
                row += f"{'—':>{col_w}}"
            else:
                row += f"{r['score']:>{col_w}.4f}"
        print(row)
        # Also show LAS for DEP
        if hk == "dep":
            row = f"  {'  (LAS)':<22}"
            for b in backends_run:
                las = all_results[b].get("dep", {}).get("las")
                if las is None:
                    row += f"{'—':>{col_w}}"
                else:
                    row += f"{las:>{col_w}.4f}"
            print(row)

    # ── Quality drop relative to PyTorch ──
    if "pytorch_" + args.device in all_results and len(backends_run) > 1:
        pt_key = "pytorch_" + args.device
        print(f"\n{'─'*80}")
        print(f"  Quality drop vs PyTorch (negative = ONNX worse, positive = ONNX better)")
        print(f"{'─'*80}")
        header = f"  {'Head':<22}"
        for b in backends_run:
            if b == pt_key:
                continue
            header += f"{'Δ ' + b:>{col_w}}"
        print(header)
        for hk in head_keys:
            if hk not in all_results.get(pt_key, {}):
                continue
            row = f"  {head_display[hk]:<22}"
            pt_score = all_results[pt_key][hk]["score"]
            for b in backends_run:
                if b == pt_key:
                    continue
                if hk in all_results[b]:
                    delta = all_results[b][hk]["score"] - pt_score
                    row += f"{delta:>+{col_w}.4f}"
                else:
                    row += f"{'—':>{col_w}}"
            print(row)

    # ── Save ──
    out_path = model_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'#'*80}")
    print(f"Saved benchmark results: {out_path}")

    # ── Optionally update metadata.json (PT scores only) ──
    if args.update_metadata:
        pt_key = "pytorch_" + args.device
        if pt_key not in all_results:
            print("WARNING: --update-metadata requires --backend pytorch (or all)")
        else:
            pt_results = all_results[pt_key]
            meta_path = model_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                heads = meta.get("heads", {})
                score_map = {
                    "pos": ("pos", "score"),
                    "ner": ("ner_conll03", "score"),
                    "dep": ("dep", "score"),
                    "srl": ("srl", "score"),
                    "cls": ("cls_internal", "score"),
                }
                for hname, (rkey, field) in score_map.items():
                    if hname in heads and rkey in pt_results:
                        heads[hname][field] = pt_results[rkey][field]
                        if hname == "dep" and "las" in pt_results["dep"]:
                            heads["dep"]["las"] = pt_results["dep"]["las"]
                        if hname == "ner":
                            heads["ner"]["conll2003_score"] = pt_results["ner_conll03"]["score"]
                        heads[hname]["status"] = "completed"
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)
                print(f"Updated metadata: {meta_path}")
    print(f"{'#'*80}")


if __name__ == "__main__":
    main()
