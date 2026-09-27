"""Run the official ATLOP checkpoint over Re-DocRED test.

ATLOP is the reference supervised document-level RE system. The released
roberta-large checkpoint was trained on the ORIGINAL DocRED, so it inherits
that dataset's systematic false negatives: scored against Re-DocRED's
corrected gold it reaches **0.952 precision at 0.302 recall** — it proposes
little, and what it proposes is nearly always right.

That makes its errors close to complementary to the LLM annotators', which
over-propose. Union with astra gains +8.9 F1 over astra alone at no extra
API cost, and the whole test set runs in ~90 seconds on a laptop.

SETUP (external code and weights, not vendored):

    cd <scratch>
    git clone --depth 1 https://github.com/wzhouad/ATLOP.git
    curl -L -o atlop-roberta \
      https://github.com/wzhouad/ATLOP/releases/download/1.0/atlop-roberta
    mkdir -p ATLOP/meta && curl -L -o ATLOP/meta/rel2id.json \
      https://raw.githubusercontent.com/tonytan48/KD-DocRE/main/meta/rel2id.json

`rel2id.json` is not in the ATLOP repo and the label ORDER must match the
checkpoint's output indices — a wrong order silently produces nonsense
rather than an error. The KD-DocRE copy was verified to contain exactly our
96 relation codes with contiguous ids and `Na`=0.

Then: ATLOP_DIR=<scratch> uv run python -m v6.experiments.atlop_runner

Three transformers-5 shims are applied below; ATLOP targets transformers 3.x.
"""
import json, sys, os
BASE = os.environ.get("ATLOP_DIR")
if not BASE:
    raise SystemExit("set ATLOP_DIR to the directory holding ATLOP/ and "
                     "atlop-roberta (see module docstring)")
sys.path.insert(0, os.path.join(BASE, "ATLOP"))
os.chdir(os.path.join(BASE, "ATLOP"))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from model import DocREModel
from prepro import read_docred

# ATLOP_CKPT overrides the released checkpoint with one we trained.
CKPT = os.environ.get("ATLOP_CKPT") or os.path.join(BASE, "atlop-roberta")
TEST = os.environ.get("ATLOP_INPUT") or \
       "/Users/rohit/Work/kniv-nlp-models/data/re-docred/test_revised.json"
OUT  = os.environ.get("ATLOP_OUT") or \
       "/Users/rohit/Work/kniv-nlp-models/data/re-docred/atlop_preds.json"
LIMIT = int(os.environ.get("ATLOP_LIMIT", "0")) or None

device = "mps" if torch.backends.mps.is_available() else "cpu"
name = "roberta-large"
tok = AutoTokenizer.from_pretrained(name)
# transformers 5.x dropped this from the tokenizer API; ATLOP's prepro
# calls it. RoBERTa wraps a single sequence as <s> ... </s>.
if not hasattr(tok, "build_inputs_with_special_tokens"):
    tok.build_inputs_with_special_tokens = (
        lambda ids, ids1=None: [tok.cls_token_id] + list(ids) + [tok.sep_token_id])

config = AutoConfig.from_pretrained(name, num_labels=97)
config.cls_token_id = tok.cls_token_id
config.sep_token_id = tok.sep_token_id
config.transformer_type = "roberta"
# ATLOP reads the last attention map for localized context pooling; SDPA
# does not materialise attention weights, so eager is required.
enc = AutoModel.from_pretrained(name, config=config, attn_implementation="eager")


# ATLOP's process_long_input reads the attention maps positionally, as
# `output[-1][-1]`, which was correct when a transformers 3.x encoder returned
# (last_hidden_state, pooler_output, attentions). transformers 5.x appends
# `cross_attentions`, so `output[-1]` is now that field -- an EMPTY tuple for a
# model without cross-attention, which is the only reason this raises
# IndexError instead of silently pooling over the wrong tensor.
#
# Patched on the INSTANCE, not by wrapping the module: a wrapper inserts a
# level into the parameter names ("model.inner.embeddings..." against the
# checkpoint's "model.embeddings..."), and because the checkpoint is loaded
# non-strictly that does not fail -- it silently runs an untrained model and
# reports zero relations. Assigning forward leaves the module tree untouched.
# Named access is used so a further reordering upstream cannot reintroduce the
# original bug.
_enc_forward = enc.forward


def _forward_old_style(*a, **kw):
    kw["output_attentions"] = True
    out = _enc_forward(*a, **kw)
    if not out.attentions:
        raise RuntimeError(
            "encoder returned no attention maps; ATLOP's localized context "
            "pooling cannot run (is attn_implementation still eager?)")
    return (out.last_hidden_state, out.pooler_output, out.attentions)


enc.forward = _forward_old_style

if LIMIT:                       # featurise a prefix only, for a smoke test
    docs = json.load(open(TEST))[:LIMIT]
    tmp = "/tmp/_redocred_slice.json"; json.dump(docs, open(tmp, "w")); src = tmp
else:
    src = TEST
features = read_docred(src, tok, max_seq_length=1024)
print(f"featurised {len(features)} documents", flush=True)

model = DocREModel(config, enc, num_labels=4).to(device)
sd = torch.load(CKPT, map_location="cpu", weights_only=False)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
if missing[:3]: print("  missing e.g.", missing[:3])
if unexpected[:3]: print("  unexpected e.g.", unexpected[:3])
model.eval()

def collate(batch):
    ml = max(len(f["input_ids"]) for f in batch)
    ids = [f["input_ids"] + [tok.pad_token_id] * (ml - len(f["input_ids"])) for f in batch]
    mask = [[1.0] * len(f["input_ids"]) + [0.0] * (ml - len(f["input_ids"])) for f in batch]
    return (torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float),
            [f["entity_pos"] for f in batch],
            [f["hts"] for f in batch])

loader = DataLoader(features, batch_size=2, shuffle=False, collate_fn=collate)
preds = []
with torch.no_grad():
    for bi, (ids, mask, ep, hts) in enumerate(loader):
        out = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                    entity_pos=ep, hts=hts)
        logits = out[0] if isinstance(out, (tuple, list)) else out
        preds.append(logits.float().cpu().numpy())
        if bi % 25 == 0:
            print(f"  batch {bi}/{len(loader)}", flush=True)
preds = np.concatenate(preds, axis=0)
np.save("/tmp/atlop_preds.npy", preds)

# map back to (doc, h, t, relation-code)
rel2id = json.load(open("meta/rel2id.json"))
id2rel = {v: k for k, v in rel2id.items()}
res, k = [], 0
for f in features:
    doc_preds = []
    for (h, t) in f["hts"]:
        row = preds[k]; k += 1
        for r in np.nonzero(row)[0]:
            if r != 0:
                doc_preds.append([int(h), int(t), id2rel[int(r)]])
    res.append({"title": f["title"], "preds": doc_preds})
json.dump(res, open(OUT, "w"))
print(f"\nwrote {OUT}: {sum(len(d['preds']) for d in res)} predicted triples "
      f"over {len(res)} documents", flush=True)
