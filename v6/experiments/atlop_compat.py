"""Runtime patches that let the ATLOP reference code run on a modern stack.

ATLOP targets transformers 3.x, apex and wandb. Rather than forking its
source, the incompatibilities are patched at import time so the upstream
checkout stays pristine and the diff between "what they published" and
"what we ran" is this file.

1. ``build_inputs_with_special_tokens`` was dropped from the tokenizer API
   in transformers 5.x; ATLOP's preprocessing calls it.
2. SDPA does not materialise attention weights, and ATLOP reads the last
   attention map for its localized context pooling — so the encoder must be
   loaded with ``attn_implementation="eager"`` (done by the caller).
3. ``output[-1][-1]`` used to reach the attentions. On a modern
   ``ModelOutput`` that index resolves to ``cross_attentions`` — an empty
   tuple — and raises. The field is now named explicitly.
"""
from __future__ import annotations

import pathlib
import re


def patch_long_seq(atlop_dir: str | pathlib.Path) -> bool:
    """Rewrite ``long_seq.py``'s output indexing. Idempotent."""
    p = pathlib.Path(atlop_dir) / "long_seq.py"
    src = p.read_text()
    if "output.attentions[-1]" in src:
        return False
    out = re.sub(r"output\[-1\]\[-1\]", "output.attentions[-1]", src)
    out = re.sub(r"output\[0\]", "output.last_hidden_state", out)
    p.write_text(out)
    return True


def patch_tokenizer(tok):
    """Restore ``build_inputs_with_special_tokens`` (RoBERTa: ``<s> … </s>``)."""
    if not hasattr(tok, "build_inputs_with_special_tokens"):
        tok.build_inputs_with_special_tokens = (
            lambda ids, ids1=None: [tok.cls_token_id] + list(ids) + [tok.sep_token_id])
    return tok
