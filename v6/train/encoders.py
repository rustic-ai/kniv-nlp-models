"""The encoders v6 is trained on, and what each one needs.

Three are compared on identical data (TRAINING.md §8): DeBERTa-v3-large,
the incumbent; Ettin-400M, the modern architecture at the same size; and
Ettin-1B, the one with headroom. Ettin is ModernBERT's architecture, so it
loads through transformers' native ``modernbert`` model type.

Every encoder sees the same window text. What differs is how many subwords
fit: DeBERTa-v3 has 512 positions and masks a window's tail past them;
Ettin takes ~8k and sees whole windows. Scoring is on sentence-level public
gold, so that difference does not tilt the evaluation -- it is the
deployment difference being measured.

Learning rates are starting points, not tuned values. ModernBERT-family
encoders usually want a higher rate than DeBERTa, so one shared rate would be
unfair to one side; the smoke runs sweep each before the full runs.

The ``-xsmall`` / ``-17m`` siblings share their family's tokenizer and
architecture and exist so the whole pipeline runs on a laptop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hf_id: str
    max_subwords: int            # including [CLS] and [SEP]
    encoder_lr: float
    head_lr: float = 1e-3
    batch_size: int = 8          # per device; windows
    grad_accum: int = 4          # batch_size * grad_accum windows per step
    grad_checkpointing: bool = True
    attn_implementation: str | None = None   # None: transformers' default

    def asdict(self) -> dict:
        return asdict(self)


# Effective batch is 32 windows (~11k words) for every encoder, so the three
# runs take the same number of optimiser steps over the same data. Per-device
# batch is an unmeasured guess until the A100 fit check (`loop --probe`).
ENCODERS: dict[str, EncoderSpec] = {s.name: s for s in [
    EncoderSpec("deberta-v3-large", "microsoft/deberta-v3-large",
                max_subwords=512, encoder_lr=2e-5),
    EncoderSpec("ettin-400m", "jhu-clsp/ettin-encoder-400m",
                max_subwords=1024, encoder_lr=5e-5),
    EncoderSpec("ettin-1b", "jhu-clsp/ettin-encoder-1b",
                max_subwords=1024, encoder_lr=3e-5,
                batch_size=4, grad_accum=8),
    # Local siblings: same tokenizer and architecture, laptop-sized.
    EncoderSpec("deberta-v3-xsmall", "microsoft/deberta-v3-xsmall",
                max_subwords=512, encoder_lr=5e-5, batch_size=4, grad_accum=1,
                grad_checkpointing=False),
    EncoderSpec("ettin-17m", "jhu-clsp/ettin-encoder-17m",
                max_subwords=1024, encoder_lr=1e-4, batch_size=4,
                grad_accum=1, grad_checkpointing=False),
]}


def get(name: str) -> EncoderSpec:
    try:
        return ENCODERS[name]
    except KeyError:
        raise SystemExit(f"unknown encoder {name!r}; "
                         f"choose from {', '.join(ENCODERS)}") from None
