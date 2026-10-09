"""Encoder + the phase-1 heads, over one pass of a window.

Phase 1 is ten heads: POS, morph, lemma, NER, DEP arc, DEP label, SRL
predicate, SRL role, CLS and sentiment. Coref and relations are phase 2,
trained from a phase-1 checkpoint (TRAINING.md §4, §8).

Shapes follow v5's conventions, reimplemented rather than imported:

- a **ScalarMix** per head over every encoder layer, so a head that wants
  surface form and one that wants semantics need not share a layer;
- a word is its **first subword**, and every head works at word level;
- the **cascade** feeds detached POS probabilities into NER and POS + NER
  into DEP, so heads train independently but read upstream predictions;
- **DEP** is Dozat-Manning biaffine, with a learned root in column 0 and
  candidate heads restricted to the word's own sentence;
- **SRL** scores every predicate in one pass (MODEL_CHANGES.md §1): a
  predicate flag per word, then a biaffine between each predicate and the
  words of its sentence. Training scores gold predicates; inference scores
  predicted ones.

Label scores are computed for one head per word -- the gold head in
training, the predicted one at inference -- rather than the full
``[W, W, 53]`` tensor, which at 1,024 words is most of the memory.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IGNORE

NEG = -1e4          # fits bf16; large enough to zero a softmax entry


class ScalarMix(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, layers: tuple[torch.Tensor, ...]) -> torch.Tensor:
        w = torch.softmax(self.weights, dim=0)
        return self.scale * sum(wi * li for wi, li in zip(w, layers))


def mlp(i: int, o: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(i, o), nn.LayerNorm(o), nn.GELU(), nn.Dropout(p))


def classifier(i: int, hidden: int, o: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, hidden), nn.GELU(),
                         nn.Dropout(p), nn.Linear(hidden, o))


class Bilinear(nn.Module):
    """``x^T U_l y + w_l . [x; y] + b_l`` for each label ``l``."""

    def __init__(self, d: int, labels: int):
        super().__init__()
        self.U = nn.Parameter(torch.zeros(labels, d, d))
        self.lin = nn.Linear(2 * d, labels)
        nn.init.xavier_uniform_(self.U)

    def pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``x [B, N, d]``, ``y [B, M, d]`` -> ``[B, N, M, labels]``."""
        s = torch.einsum("bnd,lde,bme->bnml", x, self.U, y)
        a, b = self.lin.weight[:, :x.size(-1)], self.lin.weight[:, x.size(-1):]
        return s + (x @ a.T).unsqueeze(2) + (y @ b.T).unsqueeze(1) + self.lin.bias

    def paired(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Elementwise pairs: ``x, y [..., d]`` -> ``[..., labels]``."""
        return torch.einsum("...d,lde,...e->...l", x, self.U, y) \
            + self.lin(torch.cat([x, y], -1))

    def against(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """One ``x [F, d]`` against many ``y [F, W, d]`` -> ``[F, W, labels]``.

        Contracts ``x`` with ``U`` first. Left to a generic einsum, the
        ``[F, W, d, d, labels]`` order costs ~400 GFLOP per SRL batch.
        """
        xu = torch.einsum("fd,lde->fle", x, self.U)
        a, b = self.lin.weight[:, :x.size(-1)], self.lin.weight[:, x.size(-1):]
        return torch.einsum("fwe,fle->fwl", y, xu) + (x @ a.T).unsqueeze(1) \
            + y @ b.T + self.lin.bias


class V6Model(nn.Module):
    def __init__(self, hf_id: str, sizes: dict[str, int], dropout: float = 0.1,
                 attn_implementation: str | None = None,
                 arc_dim: int = 512, rel_dim: int = 128, srl_dim: int = 256,
                 pretrained: bool = True):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
        if pretrained:
            self.encoder = AutoModel.from_pretrained(hf_id, **kw)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(hf_id), **kw)
        H = self.encoder.config.hidden_size
        L = self.encoder.config.num_hidden_layers + 1
        self.sizes = dict(sizes)
        p = dropout
        self.drop = nn.Dropout(p)
        self.mix = nn.ModuleDict({k: ScalarMix(L) for k in
                                  ("pos", "morph", "lemma", "ner", "dep", "srl",
                                   "cls", "sentiment")})
        S = sizes
        self.pos = nn.Linear(H, S["pos"])
        self.morph = nn.Linear(H, S["morph"])
        self.lemma = classifier(H, H, S["lemma"], p)
        self.ner = classifier(H + S["pos"], H, S["ner"], p)
        dep_in = H + S["pos"] + S["ner"]
        self.arc_dep, self.arc_head = mlp(dep_in, arc_dim, p), mlp(dep_in, arc_dim, p)
        self.rel_dep, self.rel_head = mlp(dep_in, rel_dim, p), mlp(dep_in, rel_dim, p)
        self.root = nn.Parameter(torch.zeros(1, 1, dep_in))
        self.arc = Bilinear(arc_dim, 1)
        self.rel = Bilinear(rel_dim, S["dep"])
        self.srl_pred = nn.Linear(H, 2)
        self.srl_p, self.srl_a = mlp(H, srl_dim, p), mlp(H, srl_dim, p)
        self.srl_role = Bilinear(srl_dim, S["srl"])
        self.pool = nn.ModuleDict({k: nn.Linear(H, 1) for k in ("cls", "sentiment")})
        self.cls = classifier(H, H, S["cls"], p)
        self.sentiment = classifier(H, H, S["sentiment"], p)

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    # ── pieces ─────────────────────────────────────────────────────
    def _words(self, layers, mix: str, first: torch.Tensor) -> torch.Tensor:
        h = self.mix[mix](layers)
        idx = first.unsqueeze(-1).expand(-1, -1, h.size(-1))
        return self.drop(torch.gather(h, 1, idx))

    @staticmethod
    def _same_sentence(sent: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        """``[B, W, W]``: may word j be word i's head / argument?"""
        return (sent.unsqueeze(2) == sent.unsqueeze(1)) & word_mask.unsqueeze(1) \
            & (sent.unsqueeze(2) >= 0)

    def _pool(self, words: torch.Tensor, batch: dict, kind: str) -> torch.Tensor:
        """Attention-pool each sentence's words: ``[B, S, H]``."""
        S = batch["sent_whole"].size(1)
        member = (batch["sent_of_word"].unsqueeze(1)
                  == torch.arange(S, device=words.device).view(1, S, 1)) \
            & batch["word_mask"].unsqueeze(1)                          # [B,S,W]
        score = self.pool[kind](words).squeeze(-1).unsqueeze(1)        # [B,1,W]
        score = score.masked_fill(~member, NEG)
        att = torch.softmax(score.float(), -1).to(words.dtype) * member
        return att @ words

    # ── forward ────────────────────────────────────────────────────
    def forward(self, batch: dict, predict: bool = False,
                predict_heads: bool | None = None) -> dict:
        """Logits per head. ``predict`` decodes DEP heads and SRL predicates
        from the model instead of reading them from the batch; dev evaluation
        sets ``predict_heads`` alone, so LAS is scored on predicted arcs while
        SRL roles are still scored on gold predicates."""
        predict_heads = predict if predict_heads is None else predict_heads
        out = self.encoder(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"],
                           output_hidden_states=True)
        layers = out.hidden_states
        first, wm = batch["first_subword"], batch["word_mask"]
        o = {}
        o["pos"] = self.pos(self._words(layers, "pos", first))
        o["morph"] = self.morph(self._words(layers, "morph", first))
        o["lemma"] = self.lemma(self._words(layers, "lemma", first))
        pos_p = torch.softmax(o["pos"].float(), -1).detach().to(o["pos"].dtype)
        o["ner"] = self.ner(torch.cat([self._words(layers, "ner", first), pos_p], -1))
        ner_p = torch.softmax(o["ner"].float(), -1).detach().to(o["ner"].dtype)

        # DEP
        d = torch.cat([self._words(layers, "dep", first), pos_p, ner_p], -1)
        B = d.size(0)
        cand = torch.cat([self.root.expand(B, 1, -1), d], 1)           # [B,W+1,·]
        arc = self.arc.pairwise(self.arc_dep(d), self.arc_head(cand)).squeeze(-1)
        allowed = torch.cat([wm.unsqueeze(-1),
                             self._same_sentence(batch["sent_of_word"], wm)], -1)
        o["arc"] = arc.masked_fill(~allowed, NEG)                      # [B,W,W+1]
        head = o["arc"].argmax(-1) if predict_heads else batch["dep_head"].clamp(min=0)
        hrep = torch.gather(self.rel_head(cand), 1,
                            head.unsqueeze(-1).expand(-1, -1, self.rel.U.size(1)))
        o["rel"] = self.rel.paired(self.rel_dep(d), hrep)              # [B,W,53]
        o["rel_head"] = head

        # SRL
        s = self._words(layers, "srl", first)
        o["srl_pred"] = self.srl_pred(s)
        if predict:
            fb, fp = (o["srl_pred"].argmax(-1).bool() & wm).nonzero(as_tuple=True)
        else:
            fb, fp = batch["frame_batch"], batch["frame_pred"]
        o["frame_batch"], o["frame_pred"] = fb, fp
        if len(fb):
            pr = self.srl_p(s)[fb, fp]                                 # [F,d]
            ar = self.srl_a(s)[fb]                                     # [F,W,d]
            role = self.srl_role.against(pr, ar)
            same = self._same_sentence(batch["sent_of_word"], wm)[fb, fp]
            o["srl_role"] = role.masked_fill(~same.unsqueeze(-1), NEG)  # [F,W,42]
        else:
            o["srl_role"] = s.new_zeros(0, s.size(1), self.sizes["srl"])

        # Sentence layers
        o["cls"] = self.cls(self._pool(self._words(layers, "cls", first), batch, "cls"))
        o["sentiment"] = self.sentiment(
            self._pool(self._words(layers, "sentiment", first), batch, "sentiment"))
        return o


# ── losses ──────────────────────────────────────────────────────────
HEADS = ("pos", "morph", "lemma", "ner", "arc", "rel", "srl_pred", "srl_role",
         "cls", "sentiment")
TARGET = {"pos": "pos", "morph": "morph", "lemma": "lemma", "ner": "ner",
          "arc": "dep_head", "rel": "dep_rel", "srl_pred": "srl_pred",
          "srl_role": "frame_roles", "sentiment": "sentiment"}


def _ce(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Mean CE over supervised positions, and how many there were.

    An all-ignored batch returns a zero that still carries a graph, rather
    than the NaN ``F.cross_entropy`` produces for an empty mean.
    """
    flat_t = target.reshape(-1)
    n = int((flat_t != IGNORE).sum())
    if n == 0:
        return logits.sum() * 0.0, 0
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), flat_t,
                           ignore_index=IGNORE), n


def losses(o: dict, batch: dict) -> dict[str, tuple[torch.Tensor, int]]:
    out = {}
    for h in HEADS:
        if h == "cls":
            ok = batch["cls_ok"]
            n = int(ok.sum())
            out[h] = ((F.binary_cross_entropy_with_logits(
                o["cls"][ok].float(), batch["cls"][ok]), n) if n
                else (o["cls"].sum() * 0.0, 0))
        elif h == "rel":
            # Only where the gold arc is supervised: the label is scored on it.
            t = batch["dep_rel"].masked_fill(batch["dep_head"] == IGNORE, IGNORE)
            out[h] = _ce(o["rel"], t)
        else:
            out[h] = _ce(o[h], batch[TARGET[h]])
    return out
