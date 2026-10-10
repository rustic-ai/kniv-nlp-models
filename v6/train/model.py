"""Encoder + the phase-1 heads, over one pass of a window.

Phase 1 is ten heads: POS, morph, lemma, NER, DEP arc, DEP label, SRL
predicate, SRL role, CLS and sentiment. Coref and relations are phase 2,
trained from a phase-1 checkpoint (TRAINING.md §4, §8).

Shapes follow v5's conventions, reimplemented rather than imported:

- a **ScalarMix** per head over every encoder layer, so a head that wants
  surface form and one that wants semantics need not share a layer. Each
  layer is layer-normalised before mixing: Ettin's residual stream is not
  normalised between layers (hidden-state RMS rises to 166 in Ettin-400M and
  17 in Ettin-1B, against ~1 throughout DeBERTa), so an unnormalised mix is
  dominated by the loudest layers whatever its weights say, and the heads
  start from losses of 9-11 where ln(classes) is 3-6. The same treatment is
  a no-op for DeBERTa, so it does not tilt the comparison;
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
                 pretrained: bool = True, srl_mode: str = "joint"):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        # fp32 master weights for every encoder; bf16 autocast does the
        # compute. transformers 5 keeps a checkpoint's stored dtype by default,
        # and DeBERTa-v3-large is stored as fp16 while Ettin is fp32: left to
        # the default, DeBERTa trained fp16 weights with no fp32 copy, where
        # AdamW updates at its learning rate are mostly lost to rounding -- a
        # silent handicap in exactly the comparison these runs exist for.
        kw = {"attn_implementation": attn_implementation} if attn_implementation else {}
        if pretrained:
            self.encoder = AutoModel.from_pretrained(hf_id, dtype=torch.float32, **kw)
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(hf_id),
                                                 dtype=torch.float32, **kw)
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
        if srl_mode not in ("joint", "conditioned"):
            raise ValueError(f"srl_mode must be joint or conditioned, not {srl_mode!r}")
        self.srl_mode = srl_mode
        if srl_mode == "joint":
            self.srl_p, self.srl_a = mlp(H, srl_dim, p), mlp(H, srl_dim, p)
            self.srl_role = Bilinear(srl_dim, S["srl"])
        else:
            # v5's design, for the controlled comparison of MODEL_CHANGES §1:
            # mark the predicate at the input so every encoder layer sees it,
            # one pass per predicate over its sentence.
            self.pred_embedding = nn.Embedding(2, H)
            nn.init.zeros_(self.pred_embedding.weight)
            self.mix["srl_cond"] = ScalarMix(L)
            self.srl_cond = classifier(2 * H, H, S["srl"], p)
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

    def _srl_conditioned(self, batch: dict, W: int,
                         max_tokens: int = 8192) -> torch.Tensor:
        """``[F, W, roles]`` logits from one encoder pass per frame over the
        predicate's sentence, predicate marked at the input. Frames run in
        length-sorted chunks of at most ``max_tokens`` subwords; positions
        outside the sentence are NEG, exactly as the joint head masks them."""
        C = self.sizes["srl"]
        if "frame_sent_ids" not in batch:          # a batch with no frames at all
            return torch.full((0, W, C), NEG, device=batch["input_ids"].device)
        ids = batch["frame_sent_ids"]
        F_ = ids.size(0)
        out = torch.full((F_, W, C), NEG, device=ids.device, dtype=torch.float32)
        if F_ == 0:
            return out
        lengths = batch["frame_sent_mask"].sum(1)
        order = torch.argsort(lengths, descending=True)
        embed = self.encoder.get_input_embeddings()
        chunks, cur, cur_len = [], [], 0
        for f in order.tolist():
            n = int(lengths[f])
            if cur and (len(cur) + 1) * max(cur_len, n) > max_tokens:
                chunks.append(cur); cur, cur_len = [], 0
            cur.append(f); cur_len = max(cur_len, n)
        chunks.append(cur)
        for chunk in chunks:
            idx = torch.tensor(chunk, device=ids.device)
            # Rounded up to a multiple of 16: far fewer distinct shapes (the MPS
            # backend caches a graph per shape and ran out of memory on them),
            # and tensor-core friendly on CUDA. Padding is masked.
            Lc = min(-(-int(lengths[idx].max()) // 16) * 16, ids.size(1))
            cid, cmask = ids[idx, :Lc], batch["frame_sent_mask"][idx, :Lc]
            psub = batch["frame_pred_sub"][idx]
            ind = torch.zeros_like(cid)
            ind[torch.arange(len(chunk), device=ids.device), psub] = 1
            emb = embed(cid) + self.pred_embedding(ind)
            hs = self.encoder(inputs_embeds=emb, attention_mask=cmask,
                              output_hidden_states=True).hidden_states
            h = self.mix["srl_cond"](tuple(F.layer_norm(x, x.shape[-1:]) for x in hs))
            first = batch["frame_sent_first"][idx]
            words = self.drop(torch.gather(h, 1, first.unsqueeze(-1).expand(-1, -1, h.size(-1))))
            pred = h[torch.arange(len(chunk), device=ids.device), psub]
            logits = self.srl_cond(torch.cat(
                [words, pred.unsqueeze(1).expand_as(words)], -1)).float()   # [c,Ws,C]
            pos = batch["frame_sent_start"][idx].unsqueeze(1) + torch.arange(
                words.size(1), device=ids.device)
            ok = batch["frame_sent_wmask"][idx] & (pos < W)
            rows = idx.unsqueeze(1).expand_as(pos)
            out[rows[ok], pos[ok]] = logits[ok]
        return out

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
        layers = tuple(F.layer_norm(h, h.shape[-1:]) for h in out.hidden_states)
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
        if self.srl_mode == "conditioned":
            if predict:
                raise NotImplementedError(
                    "the conditioned SRL head scores given predicates; decoding "
                    "predicted ones is the hybrid's job and is not built")
            o["srl_role"] = self._srl_conditioned(batch, s.size(1))
        elif len(fb):
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
