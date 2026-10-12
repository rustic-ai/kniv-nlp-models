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


def _dist_bucket(d: torch.Tensor) -> torch.Tensor:
    """0,1,2,3,4 exact, then 5-7, 8-15, 16-31, 32-63, 64+ -> 10 buckets."""
    d = d.clamp(min=0)
    log = torch.floor(torch.log2(d.clamp(min=1).float())).long()
    return torch.where(d < 5, d, (log + 3).clamp(max=9))


def context_layer(H: int, n: int, p: float) -> nn.Module | None:
    """``n`` pre-norm transformer layers over words, for a head on a frozen
    encoder: feed-forward heads transform each word alone, and a frozen
    encoder never adapts its context to the task. ``None`` when ``n`` is 0."""
    if not n:
        return None
    layer = nn.TransformerEncoderLayer(H, nhead=max(1, H // 64), dim_feedforward=4 * H,
                                       dropout=p, activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, n, enable_nested_tensor=False)


def contextualise(ctx: nn.Module | None, words: torch.Tensor, word_mask: torch.Tensor):
    if ctx is None:
        return words
    return ctx(words, src_key_padding_mask=~word_mask)


class CorefHead(nn.Module):
    """Word-level coreference, as wl-coref (Dobrovolskii 2021) builds it.

    A word is an attention-weighted average of its subwords. Antecedents are
    scored between words: a bilinear coarse score keeps the top ``k`` earlier
    words, and a feed-forward scorer over ``[a_i, a_j, a_i*a_j, distance]``
    adds to it. Column 0 of the result is the dummy "no antecedent" at 0. The
    span predictor grows each mention's head word into its span: a
    feed-forward net over ``[head, word, offset]`` for the words within
    ``span_radius`` in the head's sentence, then two width-3 convolutions so a
    boundary sees its neighbours. Widths and dropout are wl-coref's; the
    genre and speaker features are not, the corpus has neither.
    """

    def __init__(self, H: int, k: int = 50, span_radius: int = 32, p: float = 0.3,
                 context: int = 0):
        super().__init__()
        self.k, self.r = k, span_radius
        self.drop = nn.Dropout(p)
        self.attn = nn.Linear(H, 1)
        self.context = context_layer(H, context, 0.1)
        self.coarse = nn.Linear(H, H)
        self.dist = nn.Embedding(10, 32)
        self.fine = nn.Sequential(nn.Linear(3 * H + 32, H), nn.LeakyReLU(), nn.Dropout(p),
                                  nn.Linear(H, 1))
        self.rel_pos = nn.Embedding(2 * span_radius + 1, 64)
        self.span = nn.Sequential(nn.Linear(2 * H + 64, H), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(H, 256), nn.ReLU(), nn.Dropout(p),
                                  nn.Linear(256, 64))
        self.conv = nn.Sequential(nn.Conv1d(64, 4, 3, 1, 1), nn.Conv1d(4, 2, 3, 1, 1))

    def words(self, h, first, word_mask, attention_mask):
        """``[B,W,H]`` from subword states ``h [B,T,H]``: a word's subwords run
        from its first to the next represented word's first (or the last real
        subword before [SEP])."""
        B, T, _ = h.shape
        W = first.size(1)
        last = attention_mask.sum(1, keepdim=True) - 1                     # [SEP]
        f = torch.where(word_mask, first, last.expand(-1, W))
        nxt = torch.cat([f[:, 1:], last], 1)
        nxt = torch.flip(torch.cummin(torch.flip(nxt, [1]), 1).values, [1])
        t = torch.arange(T, device=h.device).view(1, 1, T)
        inside = (t >= f.unsqueeze(-1)) & (t < nxt.unsqueeze(-1)) & word_mask.unsqueeze(-1)
        sc = self.attn(h).float().squeeze(-1).unsqueeze(1)                # [B,1,T]
        att = sc.masked_fill(~inside, NEG).softmax(-1)
        att = att * inside.any(-1, keepdim=True)                           # padding -> 0
        w = torch.bmm(att.to(h.dtype), h)
        return contextualise(self.context, self.drop(w), word_mask)

    def antecedents(self, a, word_mask):
        """``(scores [B,W,k+1], cand [B,W,k])``; invalid = NEG."""
        B, W, D = a.shape
        coarse = (self.drop(self.coarse(a)) @ a.transpose(1, 2)) / D ** 0.5   # [B,W,W]
        i = torch.arange(W, device=a.device)
        ok = (i.view(1, W, 1) > i.view(1, 1, W)) & word_mask.unsqueeze(1) \
            & word_mask.unsqueeze(2)
        coarse = coarse.float().masked_fill(~ok, NEG)
        k = min(self.k, W)
        top, cand = coarse.topk(k, dim=-1)                              # [B,W,k]
        aj = torch.gather(a.unsqueeze(1).expand(B, W, W, D), 2,
                          cand.unsqueeze(-1).expand(B, W, k, D))
        ai = a.unsqueeze(2).expand_as(aj)
        dist = self.dist(_dist_bucket(i.view(1, W, 1) - cand)).to(a.dtype)
        fine = self.fine(torch.cat([ai, aj, ai * aj, dist], -1)).squeeze(-1).float()
        valid = top > NEG / 2
        s = torch.where(valid, top + fine, torch.full_like(top, NEG))
        dummy = torch.zeros(B, W, 1, device=a.device)
        return torch.cat([dummy, s], -1), cand

    def spans(self, a, batch_idx, heads, sent_of_word):
        """Start/end logits ``[M, 2r+1, 2]`` over positions ``head-r .. head+r``
        in the head's sentence, and those positions."""
        W = a.size(1)
        off = torch.arange(-self.r, self.r + 1, device=a.device)
        pos = heads.unsqueeze(1) + off                                  # [M,2r+1]
        inb = (pos >= 0) & (pos < W)
        posc = pos.clamp(0, W - 1)
        same = inb & (sent_of_word[batch_idx.unsqueeze(1), posc] ==
                      sent_of_word[batch_idx, heads].unsqueeze(1))
        aj = a[batch_idx.unsqueeze(1), posc]                            # [M,2r+1,H]
        ah = a[batch_idx, heads].unsqueeze(1).expand_as(aj)
        rp = self.rel_pos(off + self.r).to(a.dtype).expand(len(heads), -1, -1)
        x = self.span(torch.cat([ah, aj, rp], -1))
        x = x * same.unsqueeze(-1)                     # outside the sentence: zeros, as padding
        lg = self.conv(x.transpose(1, 2)).transpose(1, 2).float()      # [M,2r+1,2]
        return lg.masked_fill(~same.unsqueeze(-1), NEG), pos


class RelationHead(nn.Module):
    """ATLOP-style document relation extraction over entity clusters.

    An entity is the log-sum-exp of its mentions' first-word vectors. ATLOP's
    localized context pooling reads the encoder's attention; this learns a
    pair-specific attention over the window instead, computed per window so
    no ``[pairs, words, d]`` tensor exists. Grouped bilinear over the
    relations, with column 0 the adaptive threshold class.
    """

    def __init__(self, H: int, n_rel: int, d: int = 768, block: int = 64, p: float = 0.1,
                 context: int = 0, pool: str = "first"):
        super().__init__()
        self.block = block
        # A mention as its first word ("first"), or the mean over its words
        # ("mean") -- ATLOP's marker token stands for the whole span.
        self.pool = pool
        self.context = context_layer(H, context, p)
        self.proj = mlp(H, d, p)
        self.q = nn.Linear(2 * d, d)
        self.kp = nn.Linear(d, d)
        self.head_ext = nn.Linear(2 * d, d)
        self.tail_ext = nn.Linear(2 * d, d)
        self.bilinear = nn.Linear(d * block, n_rel + 1)

    def forward(self, words, batch):
        x = self.proj(contextualise(self.context, words, batch["word_mask"]))  # [B,W,d]
        E = len(batch["ent_batch"])
        if E == 0 or len(batch["pair_h"]) == 0:
            return x.new_zeros(0, self.bilinear.out_features), None
        mb = batch["ent_batch"][batch["men_ent"]]
        if self.pool == "mean":
            cs = torch.cat([torch.zeros_like(x[:, :1]), x.float().cumsum(1)], 1)   # [B,W+1,d]
            n = (batch["men_end"] - batch["men_start"]).clamp(min=1).unsqueeze(1).float()
            men = (cs[mb, batch["men_end"]] - cs[mb, batch["men_start"]]) / n        # [Mm,d]
        else:
            men = x[mb, batch["men_start"]].float()                                  # [Mm,d]
        # log-sum-exp per entity, via a max shift for stability
        ent_max = torch.full((E, men.size(1)), -1e4, device=x.device)
        ent_max = ent_max.scatter_reduce(0, batch["men_ent"].unsqueeze(1).expand_as(men),
                                         men, reduce="amax", include_self=True)
        ex = torch.exp(men - ent_max[batch["men_ent"]])
        ent = torch.zeros(E, men.size(1), device=x.device).index_add_(0, batch["men_ent"], ex)
        ent = ent_max + torch.log(ent.clamp(min=1e-30))
        eh, et = ent[batch["pair_h"]], ent[batch["pair_t"]]
        ctx = torch.zeros_like(eh)
        keys = self.kp(x).float()
        qs = self.q(torch.cat([eh, et], -1).to(x.dtype)).float()
        wm = batch["word_mask"]
        evid = "pair_ev_ok" in batch and bool(batch["pair_ev_ok"].any())
        sent_att = (torch.zeros(len(eh), batch["pair_evidence"].size(1), device=x.device)
                    if evid else None)
        for b in batch["pair_batch"].unique().tolist():
            sel = batch["pair_batch"] == b
            att = (qs[sel] @ keys[b].T) / keys.size(-1) ** 0.5             # [Pb,W]
            att = att.masked_fill(~wm[b].unsqueeze(0), NEG).softmax(-1)
            ctx[sel] = (att @ x[b].float()).to(ctx.dtype)   # autocast returns bf16
            if evid and batch["pair_ev_ok"][sel].any():
                # The pair's attention summed per sentence (DREEAM).
                sw = batch["sent_of_word"][b]
                onehot = F.one_hot(sw.clamp(min=0), sent_att.size(1)).float() \
                    * (sw >= 0).unsqueeze(-1)
                sent_att[sel] = (att @ onehot).to(sent_att.dtype)
        zs = torch.tanh(self.head_ext(torch.cat([eh, ctx], -1)))
        zo = torch.tanh(self.tail_ext(torch.cat([et, ctx], -1)))
        P, d = zs.shape
        b1 = zs.view(P, d // self.block, self.block)
        b2 = zo.view(P, d // self.block, self.block)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(P, d * self.block)
        return self.bilinear(bl).float(), sent_att                      # [P, R+1]


class MarkerRelationHead(nn.Module):
    """ATLOP (Zhou et al. 2021), ported from its ``model.py``, over the marker
    pass: the document encoded with ``*`` around every entity mention.

    A mention is its start marker's last-layer vector; an entity, the
    log-sum-exp of its mentions'. Localized context pooling: each entity's
    last-layer attention is the mean over its start markers' attention rows;
    a pair's context is the encoder output weighted by the product of the two
    entities' attentions, averaged over heads. Head and tail extractors and the
    grouped bilinear are ATLOP's (768 wide, blocks of 64). The pooled
    attention, summed per sentence, is what DREEAM's evidence term supervises.
    """

    def __init__(self, H: int, n_rel: int, emb: int = 768, block: int = 64):
        super().__init__()
        self.emb, self.block = emb, block
        self.head_extractor = nn.Linear(2 * H, emb)
        self.tail_extractor = nn.Linear(2 * H, emb)
        self.bilinear = nn.Linear(emb * block, n_rel + 1)

    def forward(self, seq, att, batch):
        """``seq [B,T,H]`` last hidden layer, ``att [B,heads,T,T]`` last-layer
        attention. Returns ``(logits [P, R+1], sentence attention [P, S] or None)``."""
        if len(batch["ent_batch"]) == 0 or len(batch["pair_h"]) == 0:
            return seq.new_zeros(0, self.bilinear.out_features), None
        bl, sent_att = self.pair_features(seq, att, batch)
        return self.bilinear(bl).float(), sent_att

    def pair_features(self, seq, att, batch):
        """``(bl [P, emb*block], sentence attention)``: everything up to the
        grouped bilinear's input, shared with the label-conditioned head."""
        E = len(batch["ent_batch"])
        seq = seq.float()
        mb = batch["ent_batch"][batch["men_ent"]]
        pos = batch["rm_men_pos"]
        ok = pos >= 0
        pc = pos.clamp(min=0)
        m_emb = seq[mb, pc]                                              # [M,H]
        m_att = att[mb, :, pc].float()                                   # [M,heads,T]
        ent_idx = batch["men_ent"]
        # log-sum-exp over each entity's mentions (missing ones excluded)
        m_emb = m_emb.masked_fill(~ok.unsqueeze(1), -1e4)
        e_max = torch.full((E, seq.size(-1)), -1e4, device=seq.device).scatter_reduce(
            0, ent_idx.unsqueeze(1).expand_as(m_emb), m_emb, reduce="amax", include_self=True)
        ex = torch.exp(m_emb - e_max[ent_idx]) * ok.unsqueeze(1)
        ent = e_max + torch.log(torch.zeros(E, seq.size(-1), device=seq.device)
                                .index_add_(0, ent_idx, ex).clamp(min=1e-30))
        ent = torch.where(ent > -1e3, ent, torch.zeros_like(ent))       # no usable mention
        cnt = torch.zeros(E, device=seq.device).index_add_(0, ent_idx, ok.float()).clamp(min=1)
        e_att = torch.zeros(E, *m_att.shape[1:], device=seq.device).index_add_(
            0, ent_idx, m_att * ok.view(-1, 1, 1)) / cnt.view(-1, 1, 1)  # [E,heads,T]
        hs, ts = ent[batch["pair_h"]], ent[batch["pair_t"]]
        rs = torch.zeros_like(hs)
        evid = "pair_ev_ok" in batch and bool(batch["pair_ev_ok"].any())
        S = batch["pair_evidence"].size(1) if evid else 0
        sent_att = torch.zeros(len(hs), S, device=seq.device) if evid else None
        for b in batch["pair_batch"].unique().tolist():
            sel = batch["pair_batch"] == b
            ht = (e_att[batch["pair_h"][sel]] * e_att[batch["pair_t"][sel]]).mean(1)   # [Pb,T]
            ht = ht / (ht.sum(1, keepdim=True) + 1e-5)
            rs[sel] = (ht @ seq[b]).to(rs.dtype)            # autocast returns bf16
            if evid and batch["pair_ev_ok"][sel].any():
                ts_ = batch["rm_sent_of_tok"][b]
                onehot = F.one_hot(ts_.clamp(min=0), S).float() * (ts_ >= 0).unsqueeze(-1)
                sent_att[sel] = (ht @ onehot).to(sent_att.dtype)
        zs = torch.tanh(self.head_extractor(torch.cat([hs, rs], -1)))
        zo = torch.tanh(self.tail_extractor(torch.cat([ts, rs], -1)))
        P = zs.size(0)
        b1 = zs.view(P, self.emb // self.block, self.block)
        b2 = zo.view(P, self.emb // self.block, self.block)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(P, self.emb * self.block)
        return bl, sent_att


class LabelRelationHead(MarkerRelationHead):
    """ATLOP's head with the relation inventory as input: each relation is a
    vector from the label pass (its name and description, read by the
    encoder), so a caller can ask for relations the model never trained on.

    The grouped bilinear's fixed ``[R+1]`` projection becomes ``p =
    pair_proj(bl)`` scored against ``q_r = label_proj(e_r)``, plus a per-label
    bias predicted from ``e_r``; ATLOP's adaptive threshold becomes a learned
    vector ``th`` scored the same way, so it still moves with each pair.
    Column 0 stays the threshold: logits are ``[P, K+1]`` for the batch's K
    labels, the shape ``atlop_loss`` and decoding already use.
    """

    def __init__(self, H: int, label_dim: int = 1024, dim: int = 1024,
                 emb: int = 768, block: int = 64):
        nn.Module.__init__(self)
        self.emb, self.block = emb, block
        self.head_extractor = nn.Linear(2 * H, emb)
        self.tail_extractor = nn.Linear(2 * H, emb)
        self.pair_proj = nn.Linear(emb * block, dim)
        self.label_proj = nn.Linear(label_dim, dim, bias=False)
        self.label_bias = nn.Linear(label_dim, 1)
        nn.init.zeros_(self.label_bias.weight); nn.init.zeros_(self.label_bias.bias)
        self.th = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.th, std=0.02)

    def forward(self, seq, att, batch, label_emb):
        """``label_emb [K, label_dim]`` for the batch's labels, in column order."""
        K = label_emb.size(0)
        if len(batch["ent_batch"]) == 0 or len(batch["pair_h"]) == 0:
            return seq.new_zeros(0, K + 1), None
        bl, sent_att = self.pair_features(seq, att, batch)
        if getattr(self, "kd_w", None) is not None and self.training:
            # The transplanted ATLOP head, frozen: its logits for every known
            # relation, for the distillation term (``kd_margins``).
            self.kd_logits = (bl @ self.kd_w.T + self.kd_b).float()
        p = self.pair_proj(bl).float()                                   # [P,D]
        q = self.label_proj(label_emb).float()                           # [K,D]
        b = self.label_bias(label_emb).float().squeeze(-1)               # [K]
        logits = torch.cat([(p @ self.th.float()).unsqueeze(1), p @ q.T + b], 1)
        return logits, sent_att

    def keep_teacher(self, bilinear: nn.Linear) -> None:
        """Keep the bilinear being replaced, frozen and unsaved, so training
        can distil from it (``--kd-weight``)."""
        self.register_buffer("kd_w", bilinear.weight.detach().clone(), persistent=False)
        self.register_buffer("kd_b", bilinear.bias.detach().clone(), persistent=False)

    @torch.no_grad()
    def transplant(self, bilinear: nn.Linear, label_emb: torch.Tensor) -> float:
        """Reproduce a trained ATLOP bilinear exactly: ``pair_proj``'s first
        ``R+1`` rows take its weights, ``th`` the threshold column, and
        ``label_proj`` is solved by least squares so the R known labels'
        embeddings land on their own columns. Returns the residual."""
        R1 = bilinear.out_features
        if R1 > self.pair_proj.out_features:
            raise ValueError("pair_proj narrower than the bilinear it replaces")
        if label_emb.size(0) != R1 - 1:
            raise ValueError(f"{label_emb.size(0)} label embeddings for {R1 - 1} relations")
        self.pair_proj.weight.zero_(); self.pair_proj.bias.zero_()
        self.pair_proj.weight[:R1] = bilinear.weight
        self.pair_proj.bias[:R1] = bilinear.bias
        self.th.zero_(); self.th[0] = 1.0
        # Solved in float64 on the CPU (MPS has no float64).
        E = label_emb.detach().float().cpu().double()
        target = torch.zeros(R1 - 1, self.pair_proj.out_features, dtype=torch.float64)
        target[torch.arange(R1 - 1), torch.arange(1, R1)] = 1.0           # label r -> column r+1
        sol = torch.linalg.lstsq(E, target).solution                     # [d, D]
        self.label_proj.weight.copy_(sol.T.float().to(self.label_proj.weight.device))
        self.label_bias.weight.zero_(); self.label_bias.bias.zero_()
        return float((E @ sol - target).abs().max())


class LabelEncoder(nn.Module):
    """A relation or entity type as a vector: the encoder's last layer over
    its label text, mean-pooled, normalised and projected."""

    def __init__(self, H: int, label_dim: int = 1024):
        super().__init__()
        self.norm = nn.LayerNorm(H)
        self.proj = nn.Linear(H, label_dim)

    def forward(self, hidden, mask):
        m = mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
        return self.proj(self.norm(pooled.float()))


def kd_margin_loss(new: torch.Tensor, old: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """Distillation from the transplanted head: the new head's margins (each
    listed relation's logit minus the threshold's) pulled toward the old
    head's for the same relations. ``cols`` are the batch's label-bank ids,
    which for Re-DocRED are the old head's columns minus one; labels the old
    head never had (the rest of the bank) are left out."""
    had = cols < old.size(1) - 1
    if not bool(had.any()):
        return new.sum() * 0.0
    nm = new[:, 1:][:, had] - new[:, :1]
    om = old[:, 1 + cols[had]] - old[:, :1]
    return F.smooth_l1_loss(nm, om.detach())


def atlop_loss(logits: torch.Tensor, labels: torch.Tensor,
               mask: torch.Tensor | None = None) -> torch.Tensor:
    """ATLOP's adaptive-threshold loss. ``labels[:, 0]`` marks no-relation
    pairs; positive relations must beat the threshold class and the threshold
    must beat every negative relation.

    ``mask [P, K+1]``: 0 where a relation was not in that pair's schema. Such
    a column is neither a positive nor a negative -- a teacher shown 30
    relations says nothing about the other 66 -- so it is removed from both
    terms. Without a mask every column counts, as in ATLOP."""
    labels = labels.clone()
    th = torch.zeros_like(labels)
    th[:, 0] = 1.0
    labels[:, 0] = 0.0
    if mask is not None:
        mask = mask.to(logits.dtype).clone()
        mask[:, 0] = 1.0
        labels = labels * mask
        logits = logits - (1 - mask) * 1e4
    p_mask, n_mask = labels + th, 1 - labels
    l1 = -(F.log_softmax(logits - (1 - p_mask) * 1e4, -1) * labels).sum(1)
    l2 = -(F.log_softmax(logits - (1 - n_mask) * 1e4, -1) * th).sum(1)
    return (l1 + l2).mean()


class RelationAdapter(nn.Module):
    """LoRA on the encoder's transformer-layer linears, applied only while
    switched on: the relation pass runs the encoder with it, every other head
    reads the plain pass. Installed as forward hooks, so the encoder's own
    parameter names -- and every existing checkpoint -- are unchanged.

    ``B`` starts at zero, so the adapted pass starts identical to the plain
    one and a trained relation head starts where it was."""

    def __init__(self, encoder: nn.Module, r: int = 16, alpha: float = 32.0, p: float = 0.1):
        super().__init__()
        import math
        import re
        self.on = False
        # Held on through backward in relation-only training: gradient
        # checkpointing re-runs the forward during backward, and a switch
        # turned off by then would recompute the plain pass instead.
        self.sticky = False
        self.scale = alpha / r
        self.drop = nn.Dropout(p)
        self.A, self.B = nn.ParameterDict(), nn.ParameterDict()
        layer = re.compile(r"(^|\.)layers?\.\d+\.")
        for name, m in encoder.named_modules():
            if isinstance(m, nn.Linear) and layer.search(name):
                key = name.replace(".", "__")
                a = nn.Parameter(torch.empty(r, m.in_features))
                nn.init.kaiming_uniform_(a, a=math.sqrt(5))
                self.A[key] = a
                self.B[key] = nn.Parameter(torch.zeros(m.out_features, r))
                m.register_forward_hook(self._hook(key))
        if not self.A:
            raise ValueError("no transformer-layer linears found for the relation adapter")

    def _hook(self, key):
        def hook(module, inputs, output):
            if not self.on:
                return output
            x = self.drop(inputs[0])
            return output + (x @ self.A[key].to(x.dtype).T @ self.B[key].to(x.dtype).T) * self.scale
        return hook


class V6Model(nn.Module):
    def __init__(self, hf_id: str, sizes: dict[str, int], dropout: float = 0.1,
                 attn_implementation: str | None = None,
                 arc_dim: int = 512, rel_dim: int = 128, srl_dim: int = 256,
                 pretrained: bool = True, srl_mode: str = "joint",
                 phase2: bool = False, p2_context: int = 0, ent_head: bool = True,
                 rel_adapter: int = 0, rel_markers: bool = False,
                 rel_head: str = "bilinear", label_dim: int = 1024,
                 label_plain: bool = True):
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
        self.phase2 = phase2
        if phase2:
            # Phase 2 (TRAINING.md §4): coreference and relations, each with
            # its own layer mix; relations ride on entity clusters built from
            # NER and coref.
            for k in ("coref", "relation") + (("ent",) if ent_head else ()):
                self.mix[k] = ScalarMix(L)
            # DocRED-style entity mentions, the relation head's entities at
            # inference (OntoNotes NER disagrees with DocRED on boundaries).
            # Off for checkpoints saved before the head existed.
            self.ent = None
            if ent_head:
                from .data import ENT_TAGS
                self.ent = classifier(H, H, len(ENT_TAGS), p)
            self.coref = CorefHead(H, context=p2_context)
            # ATLOP over a marker pass, or the shared-pass head.
            if rel_head not in ("bilinear", "label"):
                raise ValueError(f"rel_head must be bilinear or label, not {rel_head!r}")
            if rel_head == "label" and not rel_markers:
                raise ValueError("the label-conditioned relation head needs rel_markers")
            # ATLOP over a marker pass -- with a fixed relation inventory, or
            # with relations as label texts -- or the shared-pass head.
            if rel_head == "label":
                self.relation = LabelRelationHead(H, label_dim=label_dim)
                self.label_enc = LabelEncoder(H, label_dim)
            else:
                self.relation = (MarkerRelationHead(H, S["relations"]) if rel_markers else
                                 RelationHead(H, S["relations"], p=p, context=p2_context))
        self.rel_kind = rel_head if phase2 else "bilinear"
        self.label_plain = label_plain
        self.p2_context = p2_context
        self.rel_markers = phase2 and rel_markers
        # A rank-``rel_adapter`` LoRA for a second, relation-only encoder pass
        # (0 = none): relations get encoder capacity of their own while every
        # other head keeps the shared pass.
        self.rel_lora = (RelationAdapter(self.encoder, rel_adapter)
                         if phase2 and rel_adapter else None)
        # Set by a run that trains only heads: the encoder then runs without
        # gradient and without dropout, the features it will give at inference.
        self.freeze_encoder = False
        self.cls = classifier(H, H, S["cls"], p)
        self.sentiment = classifier(H, H, S["sentiment"], p)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

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
    def _relation_layers(self, batch: dict):
        """The adapted encoder pass, for the relation and entity heads (the
        plain pass when there is no adapter)."""
        if self.rel_lora is None:
            out = self.encoder(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],
                               output_hidden_states=True)
            return tuple(F.layer_norm(h, h.shape[-1:]) for h in out.hidden_states)
        self.rel_lora.on = True
        try:
            out = self.encoder(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],
                               output_hidden_states=True)
        finally:
            if not self.rel_lora.sticky:
                self.rel_lora.on = False
        return tuple(F.layer_norm(h, h.shape[-1:]) for h in out.hidden_states)

    def _marker_pass(self, batch: dict):
        """The marker pass, through the relation adapter when there is one:
        last hidden layer and last-layer attention."""
        if self.rel_lora is not None:
            self.rel_lora.on = True
        try:
            out = self.encoder(input_ids=batch["rm_input_ids"],
                               attention_mask=batch["rm_attention_mask"],
                               output_attentions=True)
        finally:
            if self.rel_lora is not None and not self.rel_lora.sticky:
                self.rel_lora.on = False
        return out.last_hidden_state, out.attentions[-1]

    def label_embeddings(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """The label pass: label texts through the encoder into ``[K,
        label_dim]``. At inference a schema's embeddings are computed once and
        passed as ``batch["rel_label_emb"]``.

        ``label_plain`` (default): the plain, frozen encoder reads the label
        texts and only ``label_enc`` learns -- the relation adapter then
        serves the marker pass alone, so learning label embeddings cannot
        move the pair features. Otherwise the texts go through the adapter
        too."""
        if self.label_plain:
            prev = self.rel_lora.on if self.rel_lora is not None else False
            if self.rel_lora is not None:
                self.rel_lora.on = False
            try:
                with torch.no_grad():
                    out = self.encoder(input_ids=ids, attention_mask=mask)
            finally:
                if self.rel_lora is not None:
                    self.rel_lora.on = prev
            return self.label_enc(out.last_hidden_state, mask)
        if self.rel_lora is not None:
            self.rel_lora.on = True
        try:
            out = self.encoder(input_ids=ids, attention_mask=mask)
        finally:
            if self.rel_lora is not None and not self.rel_lora.sticky:
                self.rel_lora.on = False
        return self.label_enc(out.last_hidden_state, mask)

    def _marker_heads(self, plain_layers, batch: dict, o: dict) -> dict:
        """Marker mode: the entity head reads the plain pass (entities must be
        known before the markers can be placed), relations the marker pass."""
        first = batch["first_subword"]
        if self.ent is not None:
            o["ent"] = self.ent(self._words(plain_layers, "ent", first))
            o["_anchor"] = o["ent"].sum() * 0.0
        if "rm_input_ids" in batch and bool(batch["rm_ok"].all()):
            seq, att = self._marker_pass(batch)
            if isinstance(self.relation, LabelRelationHead):
                emb = batch.get("rel_label_emb")
                if emb is None:
                    emb = self.label_embeddings(batch["rel_label_ids"], batch["rel_label_mask"])
                o["relation"], o["relation_sent_att"] = self.relation(seq, att, batch, emb)
                if getattr(self.relation, "kd_logits", None) is not None and self.training:
                    o["relation_kd_old"] = self.relation.kd_logits
                    self.relation.kd_logits = None
            else:
                o["relation"], o["relation_sent_att"] = self.relation(seq, att, batch)
            o["_anchor"] = seq.sum() * 0.0
        return o

    def _relation_heads(self, layers, batch: dict, o: dict) -> dict:
        first = batch["first_subword"]
        rw = self._words(layers, "relation", first)
        o["_anchor"] = rw.sum() * 0.0          # a graph for an all-unsupervised batch
        o["relation"], o["relation_sent_att"] = self.relation(rw, batch)
        if self.ent is not None:
            o["ent"] = self.ent(self._words(layers, "ent", first))
        return o

    def forward(self, batch: dict, predict: bool = False,
                predict_heads: bool | None = None, relation_only: bool = False) -> dict:
        """Logits per head. ``predict`` decodes DEP heads and SRL predicates
        from the model instead of reading them from the batch; dev evaluation
        sets ``predict_heads`` alone, so LAS is scored on predicted arcs while
        SRL roles are still scored on gold predicates."""
        predict_heads = predict if predict_heads is None else predict_heads
        first, wm = batch["first_subword"], batch["word_mask"]
        if relation_only:
            if self.rel_markers:
                if self.rel_lora is not None:
                    self.rel_lora.on = False      # the plain pass is plain
                with torch.no_grad():
                    out = self.encoder(input_ids=batch["input_ids"],
                                       attention_mask=batch["attention_mask"],
                                       output_hidden_states=True)
                plain = tuple(F.layer_norm(h, h.shape[-1:]) for h in out.hidden_states)
                return self._marker_heads(plain, batch, {})
            return self._relation_heads(self._relation_layers(batch), batch, {})
        if self.rel_lora is not None and not self.rel_lora.sticky:
            self.rel_lora.on = False              # the plain pass is plain
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_encoder):
            out = self.encoder(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],
                               output_hidden_states=True)
        layers = tuple(F.layer_norm(h, h.shape[-1:]) for h in out.hidden_states)
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

        if self.phase2 and "coref_ok" in batch:
            cw = self.coref.words(self.mix["coref"](layers), first, wm,
                                  batch["attention_mask"])
            o["coref_scores"], o["coref_cand"] = self.coref.antecedents(cw, wm)
            o["coref_a"] = cw
            if len(batch["coref_head"]):
                o["coref_span"], o["coref_span_pos"] = self.coref.spans(
                    o["coref_a"], batch["coref_batch"], batch["coref_head"],
                    batch["sent_of_word"])
            if self.rel_markers:
                o = self._marker_heads(layers, batch, o)
            else:
                o = self._relation_heads(self._relation_layers(batch) if self.rel_lora is not None
                                         else layers, batch, o)
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


PHASE2_HEADS = ("coref", "coref_span", "relation", "relation_evidence", "ent")
RELATION_HEADS = ("relation", "relation_evidence", "ent", "relation_kd")
# Fixed weights on top of the per-head loss scale: DREEAM weighs its evidence
# term at 0.1 of the relation loss.
HEAD_WEIGHT = {"relation_evidence": 0.1}


def coref_targets(batch: dict, cand: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """``[B,W,k+1]`` gold antecedent mask; column 0 (dummy) where a word has
    no gold antecedent among its *valid* candidates, including non-mentions.
    Early words have fewer than k real candidates and ``topk`` pads with
    masked ones; a gold match there would put the target on a NEG score."""
    B, W, k = cand.shape
    cl = torch.full((B, W), -1, dtype=torch.long, device=cand.device)
    cl[batch["coref_batch"], batch["coref_head"]] = batch["coref_cluster"]
    cj = torch.gather(cl, 1, cand.view(B, -1)).view(B, W, k)
    gold = (cj == cl.unsqueeze(-1)) & (cl.unsqueeze(-1) >= 0) & (scores[..., 1:] > NEG / 2)
    dummy = ~gold.any(-1, keepdim=True)
    return torch.cat([dummy, gold], -1)


def _zero(o: dict) -> torch.Tensor:
    return (o["pos"] if "pos" in o else o["_anchor"]).sum() * 0.0


def phase2_losses(o: dict, batch: dict) -> dict[str, tuple[torch.Tensor, int]]:
    out = {}
    ok = batch["coref_ok"]
    if "coref_scores" in o and ok.any():
        s = o["coref_scores"]                                          # [B,W,k+1]
        gold = coref_targets(batch, o["coref_cand"], s)
        wm = batch["word_mask"] & ok.unsqueeze(1)
        nll = torch.logsumexp(s, -1) - torch.logsumexp(s.masked_fill(~gold, NEG), -1)
        n = int(wm.sum())
        # wl-coref's auxiliary BCE (weight 0.5) over the real candidates and
        # the dummy: it keeps teaching after the likelihood term has converged.
        real = torch.cat([torch.ones_like(s[..., :1], dtype=torch.bool), s[..., 1:] > NEG / 2], -1)
        real = real & wm.unsqueeze(-1)
        bce = F.binary_cross_entropy_with_logits(s[real].clamp(-50, 50), gold[real].float())
        out["coref"] = (nll[wm].mean() + 0.5 * bce, n)
    else:
        out["coref"] = (_zero(o), 0)
    if "coref_span" in o:
        lg, pos = o["coref_span"], o["coref_span_pos"]
        st = (pos == batch["coref_start"].unsqueeze(1)).float().argmax(1)
        en = (pos == batch["coref_end"].unsqueeze(1)).float().argmax(1)
        r = (lg.size(1) - 1) // 2
        inr = ((batch["coref_start"] - batch["coref_head"]).abs() <= r) & \
              ((batch["coref_end"] - batch["coref_head"]).abs() <= r)
        # A target outside the head's sentence sits on a masked position:
        # LingMess mentions can cross our sentence splits.
        m_ = torch.arange(len(st), device=lg.device)
        inr = inr & (lg[m_, st, 0] > NEG / 2) & (lg[m_, en, 1] > NEG / 2)
        if inr.any():
            loss = F.cross_entropy(lg[inr, :, 0], st[inr]) + F.cross_entropy(lg[inr, :, 1], en[inr])
            out["coref_span"] = (loss, int(inr.sum()))
        else:
            out["coref_span"] = (lg.sum() * 0.0, 0)
    else:
        out["coref_span"] = (_zero(o), 0)
    if "relation" in o and len(o["relation"]):
        out["relation"] = (atlop_loss(o["relation"], batch["pair_labels"],
                                      batch.get("pair_label_mask")), len(o["relation"]))
    else:
        out["relation"] = (_zero(o), 0)
    if "ent" in o and "ent" in batch:
        out["ent"] = _ce(o["ent"], batch["ent"])
    else:
        out["ent"] = (_zero(o), 0)
    kd = o.get("relation_kd_old")
    if kd is not None and len(o.get("relation", [])) and "rel_label_cols" in batch:
        out["relation_kd"] = (kd_margin_loss(o["relation"], kd, batch["rel_label_cols"]),
                              len(o["relation"]))
    else:
        out["relation_kd"] = (_zero(o), 0)
    sa = o.get("relation_sent_att")
    if sa is not None and batch["pair_ev_ok"].any():
        ok = batch["pair_ev_ok"]
        t = batch["pair_evidence"][ok]
        kl = (t * (torch.log(t.clamp(min=1e-12)) - torch.log(sa[ok].clamp(min=1e-12)))).sum(-1)
        out["relation_evidence"] = (kl.mean(), int(ok.sum()))
    else:
        out["relation_evidence"] = (_zero(o), 0)
    return out


def losses(o: dict, batch: dict) -> dict[str, tuple[torch.Tensor, int]]:
    out = phase2_losses(o, batch) if "coref_ok" in batch else {}
    if "pos" not in o:                         # a relation-only pass
        return {h: v for h, v in out.items() if h in RELATION_HEADS}
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
