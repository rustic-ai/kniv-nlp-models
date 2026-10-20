"""The v7 model as eight exportable graphs, one window per call.

Each graph is a thin module over a trained ``V6Model``'s own submodules: the
computation is the model's forward, rewritten only where tracing needs it
(batch 1, no Python loops over the batch, no data-dependent Python ints).
The LoRA adapters are merged into their own copies of the encoder, so each
graph runs a plain encoder.

    plain        the window: every linguistic head, coreference scores, the
                 DocRED entity head, the span head's word vectors
    dep_rel      DEP labels for the chosen heads
    srl_roles    role logits per predicate frame
    coref_spans  mention spans grown from head words
    label        label texts -> label embeddings (label LoRA)
    span         candidate spans x entity types
    marker       relation logits per entity pair (relation LoRA)
    attr         fact attributes for the facts kept

``check(model, ...)`` compares every graph with the model's own forward on
real windows; the host runtime (``kniv_onnx``) does the rest.
"""
from __future__ import annotations

import copy

import torch
import torch.nn.functional as F
from torch import nn

from ..train.model import NEG, NerLSTMHead, V6Model, _dist_bucket

COREF_K = 50


def merged_encoder(model: V6Model, adapter) -> nn.Module:
    """A copy of the encoder with ``adapter``'s LoRA folded into its weights
    (``W + scale * B @ A``: what the forward hook adds), hooks removed."""
    enc = copy.deepcopy(model.encoder)
    for m in enc.modules():
        m._forward_hooks.clear()
    with torch.no_grad():
        for key in adapter.A:
            lin = enc.get_submodule(key.replace("__", "."))
            lin.weight += adapter.scale * (adapter.B[key] @ adapter.A[key]).to(lin.weight.dtype)
    return enc.eval()


def _plain_encoder(model: V6Model) -> nn.Module:
    enc = copy.deepcopy(model.encoder)
    for m in enc.modules():
        m._forward_hooks.clear()
    return enc.eval()


def mha(attn: nn.MultiheadAttention, q, k, v, pad=None):
    """``nn.MultiheadAttention`` (batch first, packed in-projection, eval) with
    every reshape written with -1: the module's own forward bakes the traced
    lengths into the graph."""
    D = attn.embed_dim
    h = attn.num_heads
    hd = D // h
    w, b = attn.in_proj_weight, attn.in_proj_bias
    qp = F.linear(q, w[:D], b[:D]).reshape(q.size(0), -1, h, hd).transpose(1, 2)
    kp = F.linear(k, w[D:2 * D], b[D:2 * D]).reshape(k.size(0), -1, h, hd).transpose(1, 2)
    vp = F.linear(v, w[2 * D:], b[2 * D:]).reshape(v.size(0), -1, h, hd).transpose(1, 2)
    sc = (qp @ kp.transpose(-1, -2)) / hd ** 0.5
    if pad is not None:
        sc = sc.masked_fill(pad.unsqueeze(1).unsqueeze(2), float("-inf"))
    o = (sc.softmax(-1) @ vp).transpose(1, 2).reshape(q.size(0), -1, D)
    return attn.out_proj(o)


def encoder_layers(enc: nn.TransformerEncoder, x, pad):
    """``nn.TransformerEncoder`` of pre-norm layers (as every head here
    builds them), eval, through ``mha``."""
    for layer in enc.layers:
        assert layer.norm_first
        y = layer.norm1(x)
        x = x + mha(layer.self_attn, y, y, y, pad)
        y = layer.norm2(x)
        x = x + layer.linear2(layer.activation(layer.linear1(y)))
    return enc.norm(x) if enc.norm is not None else x


def _layers(enc, ids, mask):
    out = enc(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    H = enc.config.hidden_size                    # a constant: traced shapes would not be
    return tuple(F.layer_norm(h, (H,)) for h in out.hidden_states)


class Plain(nn.Module):
    """``input_ids, attention_mask [1,T]``; ``first_subword, word_end,
    word_mask, sent_of_word [1,W]``; ``sent_ids [S]`` (0..S-1).

    Out: ``pos, morph, lemma, ner [1,W,*]``; ``arc [1,W,W+1]`` (masked);
    ``rel_dep [1,W,d], rel_head [1,W+1,d]`` (``dep_rel``'s inputs);
    ``srl_pred [1,W,2]``, ``srl_words [1,W,H]``; ``cls, sentiment [1,S,*]``;
    ``coref_words [1,W,H]``, ``coref_scores [1,W,K+1]``, ``coref_cand
    [1,W,K]`` (K = 50; columns past min(50, W) are NEG); ``ent [1,W,13]``;
    ``span_words [1,W,H]``."""

    def __init__(self, model: V6Model):
        super().__init__()
        self.m = model
        self.enc = _plain_encoder(model)

    def forward(self, input_ids, attention_mask, first_subword, word_end, word_mask,
                sent_of_word, sent_ids):
        m = self.m
        layers = _layers(self.enc, input_ids, attention_mask)
        first, wm = first_subword, word_mask
        m._word_span = (first, word_end)
        words = lambda k: m._words(layers, k, first)
        pos = m.pos(words("pos"))
        morph = m.morph(words("morph"))
        lemma = m.lemma(words("lemma"))
        pos_p = torch.softmax(pos.float(), -1).to(pos.dtype)
        ner_in = torch.cat([words("ner"), pos_p], -1)
        if isinstance(m.ner, NerLSTMHead):
            # The packed BiLSTM, batch 1: run to the last real word, zeros
            # after it (what pad_packed_sequence gives).
            h = m.ner.proj(ner_in)
            n = (wm.long() * torch.arange(1, wm.size(1) + 1)).max().clamp(min=1)
            r, _ = m.ner.rnn(h[:, :n])
            r = torch.cat([r, torch.zeros_like(h[:, n:])], 1)
            ner = m.ner.out(r)
        else:
            ner = m.ner(ner_in)
        ner_p = torch.softmax(ner.float(), -1).to(ner.dtype)
        d = torch.cat([words("dep"), pos_p, ner_p], -1)
        cand = torch.cat([m.root.expand(1, 1, -1), d], 1)
        arc = m.arc.pairwise(m.arc_dep(d), m.arc_head(cand)).squeeze(-1)
        allowed = torch.cat([wm.unsqueeze(-1), m._same_sentence(sent_of_word, wm)], -1)
        arc = arc.masked_fill(~allowed, NEG)
        rel_dep, rel_head = m.rel_dep(d), m.rel_head(cand)
        s = words("srl")
        srl_pred = m.srl_pred(s)
        batch = {"sent_whole": torch.zeros_like(sent_ids).unsqueeze(0),
                 "sent_of_word": sent_of_word, "word_mask": wm}
        cls = m.cls(m._pool(words("cls"), batch, "cls"))
        sentiment = m.sentiment(m._pool(words("sentiment"), batch, "sentiment"))
        cw = self.coref_words(m.mix["coref"](layers), first, wm, attention_mask)
        scores, ccand = self.antecedents(cw, wm)
        ent = m.ent(words("ent"))
        span_words = words("ent_span")
        return (pos, morph, lemma, ner, arc, rel_dep, rel_head, srl_pred, s, cls, sentiment,
                cw, scores, ccand, ent, span_words)

    def coref_words(self, h, first, word_mask, attention_mask):
        """``CorefHead.words`` with the suffix minimum (``cummin``, which ONNX
        lacks) taken as a masked minimum over a ``[W, W]`` grid."""
        c = self.m.coref
        T = h.size(1)
        W = first.size(1)
        last = attention_mask.sum(1, keepdim=True) - 1
        f = torch.where(word_mask, first, last.expand(-1, W))
        nxt = torch.cat([f[:, 1:], last], 1)                                # [1,W]
        i = torch.arange(W)
        later = i.view(W, 1) <= i.view(1, W)                                # j >= i
        big = torch.full_like(nxt, 2 ** 30)
        nxt = torch.where(later.unsqueeze(0), nxt.unsqueeze(1), big.unsqueeze(1)).amin(-1)
        t = torch.arange(T).view(1, 1, T)
        inside = (t >= f.unsqueeze(-1)) & (t < nxt.unsqueeze(-1)) & word_mask.unsqueeze(-1)
        sc = c.attn(h).float().squeeze(-1).unsqueeze(1)
        att = sc.masked_fill(~inside, NEG).softmax(-1)
        att = att * inside.any(-1, keepdim=True)
        w = torch.bmm(att.to(h.dtype), h)
        return w if c.context is None else encoder_layers(c.context, w, ~word_mask)

    def antecedents(self, a, wm):
        """``CorefHead.antecedents`` with a fixed K = 50: the coarse scores
        are padded with NEG columns so top-k never needs K <= W."""
        c = self.m.coref
        _, W, D = a.shape
        coarse = (c.coarse(a) @ a.transpose(1, 2)) / D ** 0.5
        i = torch.arange(W)
        ok = (i.view(1, W, 1) > i.view(1, 1, W)) & wm.unsqueeze(1) & wm.unsqueeze(2)
        coarse = coarse.float().masked_fill(~ok, NEG)
        coarse = torch.cat([coarse, torch.full((1, W, COREF_K), NEG)], -1)
        top, cand = coarse.topk(COREF_K, dim=-1)
        ap = torch.cat([a, torch.zeros(1, COREF_K, D, dtype=a.dtype)], 1)
        aj = ap[0][cand[0]].unsqueeze(0)                                   # [1,W,K,D]
        ai = a.unsqueeze(2).expand_as(aj)
        dist = c.dist(_dist_bucket(i.view(1, W, 1) - cand)).to(a.dtype)
        fine = c.fine(torch.cat([ai, aj, ai * aj, dist], -1)).squeeze(-1).float()
        s = torch.where(top > NEG / 2, top + fine, torch.full_like(top, NEG))
        return torch.cat([torch.zeros(1, W, 1), s], -1), cand


class DepRel(nn.Module):
    """``rel_dep [1,W,d], rel_head [1,W+1,d], head [1,W]`` (0 = root, j+1 =
    word j) -> DEP label logits ``[1,W,labels]``."""

    def __init__(self, model: V6Model):
        super().__init__()
        self.rel = model.rel

    def forward(self, rel_dep, rel_head, head):
        hrep = torch.gather(rel_head, 1, head.unsqueeze(-1).expand(-1, -1, rel_head.size(-1)))
        return self.rel.paired(rel_dep, hrep)


class SrlRoles(nn.Module):
    """``words [F,L,H]`` (each frame's sentence, left-aligned), ``pred [F]``,
    ``mask [F,L]`` -> role logits ``[F,L,tags]``."""

    def __init__(self, model: V6Model):
        super().__init__()
        self.head = model.srl_ctx

    def forward(self, words, pred, mask):
        """``SrlContextHead.forward`` through ``encoder_layers``."""
        hd = self.head
        words = words * mask.unsqueeze(-1)
        pos = torch.arange(words.size(1)).unsqueeze(0)
        x = hd.proj(words) + hd.is_pred((pos == pred.unsqueeze(1)).long()) \
            + hd.rel((pos - pred.unsqueeze(1)).clamp(-hd.max_rel, hd.max_rel) + hd.max_rel)
        h = encoder_layers(hd.enc, x, ~mask)
        hp = h[torch.arange(h.size(0)), pred].unsqueeze(1).expand_as(h)
        return hd.out(torch.cat([h, hp], -1))


class CorefSpans(nn.Module):
    """``coref_words [1,W,H]``, mention head words ``heads [M]``,
    ``sent_of_word [1,W]`` -> start/end logits ``[M,2r+1,2]``, positions."""

    def __init__(self, model: V6Model):
        super().__init__()
        self.c = model.coref

    def forward(self, coref_words, heads, sent_of_word):
        """``CorefHead.spans`` with the relative-position embeddings broadcast
        over the mentions instead of expanded to a traced count."""
        c, a = self.c, coref_words
        W = a.size(1)
        off = torch.arange(-c.r, c.r + 1)
        pos = heads.unsqueeze(1) + off
        inb = (pos >= 0) & (pos < W)
        posc = pos.clamp(0, W - 1)
        sow = sent_of_word[0]
        same = inb & (sow[posc] == sow[heads].unsqueeze(1))
        aj = a[0][posc]                                                     # [M,2r+1,H]
        ah = a[0][heads].unsqueeze(1).expand_as(aj)
        ones = torch.ones_like(heads, dtype=a.dtype).view(-1, 1, 1)
        rp = c.rel_pos(off + c.r).to(a.dtype).unsqueeze(0) * ones
        x = c.span(torch.cat([ah, aj, rp], -1))
        x = x * same.unsqueeze(-1)
        lg = c.conv(x.transpose(1, 2)).transpose(1, 2).float()
        return lg.masked_fill(~same.unsqueeze(-1), NEG), pos


class Label(nn.Module):
    """``label_ids, label_mask [K,L]`` -> ``label_emb [K,label_dim]``: the
    label texts through the encoder with the label adapter merged."""

    def __init__(self, model: V6Model):
        super().__init__()
        self.enc = merged_encoder(model, model.label_lora)
        self.label_enc = model.label_enc

    def forward(self, label_ids, label_mask):
        out = self.enc(input_ids=label_ids, attention_mask=label_mask)
        return self.label_enc(out.last_hidden_state, label_mask)


class Span(nn.Module):
    """``span_words [1,W,H]``, candidate spans ``start, end [N]`` (end
    exclusive), ``type_emb [K,label_dim]`` -> logits ``[N,K+1]`` (column 0
    the threshold)."""

    def __init__(self, model: V6Model):
        super().__init__()
        self.span = model.span

    def forward(self, span_words, start, end, type_emb):
        batch = {"span_batch": torch.zeros_like(start), "span_start": start, "span_end": end}
        return self.span(span_words, batch, type_emb)


class Marker(nn.Module):
    """The relation pass over the marker text. ``rm_input_ids,
    rm_attention_mask [1,T]``; each mention's start-marker position
    ``men_pos [M]`` (-1: past the limit) and entity ``men_ent [M]``;
    ``ent_ids [E]`` (0..E-1); ``pair_h, pair_t [P]``; ``label_emb
    [K,label_dim]``.

    Out: ``logits [P,K+1]`` (column 0 the threshold), ``pair [P,d]``,
    ``labels [K,d]`` (after the label-to-document attention) and ``ent
    [E,H]``: the fact attribute head's inputs."""

    def __init__(self, model: V6Model):
        super().__init__()
        self.enc = merged_encoder(model, model.rel_lora)
        self.r = model.relation

    def forward(self, rm_input_ids, rm_attention_mask, men_pos, men_ent, ent_ids, pair_h, pair_t,
                label_emb):
        r = self.r
        out = self.enc(input_ids=rm_input_ids, attention_mask=rm_attention_mask,
                       output_attentions=True)
        raw, att = out.last_hidden_state, out.attentions[-1]
        seq, att = raw[0].float(), att[0].float()                          # [T,H], [heads,T,T]
        ok = men_pos >= 0
        pc = men_pos.clamp(min=0)
        member = (men_ent.unsqueeze(0) == ent_ids.unsqueeze(1)) & ok.unsqueeze(0)   # [E,M]
        m_emb = seq[pc].masked_fill(~ok.unsqueeze(1), -1e4)                # [M,H]
        e_max = torch.where(member.unsqueeze(-1), m_emb.unsqueeze(0),
                            torch.full_like(m_emb, -1e4).unsqueeze(0)).amax(1)
        e_max = torch.maximum(e_max, torch.full_like(e_max, -1e4))
        ex = torch.exp(m_emb - e_max[men_ent]) * ok.unsqueeze(1).float()
        ent = e_max + torch.log((member.float() @ ex).clamp(min=1e-30))
        ent = torch.where(ent > -1e3, ent, torch.zeros_like(ent))
        cnt = member.float().sum(1).clamp(min=1)
        m_att = att[:, pc].transpose(0, 1)                                 # [M,heads,T]
        e_att = (member.float() @ m_att.reshape(m_att.size(0), -1)).view(
            -1, att.size(0), att.size(-1)) / cnt.view(-1, 1, 1)
        ht = (e_att[pair_h] * e_att[pair_t]).mean(1)
        ht = ht / (ht.sum(1, keepdim=True) + 1e-5)
        rs = ht @ seq
        hs, ts = ent[pair_h], ent[pair_t]
        zs = torch.tanh(r.head_extractor(torch.cat([hs, rs], -1)))
        zo = torch.tanh(r.tail_extractor(torch.cat([ts, rs], -1)))
        pr = r.pair(torch.cat([zs, zo, zs * zo], -1)).float()
        q = r.lab_in(label_emb).unsqueeze(0)
        if not r.xattn_off:
            doc = r.doc_proj(raw)
            pad = rm_attention_mask == 0
            for blk in r.xattn:                       # LabelXBlock.forward through mha
                q = q + torch.tanh(blk.g_att) * mha(blk.att, blk.n1(q), doc, doc, pad)
                q = q + torch.tanh(blk.g_ff) * blk.ff(blk.n2(q))
        q = q[0].float()
        scale = torch.exp(r.log_scale.float()) / r.d ** 0.5
        bias = r.bias(q).float().squeeze(-1)
        logits = torch.cat([((pr @ r.th.float()) * scale).unsqueeze(1),
                            (pr @ q.T) * scale + bias], 1)
        return logits, pr, q, ent


class Attr(nn.Module):
    """Facts kept: ``pair [A,d]``, ``labels [A,d]`` (each fact's relation
    vector), the window's ``ent [E,H]`` -> polarity ``[A,2]``, modality
    ``[A,6]``, holder and time pointers ``[A,E+1]`` (column 0: none)."""

    def __init__(self, model: V6Model):
        super().__init__()
        self.attr = model.attr

    def forward(self, pair, labels, ent):
        same = torch.ones(pair.size(0), ent.size(0), dtype=torch.bool)
        o = self.attr.score(pair, labels, ent, same)
        return o["polarity"], o["modality"], o["holder"], o["time"]


GRAPHS = {"plain": Plain, "dep_rel": DepRel, "srl_roles": SrlRoles, "coref_spans": CorefSpans,
          "label": Label, "span": Span, "marker": Marker, "attr": Attr}
