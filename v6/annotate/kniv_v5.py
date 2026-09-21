"""kniv v5 teacher as a bake-off annotator.

The published `vs v5` column compares an LLM measured on N sampled sentences
against v5's headline score on the full 2,077-sentence test set. Those are
different samples, so at small N the delta is indicative rather than decided.
Running the teacher over the *same* items closes that gap.

Structural layers only — POS and DEP. v5 has no lemma or morph head, and SRL
needs a predicate index per item, handled by the caller.

Architecture is reconstructed from the checkpoint's own state dict rather
than imported from ``models/``: v6 keeps no code dependency on v5.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from ..schemas import DEPRELS, NER_LABELS, SRL_TAGS, UPOS_TAGS
from .base import AnnotationResult, tree_is_wellformed

TEACHER_DIR = Path("models/kniv-deberta-nlp-base-en-large")


def viterbi_decode(logits: torch.Tensor, tag2id: dict[str, int]) -> list[int]:
    """Constrained BIO Viterbi: I-X may only follow B-X or I-X of the same type.

    v5 decodes NER and SRL this way, so scoring without it would understate
    the teacher.
    """
    n_tags = logits.size(-1)
    seq = logits.size(0)
    lp = torch.log_softmax(logits, dim=-1)
    names = {v: k for k, v in tag2id.items()}
    allowed = torch.ones(n_tags, n_tags, dtype=torch.bool)
    for j in range(n_tags):
        nm = names.get(j, "O")
        if nm.startswith("I-"):
            b = tag2id.get(f"B-{nm[2:]}", -1)
            for i in range(n_tags):
                if i != b and i != j:
                    allowed[i][j] = False
    NEG = -1e9
    vit = torch.full((seq, n_tags), NEG)
    bp = torch.zeros(seq, n_tags, dtype=torch.long)
    vit[0] = lp[0]
    for t in range(1, seq):
        for j in range(n_tags):
            sc = vit[t - 1].clone()
            sc[~allowed[:, j]] = NEG
            best = sc.argmax()
            vit[t, j] = sc[best] + lp[t, j]
            bp[t, j] = best
    path = [0] * seq
    path[-1] = int(vit[-1].argmax())
    for t in range(seq - 2, -1, -1):
        path[t] = int(bp[t + 1, path[t + 1]])
    return path


class ScalarMix(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n))
        self.scale = nn.Parameter(torch.ones(1))

    def forward(self, layers):
        w = torch.softmax(self.weights, dim=0)
        return self.scale * sum(wi * li for wi, li in zip(w, layers))


class Biaffine(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 1):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_dim, in_dim + 1, in_dim + 1))

    def forward(self, h_dep, h_head):
        B, S, _ = h_dep.size()
        h_dep = torch.cat([h_dep, torch.ones(B, S, 1, device=h_dep.device)], -1)
        h_head = torch.cat([h_head, torch.ones(B, S, 1, device=h_head.device)], -1)
        scores = torch.einsum("bxi,oij,byj->boxy", h_dep, self.weight, h_head)
        return (scores.squeeze(1).contiguous() if scores.size(1) == 1
                else scores.permute(0, 2, 3, 1).contiguous())


class BiaffineDEPHead(nn.Module):
    def __init__(self, H: int, arc_dim=512, label_dim=128, num_labels=53, dropout=0.1):
        super().__init__()
        mk = lambda d: nn.Sequential(  # noqa: E731
            nn.Linear(H, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.arc_dep, self.arc_head = mk(arc_dim), mk(arc_dim)
        self.label_dep, self.label_head = mk(label_dim), mk(label_dim)
        self.biaffine_arc = Biaffine(arc_dim, 1)
        self.biaffine_label = Biaffine(label_dim, num_labels)

    def forward(self, hidden):
        return (self.biaffine_arc(self.arc_dep(hidden), self.arc_head(hidden)),
                self.biaffine_label(self.label_dep(hidden), self.label_head(hidden)))


class KnivV5Annotator:
    """Drop-in annotator exposing the same surface as :class:`LLMAnnotator`."""

    name = "kniv-v5"

    def __init__(self, spec, cache, model_dir: Path | None = None,
                 device: str | None = None, max_length: int = 128, **_):
        self.spec = spec
        self.cache = cache
        self.model_dir = Path(model_dir or TEACHER_DIR)
        self.max_length = max_length
        # Apple Metal is a real accelerator on this hardware and was
        # missing from the fallback chain, so every local run silently used
        # the CPU.
        if device:
            picked = device
        elif torch.cuda.is_available():
            picked = "cuda"
        elif torch.backends.mps.is_available():
            picked = "mps"
        else:
            picked = "cpu"
        self.device = torch.device(picked)
        self._loaded = False

    def _load(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        ckpt = self.model_dir / "model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(
                f"{ckpt} not found. Fetch it with:\n"
                f"  huggingface-cli download dragonscale-ai/"
                f"kniv-deberta-nlp-base-en-large model.pt "
                f"--local-dir {self.model_dir}"
            )
        state = torch.load(ckpt, map_location="cpu", weights_only=True)

        self.tok = AutoTokenizer.from_pretrained(str(self.model_dir))
        enc = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        enc.load_state_dict(state["deberta"])
        H = enc.config.hidden_size
        NL = enc.config.num_hidden_layers + 1

        self.pos_sm = ScalarMix(NL)
        self.pos_sm.load_state_dict(state["pos_scalar_mix"])
        self.pos_head = nn.Linear(H, len(UPOS_TAGS))
        self.pos_head.load_state_dict(state["pos_head"])

        self.ner_sm = ScalarMix(NL)
        self.ner_sm.load_state_dict(state["ner_scalar_mix"])
        self.ner_lstm = nn.LSTM(H, 256, bidirectional=True, batch_first=True)
        self.ner_lstm.load_state_dict(state["ner_lstm"])
        self.ner_proj = nn.Linear(512, H)
        self.ner_proj.load_state_dict(state["ner_proj"])
        n_ner = state["ner_head.4.bias"].shape[0] if "ner_head.4.bias" in state \
            else state["ner_head"]["4.bias"].shape[0]
        self.ner_head = nn.Sequential(
            nn.LayerNorm(H + len(UPOS_TAGS)), nn.Linear(H + len(UPOS_TAGS), H),
            nn.GELU(), nn.Dropout(0.1), nn.Linear(H, n_ner))
        self.ner_head.load_state_dict(state["ner_head"])

        self.dep_sm = ScalarMix(NL)
        self.dep_sm.load_state_dict(state["dep_scalar_mix"])
        in_dim = H + len(UPOS_TAGS) + n_ner
        self.dep_proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, H), nn.GELU())
        self.dep_proj.load_state_dict(state["dep_proj"])
        self.dep_biaffine = BiaffineDEPHead(H, num_labels=len(DEPRELS))
        self.dep_biaffine.load_state_dict(state["dep_biaffine"])

        self.pred_embedding = nn.Embedding(2, H)
        self.pred_embedding.load_state_dict(state["pred_embedding"])
        self.srl_classifier = nn.Sequential(
            nn.Dropout(0.1), nn.Linear(H, H), nn.GELU(),
            nn.Dropout(0.1), nn.Linear(H, len(SRL_TAGS)))
        self.srl_classifier.load_state_dict(state["classifier"])

        self.encoder = enc
        for m in (enc, self.pos_sm, self.pos_head, self.ner_sm, self.ner_lstm,
                  self.ner_proj, self.ner_head, self.dep_sm, self.dep_proj,
                  self.dep_biaffine, self.pred_embedding, self.srl_classifier):
            m.float().to(self.device).eval()
        self._loaded = True
        print(f"  [kniv-v5] teacher loaded on {self.device}", flush=True)

    @torch.no_grad()
    def predict_batch(self, layer: str, batch: list[list[str]]):
        """Encode a batch in one pass, then decode each item.

        Batch size 1 leaves the accelerator idle: measured on this machine,
        deberta-v3-large runs 22.4 sentences/s at batch 1 on MPS and 116.9
        at batch 64, against 3.7 on CPU. The decode stays per item because
        Viterbi and the biaffine head are sequential, but the encoder —
        the dominant cost — is shared.
        """
        self._load()
        enc = self.tok(batch, is_split_into_words=True, truncation=True,
                       max_length=self.max_length, padding=True,
                       return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out = self.encoder(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True)
        hs_all = list(out.hidden_states)

        results = []
        for bi, tokens in enumerate(batch):
            hs = [h[bi:bi + 1] for h in hs_all]
            w2t, prev = {}, None
            for k, wid in enumerate(enc.word_ids(batch_index=bi)):
                if wid is not None and wid != prev:
                    w2t[wid] = k
                prev = wid
            results.append(self._decode(layer, tokens, hs, w2t))
        return results

    @torch.no_grad()
    def _predict(self, layer: str, tokens: list[str]):
        """Single-item path, kept so the bake-off numbers stay reproducible."""
        self._load()
        enc = self.tok(tokens, is_split_into_words=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)
        out = self.encoder(input_ids=ids, attention_mask=mask,
                           output_hidden_states=True)
        hs = list(out.hidden_states)
        w2t, prev = {}, None
        for k, wid in enumerate(enc.word_ids()):
            if wid is not None and wid != prev:
                w2t[wid] = k
            prev = wid
        return self._decode(layer, tokens, hs, w2t)

    @torch.no_grad()
    def _decode(self, layer: str, tokens: list[str], hs, w2t):
        """Head decode for one item, given its encoder hidden states."""
        pos_logits = self.pos_head(self.pos_sm(hs))
        pos_p = torch.softmax(pos_logits, -1)

        if layer == "pos":
            idx = pos_logits[0].argmax(-1).tolist()
            return [UPOS_TAGS[idx[w2t[i]]] if i in w2t else "X"
                    for i in range(len(tokens))]

        ner_h = self.ner_sm(hs)
        lo, _ = self.ner_lstm(ner_h)
        ner_logits = self.ner_head(torch.cat([self.ner_proj(lo) + ner_h, pos_p], -1))
        ner_p = torch.softmax(ner_logits, -1)

        if layer == "ner":
            idxs = [w2t[i] for i in range(len(tokens)) if i in w2t]
            path = viterbi_decode(ner_logits[0, idxs].cpu(),
                                  {t: i for i, t in enumerate(NER_LABELS)})
            out, k = [], 0
            for i in range(len(tokens)):
                if i in w2t:
                    out.append(NER_LABELS[path[k]])
                    k += 1
                else:
                    out.append("O")
            return out
        dep_h = self.dep_sm(hs)
        arc, lab = self.dep_biaffine(
            self.dep_proj(torch.cat([dep_h, pos_p, ner_p], -1)))

        # Restrict arc selection to real word positions, plus a root slot.
        #
        # Vectorised: the original scored every (token, candidate-head) pair
        # with a separate `.item()`, which is an O(n^2) round trip to the
        # accelerator per sentence and made DEP the slowest layer by far.
        # Identical results, one gather and one argmax.
        words = [i for i in range(len(tokens)) if i in w2t]
        heads = [0] * len(tokens)
        rels = ["dep"] * len(tokens)
        if words:
            ti = torch.tensor([w2t[i] for i in words], device=arc.device)
            # candidate heads: the root slot (self position) then every word
            cand_t = torch.cat([ti.new_tensor([-1]), ti])       # -1 = root marker
            cand_ids = [0] + [j + 1 for j in words]
            sub = arc[0].index_select(0, ti)                    # [W, T]
            # root score for token i is arc[ti, ti]; others are arc[ti, tj]
            root_s = sub.gather(1, ti.unsqueeze(1))             # [W, 1]
            other_s = sub.index_select(1, ti)                   # [W, W]
            scores = torch.cat([root_s, other_s], dim=1)        # [W, 1+W]
            best = scores.argmax(dim=1).tolist()
            lab_sel = lab[0].index_select(0, ti)                # [W, T, R]
            for k, i in enumerate(words):
                h_id = cand_ids[best[k]]
                heads[i] = h_id
                th = w2t.get(h_id - 1, w2t[i]) if h_id > 0 else w2t[i]
                rels[i] = DEPRELS[int(lab_sel[k, th].argmax())]
        return {"heads": heads, "rels": rels}

    @torch.no_grad()
    def _predict_srl(self, tokens: list[str], predicate_idx: int) -> list[str]:
        """Predicate-conditioned pass: the marker is injected at the embedding
        layer so every attention layer sees which token is the predicate."""
        enc = self.tok(tokens, is_split_into_words=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        ids = enc["input_ids"].to(self.device)
        mask = enc["attention_mask"].to(self.device)

        w2t, prev = {}, None
        for k, wid in enumerate(enc.word_ids()):
            if wid is not None and wid != prev:
                w2t[wid] = k
            prev = wid

        emb = self.encoder.embeddings(ids)
        indicator = torch.zeros_like(ids)
        indicator[0, w2t.get(predicate_idx, 0)] = 1
        emb = emb + self.pred_embedding(indicator)
        hidden = self.encoder.encoder(emb, mask).last_hidden_state
        logits = self.srl_classifier(hidden)

        idxs = [w2t[i] for i in range(len(tokens)) if i in w2t]
        path = viterbi_decode(logits[0, idxs].cpu(),
                              {t: i for i, t in enumerate(SRL_TAGS)})
        out, k = [], 0
        for i in range(len(tokens)):
            if i in w2t:
                out.append(SRL_TAGS[path[k]])
                k += 1
            else:
                out.append("O")
        return out

    async def annotate_and_cache(self, layer: str, item) -> AnnotationResult:
        if layer not in ("pos", "ner", "dep", "srl"):
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=False,
                error=f"v5 has no {layer} head", error_kind="unsupported")
        cached = self.cache.get(self.name, layer, item.id)
        if cached is not None:
            p = cached["payload"]
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=True,
                payload=p, cached=True,
                well_formed=tree_is_wellformed(p["heads"]) if layer == "dep" else None)
        if not self._loaded:
            self._load()
        t0 = time.time()
        try:
            if layer == "srl":
                payload = self._predict_srl(item.tokens, item.predicate_idx or 0)
            else:
                payload = self._predict(layer, item.tokens)
        except Exception as exc:                                # noqa: BLE001
            return AnnotationResult(
                item_id=item.id, layer=layer, annotator=self.name, ok=False,
                error=f"{type(exc).__name__}: {exc}", error_kind="api",
                latency_ms=(time.time() - t0) * 1000)
        self.cache.put(self.name, layer, item.id,
                       {"payload": payload, "repairs": 0})
        return AnnotationResult(
            item_id=item.id, layer=layer, annotator=self.name, ok=True,
            payload=payload, latency_ms=(time.time() - t0) * 1000,
            well_formed=tree_is_wellformed(payload["heads"]) if layer == "dep" else None)
