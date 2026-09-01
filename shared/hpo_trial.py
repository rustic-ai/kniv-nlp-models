"""Hyperparameter trial interface for student cascade training.

Defines the contract between the *thing that trains and evaluates a student*
(the trial runner) and the *thing that picks hyperparameters* (Optuna, a grid
search, manual scripting). The runner doesn't need to know about Optuna; the
sweep driver doesn't need to know about training.

The interface:

    runner: Callable[[TrialParams], TrialResult]

A `TrialRunner` takes a `TrialParams` (one sampled point in the search space)
and returns a `TrialResult` (per-head benchmark scores). The sweep driver
turns `TrialResult` into a single scalar via `TrialResult.aggregate()` and
hands it back to Optuna.

This module is intentionally **dependency-free** — no optuna, no torch — so
it can be imported from Colab notebooks where the actual training loop
lives, as well as from local CLI scripts.

Usage from Colab:

    from shared.hpo_trial import TrialParams, TrialResult

    def my_runner(params: TrialParams) -> TrialResult:
        # ... train one student with these params, eval, return scores
        return TrialResult(pos=..., ner=..., dep_uas=..., ...)

Usage from a sweep driver (see ``scripts/optuna_sweep_student.py``):

    import optuna
    from shared.hpo_trial import sample_params, default_aggregate

    def objective(trial):
        params = sample_params(trial)
        result = my_runner(params)
        return default_aggregate(result)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Protocol


# ── Search space ─────────────────────────────────────────────────────

@dataclass
class TrialParams:
    """One sampled point in the student-training hyperparameter space.

    Defaults match the values currently hardcoded in
    ``models/kniv-deberta-nlp-base-en-large/train_student.py`` so an
    "all-defaults" trial reproduces the existing run.
    """

    # Optimizer
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # Distillation
    temperature: float = 3.0
    alpha: float = 0.7  # weight on distillation loss vs hard labels
    alpha_hard: float = 0.5  # multiplier on the (1 - alpha) hard-label term

    # Per-head loss weights (cascade combined loss)
    loss_w_pos: float = 1.0
    loss_w_ner: float = 1.5
    loss_w_dep: float = 1.5  # DEP arc CE
    loss_w_dep_rel: float = 0.5  # DEP relation CE (gather at predicted head)
    loss_w_srl: float = 1.0
    loss_w_cls: float = 1.0

    # Hidden-state distillation (PKD-style, layer-norm'd MSE)
    use_pkd: bool = True
    gamma_hidden: float = 0.3  # global PKD weight
    hidden_mult_pred: float = 0.5  # extra weight on predicate-aware hidden
    pkd_layers: tuple[int, ...] = (12, 18, 24)  # teacher layers stored in shards

    # R-Drop consistency regularization
    use_rdrop: bool = True
    rdrop_gamma: float = 0.2  # weight on symmetric KL between two forward passes

    # Regularization
    dropout: float = 0.1

    # Three-stage schedule (see Pattern C in shared/student_train.py)
    epochs: int = 8       # Stage 1 (joint distillation)
    epochs_s2a: int = 8   # Stage 2a (SRL fine-tune, encoder + non-CLS heads trainable)
    epochs_s2b: int = 4   # Stage 2b (CLS-only, encoder + other heads frozen)
    lr_encoder_s2a: float = 3e-6  # very small encoder LR in Stage 2a
    lr_cls_s2b: float = 5e-4
    batch_size: int = 32

    # Free-form: anything else the trial runner wants to thread through
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrialParams":
        known = {f.name for f in fields(cls) if f.name != "extras"}
        kw = {k: v for k, v in d.items() if k in known}
        extras = dict(d.get("extras") or {})
        for k, v in d.items():
            if k not in known and k != "extras":
                extras[k] = v
        return cls(**kw, extras=extras)


# ── Result ───────────────────────────────────────────────────────────

@dataclass
class TrialResult:
    """Per-head benchmark scores from one trained student.

    All scores are 0..1. `dep_uas` is the primary DEP metric (LAS is
    secondary). Use ``aggregate()`` to collapse to a single scalar.
    """

    pos: float
    ner: float
    dep_uas: float
    dep_las: float
    srl: float
    cls: float

    # Optional diagnostics — runner may populate these for the sweep DB
    train_loss: float | None = None
    epoch: int | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Default scoring weights — equal across heads. Tune these if the sweep should
# prioritize a specific head (e.g., bumping `srl` if the goal is closing the
# SRL gap to the teacher).
DEFAULT_AGGREGATE_WEIGHTS = {
    "pos": 1.0,
    "ner": 1.0,
    "dep_uas": 1.0,
    "srl": 1.0,
    "cls": 1.0,
}


def default_aggregate(
    result: TrialResult,
    weights: dict[str, float] | None = None,
) -> float:
    """Collapse per-head scores to one scalar for Optuna.

    Weighted arithmetic mean — simple and interpretable. If a head failed
    to train (score = 0.0) it pulls the average down proportionally to its
    weight, which is the right behavior: a trial that nukes one head is
    worse than a trial that's mediocre across all heads.
    """
    w = weights or DEFAULT_AGGREGATE_WEIGHTS
    total = (
        w.get("pos", 0.0) * result.pos
        + w.get("ner", 0.0) * result.ner
        + w.get("dep_uas", 0.0) * result.dep_uas
        + w.get("srl", 0.0) * result.srl
        + w.get("cls", 0.0) * result.cls
    )
    denom = sum(w.values())
    return total / denom if denom > 0 else 0.0


# ── Runner protocol ──────────────────────────────────────────────────

class TrialRunner(Protocol):
    """Anything that turns ``TrialParams`` into ``TrialResult``."""

    def __call__(self, params: TrialParams) -> TrialResult: ...


# ── Optuna search space helper ───────────────────────────────────────
#
# This function takes an `optuna.Trial` and samples a `TrialParams`. We
# import optuna lazily so this module stays importable in environments that
# don't have optuna (e.g., Colab notebooks running a single trial).

def sample_params(trial: Any, fixed: dict[str, Any] | None = None) -> TrialParams:
    """Sample ``TrialParams`` from an Optuna trial.

    Args:
        trial: An ``optuna.Trial`` (passed as ``Any`` so this module
            doesn't import optuna).
        fixed: Keys to pin to a specific value instead of sampling.
            Useful when narrowing the search around a known-good region.

    Default ranges below were chosen from prior manual ablations:
      - encoder LR: 5e-6 .. 5e-5 (log) — typical fine-tune band
      - head LR: 3e-4 .. 3e-3 (log)   — heads can take much bigger steps
      - temperature: 2 .. 6           — distillation softness
      - alpha: 0.3 .. 0.9             — distill vs hard mix
      - loss weights: 0.5 .. 3.0      — for the four cascade-tuning heads
        (POS weight pinned to 1.0 as the reference; others tune relative to it)
      - dropout: 0.05 .. 0.25
      - rdrop_gamma: 0.0 .. 1.5 (0 disables it)
    """
    fixed = fixed or {}

    def pick(name, sampler):
        return fixed[name] if name in fixed else sampler()

    return TrialParams(
        lr_encoder=pick("lr_encoder",
            lambda: trial.suggest_float("lr_encoder", 5e-6, 5e-5, log=True)),
        lr_heads=pick("lr_heads",
            lambda: trial.suggest_float("lr_heads", 3e-4, 3e-3, log=True)),
        weight_decay=pick("weight_decay",
            lambda: trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True)),
        warmup_ratio=pick("warmup_ratio",
            lambda: trial.suggest_float("warmup_ratio", 0.05, 0.2)),

        temperature=pick("temperature",
            lambda: trial.suggest_float("temperature", 2.0, 6.0)),
        alpha=pick("alpha",
            lambda: trial.suggest_float("alpha", 0.3, 0.9)),
        alpha_hard=pick("alpha_hard",
            lambda: trial.suggest_float("alpha_hard", 0.2, 1.0)),

        loss_w_pos=pick("loss_w_pos", lambda: 1.0),  # reference, not searched
        loss_w_ner=pick("loss_w_ner",
            lambda: trial.suggest_float("loss_w_ner", 0.5, 3.0)),
        loss_w_dep=pick("loss_w_dep",
            lambda: trial.suggest_float("loss_w_dep", 0.5, 3.0)),
        loss_w_dep_rel=pick("loss_w_dep_rel",
            lambda: trial.suggest_float("loss_w_dep_rel", 0.1, 1.5)),
        loss_w_srl=pick("loss_w_srl",
            lambda: trial.suggest_float("loss_w_srl", 0.5, 3.0)),
        loss_w_cls=pick("loss_w_cls",
            lambda: trial.suggest_float("loss_w_cls", 0.5, 3.0)),

        # PKD + R-Drop are gated by booleans the sweep typically pins.
        use_pkd=pick("use_pkd", lambda: True),
        gamma_hidden=pick("gamma_hidden",
            lambda: trial.suggest_float("gamma_hidden", 0.0, 1.0)),
        hidden_mult_pred=pick("hidden_mult_pred",
            lambda: trial.suggest_float("hidden_mult_pred", 0.0, 1.5)),
        use_rdrop=pick("use_rdrop", lambda: True),
        rdrop_gamma=pick("rdrop_gamma",
            lambda: trial.suggest_float("rdrop_gamma", 0.0, 1.0)),

        dropout=pick("dropout",
            lambda: trial.suggest_float("dropout", 0.05, 0.25)),

        epochs=pick("epochs", lambda: 2),  # sweep epoch budget — pin externally
        epochs_s2a=pick("epochs_s2a", lambda: 0),  # 0 = skip Stage 2a in sweeps
        epochs_s2b=pick("epochs_s2b", lambda: 0),  # 0 = skip Stage 2b in sweeps
        lr_encoder_s2a=pick("lr_encoder_s2a", lambda: 3e-6),
        lr_cls_s2b=pick("lr_cls_s2b", lambda: 5e-4),
        batch_size=pick("batch_size", lambda: 32),
    )


# ── Reference dummy runner — for testing the sweep mechanics ─────────

def dummy_runner(params: TrialParams) -> TrialResult:
    """Cheap synthetic runner. Returns plausible scores derived from params.

    Use this to verify the sweep driver, study DB, and aggregation logic
    without spending GPU time. Not a real surrogate model — the score
    surface is deliberately simple but non-monotonic so Optuna has
    something to learn.
    """
    import math
    # Plausible "best" point near the manual defaults
    def gauss(x, mu, sigma):
        return math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    base = (
        gauss(math.log10(params.lr_encoder), math.log10(2e-5), 0.4) *
        gauss(math.log10(params.lr_heads), math.log10(1e-3), 0.5) *
        gauss(params.temperature, 3.0, 1.5) *
        gauss(params.alpha, 0.7, 0.2) *
        gauss(params.dropout, 0.1, 0.08)
    )
    # Per-head responses biased differently so head weights matter
    pos = 0.95 + 0.03 * base
    ner = 0.75 + 0.07 * base * gauss(params.loss_w_ner, 1.5, 0.7)
    dep_uas = 0.88 + 0.06 * base * gauss(params.loss_w_dep, 1.5, 0.7)
    srl = 0.78 + 0.07 * base * gauss(params.loss_w_srl, 2.0, 0.8)
    cls = 0.85 + 0.10 * base * gauss(params.loss_w_cls, 1.0, 0.6)
    return TrialResult(
        pos=pos, ner=ner, dep_uas=dep_uas, dep_las=dep_uas - 0.02,
        srl=srl, cls=cls,
        notes="dummy_runner — no actual training",
    )


__all__ = [
    "TrialParams",
    "TrialResult",
    "TrialRunner",
    "sample_params",
    "default_aggregate",
    "DEFAULT_AGGREGATE_WEIGHTS",
    "dummy_runner",
]
