"""Optuna sweep driver for the student cascade hyperparameter search.

Decoupled from training: this script knows about Optuna and search space
sampling; the actual training+eval lives in a `TrialRunner` that the user
plugs in (see ``shared/hpo_trial.py``). That keeps the sweep usable from
both Colab notebooks (where most student training happens) and local CLI.

Quick start — verify the sweep mechanics with the dummy runner:

    poetry run python scripts/optuna_sweep_student.py \\
        --runner shared.hpo_trial:dummy_runner \\
        --n-trials 50 \\
        --study-name dummy

The dummy runner returns synthetic scores in milliseconds; useful for
validating the search space, pruner, and study DB before burning GPU.

Real run — point at your trial function (here `my_module:train_one`):

    poetry run python scripts/optuna_sweep_student.py \\
        --runner my_module:train_one \\
        --n-trials 30 \\
        --storage sqlite:///runs/optuna_student.db \\
        --study-name student_xsmall_v1

The runner string is `module.path:callable_name`. The callable must match
``shared.hpo_trial.TrialRunner`` — i.e., take a ``TrialParams`` and return
a ``TrialResult``.

Resumability: studies persist in the SQLite DB at ``--storage``. Re-running
with the same ``--study-name`` continues the existing study.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from shared.hpo_trial import (  # noqa: E402
    TrialParams, TrialResult, sample_params, default_aggregate,
)


def load_runner(spec: str):
    """Resolve a `module.path:callable` string to the actual callable."""
    if ":" not in spec:
        raise ValueError(
            f"--runner must be 'module.path:callable_name', got: {spec}"
        )
    mod_path, attr = spec.split(":", 1)
    module = importlib.import_module(mod_path)
    runner = getattr(module, attr)
    if not callable(runner):
        raise TypeError(f"{spec} is not callable")
    return runner


def parse_fixed(items: list[str] | None) -> dict[str, float]:
    """Parse `--fix key=value` pairs into a dict, coercing to float when possible."""
    out: dict[str, float] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--fix expects key=value, got: {item}")
        k, v = item.split("=", 1)
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v  # leave as str if not numeric (e.g., epochs as int → handled below)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", required=True,
                    help="Trial runner as 'module.path:callable'")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--timeout-sec", type=int, default=None,
                    help="Stop the study after this many seconds")
    ap.add_argument("--storage", default=None,
                    help="Optuna storage URL (e.g., sqlite:///path.db). "
                         "If unset, an in-memory study is used (not resumable).")
    ap.add_argument("--study-name", default="student_hpo")
    ap.add_argument("--direction", default="maximize", choices=["maximize", "minimize"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fix", action="append",
                    help="Pin a hyperparameter: --fix key=value (repeatable)")
    ap.add_argument("--dump-best", default=None,
                    help="After the sweep, write best params + score to this JSON path")
    ap.add_argument("--pruner", default="median",
                    choices=["none", "median", "hyperband"])
    args = ap.parse_args()

    try:
        import optuna
    except ImportError:
        print("optuna is not installed. Add to dev deps:", file=sys.stderr)
        print("  uv pip install optuna", file=sys.stderr)
        sys.exit(2)

    runner = load_runner(args.runner)
    fixed = parse_fixed(args.fix)
    if "epochs" in fixed:
        fixed["epochs"] = int(fixed["epochs"])
    if "batch_size" in fixed:
        fixed["batch_size"] = int(fixed["batch_size"])
    print(f"Runner: {args.runner}")
    print(f"Pinned: {fixed or '(none)'}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    if args.pruner == "median":
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    elif args.pruner == "hyperband":
        pruner = optuna.pruners.HyperbandPruner()
    else:
        pruner = optuna.pruners.NopPruner()

    study = optuna.create_study(
        study_name=args.study_name,
        direction=args.direction,
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    def objective(trial: "optuna.Trial") -> float:
        params: TrialParams = sample_params(trial, fixed=fixed)
        # Log the sampled point for downstream diagnosis (visible in optuna-dashboard)
        for k, v in params.to_dict().items():
            if k != "extras":
                trial.set_user_attr(f"param/{k}", v)

        result: TrialResult = runner(params)

        # Persist per-head scores so we can analyze the Pareto front later
        for head in ("pos", "ner", "dep_uas", "dep_las", "srl", "cls"):
            trial.set_user_attr(f"score/{head}", getattr(result, head))
        if result.notes:
            trial.set_user_attr("notes", result.notes)

        return default_aggregate(result)

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout_sec,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    print()
    print("=" * 60)
    print(f"Best trial #{study.best_trial.number}: score={study.best_value:.4f}")
    print("=" * 60)
    for k, v in study.best_trial.params.items():
        print(f"  {k:>16} = {v}")
    print("\nPer-head scores at best trial:")
    for head in ("pos", "ner", "dep_uas", "dep_las", "srl", "cls"):
        s = study.best_trial.user_attrs.get(f"score/{head}")
        if s is not None:
            print(f"  {head:>8}: {s:.4f}")

    if args.dump_best:
        Path(args.dump_best).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_best).write_text(json.dumps({
            "study_name": args.study_name,
            "n_trials": len(study.trials),
            "best_value": study.best_value,
            "best_params": study.best_trial.params,
            "best_user_attrs": study.best_trial.user_attrs,
        }, indent=2))
        print(f"\nBest trial dumped to {args.dump_best}")


if __name__ == "__main__":
    main()
