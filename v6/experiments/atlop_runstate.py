"""Run state for long training jobs: resume, status, event log, heartbeat.

Written for Colab, where the assumptions are hostile: the VM disconnects
without warning, there is a wall-clock cap, and the filesystem disappears
with the instance. Anything not written to a mounted Drive and not
resumable is work you will do twice.

The design mirrors what the bake-off already relies on — a durable artifact
is the resume point, writes are atomic, and failures are recorded by kind
rather than printed and lost:

``latest.pt``   full training state (model, optimizer, scheduler, scaler,
                epoch, step, best) written every ``save_every`` steps and at
                every epoch boundary. Resume reads this and nothing else.
``best.pt``     model weights only, at the best dev F1 so far.
``status.json`` one small file, rewritten atomically, holding everything a
                human or a poller needs: step, epoch, loss, rate, ETA, best
                F1, and a heartbeat. Readable from another machine while the
                job runs.
``events.jsonl`` append-only history — one line per checkpoint, evaluation,
                signal and error. This is what you read *after* a crash.

A SIGTERM or SIGINT (Colab preemption, "stop" in the UI) is caught and
turned into a clean checkpoint before exit, so a kill costs at most
``save_every`` steps rather than the epoch.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).resolve().parents[2]
                              ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _versions() -> dict:
    out = {"git_sha": _git_sha()}
    for mod in ("torch", "transformers", "numpy"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = "absent"
    return out


def atomic_write_json(path: Path, obj: dict) -> None:
    """Write via a temp file + rename so a reader never sees a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


@dataclass
class Progress:
    run_id: str
    state: str = "starting"          # starting|featurising|training|evaluating|done|interrupted|failed
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float | None = None
    best_f1: float = -1.0
    best_epoch: int = -1
    last_eval: dict = field(default_factory=dict)
    steps_per_sec: float | None = None
    eta_minutes: float | None = None
    started_at: float = 0.0
    heartbeat: float = 0.0
    elapsed_minutes: float = 0.0
    device: str = ""
    resumed_from_step: int = 0
    errors: int = 0
    meta: dict = field(default_factory=dict)


class RunState:
    def __init__(self, out_dir: Path, run_id: str, save_every: int = 200):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.latest = self.dir / "latest.pt"
        self.best = self.dir / "best.pt"
        self.status_path = self.dir / "status.json"
        self.events_path = self.dir / "events.jsonl"
        self.p = Progress(run_id=run_id, started_at=time.time(),
                          meta=_versions())
        self._t0 = time.time()
        self._step0 = 0
        self.stop_requested = False
        self._install_signals()

    # ── signals ──────────────────────────────────────────────────
    def _install_signals(self):
        def handler(signum, _frame):
            # Do not checkpoint inside the handler: the tensors may be
            # mid-update. Raise a flag; the training loop saves at the next
            # safe boundary and exits.
            self.stop_requested = True
            self.event("signal", signal=int(signum),
                       note="checkpoint at next step boundary, then exit")
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass                     # not the main thread; non-fatal

    # ── logging ──────────────────────────────────────────────────
    def event(self, kind: str, **fields) -> None:
        rec = {"ts": time.time(), "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                       time.gmtime()),
               "run_id": self.p.run_id, "kind": kind, **fields}
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def status(self, **updates) -> None:
        for k, v in updates.items():
            setattr(self.p, k, v)
        now = time.time()
        self.p.heartbeat = now
        self.p.elapsed_minutes = (now - self.p.started_at) / 60
        done = self.p.step - self._step0
        if done > 0:
            rate = done / max(now - self._t0, 1e-6)
            self.p.steps_per_sec = round(rate, 4)
            left = max(self.p.total_steps - self.p.step, 0)
            self.p.eta_minutes = round(left / rate / 60, 1) if rate else None
        atomic_write_json(self.status_path, asdict(self.p))

    # ── checkpointing ────────────────────────────────────────────
    def save_latest(self, model, optimizer, scheduler, scaler) -> None:
        import torch
        tmp = self.latest.with_suffix(".pt.tmp")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler else None,
            "epoch": self.p.epoch, "step": self.p.step,
            "best_f1": self.p.best_f1, "best_epoch": self.p.best_epoch,
            "run_id": self.p.run_id, "meta": self.p.meta,
        }, tmp)
        tmp.replace(self.latest)          # atomic: never a half-written resume point
        self.event("checkpoint", step=self.p.step, epoch=self.p.epoch,
                   path=str(self.latest))

    def save_best(self, model, f1: float, epoch: int) -> None:
        import torch
        tmp = self.best.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), tmp)
        tmp.replace(self.best)
        self.p.best_f1, self.p.best_epoch = f1, epoch
        self.event("best", f1=f1, epoch=epoch, path=str(self.best))

    def try_resume(self, model, optimizer, scheduler, scaler) -> tuple[int, int]:
        """Restore from ``latest.pt`` if present. Returns (epoch, step)."""
        import torch
        if not self.latest.exists():
            self.event("resume", found=False)
            return 0, 0
        ck = torch.load(self.latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        self.p.epoch, self.p.step = ck["epoch"], ck["step"]
        self.p.best_f1 = ck.get("best_f1", -1.0)
        self.p.best_epoch = ck.get("best_epoch", -1)
        self.p.resumed_from_step = ck["step"]
        self._step0 = ck["step"]
        self.event("resume", found=True, step=ck["step"], epoch=ck["epoch"],
                   best_f1=self.p.best_f1,
                   previous_run=ck.get("run_id"))
        print(f"resumed from step {ck['step']} (epoch {ck['epoch']}, "
              f"best F1 {self.p.best_f1:.4f})", flush=True)
        return ck["epoch"], ck["step"]
