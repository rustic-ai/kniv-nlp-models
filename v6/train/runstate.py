"""Run state for a v6 training job: resume, status, events, checkpoints.

Generalised from ``v6/experiments/atlop_runstate.py`` with what TRAINING.md
§7 adds: RNG state in ``latest.pt`` so a resumed run is the same sample
order, a ``best.metric`` beside ``best.pt`` so a checkpoint is never replaced
by a worse one, memory instrumentation that sees swap, and an optional
private HF repo as the durable copy.

``latest.pt``    model, optimizer, scheduler, step, epoch, best, RNG state
``best.pt``      weights only, at the best dev composite so far
``best.metric``  the score ``best.pt`` was selected on
``status.json``  rewritten atomically every interval; a heartbeat older than
                 15 minutes means dead, not slow
``events.jsonl`` append-only: checkpoints, evals, signals, errors

SIGTERM and SIGINT set a flag; the loop checkpoints at the next step
boundary and exits, so a preemption costs at most one save interval.
"""
from __future__ import annotations

import json
import os
import random
import resource
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).resolve().parents[2]
                              ).stdout.strip() or "unknown"
    except Exception:                                       # noqa: BLE001
        return "unknown"


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def memory() -> dict:
    """GPU allocated/reserved/peak and host RSS + swap, in GB.

    RSS alone is not enough: ATLOP stalled for eleven hours holding ~47 GB
    while reporting 108 MB RSS, because RSS does not count swapped pages.
    """
    import torch
    out = {}
    if torch.cuda.is_available():
        g = 1 << 30
        out.update(gpu_alloc=round(torch.cuda.memory_allocated() / g, 2),
                   gpu_reserved=round(torch.cuda.memory_reserved() / g, 2),
                   gpu_peak=round(torch.cuda.max_memory_allocated() / g, 2))
    try:
        import psutil
        p = psutil.Process()
        out["rss"] = round(p.memory_info().rss / (1 << 30), 2)
        out["swap_used_system"] = round(psutil.swap_memory().used / (1 << 30), 2)
    except ImportError:
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["rss_peak"] = round(kb / (1 << 30 if sys.platform == "darwin" else 1 << 20), 2)
    return out


@dataclass
class Progress:
    run_id: str
    state: str = "starting"       # starting|featurising|training|evaluating|done|interrupted|failed
    epoch: int = 0
    step: int = 0
    total_steps: int = 0
    loss: float | None = None
    head_loss: dict = field(default_factory=dict)
    best: float = -1.0
    best_step: int = -1
    last_eval: dict = field(default_factory=dict)
    steps_per_sec: float | None = None
    eta_minutes: float | None = None
    started_at: float = 0.0
    heartbeat: float = 0.0
    elapsed_minutes: float = 0.0
    memory: dict = field(default_factory=dict)
    resumed_from_step: int = 0
    meta: dict = field(default_factory=dict)


class Hub:
    """Mirror run files to ``<repo>/<run_name>/`` on a private HF repo.

    Uploads run on one background thread so a multi-GB ``latest.pt`` does not
    stall training; a new upload of the same file waits for the previous one.
    """

    def __init__(self, repo: str, run_name: str):
        from huggingface_hub import HfApi
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()           # they would flood train.log
        self.api, self.repo, self.prefix = HfApi(), repo, run_name
        self.api.create_repo(repo, private=True, exist_ok=True)
        self._busy: dict[str, threading.Thread] = {}

    def _path(self, name: str) -> str:
        return f"{self.prefix}/{name}"

    def push(self, *files: Path, wait: bool = False, on_error=None) -> None:
        """Upload ``files`` in order on one background thread.

        Order matters for ``best.pt`` then ``best.metric``: uploaded in
        parallel, a VM lost between the two could leave a remote score
        describing a checkpoint that never arrived, and a genuinely better
        model would later be refused against it.
        """
        key = files[0].name
        prev = self._busy.get(key)
        if prev and prev.is_alive():
            prev.join()

        def run():
            for local in files:
                try:
                    self.api.upload_file(path_or_fileobj=str(local),
                                         path_in_repo=self._path(local.name),
                                         repo_id=self.repo)
                except Exception as exc:                    # noqa: BLE001
                    if on_error:
                        on_error(local.name, exc)
                    return                  # never push a score without its file
        t = threading.Thread(target=run, daemon=False)
        t.start()
        self._busy[key] = t
        if wait:
            t.join()

    def pull(self, name: str, dest: Path) -> bool:
        """Download to a temp path and rename, so a half-pulled file is never
        mistaken for a whole one."""
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            got = hf_hub_download(self.repo, self._path(name),
                                  local_dir=str(dest.parent / ".hub"))
        except EntryNotFoundError:
            return False
        tmp = dest.with_suffix(dest.suffix + ".pulling")
        os.replace(got, tmp)
        tmp.replace(dest)
        return True

    def remote_best(self) -> float | None:
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "best.metric"
            if self.pull("best.metric", p):
                return float(json.loads(p.read_text())["value"])
        return None

    def wait(self) -> None:
        for t in self._busy.values():
            t.join()


class RunState:
    def __init__(self, out_dir: Path, run_id: str, meta: dict,
                 hub: Hub | None = None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.latest, self.best = self.dir / "latest.pt", self.dir / "best.pt"
        self.best_metric = self.dir / "best.metric"
        self.status_path, self.events_path = self.dir / "status.json", self.dir / "events.jsonl"
        self.hub = hub
        import torch, transformers
        self.p = Progress(run_id=run_id, started_at=time.time(), meta={
            **meta, "git_sha": _git_sha(), "torch": torch.__version__,
            "transformers": transformers.__version__})
        self._t0, self._step0 = time.time(), 0
        self.stop_requested = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                pass

    def _on_signal(self, signum, _frame):
        # Never checkpoint inside the handler: tensors may be mid-update.
        self.stop_requested = True
        self.event("signal", signal=int(signum),
                   note="checkpoint at next step boundary, then exit")

    def event(self, kind: str, **fields) -> None:
        rec = {"ts": time.time(),
               "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "run_id": self.p.run_id, "kind": kind, **fields}
        with self.events_path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def _hub_error(self, name, exc):
        self.event("hub_error", file=name, error=f"{type(exc).__name__}: {exc}")
        print(f"  HUB UPLOAD FAILED for {name}: {exc}", flush=True)

    def status(self, push: bool = False, **updates) -> None:
        for k, v in updates.items():
            setattr(self.p, k, v)
        now = time.time()
        self.p.heartbeat = now
        self.p.elapsed_minutes = round((now - self.p.started_at) / 60, 2)
        self.p.memory = memory()
        done = self.p.step - self._step0
        if done > 0:
            rate = done / max(now - self._t0, 1e-6)
            self.p.steps_per_sec = round(rate, 4)
            self.p.eta_minutes = round(max(self.p.total_steps - self.p.step, 0)
                                       / rate / 60, 1)
        atomic_write_json(self.status_path, asdict(self.p))
        if push and self.hub:
            self.hub.push(self.status_path, on_error=self._hub_error)

    # ── checkpoints ───────────────────────────────────────────────
    def save_latest(self, model, optimizer, scheduler, extra: dict) -> None:
        import numpy as np
        import torch
        tmp = self.latest.with_suffix(".pt.tmp")
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": self.p.epoch, "step": self.p.step,
            "best": self.p.best, "best_step": self.p.best_step,
            "run_id": self.p.run_id, "meta": self.p.meta,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": (torch.cuda.get_rng_state_all()
                             if torch.cuda.is_available() else None),
                    # Dropout on Apple GPUs draws from the MPS generator.
                    "mps": (torch.mps.get_rng_state()
                            if torch.backends.mps.is_available() else None)},
            **extra,
        }, tmp)
        tmp.replace(self.latest)
        self.event("checkpoint", step=self.p.step, epoch=self.p.epoch)
        if self.hub:
            self.hub.push(self.latest, on_error=self._hub_error)

    def maybe_save_best(self, model, value: float, extra: dict) -> bool:
        """Save ``best.pt`` only if ``value`` beats both the local and the
        remote ``best.metric``. A puller once replaced a 0.7161 checkpoint
        with a 0.6278 one; the comparison is what prevents that."""
        import torch
        local = (float(json.loads(self.best_metric.read_text())["value"])
                 if self.best_metric.exists() else None)
        remote = self.hub.remote_best() if self.hub else None
        bar = max([x for x in (local, remote) if x is not None], default=None)
        if bar is not None and value <= bar:
            return False
        tmp = self.best.with_suffix(".pt.tmp")
        torch.save({"model": model.state_dict(), **extra}, tmp)
        tmp.replace(self.best)
        atomic_write_json(self.best_metric, {"value": value, "step": self.p.step,
                                             "metric": "dev composite"})
        self.p.best, self.p.best_step = value, self.p.step
        self.event("best", value=value, step=self.p.step)
        if self.hub:
            self.hub.push(self.best, self.best_metric, on_error=self._hub_error)
        return True

    def try_resume(self, model, optimizer, scheduler) -> dict | None:
        """Restore from ``latest.pt``, pulling it from the hub if absent locally."""
        import numpy as np
        import torch
        if not self.latest.exists() and self.hub:
            if self.hub.pull("latest.pt", self.latest):
                self.event("pulled", file="latest.pt")
            if not self.best_metric.exists():
                self.hub.pull("best.metric", self.best_metric)
        if not self.latest.exists():
            self.event("resume", found=False)
            return None
        ck = torch.load(self.latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        r = ck["rng"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(r["cuda"])
        if r.get("mps") is not None and torch.backends.mps.is_available():
            torch.mps.set_rng_state(r["mps"])
        self.p.epoch, self.p.step = ck["epoch"], ck["step"]
        self.p.best, self.p.best_step = ck["best"], ck["best_step"]
        self.p.resumed_from_step = self._step0 = ck["step"]
        self.event("resume", found=True, step=ck["step"], epoch=ck["epoch"],
                   best=ck["best"], previous_run=ck.get("run_id"))
        print(f"resumed from step {ck['step']} (epoch {ck['epoch']}, "
              f"best {ck['best']:.4f})", flush=True)
        return ck
