"""Launch and watch v6 training runs on Colab A100s, one VM per encoder.

    uv run python -m v6.train.colab launch --encoder ettin-1b 
    uv run python -m v6.train.colab status --encoder ettin-1b
    uv run python -m v6.train.colab probe  --encoder ettin-1b      # A100 fit check
    uv run python -m v6.train.colab watch                          # report changes until done
    uv run python -m v6.train.colab stop   --run-name ettin-1b --release

It allocates billable VMs. A run that mirrors to the hub (the default for
``launch``) has your local HuggingFace token copied to its VM as a 0600 file,
so checkpoints can be pushed to the private repo; the token never appears in a
command line or a log. ``probe``, and ``launch --hf-repo ''``, copy no token:
the models are public.

What it does, per encoder (TRAINING.md §7 says why each step is this way):

1. **Bundles the code at a commit**, via ``git archive HEAD`` -- not the
   working tree. A run is reproducible from its sha, and the launcher refuses
   to bundle while ``v6/`` has uncommitted changes.
2. **Bundles the corpus** (parquet, ``vocab.json``, ``MANIFEST.json``). It is
   local-only and gitignored, so the VM cannot fetch it from anywhere else.
3. **Allocates** ``colab new --gpu A100`` and checks the GPU really is one.
4. **Starts training inside the kernel**, not as a detached process: a
   detached process leaves the kernel IDLE and Colab reclaims the VM within
   the hour. ``colab exec`` returns after a short client-side timeout while
   the kernel keeps running; its output goes to ``train.log`` on the VM.
5. **Resumes** on a re-launch: ``latest.pt`` is pulled from the HF repo when
   the VM is new, so a reclaimed VM costs at most one save interval.

``status`` reads ``status.json`` from the VM (falling back to the HF mirror
when the VM is gone) and reports STALE past 15 minutes: a stale heartbeat
means dead, not slow.

``watch`` polls every run and prints only what changed -- state, a stale
heartbeat, a lost VM, non-finite steps, a new best, low disk or GPU headroom
-- so it can run unattended under a monitor. With ``--relaunch`` it also
restarts a run whose VM died or went stale, up to ``--max-relaunches`` times;
that spends compute units, so it is opt-in. It never relaunches a run that
stopped, finished or failed on its own: those need a person.

``stop`` asks the loop to checkpoint and exit (a ``STOP`` file it checks every
step) instead of tearing the VM down under it; ``--release`` then waits until
the hub shows the run stopped -- which the loop writes only after
``latest.pt`` has uploaded -- and releases the VM. ``checkpoint`` writes
``latest.pt`` now without stopping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

from ..config import DATA_DIR, REPO, RUNS_DIR
from . import encoders

BUNDLE = RUNS_DIR / "colab-bundle"
LAUNCHES = RUNS_DIR / "colab-launches"      # what was launched, for --relaunch
TERMINAL = ("done", "stopped", "failed")
VM_REPO = "/content/kniv"
VM_RUNS = "/content/runs"
STALE_MINUTES = 15
HF_REPO = "dragonscale-ai/kniv-v6-runs"      # private; the durable copy of every run
PIP = ["transformers==5.6.2", "huggingface_hub>=1.0", "psutil", "pyarrow",
       "sentencepiece"]


def session(run_name: str) -> str:
    return f"kniv-v6-{run_name}"


def run_name(a) -> str:
    """One name keys the VM, the output directory and the HF prefix. Two runs
    sharing a name share a resume point: a sweep run would silently resume
    another learning rate's weights. Name every sweep run distinctly."""
    return a.run_name or a.encoder


def sh(cmd: list[str], check: bool = True, timeout: float | None = None,
       quiet: bool = False) -> subprocess.CompletedProcess:
    """Run a command and surface its stderr. Several wrong conclusions this
    cycle came from commands whose failure output was discarded."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    # Drop only the CLI's own version notice; anything else may be the error.
    notice = ("[colab] A new version", "[colab] Run 'colab update'",
              "[colab] Run 'uv tool install", "[colab] You can run 'colab update",
              "[colab] To silence this check")
    noise = [l for l in r.stderr.splitlines() if l.strip() and not l.startswith(notice)]
    if not quiet and r.stdout.strip():
        print(redact(r.stdout.rstrip()))
    if r.returncode and check:
        # The CLI reports some failures on stdout, so show both streams.
        print(redact("\n".join(noise + r.stdout.strip().splitlines())), file=sys.stderr)
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd[:4])} ...")
    return r


def redact(text: str) -> str:
    """Colab error URLs carry a runtime proxy token; never print one."""
    return re.sub(r"(token=)[A-Za-z0-9._-]+", r"\1<redacted>", text)


def colab(*args: str, **kw) -> subprocess.CompletedProcess:
    return sh(["colab", *args], **kw)


CHUNK = 16 << 20     # the VM's file API rejects a 96 MB upload with a 500


def upload_big(sess: str, local: Path, remote: str) -> None:
    """Upload in chunks of at most ``CHUNK`` and reassemble on the VM,
    checking the SHA-256 against the local file."""
    size = local.stat().st_size
    if size <= CHUNK:
        upload(sess, str(local), remote)
        return
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    parts = []
    with tempfile.TemporaryDirectory() as d, local.open("rb") as fh:
        i = 0
        while chunk := fh.read(CHUNK):
            part = Path(d) / f"{local.name}.part{i:03d}"
            part.write_bytes(chunk)
            upload(sess, str(part), f"{remote}.part{i:03d}")
            parts.append(f"{remote}.part{i:03d}")
            i += 1
    print(f"  {local.name}: {len(parts)} chunks uploaded; reassembling on the VM")
    vm_exec(sess, f"""
import hashlib, os
parts = {parts!r}
h = hashlib.sha256()
with open({remote!r}, "wb") as out:
    for p in parts:
        b = open(p, "rb").read(); h.update(b); out.write(b); os.remove(p)
if h.hexdigest() != {digest!r}:
    os.remove({remote!r})
    raise SystemExit("reassembled {local.name} does not match its SHA-256")
print("reassembled {local.name}: sha256 ok")
""", timeout=600)


def upload(sess: str, local: str, remote: str, attempts: int = 4) -> None:
    """Upload with retries: the first upload to a freshly created session can
    fail while the VM's file API is still coming up."""
    for i in range(attempts):
        r = colab("upload", "-s", sess, local, remote, check=False, quiet=True)
        if r.returncode == 0:
            return
        time.sleep(15 * (i + 1))
    colab("upload", "-s", sess, local, remote, quiet=True)   # last try raises


def vm_exec(sess: str, code: str, timeout: float = 600, check: bool = True):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    try:
        return colab("exec", "-s", sess, "-f", f.name, "--timeout", str(timeout),
                     check=check)
    finally:
        os.unlink(f.name)


# ── bundles ─────────────────────────────────────────────────────────
def build_bundles() -> tuple[Path, Path, str]:
    dirty = sh(["git", "-C", str(REPO), "status", "--porcelain", "--", "v6"],
               quiet=True).stdout.strip()
    if dirty:
        raise SystemExit("v6/ has uncommitted changes; commit them so the run "
                         "is reproducible from its sha:\n" + dirty)
    sha = sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
             quiet=True).stdout.strip()
    BUNDLE.mkdir(parents=True, exist_ok=True)
    code = BUNDLE / f"code-{sha}.tar.gz"
    if not code.exists():
        sh(["git", "-C", str(REPO), "archive", "--format=tar.gz", "-o", str(code),
            "HEAD", "v6", "pyproject.toml"], quiet=True)
    corpus = DATA_DIR / "v6-corpus"
    vocab = corpus / "vocab.json"
    if not vocab.exists():
        raise SystemExit("data/v6-corpus/vocab.json missing; run python -m v6.train.vocab")
    stamp = json.loads((corpus / "MANIFEST.json").read_text()).get("git_sha", "x")[:7]
    data = BUNDLE / f"corpus-{stamp}-{int(vocab.stat().st_mtime)}.tar.gz"
    if not data.exists():
        with tarfile.open(data, "w:gz") as t:
            for p in [corpus / "corpus", vocab, corpus / "MANIFEST.json"]:
                t.add(p, arcname=str(p.relative_to(REPO)))
    print(f"bundles: {code.name} ({code.stat().st_size >> 10} KB), "
          f"{data.name} ({data.stat().st_size >> 20} MB)")
    return code, data, sha


# ── VM setup ────────────────────────────────────────────────────────
SETUP = r'''
import os, subprocess, sys, tarfile, json
os.makedirs("{repo}", exist_ok=True)
for b in ("{code}", "{data}"):
    with tarfile.open(b) as t:
        t.extractall("{repo}", filter="data")
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *{pip}],
                   capture_output=True, text=True)
if r.returncode:
    raise SystemExit("pip failed:\n" + r.stderr[-2000:])
import torch
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
if "A100" not in name:
    raise SystemExit(f"expected an A100, got {{name}}")
user = None
if os.path.exists("/content/.hf_token"):        # only when the run mirrors to the hub
    os.chmod("/content/.hf_token", 0o600)
    os.environ["HF_TOKEN"] = open("/content/.hf_token").read().strip()
    from huggingface_hub import whoami
    user = whoami()["name"]
print(json.dumps({{"gpu": name, "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                  "torch": torch.__version__, "hf_user": user}}))
'''

RUN_MARKER = "=== kniv run start"

START = r'''
import os, sys, runpy, threading
os.chdir("{repo}")
sys.path.insert(0, "{repo}")
if os.path.exists("/content/.hf_token"):
    os.environ["HF_TOKEN"] = open("/content/.hf_token").read().strip()
os.makedirs("{out}", exist_ok=True)

# The kernel outlives each run, and so do logging handlers that captured
# sys.stderr during a run. The tee is therefore created once per kernel and
# only its target file changes: a handler holding it writes to the current
# run's train.log, never to a closed file from an earlier one.
# Output goes to the KERNEL's stream, not the process's raw stdout. A
# training cell that writes only to raw stdout emits no kernel messages, and
# Colab reclaimed such VMs about 20 minutes into a run with no client
# attached -- three times in one sweep -- while VMs whose launcher reconnected
# every few minutes survived.
if "_KNIV_TEE" not in globals():
    class _Tee:
        """Copy output to the current train.log. Anything else -- isatty,
        fileno, encoding -- answers as the real stream does."""
        def __init__(self, base): self.base, self.log = base, None
        def write(self, x):
            self.base.write(x)
            if self.log is not None:
                self.log.write(x); self.log.flush()
            return len(x)
        def flush(self):
            self.base.flush()
            if self.log is not None: self.log.flush()
        def __getattr__(self, name): return getattr(self.base, name)
    _KNIV_TEE = (_Tee(sys.stdout), _Tee(sys.stderr))

_log = open("{out}/train.log", "a", buffering=1)
for _t in _KNIV_TEE:
    _t.log = _log
_prev = (sys.stdout, sys.stderr)     # the kernel's own streams, to restore
sys.stdout, sys.stderr = _KNIV_TEE

# Featurising and evaluation can be silent for minutes; a keepalive line to
# the kernel stream (not train.log) every 5 minutes covers them.
import threading as _threading
_alive = _threading.Event()
def _keepalive():
    import time as _t
    while not _alive.wait(300):
        _KNIV_TEE[0].base.write(f"[kniv keepalive {{_t.strftime('%H:%M:%S')}}]\n")
        _KNIV_TEE[0].base.flush()
_threading.Thread(target=_keepalive, daemon=True).start()
# The kernel caches imports. Without this, a run after a code upload imports
# the previous run's v6 modules -- on the A100 it ran a new loop.py against
# an old model.py.
for _m in [m for m in sys.modules if m == "v6" or m.startswith("v6.")]:
    del sys.modules[_m]
import time as _time
print(f"{RUN_MARKER} {{_time.strftime('%Y-%m-%dT%H:%M:%S')}} {argv}")
sys.argv = ["v6.train.loop", *{argv}]
try:
    runpy.run_module("v6.train.loop", run_name="__main__", alter_sys=True)
except SystemExit as e:
    print(f"loop exited: {{e.code}}")
finally:
    _alive.set()
    # Restore the kernel's streams, not the process's raw stdout, or every
    # later cell's output disappears.
    sys.stdout, sys.stderr = _prev
    for _t in _KNIV_TEE:
        _t.log = None
    _log.close()
'''


def _upload_token(sess: str) -> None:
    from huggingface_hub import get_token
    token = get_token()
    if not token:
        raise SystemExit("no local HF token; run `hf auth login` first")
    fd, path = tempfile.mkstemp()
    try:
        os.fchmod(fd, 0o600)
        os.write(fd, token.encode())
        os.close(fd)
        upload(sess, path, "/content/.hf_token")
    finally:
        os.unlink(path)


def prepare(sess: str, token: bool) -> str:
    """Bundle, allocate, upload, set up. ``token``: copy the local HF token
    to the VM -- only for a run that mirrors to the hub; models are public."""
    code, data, sha = build_bundles()
    existing = colab("sessions", check=False, quiet=True).stdout
    if sess not in existing:
        print(f"allocating {sess} (A100)...")
        colab("new", "-s", sess, "--gpu", "A100")
    for b in (code, data):
        upload_big(sess, b, f"/content/{b.name}")
    if token:
        _upload_token(sess)
    print("setting up the VM (pip, GPU check, HF auth)...")
    vm_exec(sess, SETUP.format(repo=VM_REPO, code=f"/content/{code.name}",
                               data=f"/content/{data.name}", pip=PIP), timeout=900)
    return sess


HOLDERS = RUNS_DIR / "colab-holders"
HOLDER_CODE = 'print("kniv holder: the training cell has finished")\n'


def _holder_alive(sess: str) -> bool:
    f = HOLDERS / f"{sess}.pid"
    if not f.exists():
        return False
    try:
        os.kill(int(f.read_text()), 0)
        return True
    except (OSError, ValueError):
        return False


def ensure_holder(sess: str) -> bool:
    """Keep a client attached to the kernel for as long as the run cell runs.

    Colab reclaims a VM once no client has been attached for ~20 minutes,
    however busy its kernel: on the first sweep, six VMs died 22-26 minutes
    into a run with no client attached, kernel output notwithstanding, while
    a VM with one attached ran on. A holder is a ``colab exec`` of a trivial
    cell with a long timeout: it queues behind the training cell, so its
    connection stays open until training ends, then prints and exits.

    It runs on this machine, so this machine must stay awake and online for
    the run; ``watch`` re-attaches one if it drops. Returns True if one was
    started.
    """
    if _holder_alive(sess):
        return False
    HOLDERS.mkdir(parents=True, exist_ok=True)
    code = HOLDERS / "holder.py"
    code.write_text(HOLDER_CODE)
    log = (HOLDERS / f"{sess}.log").open("a")
    proc = subprocess.Popen(["colab", "exec", "-s", sess, "-f", str(code),
                             "--timeout", str(24 * 3600)],
                            stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)     # survives this process
    (HOLDERS / f"{sess}.pid").write_text(str(proc.pid))
    return True


def _kill_holder(sess: str) -> None:
    f = HOLDERS / f"{sess}.pid"
    if f.exists():
        try:
            os.kill(int(f.read_text()), 15)
        except (OSError, ValueError):
            pass
        f.unlink()


def start(sess: str, out: str, argv: list[str], wait_minutes: float = 25) -> bool:
    """Start the loop in the kernel and confirm it is training.

    ``colab status`` is no evidence either way: it reports the CLI's own
    bookkeeping, and showed IDLE while the kernel trained. The evidence is the
    run itself -- a ``status.json`` written after this launch, in state
    training. Featurising takes ~9 minutes on a fresh VM, hence the wait.
    """
    print(f"starting in-kernel: python -m v6.train.loop {shlex.join(argv)}")
    t0 = time.time()
    # The client times out; the kernel keeps running the cell.
    vm_exec(sess, START.format(repo=VM_REPO, out=out, argv=argv, RUN_MARKER=RUN_MARKER),
            timeout=90, check=False)
    # The training cell is now running; attach a holder before Colab's ~20
    # minute no-client clock can run out.
    _kill_holder(sess)           # one left from a previous run would not hold this one
    ensure_holder(sess)
    with tempfile.TemporaryDirectory() as d:
        while time.time() - t0 < wait_minutes * 60:
            r = colab("download", "-s", sess, f"{out}/status.json", f"{d}/s.json",
                      check=False, quiet=True)
            if r.returncode == 0:
                st = json.loads(Path(f"{d}/s.json").read_text())
                if st.get("started_at", 0) >= t0 - 60 and st.get("state") in (
                        "training", "evaluating"):
                    print(f"running: step {st['step']}/{st['total_steps']}, "
                          f"heartbeat {time.time() - st['heartbeat']:.0f}s ago")
                    return True
            r = colab("download", "-s", sess, f"{out}/train.log", f"{d}/t.log",
                      check=False, quiet=True)
            if r.returncode == 0:
                lines = Path(f"{d}/t.log").read_text().splitlines()
                starts = [i for i, l in enumerate(lines) if l.startswith(RUN_MARKER)]
                mine = lines[starts[-1]:] if starts else []
                if any(l.startswith("loop exited") for l in mine):
                    print("the loop exited before training; last lines of train.log:")
                    print(redact("\n".join(mine[-15:])))
                    return False
            time.sleep(30)
    print(f"no heartbeat within {wait_minutes} minutes; inspect train.log:\n"
          f"  colab download -s {sess} {out}/train.log ./train.log")
    return False


# ── commands ────────────────────────────────────────────────────────
def _record(name: str, a, **more) -> dict:
    LAUNCHES.mkdir(parents=True, exist_ok=True)
    f = LAUNCHES / f"{name}.json"
    rec = json.loads(f.read_text()) if f.exists() else {"relaunches": 0}
    rec.update({"encoder": a.encoder, "run_name": name, "session": a.session,
                "hf_repo": a.hf_repo, "extra": a.extra, **more})
    f.write_text(json.dumps(rec, indent=1))
    return rec


def cmd_launch(a) -> int:
    spec, name = encoders.get(a.encoder), run_name(a)
    _record(name, a, launched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    sess = prepare(a.session or session(name), token=bool(a.hf_repo))
    out = f"{VM_RUNS}/{name}"
    argv = ["--encoder", spec.name, "--run-name", name, "--out", out, *a.extra]
    if a.hf_repo:
        argv += ["--hf-repo", a.hf_repo]
    else:
        print("WARNING: no --hf-repo; checkpoints die with the VM")
    if not start(sess, out, argv):
        return 1
    print(f"watch:  uv run python -m v6.train.colab status --run-name {name}"
          + (f" --session {a.session}" if a.session else "")
          + (f" --hf-repo {a.hf_repo}" if a.hf_repo else ""))
    return 0


SWEEP_DIR = RUNS_DIR / "sweeps"
SWEEP_METRICS = ("composite", "pos", "las", "ner_f1", "srl_role_f1", "cls_f1")


def _vm_status(sess: str, out: str) -> dict | None:
    with tempfile.TemporaryDirectory() as d:
        r = colab("download", "-s", sess, f"{out}/status.json", f"{d}/s.json",
                  check=False, quiet=True)
        return json.loads(Path(f"{d}/s.json").read_text()) if r.returncode == 0 else None


def cmd_sweep(a) -> int:
    """Learning-rate sweep for one encoder on one VM (OPERATIONS.md §4.2).

    Sets the VM up once, runs each rate under its own run name -- so no run
    resumes another's weights -- waits for each to finish, records the dev
    results in ``v6/runs/sweeps/<encoder>.json``, and releases the VM. No hub
    mirror by default: a sweep run is short, its output is a score, and losing
    one costs a rerun, not a checkpoint.
    """
    spec = encoders.get(a.encoder)
    lrs = [float(x) for x in a.lrs.split(",")]
    sess = a.session or session(f"sweep-{spec.name}")
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    results_f = SWEEP_DIR / f"{spec.name}.json"
    results = json.loads(results_f.read_text()) if results_f.exists() else {}
    sha = sh(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], quiet=True).stdout.strip()
    todo = [lr for lr in lrs if a.rerun or results.get(
        f"{spec.name}-lr{lr:g}", {}).get("state") != "done"]
    for lr in lrs:
        if lr not in todo:
            print(f"{spec.name}-lr{lr:g}: already done, skipping (--rerun to repeat)")
    if todo:
        prepare(sess, token=bool(a.hf_repo))
    rc, retries = 0, a.retries
    try:
        queue = list(todo)
        while queue:
            lr = queue.pop(0)
            name = f"{spec.name}-lr{lr:g}"
            out = f"{VM_RUNS}/{name}"
            argv = ["--encoder", spec.name, "--run-name", name, "--out", out,
                    "--encoder-lr", f"{lr:g}", "--max-steps", str(a.steps),
                    "--eval-every", str(a.eval_every), "--dev-limit", str(a.dev_limit),
                    "--log-every", "20", *a.extra]
            if a.hf_repo:
                argv += ["--hf-repo", a.hf_repo]
            if not start(sess, out, argv):
                results[name] = {"lr": lr, "state": "failed to start", "sha": sha}
                rc = 1
                continue
            st, lost = None, False
            while True:
                time.sleep(60)
                st = _vm_status(sess, out) or st
                if st and st["state"] in TERMINAL:
                    break
                if st and st["state"] in ("training", "evaluating") and ensure_holder(sess):
                    print(f"{name}: holder re-attached", flush=True)
                # Colab reclaims VMs server-side without warning: on the first
                # sweep two of three A100s vanished 25 minutes in. Notice within
                # a poll rather than waiting out a stale heartbeat.
                if sess not in colab("sessions", check=False, quiet=True).stdout:
                    lost = True
                    break
                if st and time.time() - st["heartbeat"] > STALE_MINUTES * 60:
                    print(f"{name}: STALE heartbeat; abandoning this rate")
                    break
            if lost:
                print(f"{name}: VM {sess} was reclaimed at step "
                      f"{(st or {}).get('step', '?')}", flush=True)
                if retries > 0:
                    retries -= 1
                    print(f"  allocating a fresh VM and retrying this rate "
                          f"({a.retries - retries}/{a.retries})", flush=True)
                    colab("stop", "-s", sess, check=False, quiet=True)
                    prepare(sess, token=bool(a.hf_repo))
                    queue.insert(0, lr)
                    continue
                results[name] = {"lr": lr, "sha": sha, "state": "vm lost",
                                 "steps": (st or {}).get("step")}
                results_f.write_text(json.dumps(results, indent=1))
                print(f"  no retries left; stopping this sweep", flush=True)
                rc = 1
                break
            ev = (st or {}).get("last_eval") or {}
            results[name] = {
                "lr": lr, "sha": sha, "state": (st or {}).get("state"),
                "reason": (st or {}).get("reason"), "steps": (st or {}).get("step"),
                "best": (st or {}).get("best"), "nonfinite_steps": (st or {}).get("nonfinite_steps"),
                "final_head_loss": (st or {}).get("head_loss"),
                "final_eval": ev}
            results_f.write_text(json.dumps(results, indent=1))
            print(f"{name}: {results[name]['state']} at step {results[name]['steps']}; "
                  + " ".join(f"{k} {ev.get(k)}" for k in SWEEP_METRICS), flush=True)
    finally:
        if not a.keep:
            _kill_holder(sess)
            colab("stop", "-s", sess, check=False)
            print(f"released {sess}")
    print(f"\n{spec.name} sweep ({a.steps} steps, dev {a.dev_limit} windows):")
    print(f"  {'lr':>8s} " + " ".join(f"{k:>12s}" for k in SWEEP_METRICS) + "  state")
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["lr"]):
        ev = r.get("final_eval") or {}
        print(f"  {r['lr']:>8g} " + " ".join(f"{ev.get(k, float('nan')):>12.4f}" for k in SWEEP_METRICS)
              + f"  {r['state']}" + (f" ({r['reason']})" if r.get("reason") else ""))
    return rc


def cmd_probe(a) -> int:
    """The A100 fit check, run synchronously: batch size, gradient
    checkpointing and bf16 are measured, not assumed."""
    spec, name = encoders.get(a.encoder), run_name(a)
    sess = prepare(a.session or session(name), token=False)
    out = f"{VM_RUNS}/{name}-probe"
    argv = ["--encoder", spec.name, "--out", out, "--probe", str(a.steps), *a.extra]
    code = START.format(repo=VM_REPO, out=out, argv=argv, RUN_MARKER=RUN_MARKER)
    vm_exec(sess, code, timeout=3600)
    with tempfile.TemporaryDirectory() as d:
        colab("download", "-s", sess, f"{out}/train.log", f"{d}/train.log", quiet=True)
        lines = Path(f"{d}/train.log").read_text().splitlines()
    # train.log is appended across runs; report only the latest.
    starts = [i for i, l in enumerate(lines) if l.startswith(RUN_MARKER)]
    lines = lines[starts[-1]:] if starts else lines
    for l in lines:
        if "probe step" in l or l.startswith("PROBE") or "Error" in l:
            print(l)
    return 0


def _read_status(a, name: str) -> tuple[dict | None, str]:
    out = f"{VM_RUNS}/{name}"
    with tempfile.TemporaryDirectory() as d:
        r = colab("download", "-s", a.session or session(name), f"{out}/status.json",
                  f"{d}/s.json", check=False, quiet=True)
        if r.returncode == 0 and Path(f"{d}/s.json").exists():
            return json.loads(Path(f"{d}/s.json").read_text()), "vm"
        if a.hf_repo:
            from huggingface_hub import hf_hub_download
            try:
                p = hf_hub_download(a.hf_repo, f"{name}/status.json",
                                    local_dir=d, force_download=True)
                return json.loads(Path(p).read_text()), "hf mirror"
            except Exception:                               # noqa: BLE001
                pass
    return None, ""


def cmd_status(a) -> int:
    rc = 0
    names = ([a.run_name] if a.run_name else [a.encoder] if a.encoder
             else ["deberta-v3-large", "ettin-400m", "ettin-1b"])
    for n in names:
        st, src = _read_status(a, n)
        if st is None:
            print(f"{n}: no status (VM gone and no HF mirror, or not started)")
            rc = 1
            continue
        age = (time.time() - st.get("heartbeat", 0)) / 60
        stale = age > STALE_MINUTES and st.get("state") not in ("done",)
        rc |= stale
        ev = st.get("last_eval") or {}
        print(f"{n} [{src}]: {st['state']}{'  ** STALE **' if stale else ''} "
              f"step {st['step']}/{st['total_steps']} ep {st['epoch']} "
              f"loss {st.get('loss')} heartbeat {age:.1f}m ago, "
              f"eta {st.get('eta_minutes')}m"
              + (f"  reason: {st['reason']}" if st.get("reason") else ""))
        print(f"   grad norm {st.get('grad_norm')} lr {st.get('lr')} "
              f"non-finite {st.get('nonfinite_steps', 0)} "
              f"(streak {st.get('nonfinite_streak', 0)}) disk free {st.get('disk_free_gb')} GB")
        if st.get("head_loss"):
            print("   heads: " + " ".join(f"{k} {v}" for k, v in st["head_loss"].items()))
        if ev:
            print(f"   dev: composite {ev.get('composite')} (best {st.get('best')} at "
                  f"{st.get('best_step')}) pos {ev.get('pos')} las {ev.get('las')} "
                  f"ner {ev.get('ner_f1')} srl {ev.get('srl_role_f1')} cls {ev.get('cls_f1')}")
        print(f"   memory: {st.get('memory')}")
    return rc


# ── control ─────────────────────────────────────────────────────────
def _put_flag(sess: str, name: str, flag: str, text: str) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(text)
    try:
        upload(sess, f.name, f"{VM_RUNS}/{name}/{flag}")
    finally:
        os.unlink(f.name)


def cmd_checkpoint(a) -> int:
    name = run_name(a)
    _put_flag(a.session or session(name), name, "CHECKPOINT", "")
    print(f"{name}: checkpoint requested; the loop saves at its next step")
    return 0


def cmd_stop(a) -> int:
    """Checkpoint and exit cleanly. ``colab stop`` alone tears the VM down
    under the loop and loses up to one save interval."""
    name = run_name(a)
    sess = a.session or session(name)
    _put_flag(sess, name, "STOP", a.reason)
    print(f"{name}: stop requested ({a.reason})")
    if not a.release:
        return 0
    deadline = time.time() + a.timeout * 60
    while time.time() < deadline:
        time.sleep(30)
        # The hub's copy is the one that matters: the loop mirrors its final
        # state only after latest.pt has uploaded.
        state = (_hf_state(a, name) if a.hf_repo
                 else ((_read_status(a, name)[0] or {}).get("state")))
        if state in TERMINAL:
            print(f"{name}: {state}; checkpoint is on the hub; releasing {sess}")
            colab("stop", "-s", sess)
            return 0
    print(f"{name}: not stopped after {a.timeout} minutes; VM left running")
    return 1


def _hf_state(a, name: str) -> str | None:
    from huggingface_hub import hf_hub_download
    with tempfile.TemporaryDirectory() as d:
        try:
            p = hf_hub_download(a.hf_repo, f"{name}/status.json", local_dir=d,
                                force_download=True)
            return json.loads(Path(p).read_text()).get("state")
        except Exception:                                   # noqa: BLE001
            return None


# ── watch ───────────────────────────────────────────────────────────
def observe(prev: dict | None, st: dict | None, alive: bool | None,
            now: float) -> tuple[list[str], bool]:
    """What changed between two polls of one run, and whether it needs a
    relaunch. Pure, so the rules can be tested without a VM.

    ``prev`` and ``st`` are status dicts (``st`` None if unreadable);
    ``alive`` is whether the run's Colab session exists (None if unknown).
    """
    msgs = []
    if st is None:
        return (["no status: not started, or VM gone with no hub mirror"]
                if prev is not None or alive is False else []), False
    state = st.get("state")
    terminal = state in TERMINAL
    age = (now - st.get("heartbeat", 0)) / 60
    stale = not terminal and age > STALE_MINUTES
    p = prev or {}
    if state != p.get("state"):
        msgs.append(f"state {p.get('state', '-')} -> {state}"
                    + (f" ({st['reason']})" if st.get("reason") else ""))
    if stale and not p.get("_stale"):
        msgs.append(f"STALE: no heartbeat for {age:.0f} minutes (dead, not slow)")
    if alive is False and not terminal and p.get("_alive") is not False:
        msgs.append("VM GONE: the Colab session no longer exists")
    nf, pnf = st.get("nonfinite_steps", 0), p.get("nonfinite_steps", 0)
    if nf > pnf:
        msgs.append(f"NON-FINITE: {nf - pnf} step(s) skipped "
                    f"(streak {st.get('nonfinite_streak', 0)})")
    if st.get("best", -1) > p.get("best", -1) and st.get("best", -1) >= 0:
        msgs.append(f"new best {st['best']:.4f} at step {st.get('best_step')}")
    disk = st.get("disk_free_gb")
    if disk is not None and disk < 25 and not (p.get("disk_free_gb") or 99) < 25:
        msgs.append(f"LOW DISK: {disk} GB free; latest.pt saves will be refused")
    peak = (st.get("memory") or {}).get("gpu_peak")
    ppeak = (p.get("memory") or {}).get("gpu_peak") or 0
    if peak and peak > 37 and ppeak <= 37:
        msgs.append(f"GPU PEAK {peak} GB of 40: little headroom")
    relaunch = not terminal and (stale or alive is False)
    return msgs, relaunch


def cmd_watch(a) -> int:
    names = a.runs or ["deberta-v3-large", "ettin-400m", "ettin-1b"]
    prev: dict[str, dict | None] = {n: None for n in names}
    relaunched: dict[str, int] = {n: 0 for n in names}
    hinted: set[str] = set()         # say "needs a relaunch" once, not every poll
    grace_until: dict[str, float] = {}
    live = set(names)
    while live:
        sessions = colab("sessions", check=False, quiet=True).stdout
        now = time.time()
        for n in sorted(live):
            rec_f = LAUNCHES / f"{n}.json"
            rec = json.loads(rec_f.read_text()) if rec_f.exists() else {}
            sess = rec.get("session") or session(n)
            alive = sess in sessions if sessions else None
            a.run_name, a.session = n, rec.get("session")
            st, src = _read_status(a, n)
            msgs, relaunch = observe(prev[n], st, alive, now)
            stamp = time.strftime("%H:%M")
            for m in msgs:
                print(f"{stamp} {n}: {m}", flush=True)
            if st is not None:
                st["_stale"] = (now - st.get("heartbeat", 0)) / 60 > STALE_MINUTES
                st["_alive"] = alive
                prev[n] = st
            if (st and st.get("state") in ("training", "evaluating") and alive
                    and ensure_holder(sess)):
                print(f"{stamp} {n}: holder re-attached", flush=True)
            if st and st.get("state") in TERMINAL:
                print(f"{stamp} {n}: finished watching ({st['state']})", flush=True)
                live.discard(n)
                continue
            if not relaunch:
                hinted.discard(n)
            elif now < grace_until.get(n, 0):
                pass        # a relaunched VM needs ~15 min before its first heartbeat
            elif a.relaunch:
                if relaunched[n] >= a.max_relaunches or not rec:
                    if n not in hinted:
                        print(f"{stamp} {n}: needs a relaunch, but "
                              + (f"{a.max_relaunches} already spent" if rec else
                                 f"there is no launch record in {LAUNCHES}")
                              + "; relaunch it by hand", flush=True)
                        hinted.add(n)
                else:
                    relaunched[n] += 1
                    print(f"{stamp} {n}: RELAUNCHING ({relaunched[n]}/{a.max_relaunches})",
                          flush=True)
                    if alive:
                        colab("stop", "-s", sess, check=False)   # a wedged kernel
                    ns = argparse.Namespace(encoder=rec["encoder"], run_name=n,
                                            session=rec.get("session"),
                                            hf_repo=rec.get("hf_repo"),
                                            extra=rec.get("extra") or [])
                    try:
                        cmd_launch(ns)
                        _record(n, ns, relaunches=rec.get("relaunches", 0) + 1)
                    except SystemExit as exc:
                        print(f"{stamp} {n}: relaunch failed: {exc}", flush=True)
                    grace_until[n] = time.time() + a.relaunch_grace * 60
            elif n not in hinted:
                print(f"{stamp} {n}: needs a relaunch: uv run python -m v6.train.colab "
                      f"launch --encoder {rec.get('encoder', '<encoder>')} --run-name {n}",
                      flush=True)
                hinted.add(n)
        if a.once or not live:
            break
        time.sleep(a.interval * 60)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "probe", "status", "stop", "checkpoint"):
        p = sub.add_parser(name)
        p.add_argument("--encoder", required=name in ("launch", "probe"),
                       choices=list(encoders.ENCODERS))
        p.add_argument("--hf-repo", default=HF_REPO,
                       help=f"checkpoint mirror (default {HF_REPO}); '' disables it")
        p.add_argument("--run-name", help="default: the encoder name; keys the VM, "
                       "output dir and HF prefix, so name every sweep run distinctly")
        p.add_argument("--session", help="reuse this Colab session instead of "
                       "kniv-v6-<run-name>")
        if name == "probe":
            p.add_argument("--steps", type=int, default=10)
        if name in ("launch", "probe"):
            p.add_argument("extra", nargs=argparse.REMAINDER,
                           help="passed through to v6.train.loop, after --")
        if name == "stop":
            p.add_argument("--reason", default="requested from colab.py stop")
            p.add_argument("--release", action="store_true",
                           help="after the hub shows the run stopped, release the VM")
            p.add_argument("--timeout", type=float, default=30.0, help="minutes")
    s = sub.add_parser("sweep")
    s.add_argument("--encoder", required=True, choices=list(encoders.ENCODERS))
    s.add_argument("--lrs", required=True, help="comma-separated encoder learning rates")
    s.add_argument("--steps", type=int, default=400)
    s.add_argument("--eval-every", type=int, default=100)
    s.add_argument("--dev-limit", type=int, default=500)
    s.add_argument("--hf-repo", default="", help="mirror sweep runs to the hub (off by default)")
    s.add_argument("--session")
    s.add_argument("--keep", action="store_true", help="do not release the VM at the end")
    s.add_argument("--rerun", action="store_true", help="repeat rates already done")
    s.add_argument("--retries", type=int, default=2,
                   help="fresh VMs to allocate if Colab reclaims the current one")
    s.add_argument("extra", nargs=argparse.REMAINDER, help="passed to v6.train.loop, after --")
    w = sub.add_parser("watch")
    w.add_argument("runs", nargs="*", help="run names (default: the three encoders)")
    w.add_argument("--hf-repo", default=HF_REPO)
    w.add_argument("--interval", type=float, default=10.0, help="minutes between polls")
    w.add_argument("--once", action="store_true")
    w.add_argument("--relaunch", action="store_true",
                   help="restart a run whose VM died or went stale (spends compute units)")
    w.add_argument("--max-relaunches", type=int, default=3)
    w.add_argument("--relaunch-grace", type=float, default=40.0,
                   help="minutes after a relaunch before the run may be judged dead again")
    a = ap.parse_args()
    for k in ("run_name", "session", "extra"):
        if not hasattr(a, k):
            setattr(a, k, None if k != "extra" else [])
    if a.cmd in ("stop", "checkpoint") and not (a.run_name or a.encoder):
        ap.error("--run-name or --encoder is required")
    if getattr(a, "extra", None) and a.extra[:1] == ["--"]:
        a.extra = a.extra[1:]
    return {"launch": cmd_launch, "probe": cmd_probe, "status": cmd_status,
            "stop": cmd_stop, "checkpoint": cmd_checkpoint, "watch": cmd_watch,
            "sweep": cmd_sweep}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
