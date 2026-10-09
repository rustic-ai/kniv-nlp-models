"""Launch and watch v6 training runs on Colab A100s, one VM per encoder.

    uv run python -m v6.train.colab launch --encoder ettin-1b 
    uv run python -m v6.train.colab status --encoder ettin-1b
    uv run python -m v6.train.colab probe  --encoder ettin-1b      # A100 fit check

Run this yourself: it allocates billable VMs, and it copies your local
HuggingFace token to each VM (as a 0600 file) so checkpoints can be pushed to
the private repo. The token never appears in a command line or a log.

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
"""
from __future__ import annotations

import argparse
import json
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
    noise = [l for l in r.stderr.splitlines() if not l.startswith("[colab] A new version")
             and "update" not in l.lower() and "enable_update_check" not in l]
    if not quiet and r.stdout.strip():
        print(r.stdout.rstrip())
    if r.returncode and check:
        print("\n".join(noise), file=sys.stderr)
        raise SystemExit(f"failed ({r.returncode}): {' '.join(cmd[:4])} ...")
    return r


def colab(*args: str, **kw) -> subprocess.CompletedProcess:
    return sh(["colab", *args], **kw)


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
os.chmod("/content/.hf_token", 0o600)
os.environ["HF_TOKEN"] = open("/content/.hf_token").read().strip()
from huggingface_hub import whoami
user = whoami()["name"]
print(json.dumps({{"gpu": name, "gpu_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                  "torch": torch.__version__, "hf_user": user}}))
'''

START = r'''
import os, sys, runpy, threading
os.chdir("{repo}")
sys.path.insert(0, "{repo}")
os.environ["HF_TOKEN"] = open("/content/.hf_token").read().strip()
os.makedirs("{out}", exist_ok=True)

class _Tee:
    """Copy output to train.log. Anything else -- isatty, fileno, encoding,
    which transformers and tqdm query -- answers as the real stream does."""
    def __init__(self, *s): self.s = s
    def write(self, x):
        for f in self.s: f.write(x); f.flush()
        return len(x)
    def flush(self):
        for f in self.s: f.flush()
    def __getattr__(self, name): return getattr(self.s[0], name)

_log = open("{out}/train.log", "a", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log)
sys.stderr = _Tee(sys.__stderr__, _log)
sys.argv = ["v6.train.loop", *{argv}]
try:
    runpy.run_module("v6.train.loop", run_name="__main__", alter_sys=True)
except SystemExit as e:
    print(f"loop exited: {{e.code}}")
finally:
    sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
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
        colab("upload", "-s", sess, path, "/content/.hf_token", quiet=True)
    finally:
        os.unlink(path)


def prepare(sess: str) -> str:
    code, data, sha = build_bundles()
    existing = colab("sessions", check=False, quiet=True).stdout
    if sess not in existing:
        print(f"allocating {sess} (A100)...")
        colab("new", "-s", sess, "--gpu", "A100")
    for b in (code, data):
        colab("upload", "-s", sess, str(b), f"/content/{b.name}", quiet=True)
    _upload_token(sess)
    print("setting up the VM (pip, GPU check, HF auth)...")
    vm_exec(sess, SETUP.format(repo=VM_REPO, code=f"/content/{code.name}",
                               data=f"/content/{data.name}", pip=PIP), timeout=900)
    return sess


def start(sess: str, out: str, argv: list[str]) -> None:
    print(f"starting in-kernel: python -m v6.train.loop {shlex.join(argv)}")
    # The client times out; the kernel keeps running the cell.
    vm_exec(sess, START.format(repo=VM_REPO, out=out, argv=argv), timeout=90,
            check=False)
    st = colab("status", "-s", sess, check=False, quiet=True).stdout
    busy = "BUSY" in st.upper()
    print(f"kernel {'BUSY: training is running' if busy else 'NOT BUSY'}")
    if not busy:
        print(f"check {out}/train.log on the VM:  "
              f"colab download -s {sess} {out}/train.log ./train.log")


# ── commands ────────────────────────────────────────────────────────
def cmd_launch(a) -> int:
    spec, name = encoders.get(a.encoder), run_name(a)
    sess = prepare(a.session or session(name))
    out = f"{VM_RUNS}/{name}"
    argv = ["--encoder", spec.name, "--run-name", name, "--out", out, *a.extra]
    if a.hf_repo:
        argv += ["--hf-repo", a.hf_repo]
    else:
        print("WARNING: no --hf-repo; checkpoints die with the VM")
    start(sess, out, argv)
    print(f"watch:  uv run python -m v6.train.colab status --run-name {name}"
          + (f" --session {a.session}" if a.session else "")
          + (f" --hf-repo {a.hf_repo}" if a.hf_repo else ""))
    return 0


def cmd_probe(a) -> int:
    """The A100 fit check, run synchronously: batch size, gradient
    checkpointing and bf16 are measured, not assumed."""
    spec, name = encoders.get(a.encoder), run_name(a)
    sess = prepare(a.session or session(name))
    out = f"{VM_RUNS}/{name}-probe"
    argv = ["--encoder", spec.name, "--out", out, "--probe", str(a.steps), *a.extra]
    code = START.format(repo=VM_REPO, out=out, argv=argv)
    vm_exec(sess, code, timeout=3600)
    with tempfile.TemporaryDirectory() as d:
        colab("download", "-s", sess, f"{out}/train.log", f"{d}/train.log", quiet=True)
        lines = Path(f"{d}/train.log").read_text().splitlines()
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
              f"eta {st.get('eta_minutes')}m")
        if st.get("head_loss"):
            print("   heads: " + " ".join(f"{k} {v}" for k, v in st["head_loss"].items()))
        if ev:
            print(f"   dev: composite {ev.get('composite')} (best {st.get('best')} at "
                  f"{st.get('best_step')}) pos {ev.get('pos')} las {ev.get('las')} "
                  f"ner {ev.get('ner_f1')} srl {ev.get('srl_role_f1')} cls {ev.get('cls_f1')}")
        print(f"   memory: {st.get('memory')}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("launch", "probe", "status"):
        p = sub.add_parser(name)
        p.add_argument("--encoder", required=name != "status",
                       choices=list(encoders.ENCODERS))
        p.add_argument("--hf-repo", default=HF_REPO,
                       help=f"checkpoint mirror (default {HF_REPO}); '' disables it")
        p.add_argument("--run-name", help="default: the encoder name; keys the VM, "
                       "output dir and HF prefix, so name every sweep run distinctly")
        p.add_argument("--session", help="reuse this Colab session instead of "
                       "kniv-v6-<run-name>")
        if name == "probe":
            p.add_argument("--steps", type=int, default=10)
        if name != "status":
            p.add_argument("extra", nargs=argparse.REMAINDER,
                           help="passed through to v6.train.loop, after --")
    a = ap.parse_args()
    if getattr(a, "extra", None) and a.extra[:1] == ["--"]:
        a.extra = a.extra[1:]
    return {"launch": cmd_launch, "probe": cmd_probe, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
