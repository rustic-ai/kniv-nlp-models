"""Hold a client connection to a Colab kernel for as long as a run trains.

    <colab-cli python> colab_holder.py <session> <run-dir-on-vm> [max-hours]
    <colab-cli python> colab_holder.py --refresh-tokens

Colab reclaims a VM once no client has been attached to its kernel for about
20 minutes, however busy the kernel is (OPERATIONS.md §4.3). This process is
that client: it opens the kernel's websocket and holds it, reattaching if it
drops, and executes nothing. It exits when the run's ``status.json`` reaches a
terminal state -- so a finished run's VM is not kept alive and billing -- when
the session disappears, or after ``max-hours``.

It runs under the Colab CLI's own interpreter, because it uses the CLI's
session store and runtime client; ``colab.ensure_holder`` starts it that way.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TERMINAL = ("done", "stopped", "failed")
OURS = "kniv-v6-"             # session-name prefix; nothing else is touched


def log(msg: str) -> None:
    msg = re.sub(r"(token=)[A-Za-z0-9._-]+", r"\1<redacted>", msg)   # never log a proxy token
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def run_state(session: str, run_dir: str) -> str | None:
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run(["colab", "download", "-s", session,
                            f"{run_dir}/status.json", f"{d}/s.json"],
                           capture_output=True, text=True, timeout=120)
        if r.returncode == 0:
            return json.loads(Path(f"{d}/s.json").read_text()).get("state")
    return None


def refresh_tokens() -> int:
    """Write each session's current runtime proxy token back to the CLI's store.

    The CLI stores a VM's proxy token when it allocates the VM and never
    renews it, and the token expires after one hour: from then on every file
    operation fails while the VM trains on. That made healthy runs look dead
    exactly one hour after allocation. The backend's assignment listing
    carries the current token, so copy it across. Returns how many changed.
    """
    from colab_cli.common import state
    current = {a.endpoint: a.runtime_proxy_info for a in state.client.list_assignments()}
    changed = 0
    for name, s in state.store.list().items():
        if not name.startswith(OURS):
            continue                # another project's session: not ours to touch
        info = current.get(s.endpoint)
        if info and info.token != s.token:
            s.token = info.token
            if getattr(info, "url", None):
                s.url = info.url
            state.store.add(s)
            changed += 1
    return changed


def connected(runtime) -> bool:
    try:
        ws = runtime._kernel_client._manager.client
        ready = getattr(ws, "connection_ready", None)
        return ready.is_set() if ready is not None else True
    except Exception:                                       # noqa: BLE001
        return False


def main() -> int:
    from colab_cli.common import state
    from colab_cli.runtime import ColabRuntime

    if sys.argv[1] == "--refresh-tokens":
        print(refresh_tokens())
        return 0
    if sys.argv[1] == "--orphans":
        # VMs allocated on Colab with no local session: the CLI drops a session
        # from its store when a call fails with an expired token, and such a VM
        # keeps billing with nothing left to stop it.
        from colab_cli.common import state
        local = {s.endpoint for s in state.store.list().values()}
        for a in state.client.list_assignments():
            if a.endpoint not in local:
                print(f"{a.endpoint} {a.accelerator.name}")
        return 0
    session, run_dir = sys.argv[1], sys.argv[2]
    deadline = time.time() + float(sys.argv[3] if len(sys.argv) > 3 else 24) * 3600
    failures = 0
    while time.time() < deadline:
        try:
            refresh_tokens()        # a reattach after an hour needs a fresh token
        except Exception as exc:                            # noqa: BLE001
            log(f"token refresh failed: {type(exc).__name__}")
        s = state.store.get(session)
        if not s:
            log(f"session {session} no longer exists; exiting")
            return 0
        rt = ColabRuntime(s.url, s.token, kernel_id=s.kernel_id, session_id=s.session_id)
        try:
            rt.kernel_client                    # opens and holds the websocket
            failures = 0
            log(f"attached to {session}")
            last_check = 0.0
            while time.time() < deadline and connected(rt):
                if time.time() - last_check > 120:
                    last_check = time.time()
                    st = run_state(session, run_dir)
                    if st in TERMINAL:
                        log(f"run is {st}; detaching")
                        return 0
                time.sleep(15)
            log("connection dropped; reattaching")
        except Exception as exc:                            # noqa: BLE001
            failures += 1
            log(f"attach failed ({failures}): {type(exc).__name__}: {exc}"[:300])
            if failures >= 6:
                log("giving up: the VM is probably gone")
                return 1
        finally:
            try:
                rt.stop()
            except Exception:                               # noqa: BLE001
                pass
        time.sleep(30)
    log("max hours reached; exiting")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
