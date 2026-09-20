"""Read a training run's status from outside the job.

``status.json`` is rewritten atomically every few steps, so this is safe to
run against a live run — from another machine, against mounted Drive, while
Colab is still training.

The heartbeat is the point: a job that has stalled and a job that is merely
slow look identical in a log tail, and that ambiguity has already cost this
project an hour of silent wall-clock. Anything older than --stale-minutes
is reported as STALE rather than left to interpretation.

    uv run python -m v6.experiments.atlop_status --out-dir <dir>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stale-minutes", type=float, default=10.0)
    ap.add_argument("--events", type=int, default=5,
                    help="tail this many events")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    d = Path(args.out_dir)
    sp = d / "status.json"
    if not sp.exists():
        print(f"no status.json in {d} — run not started, or wrong directory")
        return 2
    st = json.loads(sp.read_text())
    age = (time.time() - st.get("heartbeat", 0)) / 60
    stale = age > args.stale_minutes
    st["heartbeat_age_minutes"] = round(age, 1)
    st["stale"] = stale

    if args.json:
        print(json.dumps(st, indent=2))
        return 1 if stale else 0

    flag = "  ** STALE **" if stale else ""
    print(f"run {st['run_id']}  [{st['state']}]{flag}")
    print(f"  step        {st['step']}/{st['total_steps']}"
          f"  (epoch {st['epoch']})")
    print(f"  loss        {st.get('loss')}")
    print(f"  best F1     {st['best_f1']:.4f} (epoch {st['best_epoch']})"
          if st["best_f1"] >= 0 else "  best F1     none yet")
    if st.get("last_eval"):
        e = st["last_eval"]
        print(f"  last eval   F1={e['f1']} P={e['precision']} R={e['recall']}")
    print(f"  rate        {st.get('steps_per_sec')} steps/s"
          f"   eta {st.get('eta_minutes')} min")
    print(f"  elapsed     {st['elapsed_minutes']:.1f} min"
          f"   heartbeat {age:.1f} min ago")
    if st.get("resumed_from_step"):
        print(f"  resumed     from step {st['resumed_from_step']}")
    print(f"  versions    {st.get('meta', {})}")

    ev = d / "events.jsonl"
    if args.events and ev.exists():
        lines = ev.read_text().splitlines()[-args.events:]
        print(f"\n  last {len(lines)} events:")
        for ln in lines:
            r = json.loads(ln)
            extra = {k: v for k, v in r.items()
                     if k not in ("ts", "iso", "run_id", "kind", "traceback")}
            print(f"    {r.get('iso', '')} {r['kind']:<12} {extra}")
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main())
