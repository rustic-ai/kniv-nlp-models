"""Record a run's deploy-time settings in ``<run>/calibration.json`` on the hub.

    uv run python -m v6.train.set_calibration --run R --set entity_coref=compatible

Merges into what is there (``calibrate_zs`` writes the threshold offsets to the
same file). Every value must have been chosen on dev, never on test: the
choice and its evidence go in ``--note``.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .gold_eval import HF_REPO


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="value parsed as JSON when it parses, else kept as a string")
    ap.add_argument("--note", required=True, help="how the values were chosen (on dev)")
    args = ap.parse_args()
    from huggingface_hub import HfApi, hf_hub_download
    try:
        cal = json.loads(Path(hf_hub_download(HF_REPO, f"{args.run}/calibration.json",
                                              force_download=True)).read_text())
    except Exception:
        cal = {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cal[k] = json.loads(v)
        except json.JSONDecodeError:
            cal[k] = v
    cal.setdefault("notes", []).append(args.note)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "calibration.json"
        p.write_text(json.dumps(cal, indent=1))
        HfApi().upload_file(path_or_fileobj=str(p), path_in_repo=f"{args.run}/calibration.json",
                            repo_id=HF_REPO, commit_message=f"{args.run}: calibration")
    print(json.dumps(cal, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
