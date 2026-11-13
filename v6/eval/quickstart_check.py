"""Quick-start check for a runtime change: both model-card examples on every
v7 model, fp32 and INT8, with the repo's ``kniv_v7`` overlaid on a released
snapshot (default the v7.1.1 tag). Expected facts are the cards'. Gates every
runtime release.

    uv run python -m v6.eval.quickstart_check [--revision v7.1.1]
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SIZES = ["tiny", "small", "medium", "large", "xlarge-d", "xlarge-u"]

CONV = dict(turns=[("user", "I live in Berlin and work for Siemens."),
                   ("assistant", "Nice, how long have you been there?")],
            relations=[{"name": "lives in", "description": "the place where the head resides"},
                       {"name": "employer", "description": "the organization the head works for"}],
            entity_types=[{"name": "location"}, {"name": "organization"}, {"name": "person"}])
CONV_WANT = {("user", "lives in", "Berlin"), ("user", "employer", "Siemens")}
TEXT = ("Henry Wager Halleck (January 16, 1815 – January 9, 1872) was a United States Army "
        "officer, scholar, and lawyer.")
TEXT_KW = dict(relations=[{"name": "date of birth", "description": "date on which the subject was born"},
                          {"name": "date of death", "description": "date on which the subject died"},
                          {"name": "military branch", "description": "branch to which this person belongs"}],
               entity_types=[{"name": "person"}, {"name": "organization"}, {"name": "date"}])
TEXT_WANT = {("Henry Wager Halleck", "date of birth", "January 16 , 1815"),
             ("Henry Wager Halleck", "date of death", "January 9 , 1872"),
             ("Henry Wager Halleck", "military branch", "United States Army")}


def overlay(size: str, revision: str, work: Path) -> Path:
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(f"dragonscale-ai/kniv-v7-{size}-en", revision=revision))
    d = work / size
    d.mkdir(parents=True)
    for p in snap.iterdir():
        if p.name != "kniv_v7":
            (d / p.name).symlink_to(p.resolve() if p.is_file() else p)
    shutil.copytree(REPO / "kniv_v7", d / "kniv_v7", ignore=shutil.ignore_patterns("__pycache__"))
    return d


def facts(out):
    return {(w["entities"][f["head"]]["text"], f["relation"], w["entities"][f["tail"]]["text"],
             f["polarity"], f["modality"]) for w in out for f in w["facts"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--revision", default="v7.1.1")
    a = ap.parse_args()
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for size in SIZES:
            d = overlay(size, a.revision, Path(tmp))
            name = f"kniv_v7_qs_{size.replace('-', '_')}"
            spec = importlib.util.spec_from_file_location(name, d / "kniv_v7" / "__init__.py",
                                                          submodule_search_locations=[str(d / "kniv_v7")])
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            for int8 in (False, True):
                k = mod.Kniv(d, int8=int8)
                for tag, out, want in (("conversation", k.analyse(**CONV), CONV_WANT),
                                       ("text", k.analyse(TEXT, **TEXT_KW), TEXT_WANT)):
                    got, exp = facts(out), {w + ("positive", "asserted") for w in want}
                    good = got == exp
                    ok &= good
                    print(f"{size:9s} {'int8' if int8 else 'fp32'} {tag:12s} {'PASS' if good else 'FAIL'}"
                          + ("" if good else f"  missing {sorted(exp - got)} extra {sorted(got - exp)}"),
                          flush=True)
    print("ALL PASS" if ok else "SOME FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
