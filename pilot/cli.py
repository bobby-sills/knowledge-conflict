"""Single entry point: `python -m pilot.cli <stage> [args]`.

    python -m pilot.cli factset
    python -m pilot.cli prior
    python -m pilot.cli capture --batch-size 8
    python -m pilot.cli test1
    python -m pilot.cli test2
    python -m pilot.cli test3a
    python -m pilot.cli test3b --grid coarse
    python -m pilot.cli test4 --shuffles 5
    python -m pilot.cli timing
    python -m pilot.cli status
    python -m pilot.cli smoke                 # tiny end-to-end run
    python -m pilot.cli unlock-report --reason '...'

Stages stop the pipeline when a kill criterion fires: `all` refuses to continue
past a fired criterion, because "stop, write it up, and report" is a guardrail and
a convenience runner that ignored it would be the easiest possible way to violate
the spec by accident.
"""

from __future__ import annotations

import argparse
import sys

from . import config, io_utils

STAGES = {
    "factset": "pilot.stages.stage00_factset",
    "prior": "pilot.stages.stage01_prior",
    "capture": "pilot.stages.stage02_capture",
    "test1": "pilot.stages.stage03_test1",
    "test2": "pilot.stages.stage04_test2",
    "test3a": "pilot.stages.stage05_test3a",
    "test3b": "pilot.stages.stage06_test3b",
    "test4": "pilot.stages.stage07_test4",
    "timing": "pilot.stages.stage08_timing",
}

# Order for `all`, and the stage whose kill criterion gates the next one.
PIPELINE = ["factset", "prior", "capture", "test1", "test2", "test3a", "test3b",
            "test4"]


def _run_stage(name: str, argv: list[str]) -> dict:
    import importlib
    mod = importlib.import_module(STAGES[name])
    return mod.main(argv)


def status() -> dict:
    """What exists on disk, so a resumed session knows where it is."""
    paths = config.PATHS
    items = [
        ("factset", paths.factset, "fact_id"),
        ("prior samples", paths.prior, "fact_id"),
        ("conflict cases", paths.states, "case_id"),
        ("capture (prior)", paths.results / "capture_prior.jsonl", "fact_id"),
        ("capture (ctx)", paths.results / "capture_ctx.jsonl", "case_id"),
        ("test3b generations", paths.results / "test3b_generations.jsonl", None),
        ("test4 generations", paths.results / "test4_generations.jsonl", None),
    ]
    out = {"project_dir": str(paths.root), "artefacts": {}}
    print(f"project dir: {paths.root}")
    for label, path, key in items:
        if not path.exists():
            print(f"  {label:<20} -")
            out["artefacts"][label] = 0
            continue
        rows = io_utils.read_jsonl(path)
        n = len(io_utils.load_done_keys(path, key)) if key else len(rows)
        unit = "unique" if key else "rows"
        print(f"  {label:<20} {n} {unit}")
        out["artefacts"][label] = n

    for test in ("test0", "test1", "test2", "test3a", "test3b", "test4"):
        d = paths.results / test
        found = sorted(p.name for p in d.glob("*.json")) if d.is_dir() else []
        if found:
            print(f"  {test:<20} {', '.join(found)}")
    lock = io_utils.read_json(paths.report_lock)
    print(f"  report split         {'UNLOCKED: ' + lock['reason'] if lock else 'locked'}")
    out["report_unlocked"] = bool(lock)
    return out


def _kill_fired(name: str) -> tuple[bool, list[str]]:
    """Whether a completed stage's kill criterion fired, from its saved output."""
    files = {"test1": ("test1", "test1.json"), "test2": ("test2", "test2.json"),
             "test3b": ("test3b", "test3b.json"), "test4": ("test4", "test4.json")}
    if name not in files:
        return False, []
    sub, fname = files[name]
    data = io_utils.read_json(config.PATHS.results / sub / fname)
    if not data or "kill" not in data:
        return False, []
    return bool(data["kill"].get("fired")), list(data["kill"].get("reasons", []))


def run_all(extra: dict | None = None) -> dict:
    extra = extra or {}
    results = {}
    for name in PIPELINE:
        print(f"\n{'=' * 66}\n  {name}\n{'=' * 66}")
        results[name] = _run_stage(name, extra.get(name, []))
        fired, reasons = _kill_fired(name)
        if fired:
            print(f"\n{'!' * 66}")
            print(f"  {name}: KILL CRITERION FIRED — stopping the pipeline.")
            for r in reasons:
                print(f"    - {r}")
            print("  Write it up in RESULTS.md. A pilot that kills the project in")
            print("  two weeks has succeeded. Do not adjust the threshold, do not")
            print("  search for a variant that passes, do not run the next test.")
            print(f"{'!' * 66}\n")
            results["_stopped_at"] = name
            return results
    return results


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 1
    cmd, rest = argv[0], argv[1:]

    if cmd == "status":
        status()
        return 0
    if cmd == "smoke":
        from .smoke import main as smoke_main
        return smoke_main(rest)
    if cmd == "unlock-report":
        ap = argparse.ArgumentParser(prog="unlock-report")
        ap.add_argument("--reason", required=True,
                        help="why the report split is being unlocked now")
        a = ap.parse_args(rest)
        rec = io_utils.unlock_report(config.PATHS.mkdirs().report_lock, a.reason)
        print(f"report split unlocked: {rec}")
        return 0
    if cmd == "all":
        run_all()
        return 0
    if cmd in STAGES:
        _run_stage(cmd, rest)
        return 0

    print(f"unknown stage {cmd!r}\n")
    print(__doc__)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
