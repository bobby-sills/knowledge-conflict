"""Stage 01 — closed-book prior screening and conflict-state labelling. GPU.

Resumable at fact granularity: 8 samples per fact are written as soon as they are
drawn, so a session that dies 1,400 facts in resumes at 1,400.
"""

from __future__ import annotations

import argparse

from .. import config, io_utils, prior
from ..documents import build_conflict_cases


def run(limit: int | None = None, batch_report: int = 25) -> dict:
    paths = config.PATHS.mkdirs()
    facts = io_utils.read_jsonl(paths.factset)
    if not facts:
        raise SystemExit("no fact set: run stage 00 first")
    if limit:
        facts = facts[:limit]

    done = io_utils.load_done_keys(paths.prior, "fact_id")
    todo = [f for f in facts if f["fact_id"] not in done]
    print(f"[01] {len(done)} screened, {len(todo)} to go")

    with io_utils.stage("01_prior", paths.manifest) as box:
        if todo:
            from ..model import load_model
            bundle = load_model()
            buf = []
            for i, fact in enumerate(todo, 1):
                buf.append(prior.screen_fact(bundle, fact))
                if len(buf) >= batch_report or i == len(todo):
                    io_utils.append_jsonl(paths.prior, buf)
                    buf = []
                    print(f"[01] {i}/{len(todo)}", flush=True)

        io_utils.dedup_jsonl(paths.prior, "fact_id")
        rows = io_utils.read_jsonl(paths.prior)
        facts_by_id = {f["fact_id"]: f for f in facts}
        summary = prior.summarise_screening(rows, facts_by_id)

        # Conflict cases are a deterministic function of the labels, so they are
        # rebuilt rather than resumed.
        states = {r["fact_id"]: r["prior_label"] for r in rows}
        cases = build_conflict_cases(facts, states)
        if paths.states.exists():
            paths.states.unlink()
        io_utils.append_jsonl(paths.states, cases)

        box.update(summary)
        box["n_cases"] = len(cases)
        io_utils.write_json(paths.test_dir("test0") / "summary.json", summary)

        print(f"[01] prior labels {summary['prior_labels']}")
        print(f"[01] ambiguous fraction {summary['ambiguous_fraction']:.1%}")
        print(f"[01] state counts {summary['state_counts']}")
        if summary["ambiguous_fraction"] > 0.5:
            print("[01] FLAG: over half the fact set is ambiguous at 6/8 and 0/8. "
                  "The thresholds may be wrong for this model — see the spec's "
                  "'flag before assuming' guardrail. Do not adjust them silently.")
        resistance = summary["state_counts"].get("resistance", 0)
        if resistance < 100:
            print(f"[01] FLAG: only {resistance} resistance cases. Test 2's AUC "
                  "will have wide CIs; report the width rather than the point.")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="screen only the first N facts (smoke test)")
    a = ap.parse_args(argv)
    return run(limit=a.limit)


if __name__ == "__main__":
    main()
