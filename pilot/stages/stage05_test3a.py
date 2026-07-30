"""Stage 05 — Test 3a: analytic reachability. CPU only.

tau* was computed during capture, while both full distributions were in memory.
This stage only summarises and plots it, which is why it is free and why it can be
re-run after any change to the analysis without touching the GPU.

The prediction under test: tau* in (0,1) for resistance cases, meaning plain
interpolation suffices and the extrapolation regime that every existing method
lives in is not merely unnecessary but pointed away from the answer.
"""

from __future__ import annotations

import argparse

from .. import config, figures, io_utils, reachability


def run(splits=("train", "layer")) -> dict:
    paths = config.PATHS.mkdirs()
    # Guardrail before anything that could fail for a mundane reason.
    if "report" in splits:
        io_utils.assert_report_unlocked(paths.report_lock, "stage 05")
    ctx_rows = io_utils.read_jsonl(paths.results / "capture_ctx.jsonl")
    if not ctx_rows:
        raise SystemExit("no capture: run stage 02 first")

    rows = [r for r in ctx_rows if r.get("split") in set(splits)]

    with io_utils.stage("05_test3a", paths.manifest) as box:
        out_dir = paths.test_dir("test3a")
        by_ctxtop = reachability.summarise_by_state(rows, "tau_star_ctxtop")
        by_stated = reachability.summarise_by_state(rows, "tau_star_stated")

        taus_by_state: dict[str, list] = {}
        for r in rows:
            taus_by_state.setdefault(r["state"], []).append(r.get("tau_star_ctxtop"))

        result = {"splits": list(splits), "n_cases": len(rows),
                  "competitor_is_ctx_top": by_ctxtop,
                  "competitor_is_stated_object": by_stated}
        io_utils.write_json(out_dir / "test3a.json", result)
        figures.tau_star_histogram(taus_by_state,
                                   paths.figures / "test3a_tau_star.png")

        box.update(result)
        for state, s in by_ctxtop.items():
            if s.get("n_finite"):
                print(f"[05] {state:<11} tau* median {s['median']:+.3f}  "
                      f"in (0,1): {s['frac_in_interpolation']:.1%}  "
                      f"undefined: {s.get('frac_undefined', 0):.1%}")
        res = by_ctxtop.get("resistance", {})
        if res.get("frac_in_interpolation", 0) < 0.5:
            print("[05] NOTE: fewer than half of resistance cases have tau* in "
                  "(0,1). The theoretical prediction that interpolation suffices "
                  "does not hold here — report it; it changes what the eventual "
                  "method has to do.")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "layer"])
    a = ap.parse_args(argv)
    return run(tuple(a.splits))


if __name__ == "__main__":
    main()
