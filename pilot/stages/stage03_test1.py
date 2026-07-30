"""Stage 03 — Test 1: do internal and external answers diverge? CPU only.

The layer is chosen on the **layer** split and the divergence numbers are
reported on the **train** split. Both are dev splits; the report split stays
locked. Choosing the layer on the same facts the divergence is measured on would
be selecting the layer that maximises the result being reported.
"""

from __future__ import annotations

import argparse

from .. import analysis, config, figures, io_utils, scoring


def run(splits_for_report=("train",), splits_for_layer=("layer",)) -> dict:
    paths = config.PATHS.mkdirs()
    # Guardrail first: the lock must fire before anything else can fail for a
    # mundane reason, or a missing artefact would mask an illegal split request.
    for s in tuple(splits_for_report) + tuple(splits_for_layer):
        if s == "report":
            io_utils.assert_report_unlocked(paths.report_lock, "stage 03")

    prior_rows = io_utils.read_jsonl(paths.results / "capture_prior.jsonl")
    if not prior_rows:
        raise SystemExit("no capture: run stage 02 first")

    with io_utils.stage("03_test1", paths.manifest) as box:
        out_dir = paths.test_dir("test1")

        # ---- layer selection, on the layer split ------------------------- #
        layer_rows = analysis.filter_split(prior_rows, splits_for_layer)
        if not layer_rows:
            raise SystemExit(f"no captured facts in splits {splits_for_layer}")
        layer_scores = analysis.per_layer_internal_scores(layer_rows)
        chosen, per_layer_means = scoring.best_layer(layer_scores)

        # ---- divergence, on the report-side dev split -------------------- #
        rep_rows = analysis.filter_split(prior_rows, splits_for_report)
        if not rep_rows:
            raise SystemExit(
                f"no captured facts in splits {list(splits_for_report)}; "
                f"available: {sorted({r.get('split') for r in prior_rows})}")
        rep_scores = analysis.per_layer_internal_scores(rep_rows)
        records = analysis.external_knowledge_table(rep_rows, chosen)["records"]
        report = scoring.divergence_report(records)
        kill = scoring.check_test1_kill(report, config.KILL)

        # Restricting to facts where first-token scoring is not capped by a
        # gold/distractor collision. Reported both ways: the unrestricted number
        # is the honest headline, the restricted one shows whether collisions are
        # driving it.
        clean = [r for r in records if r.get("usable_first_token")]
        report_clean = scoring.divergence_result_or_none(clean)

        result = {
            "chosen_layer": chosen,
            "n_layers": int(layer_scores.shape[1]),
            "layer_selection_split": list(splits_for_layer),
            "report_split_used": list(splits_for_report),
            "per_layer_internal_mean": [float(x) for x in per_layer_means],
            **{k: v for k, v in report.items()
               if k not in ("internal_scores", "external_scores", "states")},
            "kill": kill,
            "n_usable_first_token": len(clean),
            "restricted_to_usable": report_clean,
        }
        io_utils.write_json(out_dir / "test1.json", result)

        figures.knowledge_by_layer(
            rep_scores, report["external_knowledge_mean"], chosen,
            paths.figures / "test1_knowledge_by_layer.png")
        figures.internal_vs_external_scatter(
            report["internal_scores"], report["external_scores"],
            report["states"], paths.figures / "test1_scatter.png")

        box.update({k: v for k, v in result.items()
                    if k != "per_layer_internal_mean"})
        print(f"[03] best layer {chosen}/{layer_scores.shape[1] - 1} "
              f"(mean internal {per_layer_means[chosen]:.4f})")
        print(f"[03] internal {report['internal_knowledge_mean']:.4f} vs external "
              f"{report['external_knowledge_mean']:.4f} "
              f"(gap {report['knowledge_gap']:+.4f})")
        print(f"[03] divergence rate {report['divergence_rate']:.1%} "
              f"(n={report['n_diverged']})")
        print(f"[03] on divergent: internal right "
              f"{report['on_divergent_internal_correct']:.1%}, external right "
              f"{report['on_divergent_external_correct']:.1%}")
        if kill["fired"]:
            print("\n[03] *** TEST 1 KILL CRITERION FIRED ***")
            for r in kill["reasons"]:
                print(f"      - {r}")
            print("      Stop here. Write it up in RESULTS.md. Do not adjust the "
                  "threshold and do not proceed to Test 2.\n")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report-splits", nargs="+", default=["train"])
    ap.add_argument("--layer-splits", nargs="+", default=["layer"])
    a = ap.parse_args(argv)
    return run(tuple(a.report_splits), tuple(a.layer_splits))


if __name__ == "__main__":
    main()
