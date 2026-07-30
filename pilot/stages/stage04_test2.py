"""Stage 04 — Test 2: does the internal signal predict the decision? CPU only.

The real gate. Restricted to correction and resistance cases, the binary is
"should we resist", and every predictor from the spec is scored on the same
questions with the same bootstrap resamples.

Two splits are used, for two different jobs:
  * train — fitting the direction of the sign-free divergences and the routing
    thresholds. One free parameter each, granted to the baselines.
  * layer — the split the AUC table is reported on. (Test 1 chose the layer here,
    which is a mild reuse; the alternative is reporting Test 2 on the report split,
    and the guardrail forbids touching it until Tests 0-4 are done. Recorded in
    DECISIONS.md, and the final headline number is meant to be re-run once on the
    report split after the write-up.)
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import analysis, config, figures, io_utils, permutation, signals, stats


def run(fit_split=("train",), report_split=("layer",),
        layer: int | None = None) -> dict:
    paths = config.PATHS.mkdirs()
    # Guardrail before anything that could fail for a mundane reason.
    for s in tuple(fit_split) + tuple(report_split):
        if s == "report":
            io_utils.assert_report_unlocked(paths.report_lock, "stage 04")

    ctx_rows = io_utils.read_jsonl(paths.results / "capture_ctx.jsonl")
    prior_rows = io_utils.read_jsonl(paths.results / "capture_prior.jsonl")
    screen_rows = io_utils.read_jsonl(paths.prior)
    if not ctx_rows or not prior_rows:
        raise SystemExit("no capture: run stage 02 first")

    if layer is None:
        test1 = io_utils.read_json(paths.test_dir("test1") / "test1.json")
        if not test1:
            raise SystemExit("no layer choice: run stage 03 first, or pass --layer")
        layer = int(test1["chosen_layer"])

    prior_by_id = {r["fact_id"]: r for r in prior_rows}
    consistency = analysis.consistency_from_prior_samples(screen_rows)

    with io_utils.stage("04_test2", paths.manifest) as box:
        out_dir = paths.test_dir("test2")

        fit = analysis.build_predictor_table(
            analysis.filter_split(ctx_rows, fit_split), prior_by_id, layer,
            consistency)
        rep = analysis.build_predictor_table(
            analysis.filter_split(ctx_rows, report_split), prior_by_id, layer,
            consistency)
        if rep["labels"].size == 0:
            raise SystemExit(f"no conflict cases in splits {report_split}")

        # Directions fitted on train, applied unchanged to the reported split.
        signs = signals.fit_orientations(fit["table"], fit["labels"])
        oriented = signals.apply_orientations(rep["table"], signs)

        boot = stats.bootstrap_auc_table(oriented, rep["labels"])
        kill = stats.check_test2_kill(boot, signals.INTERNAL_SIGNALS,
                                     signals.EXTERNAL_SIGNALS)

        # Thresholds fitted on train, so the error patterns being correlated are
        # out-of-sample errors rather than in-sample fits.
        fit_oriented = signals.apply_orientations(fit["table"], signs)
        thresholds = {
            name: stats.accuracy_at_best_threshold(vals, fit["labels"])[1]
            for name, vals in fit_oriented.items()}
        errors = stats.error_vectors(oriented, rep["labels"], thresholds)
        err_corr = stats.error_correlation_matrix(errors)

        # Cheap permutation control, applied to every signal (Test 4 guardrail:
        # run it every time a promising result appears).
        perm = {name: permutation.permutation_auc(vals, rep["labels"])
                for name, vals in oriented.items()}

        pairwise = {}
        for i_name in signals.INTERNAL_SIGNALS:
            for e_name in signals.EXTERNAL_SIGNALS:
                if i_name in boot["per_signal"] and e_name in boot["per_signal"]:
                    pairwise[f"{i_name}_vs_{e_name}"] = stats.paired_difference(
                        boot, i_name, e_name)

        result = {
            "layer": layer,
            "fit_split": list(fit_split),
            "report_split": list(report_split),
            "n_cases": int(rep["labels"].size),
            "n_resistance": int(rep["labels"].sum()),
            "n_correction": int((~rep["labels"]).sum()),
            "orientations": signs,
            "thresholds": thresholds,
            "auc": boot["per_signal"],
            "paired_differences": pairwise,
            "error_correlation": err_corr,
            "permutation_auc": perm,
            "kill": kill,
        }
        io_utils.write_json(out_dir / "test2.json", result)
        np.save(out_dir / "bootstrap_draws.npy", boot["_draws"])
        io_utils.write_json(out_dir / "bootstrap_names.json", boot["_names"])

        figures.auc_table(boot, signals.INTERNAL_SIGNALS,
                          paths.figures / "test2_auc.png")
        figures.error_correlation(err_corr,
                                  paths.figures / "test2_error_correlation.png")

        box.update({"layer": layer, "n_cases": result["n_cases"],
                    "kill": kill,
                    "auc": {k: round(v["auc"], 4) for k, v in boot["per_signal"].items()}})

        print(f"[04] {result['n_cases']} conflict cases "
              f"({result['n_resistance']} resistance, {result['n_correction']} correction)")
        print("[04] AUC (larger = predicts 'resist'):")
        for name in sorted(boot["per_signal"],
                           key=lambda n: -np.nan_to_num(boot["per_signal"][n]["auc"], nan=-1)):
            s = boot["per_signal"][name]
            tag = "internal" if name in signals.INTERNAL_SIGNALS else "external"
            print(f"       {name:<22} {s['auc']:.4f}  "
                  f"[{s['lo']:.4f}, {s['hi']:.4f}]  ({tag})")
        print(f"[04] best internal {kill.get('best_internal')} vs best external "
              f"{kill.get('best_external')}: margin {kill.get('margin', float('nan')):+.4f}, "
              f"CIs disjoint={kill.get('ci_disjoint')}")

        if kill["fired"]:
            print("\n[04] *** TEST 2 KILL CRITERION FIRED ***")
            for r in kill["reasons"]:
                print(f"      - {r}")
            if kill.get("narrow_failure"):
                names = err_corr["names"]
                mat = np.asarray(err_corr["matrix"])
                bi = names.index(kill["best_internal"])
                pairs = [(abs(mat[bi, names.index(e)]), e)
                         for e in signals.EXTERNAL_SIGNALS if e in names]
                usable = sorted((c, n) for c, n in pairs if np.isfinite(c))
                print("      Narrow failure. Error correlation of "
                      f"{kill['best_internal']} with the external signals "
                      "(lowest first):")
                for corr, name in usable:
                    print(f"        {name:<22} |phi| = {corr:.3f}")
                for corr, name in pairs:
                    if not np.isfinite(corr):
                        print(f"        {name:<22} undefined (constant errors)")
                print("      Low correlation means complementary, not superior: "
                      "the project becomes an ensemble method. Weaker, still "
                      "viable. Report that explicitly rather than a bare failure.")
            print("      Do not adjust the threshold. Do not search for a variant "
                  "that passes.\n")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit-splits", nargs="+", default=["train"])
    ap.add_argument("--report-splits", nargs="+", default=["layer"])
    ap.add_argument("--layer", type=int, default=None)
    a = ap.parse_args(argv)
    return run(tuple(a.fit_splits), tuple(a.report_splits), a.layer)


if __name__ == "__main__":
    main()
