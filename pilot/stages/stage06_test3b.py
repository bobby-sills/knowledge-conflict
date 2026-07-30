"""Stage 06 — Test 3b: the tau sweep, the baselines, and the oracle ceiling. GPU.

Three things run here:

1. The power family swept over tau. Every (case, tau) generation is written as it
   completes, so this is resumable at generation granularity — which matters,
   because a 51-point sweep over a few thousand cases is the longest-running
   stage in the pilot.
2. The vendored baselines (CAD, AdaCAD, CoCoA, COIECD, ARR) on the same cases,
   each first probed for the tau regime it is *actually* in rather than the one its
   name or its own `get_tau()` claims.
3. Three-way oracle routing, assembled *from the sweep* rather than regenerated.
   Selecting the row already generated at each state's chosen tau is provably the
   same as regenerating, and it guarantees the oracle cannot see anything the
   sweep did not.

The tau grid is coarse-then-fine by default: a 51-point sweep is wasteful when
the interesting structure is a crossing. `--tau-grid coarse` runs 11 points at
0.25 spacing, which is enough for the trade-off curves and roughly a fifth of the
compute.
"""

from __future__ import annotations

import argparse

from .. import config, figures, io_utils, powerfamily


COARSE_GRID = tuple(round(0.25 * i, 3) for i in range(0, 11))    # 0.00 .. 2.50


def _gen_path():
    return config.PATHS.results / "test3b_generations.jsonl"


def _probe_regimes(bundle, vendored, probe_cases) -> dict:
    """Measure each baseline's effective tau on a real conflict case."""
    from ..documents import context_prompt, prior_prompt
    from ..model import forward_last
    from ..vendor import regime_report

    if not probe_cases:
        return {}
    case = probe_cases[0]
    fact = case["_fact"]
    out_ctx = forward_last(bundle, [context_prompt(fact, case["document"],
                                                   bundle.tokenizer)],
                           want_hidden=False)
    out_pri = forward_last(bundle, [prior_prompt(fact, bundle.tokenizer)],
                           want_hidden=False)
    reports = {}
    for name, method in vendored.items():
        try:
            reports[name] = regime_report(method, out_ctx["logits"],
                                          out_pri["logits"])
        except Exception as exc:      # a baseline we cannot probe is not fatal
            reports[name] = {"method": name, "error": f"{type(exc).__name__}: {exc}",
                             "effective_tau": float("nan"), "reported_tau": None,
                             "regime": "unprobed", "self_report_matches": False}
    return reports


def _cases_with_facts(paths, splits, limit=None):
    facts = {f["fact_id"]: f for f in io_utils.read_jsonl(paths.factset)}
    cases = [c for c in io_utils.read_jsonl(paths.states)
             if c.get("split") in set(splits)]
    out = []
    for c in cases:
        fact = facts.get(c["fact_id"])
        if fact is None:
            continue
        out.append(dict(c, gold_aliases=list(fact["gold_aliases"]) + [fact["object"]],
                        _fact=fact))
    return out[:limit] if limit else out


def run(splits=("train", "layer"), grid: str = "coarse",
        limit: int | None = None, methods=config.VENDOR_METHODS,
        max_new_tokens: int = config.GEN_MAX_NEW_TOKENS) -> dict:
    paths = config.PATHS.mkdirs()
    if "report" in splits:
        io_utils.assert_report_unlocked(paths.report_lock, "stage 06")

    cases = _cases_with_facts(paths, splits, limit)
    if not cases:
        raise SystemExit("no cases: run stages 00 and 01 first")
    taus = COARSE_GRID if grid == "coarse" else config.TAU_GRID

    with io_utils.stage("06_test3b", paths.manifest) as box:
        from ..model import load_model, stop_token_ids
        from ..vendor import build_methods, load as load_vendor, using_vendored_metrics

        vendor_root = load_vendor(paths.vendor)
        bundle = load_model()
        stops = stop_token_ids(bundle)
        from ..documents import context_prompt, prior_prompt

        done = {(r["method"], r.get("tau"), r["case_id"])
                for r in io_utils.read_jsonl(_gen_path())}

        def emit(method_name, tau, todo, method):
            buf = []
            for case in todo:
                fact = case["_fact"]
                p_ctx = context_prompt(fact, case["document"], bundle.tokenizer)
                p_pri = prior_prompt(fact, bundle.tokenizer)
                gen = powerfamily.generate(bundle, p_ctx, p_pri, method,
                                          max_new_tokens=max_new_tokens,
                                          stop_ids=stops)
                buf.append({
                    "method": method_name, "tau": tau,
                    "case_id": case["case_id"], "fact_id": case["fact_id"],
                    "split": case["split"], "state": case["state"],
                    "gold_aliases": case["gold_aliases"], "text": gen["text"],
                    "n_tokens": gen["n_tokens"],
                    "mean_tau": (sum(gen["taus"]) / len(gen["taus"])
                                 if gen["taus"] else None),
                })
                if len(buf) >= 32:
                    io_utils.append_jsonl(_gen_path(), buf)
                    buf = []
            io_utils.append_jsonl(_gen_path(), buf)      # flush the partial batch
            print(f"[06] {method_name} tau={tau}: {len(todo)} generated", flush=True)

        # ---- the sweep ---------------------------------------------------- #
        for tau in taus:
            todo = [c for c in cases if ("power", tau, c["case_id"]) not in done]
            if todo:
                emit("power", tau, todo, powerfamily.PowerFamily(tau))

        # ---- the baselines ------------------------------------------------ #
        vendored = build_methods(methods)

        # Before running them, measure which regime each is actually in. A method's
        # name and its self-reported tau can both be wrong (the repo's CoCoA reports
        # 0.5 while operating at 1.5), and the interpolation/extrapolation
        # distinction is the paper's whole subject.
        regimes = _probe_regimes(bundle, vendored, cases[:1])
        box["regimes"] = regimes
        for r in regimes.values():
            flag = "" if r["self_report_matches"] else "   <- self-report disagrees"
            print(f"[06] {r['method']:<10} effective tau ~ {r['effective_tau']:+.3f} "
                  f"({r['regime']}), reports {r['reported_tau']}{flag}")

        for name in methods:
            todo = [c for c in cases if (name, None, c["case_id"]) not in done]
            if todo:
                emit(name, None, todo, vendored[name])

        # ---- analysis ----------------------------------------------------- #
        rows = io_utils.read_jsonl(_gen_path())
        keep = {c["case_id"] for c in cases}
        rows = [r for r in rows if r["case_id"] in keep]

        rows_by_tau = {}
        for r in rows:
            if r["method"] == "power":
                rows_by_tau.setdefault(float(r["tau"]), []).append(r)
        sweep = {t: powerfamily.evaluate_predictions(rs)
                 for t, rs in sorted(rows_by_tau.items())}

        baselines = {}
        for name in methods:
            rs = [r for r in rows if r["method"] == name]
            if rs:
                baselines[name] = powerfamily.evaluate_predictions(rs)

        tuned = powerfamily.tune_oracle(rows_by_tau)
        oracle_rows = powerfamily.oracle_from_sweep(rows_by_tau, tuned["tau_by_state"])
        oracle = powerfamily.evaluate_predictions(oracle_rows)
        kill = powerfamily.check_test3_kill(oracle, baselines)

        # ---- reproduction check on the published comparator ---------------- #
        # The paper reports ARR lifting resistance EM to 16-33. If our run lands far
        # outside that, we have the wrong method or a broken config regardless of
        # what the class is called — and Test 3's kill criterion is measured
        # *relative to* this number, so a wrong baseline silently moves the gate.
        arr_check = None
        if "arr" in baselines:
            lo, hi = config.ARR_RESISTANCE_EM_TARGET
            got = baselines["arr"].get("resistance", float("nan"))
            in_range = bool(lo <= got <= hi) if got == got else False
            arr_check = {"resistance_em": got, "target": [lo, hi],
                         "in_range": in_range}
            print(f"[06] ARR resistance EM {got:.3f}; paper reports "
                  f"{lo:.2f}-{hi:.2f} -> {'ok' if in_range else 'OUT OF RANGE'}")
            if not in_range:
                print("[06] FLAG: ARR does not reproduce the paper's resistance EM. "
                      "Test 3's kill criterion is relative to this baseline, so "
                      "resolve it before believing the gate. Check the prompt style "
                      "(--prompt-style raw for their format), the fact set, and the "
                      "regime probe above.")

        best_fixed_tau = max(sweep, key=lambda t: sweep[t]["overall"]) if sweep else None
        result = {
            "splits": list(splits), "grid": grid, "taus": list(taus),
            "n_cases": len(cases),
            "vendored_metrics": using_vendored_metrics(),
            "vendor_root": str(vendor_root),
            "vendor_commit": config.VENDOR_COMMIT,
            "regimes": regimes,
            "arr_reproduction": arr_check,
            "sweep": {str(t): s for t, s in sweep.items()},
            "baselines": baselines,
            "oracle_tau_by_state": tuned["tau_by_state"],
            "oracle_per_state_curves": {k: {str(t): v for t, v in c.items()}
                                        for k, c in tuned["curves"].items()},
            "oracle": oracle,
            "best_fixed_tau": best_fixed_tau,
            "best_fixed_tau_em": sweep[best_fixed_tau]["overall"] if sweep else None,
            "kill": kill,
        }
        io_utils.write_json(paths.test_dir("test3b") / "test3b.json", result)

        figures.tau_sweep_curves(sweep, paths.figures / "test3b_tau_sweep.png",
                                 oracle=oracle, baselines=baselines)
        figures.correction_vs_resistance(
            sweep, paths.figures / "test3b_tradeoff.png",
            oracle=oracle, baselines=baselines)

        box.update({"n_cases": len(cases), "oracle": oracle,
                    "baselines": baselines, "kill": kill,
                    "oracle_tau_by_state": tuned["tau_by_state"]})

        print(f"[06] oracle tau by state: {tuned['tau_by_state']}")
        print(f"[06] oracle EM {oracle['overall']:.4f} "
              f"(correction {oracle.get('correction', float('nan')):.3f}, "
              f"resistance {oracle.get('resistance', float('nan')):.3f}, "
              f"agreement {oracle.get('agreement', float('nan')):.3f})")
        for name, s in sorted(baselines.items(), key=lambda kv: -kv[1]["overall"]):
            print(f"[06]   {name:<15} {s['overall']:.4f}  "
                  f"(resistance {s.get('resistance', float('nan')):.3f})")
        if not using_vendored_metrics():
            print("[06] FLAG: EM used the fallback normaliser, not the repo's. "
                  "These numbers are not comparable to the published table.")
        if kill["fired"]:
            print("\n[06] *** TEST 3 KILL CRITERION FIRED ***")
            for r in kill["reasons"]:
                print(f"      - {r}")
            print("      No signal quality can exceed the oracle, so a low "
                  "ceiling means there is nothing to chase.\n")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=["train", "layer"])
    ap.add_argument("--grid", choices=("coarse", "fine"), default="coarse")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=list(config.VENDOR_METHODS))
    a = ap.parse_args(argv)
    return run(tuple(a.splits), a.grid, a.limit, tuple(a.methods))


if __name__ == "__main__":
    main()
