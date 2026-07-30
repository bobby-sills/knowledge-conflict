"""Stage 07 — Test 4: the permutation control. GPU for the decoding half.

Two levels, cheapest first:

**Signal level** (free, no GPU). Shuffle the winning signal and recompute its AUC.
The null should sit at 0.5. This catches a broken analysis but not a global-knob
signal, because AUC is invariant to the routing policy.

**Decoding level** (GPU). Route tau per question by thresholding the signal, then
re-run with the signal shuffled across questions. Gains are measured against the
**best fixed tau** from stage 06, not against tau=1: a signal that is really a
global rescale would beat tau=1 comfortably and pass a control designed to catch
exactly that.

Every generation is cached by (permutation index, case), so shuffles are resumable
and a repeated run costs nothing.
"""

from __future__ import annotations

import argparse

import numpy as np

from .. import (analysis, config, figures, io_utils, permutation, powerfamily,
                stats)


def _gen_path():
    return config.PATHS.results / "test4_generations.jsonl"


def run(n_shuffles: int = 5, signal: str | None = None,
        splits=("layer",), limit: int | None = None,
        within_state: bool = False,
        max_new_tokens: int = config.GEN_MAX_NEW_TOKENS,
        skip_decoding: bool = False) -> dict:
    paths = config.PATHS.mkdirs()
    if "report" in splits:
        io_utils.assert_report_unlocked(paths.report_lock, "stage 07")

    test2 = io_utils.read_json(paths.test_dir("test2") / "test2.json")
    if not test2:
        raise SystemExit("run stage 04 first")
    test3b = io_utils.read_json(paths.test_dir("test3b") / "test3b.json")

    signal = signal or test2["kill"].get("best_internal") or "internal_knowledge"
    layer = int(test2["layer"])

    ctx_rows = io_utils.read_jsonl(paths.results / "capture_ctx.jsonl")
    prior_rows = io_utils.read_jsonl(paths.results / "capture_prior.jsonl")
    screen_rows = io_utils.read_jsonl(paths.prior)
    prior_by_id = {r["fact_id"]: r for r in prior_rows}
    consistency = analysis.consistency_from_prior_samples(screen_rows)

    tbl = analysis.build_predictor_table(
        analysis.filter_split(ctx_rows, splits), prior_by_id, layer, consistency)
    if tbl["labels"].size == 0:
        raise SystemExit(f"no conflict cases in splits {splits}")

    signs = test2.get("orientations", {})
    values = np.asarray(signs.get(signal, 1) * np.asarray(tbl["table"][signal],
                                                          dtype=np.float64))

    with io_utils.stage("07_test4", paths.manifest) as box:
        out_dir = paths.test_dir("test4")

        # ---- signal level ------------------------------------------------- #
        sig_level = permutation.permutation_auc(values, tbl["labels"],
                                                n_shuffles=config.PERMUTATION_N_SHUFFLES)
        result = {"signal": signal, "layer": layer, "splits": list(splits),
                  "signal_level": sig_level}
        print(f"[07] signal-level: real AUC {sig_level['real_auc']:.4f}, "
              f"shuffled mean {sig_level['null_mean']:.4f} "
              f"(should be ~0.5), p={sig_level['p_value']:.3f}")

        if skip_decoding or test3b is None:
            if test3b is None:
                print("[07] no stage 06 output; decoding-level control skipped")
            io_utils.write_json(out_dir / "test4.json", result)
            box.update(result)
            return dict(box)

        # ---- decoding level ----------------------------------------------- #
        threshold = test2["thresholds"].get(signal)
        if threshold is None:
            threshold = stats.accuracy_at_best_threshold(values, tbl["labels"])[1]
        oracle_taus = test3b["oracle_tau_by_state"]
        tau_resist = float(oracle_taus.get("resistance", 0.3))
        tau_correct = float(oracle_taus.get("correction", 1.5))
        best_fixed = float(test3b["best_fixed_tau"])
        best_fixed_em = float(test3b["best_fixed_tau_em"])

        from ..documents import context_prompt, prior_prompt
        from ..model import load_model, stop_token_ids

        bundle = load_model()
        stops = stop_token_ids(bundle)
        facts = {f["fact_id"]: f for f in io_utils.read_jsonl(paths.factset)}
        case_by_id = {c["case_id"]: c for c in io_utils.read_jsonl(paths.states)}
        case_ids = tbl["case_ids"][:limit] if limit else tbl["case_ids"]

        done = {(r["perm"], r["case_id"]) for r in io_utils.read_jsonl(_gen_path())}

        def route_and_generate(perm_index: int, sig_map: dict) -> list[dict]:
            router = powerfamily.SignalRouter(sig_map, threshold, tau_resist,
                                              tau_correct)
            buf, out = [], []
            for cid in case_ids:
                if (perm_index, cid) in done:
                    continue
                case = case_by_id[cid]
                fact = facts[case["fact_id"]]
                p_ctx = context_prompt(fact, case["document"], bundle.tokenizer)
                p_pri = prior_prompt(fact, bundle.tokenizer)
                method = router.for_case(case)
                gen = powerfamily.generate(bundle, p_ctx, p_pri, method,
                                          max_new_tokens=max_new_tokens,
                                          stop_ids=stops)
                row = {"perm": perm_index, "case_id": cid,
                       "fact_id": case["fact_id"], "state": case["state"],
                       "split": case["split"], "routed_tau": method.tau,
                       "gold_aliases": list(fact["gold_aliases"]) + [fact["object"]],
                       "text": gen["text"]}
                buf.append(row)
                out.append(row)
                if len(buf) >= 32:
                    io_utils.append_jsonl(_gen_path(), buf)
                    buf = []
            # Flush after the loop, not on the last index: if the final cases were
            # already done and skipped, an index-based flush never fires and the
            # partial batch is lost.
            io_utils.append_jsonl(_gen_path(), buf)
            return out

        # perm == -1 is the real signal.
        real_map = {cid: float(v) for cid, v in zip(tbl["case_ids"], values)}
        route_and_generate(-1, real_map)

        rng = np.random.default_rng(config.PERMUTATION_SEED)
        for k in range(n_shuffles):
            shuffled = (permutation.permute_within_state(values, tbl["states"], rng)
                        if within_state else permutation.permute_within_all(values, rng))
            route_and_generate(k, {cid: float(v) for cid, v
                                   in zip(tbl["case_ids"], shuffled)})

        # ---- evaluate ------------------------------------------------------ #
        all_rows = io_utils.read_jsonl(_gen_path())
        keep = set(case_ids)
        by_perm: dict[int, list[dict]] = {}
        for r in all_rows:
            if r["case_id"] in keep:
                by_perm.setdefault(int(r["perm"]), []).append(r)

        real_scores = powerfamily.evaluate_predictions(by_perm.get(-1, []))
        perm_scores = [powerfamily.evaluate_predictions(by_perm[k])
                       for k in sorted(by_perm) if k >= 0 and by_perm[k]]

        real_gain = real_scores["overall"] - best_fixed_em
        perm_gains = [p["overall"] - best_fixed_em for p in perm_scores]
        kill = permutation.check_test4_kill(real_gain, perm_gains)

        result.update({
            "threshold": threshold, "tau_resist": tau_resist,
            "tau_correct": tau_correct, "within_state": within_state,
            "best_fixed_tau": best_fixed, "best_fixed_tau_em": best_fixed_em,
            "routed_real": real_scores, "routed_permuted": perm_scores,
            "real_gain_over_best_fixed": real_gain,
            "permuted_gains": perm_gains, "kill": kill,
        })
        io_utils.write_json(out_dir / "test4.json", result)

        figures.permutation_control(
            {best_fixed: real_scores}, [{best_fixed: p} for p in perm_scores],
            paths.figures / "test4_permutation.png")

        box.update({k: v for k, v in result.items() if k != "routed_permuted"})
        print(f"[07] routed real EM {real_scores['overall']:.4f} vs best fixed tau "
              f"({best_fixed:g}) {best_fixed_em:.4f}  -> gain {real_gain:+.4f}")
        print(f"[07] shuffled gains: mean {kill['permuted_mean_gain']:+.4f}, "
              f"max {kill.get('permuted_max_gain', float('nan')):+.4f}")
        if kill["fired"]:
            print("\n[07] *** TEST 4 KILL CRITERION FIRED ***")
            for r in kill["reasons"]:
                print(f"      - {r}")
            print("      The signal is functioning as a global correction-strength "
                  "knob. The per-question adaptivity — the entire claim — is doing "
                  "nothing.\n")
        return dict(box)


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--signal", default=None,
                    help="default: the winner from stage 04")
    ap.add_argument("--splits", nargs="+", default=["layer"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--within-state", action="store_true",
                    help="shuffle within each conflict state (stricter control)")
    ap.add_argument("--skip-decoding", action="store_true",
                    help="signal-level control only, no GPU")
    a = ap.parse_args(argv)
    return run(a.shuffles, a.signal, tuple(a.splits), a.limit, a.within_state,
               skip_decoding=a.skip_decoding)


if __name__ == "__main__":
    main()
