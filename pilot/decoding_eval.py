"""Scoring generated answers, tuning the oracle, and Test 3's gate. No torch.

Split out of `powerfamily.py` so that the analysis half of Test 3 runs on a
CPU-only machine: generations are written to JSONL as they are produced, and
everything after that is arithmetic over strings. A dead Colab session should
never force a regeneration just to recompute an EM.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import config


def evaluate_predictions(rows: Sequence[Mapping], metric: str = "em") -> dict:
    """Overall and per-state scores from generated rows.

    Each row needs `state`, `text`, `gold_aliases`. Both strict EM and substring EM
    are computed with the vendored normalisation; `metric` picks which one the
    trade-off curves use. Substring EM is the fairer read on an instruct model,
    which will answer "George Washington." with a period or wrap the answer in a
    short phrase — so both are reported side by side rather than one being chosen
    quietly.
    """
    from .vendor import em_any, substring_any
    per_state: dict[str, list[float]] = {}
    per_state_sub: dict[str, list[float]] = {}
    overall: list[float] = []
    overall_sub: list[float] = []
    for r in rows:
        golds = r["gold_aliases"]
        em = float(em_any(r["text"], golds))
        sub = float(substring_any(r["text"], golds))
        primary = em if metric == "em" else sub
        per_state.setdefault(r["state"], []).append(primary)
        per_state_sub.setdefault(r["state"], []).append(sub)
        overall.append(primary)
        overall_sub.append(sub)

    def mean(v):
        return float(sum(v) / len(v)) if v else float("nan")

    out = {"overall": mean(overall), "overall_substring": mean(overall_sub),
           "n": len(overall), "metric": metric}
    for state, vals in sorted(per_state.items()):
        out[state] = mean(vals)
        out[f"{state}_substring"] = mean(per_state_sub[state])
        out[f"n_{state}"] = len(vals)
    return out


def tune_oracle(rows_by_tau: Mapping[float, Sequence[Mapping]],
                metric: str = "em") -> dict:
    """Pick the three oracle tau constants: the best tau per state.

    Given a tau sweep that has already been run over every case, the best per-state
    tau is an argmax over the sweep — no extra generation. That is why the sweep
    comes first: the oracle is a *re-analysis* of it, so it cannot accidentally be
    tuned on more data than the sweep saw. Ties break toward tau=1, the
    do-nothing value, so a state with a flat curve does not get a spuriously
    extreme constant.
    """
    from .vendor import em_any, substring_any
    scores: dict[str, dict[float, list[float]]] = {}
    for tau, rows in rows_by_tau.items():
        for r in rows:
            ok = float(em_any(r["text"], r["gold_aliases"]) if metric == "em"
                       else substring_any(r["text"], r["gold_aliases"]))
            scores.setdefault(r["state"], {}).setdefault(float(tau), []).append(ok)
    best: dict[str, float] = {}
    curves: dict[str, dict[float, float]] = {}
    for state, by_tau in scores.items():
        curve = {tau: sum(v) / len(v) for tau, v in sorted(by_tau.items())}
        curves[state] = curve
        best[state] = max(curve, key=lambda t: (curve[t], -abs(t - 1.0)))
    return {"tau_by_state": best, "curves": curves}


def oracle_from_sweep(rows_by_tau: Mapping[float, Sequence[Mapping]],
                      tau_by_state: Mapping[str, float]) -> list[dict]:
    """The oracle's predictions: per case, the row generated at that case's routed
    tau. Provably identical to regenerating, and free."""
    out = []
    for tau, rows in rows_by_tau.items():
        for r in rows:
            target = tau_by_state.get(r["state"])
            if target is not None and abs(float(tau) - float(target)) < 1e-9:
                out.append(dict(r, method="oracle3", routed_tau=float(tau)))
    return out


def check_test3_kill(oracle: Mapping[str, float],
                     baselines: Mapping[str, Mapping[str, float]],
                     kill=config.KILL) -> dict:
    """Test 3's kill criterion: does oracle routing clear the best baseline?

    "Barely beats" is made concrete as `min_oracle_gain_over_best_baseline_em` on
    overall EM, against the strongest baseline actually run — which should include
    **ARR**, the method arXiv 2606.10298 publishes and the one the spec names. See
    DECISIONS.md §1.2 for why the vendored pin is not HEAD.
    """
    if not baselines:
        return {"test": "test3", "fired": True, "reasons": ["no baselines run"]}
    best_name = max(baselines,
                    key=lambda n: baselines[n].get("overall", float("-inf")))
    best = baselines[best_name].get("overall", float("nan"))
    gain = oracle.get("overall", float("nan")) - best
    fired = not (gain >= kill.min_oracle_gain_over_best_baseline_em)
    res_gain = (oracle.get("resistance", float("nan"))
                - baselines[best_name].get("resistance", float("nan")))
    return {
        "test": "test3", "fired": bool(fired),
        "reasons": ([f"oracle gain {gain:+.4f} EM over best baseline "
                     f"({best_name} {best:.4f}) < required "
                     f"{kill.min_oracle_gain_over_best_baseline_em}"] if fired else []),
        "best_baseline": best_name, "best_baseline_em": best,
        "oracle_em": oracle.get("overall"), "gain": float(gain),
        # A ceiling that is high overall but flat on resistance is still a dead end
        # for this project, so the per-state gain is surfaced next to the headline.
        "resistance_gain": float(res_gain),
    }
