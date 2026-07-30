"""Test 4: the permutation control.

Shuffle the surviving signal's values across questions, keeping its marginal
distribution fixed, and re-run the routing. If the shuffled curve matches the real
one, the signal was never doing per-question work — it was acting as a global
correction-strength knob, and the per-question adaptivity that the entire claim
rests on contributes nothing. The marginal must be held fixed for the control to
mean anything: resampling from a fitted distribution instead would change the
fraction of questions routed to each τ and confound "adaptivity does nothing" with
"a different global mixture".

Cheapest test in the pilot and the one most likely to catch a false positive, so
it runs against every promising result, not once at the end.

Two flavours:

`permute_within_all`  — shuffle over all cases. Detects a signal that is merely a
                        global knob.
`permute_within_state`— shuffle *within* each conflict state. Stricter: it
                        preserves any between-state difference in the signal's
                        distribution and asks whether the *within*-state ordering
                        carries anything. A signal that survives the first test
                        only because correction and resistance cases have
                        different signal ranges will fail this one, and the
                        difference between the two tells you which kind of
                        adaptivity you actually have.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config


def permute_within_all(values: Sequence[float], rng: np.random.Generator) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).copy()
    rng.shuffle(v)
    return v


def permute_within_state(values: Sequence[float], states: Sequence[str],
                         rng: np.random.Generator) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64).copy()
    st = np.asarray(states)
    for state in np.unique(st):
        idx = np.flatnonzero(st == state)
        v[idx] = rng.permutation(v[idx])
    return v


def permutation_auc(values: Sequence[float], labels: Sequence[bool],
                    n_shuffles: int = config.PERMUTATION_N_SHUFFLES,
                    seed: int = config.PERMUTATION_SEED) -> dict:
    """Null distribution of AUC under shuffling. The cheap version of Test 4.

    Runs on saved numbers only, so it costs nothing and can be applied to every
    signal in the Test 2 table. AUC under a shuffle should sit at 0.5; anything
    else means the resampling is broken, which makes this a self-check as much as a
    control.
    """
    from .stats import auc
    rng = np.random.default_rng(seed)
    real = auc(values, labels)
    null = [auc(permute_within_all(values, rng), labels) for _ in range(n_shuffles)]
    null = np.asarray([x for x in null if np.isfinite(x)])
    return {
        "real_auc": real,
        "null_mean": float(null.mean()) if null.size else float("nan"),
        "null_std": float(null.std()) if null.size else float("nan"),
        "null_max": float(null.max()) if null.size else float("nan"),
        "n_shuffles": int(null.size),
        "p_value": float((null >= real).mean()) if null.size else float("nan"),
    }


def check_test4_kill(real_gain: float, permuted_gains: Sequence[float],
                     kill=config.KILL) -> dict:
    """Test 4's kill criterion, on the decoding-level result.

    `real_gain` is the routed method's gain over the best fixed τ; `permuted_gains`
    are the same quantity with the signal shuffled. The criterion fires when the
    shuffled runs recover more than `max_permuted_fraction_of_real_gain` of the
    real gain — i.e. when the curve does not move much on shuffling.

    Gains are measured against the best *fixed* τ, not against τ=1. Against τ=1,
    a pure global rescale would look like a win and pass a control designed to
    catch exactly that.
    """
    g = np.asarray([x for x in permuted_gains if np.isfinite(x)], dtype=np.float64)
    if g.size == 0:
        return {"test": "test4", "fired": True,
                "reasons": ["no valid permuted runs"]}
    mean_perm = float(g.mean())
    if real_gain <= 0:
        return {"test": "test4", "fired": True,
                "reasons": [f"real gain over best fixed tau is {real_gain:+.4f}; "
                            "nothing for the control to explain"],
                "real_gain": float(real_gain), "permuted_mean_gain": mean_perm}
    ratio = mean_perm / real_gain
    fired = bool(ratio > kill.max_permuted_fraction_of_real_gain)
    return {
        "test": "test4", "fired": fired,
        "reasons": ([f"shuffled signal recovers {ratio:.2%} of the real gain "
                     f"(> {kill.max_permuted_fraction_of_real_gain:.0%}): the gain "
                     f"is a global rescale, not per-question adaptivity"]
                    if fired else []),
        "real_gain": float(real_gain),
        "permuted_mean_gain": mean_perm,
        "permuted_max_gain": float(g.max()),
        "recovered_fraction": float(ratio),
        "n_permutations": int(g.size),
    }
