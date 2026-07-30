"""AUC, bootstrap confidence intervals, and the error-correlation matrix.

Test 2's kill criterion is "+0.05 AUC with non-overlapping CIs", so both halves
of that sentence need an implementation that will survive a reviewer:

* AUC via the rank (Mann-Whitney) identity, with ties at 0.5. Threshold-sweeping
  gives the same number only when there are no ties, and coarse signals like
  `trajectory_stability` (33 possible values) are nothing but ties.

* Bootstrap by resampling *questions*, and resampling the same question indices
  for every predictor. Independent resamples per predictor would inflate the
  apparent difference between two predictors that agree, which is precisely the
  comparison the gate rests on. The paired difference CI is reported too — it is
  the right test for "does A beat B", and non-overlapping marginal CIs is the
  weaker, more conservative condition the spec actually named.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from . import config


# --------------------------------------------------------------------------- #
# AUC
# --------------------------------------------------------------------------- #

def auc(scores: Sequence[float], labels: Sequence[bool]) -> float:
    """Area under the ROC curve; P(score(pos) > score(neg)) with ties at 0.5.

    NaN scores are dropped pairwise (with their labels) rather than imputed. NaN
    here means "this predictor could not be computed for this question", and
    filling it with a mean would let one predictor's missingness pattern leak into
    another's comparison.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    if s.shape != y.shape:
        raise ValueError(f"shape mismatch: {s.shape} vs {y.shape}")
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < len(sorted_s):
        j = i
        while j + 1 < len(sorted_s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0    # average rank, 1-based
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def accuracy_at_best_threshold(scores: Sequence[float],
                               labels: Sequence[bool]) -> tuple[float, float]:
    """Best achievable accuracy and the threshold achieving it.

    Only used to turn a signal into a routing policy in Test 3/4, and fitted on
    the train split. Reported alongside AUC because a signal can have a good AUC
    and still have no threshold that routes usefully.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    keep = np.isfinite(s)
    s, y = s[keep], y[keep]
    if s.size == 0:
        return float("nan"), float("nan")
    cuts = np.unique(s)
    mids = np.concatenate([[cuts[0] - 1.0], (cuts[:-1] + cuts[1:]) / 2.0,
                           [cuts[-1] + 1.0]])
    best_acc, best_cut = -1.0, float("nan")
    for cut in mids:
        acc = float(((s >= cut) == y).mean())
        if acc > best_acc:
            best_acc, best_cut = acc, float(cut)
    return best_acc, best_cut


# --------------------------------------------------------------------------- #
# Bootstrap
# --------------------------------------------------------------------------- #

def bootstrap_auc_table(table: Mapping[str, Sequence[float]],
                        labels: Sequence[bool],
                        n_boot: int = config.BOOTSTRAP_N,
                        ci: float = config.BOOTSTRAP_CI,
                        seed: int = config.BOOTSTRAP_SEED) -> dict:
    """AUC + bootstrap CI for every predictor, on shared question resamples.

    Returns
        {name: {auc, lo, hi, n}}  plus  {"_draws": (n_boot, n_predictors)}
    so that paired differences can be computed from the same draws.
    """
    y = np.asarray(labels, dtype=bool)
    names = list(table.keys())
    mat = np.vstack([np.asarray(table[n], dtype=np.float64) for n in names])
    n = int(y.size)
    if mat.shape[1] != n:
        raise ValueError(f"predictor length {mat.shape[1]} != labels {n}")

    point = {name: auc(mat[i], y) for i, name in enumerate(names)}

    rng = np.random.default_rng(seed)
    draws = np.full((n_boot, len(names)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.all() or not yb.any():
            continue                       # degenerate resample, no AUC exists
        for i in range(len(names)):
            draws[b, i] = auc(mat[i, idx], yb)

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    out = {}
    for i, name in enumerate(names):
        col = draws[:, i]
        col = col[np.isfinite(col)]
        out[name] = {
            "auc": point[name],
            "lo": float(np.quantile(col, lo_q)) if col.size else float("nan"),
            "hi": float(np.quantile(col, hi_q)) if col.size else float("nan"),
            "n_boot_valid": int(col.size),
            "n": int(np.isfinite(mat[i]).sum()),
        }
    return {"per_signal": out, "_draws": draws, "_names": names,
            "n_boot": n_boot, "ci": ci}


def paired_difference(boot: dict, name_a: str, name_b: str,
                      ci: float | None = None) -> dict:
    """CI on AUC(a) - AUC(b) from the shared draws. The right test for "A beats B"."""
    ci = ci if ci is not None else boot["ci"]
    names = boot["_names"]
    ia, ib = names.index(name_a), names.index(name_b)
    diff = boot["_draws"][:, ia] - boot["_draws"][:, ib]
    diff = diff[np.isfinite(diff)]
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    point = boot["per_signal"][name_a]["auc"] - boot["per_signal"][name_b]["auc"]
    return {
        "a": name_a, "b": name_b, "diff": float(point),
        "lo": float(np.quantile(diff, lo_q)) if diff.size else float("nan"),
        "hi": float(np.quantile(diff, hi_q)) if diff.size else float("nan"),
        "p_a_gt_b": float((diff > 0).mean()) if diff.size else float("nan"),
    }


def non_overlapping(a: dict, b: dict) -> bool:
    """Whether two marginal CIs are disjoint."""
    return bool(a["lo"] > b["hi"] or b["lo"] > a["hi"])


# --------------------------------------------------------------------------- #
# Error correlation — the "complementary rather than superior" check
# --------------------------------------------------------------------------- #

def error_vectors(table: Mapping[str, Sequence[float]],
                  labels: Sequence[bool],
                  thresholds: Mapping[str, float] | None = None) -> dict:
    """Per-question 0/1 error indicator for each predictor.

    Thresholds come from `accuracy_at_best_threshold` on the train split unless
    supplied. Missing values are marked as errors: a predictor that cannot decide
    has not decided correctly.
    """
    y = np.asarray(labels, dtype=bool)
    out = {}
    for name, values in table.items():
        s = np.asarray(values, dtype=np.float64)
        if thresholds and name in thresholds:
            cut = thresholds[name]
        else:
            _, cut = accuracy_at_best_threshold(s, y)
        pred = np.where(np.isfinite(s), s >= cut, False)
        out[name] = (pred != y).astype(np.float64)
    return out


def error_correlation_matrix(errors: Mapping[str, np.ndarray]) -> dict:
    """Phi (Pearson-on-binary) correlation between predictors' error patterns.

    This matters even when Test 2 fails. Two signals with the same AUC and
    uncorrelated errors are complementary: the internal signal would then be worth
    keeping as an ensemble component rather than as a replacement, which is a
    weaker but real result. A bare "we lost" without this matrix throws that away.

    A predictor whose error vector is constant (never wrong, or always wrong) has
    no correlation with anything — 0/0. Those entries stay **NaN** and the
    predictor is listed in `degenerate`, rather than being coerced to 0.0: a zero
    would read as "uncorrelated errors", which is the finding that keeps the
    project alive as an ensemble method, and it must not be manufactured by a
    division by zero.
    """
    names = list(errors.keys())
    mat = np.vstack([errors[n] for n in names])
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(mat)
    corr = np.atleast_2d(np.asarray(corr, dtype=np.float64))
    degenerate = [n for n in names if np.std(errors[n]) == 0]
    for i, n in enumerate(names):
        if n in degenerate:
            corr[i, :] = np.nan
            corr[:, i] = np.nan
        corr[i, i] = 1.0
    return {"names": names, "matrix": corr.tolist(), "degenerate": degenerate,
            "error_rates": {n: float(errors[n].mean()) for n in names}}


# --------------------------------------------------------------------------- #
# Test 2 gate
# --------------------------------------------------------------------------- #

def check_test2_kill(boot: dict, internal_names: Sequence[str],
                     external_names: Sequence[str], kill=config.KILL) -> dict:
    """Apply Test 2's kill criterion. This is the real gate.

    Fires unless the best internal signal beats the best external signal by at
    least `min_auc_margin` with non-overlapping CIs. The comparison is
    best-vs-best, not internal-vs-average-external: beating a weak baseline is not
    the claim.
    """
    per = boot["per_signal"]
    avail_int = [n for n in internal_names if n in per and np.isfinite(per[n]["auc"])]
    avail_ext = [n for n in external_names if n in per and np.isfinite(per[n]["auc"])]
    if not avail_int or not avail_ext:
        return {"test": "test2", "fired": True,
                "reasons": ["missing internal or external AUCs"]}

    best_int = max(avail_int, key=lambda n: per[n]["auc"])
    best_ext = max(avail_ext, key=lambda n: per[n]["auc"])
    margin = per[best_int]["auc"] - per[best_ext]["auc"]
    disjoint = non_overlapping(per[best_int], per[best_ext])
    paired = paired_difference(boot, best_int, best_ext)

    fired = bool(margin < kill.min_auc_margin
                 or (kill.require_non_overlapping_ci and not disjoint))
    reasons = []
    if margin < kill.min_auc_margin:
        reasons.append(f"margin {margin:+.4f} < required {kill.min_auc_margin}")
    if kill.require_non_overlapping_ci and not disjoint:
        reasons.append(
            f"CIs overlap: {best_int} [{per[best_int]['lo']:.3f},"
            f"{per[best_int]['hi']:.3f}] vs {best_ext} "
            f"[{per[best_ext]['lo']:.3f},{per[best_ext]['hi']:.3f}]")
    return {
        "test": "test2", "fired": fired, "reasons": reasons,
        "best_internal": best_int, "best_external": best_ext,
        "margin": float(margin), "ci_disjoint": disjoint, "paired": paired,
        # Narrow failure -> look at the error correlation before concluding.
        "narrow_failure": bool(fired and margin > 0),
    }
