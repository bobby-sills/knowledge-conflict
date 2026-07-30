"""Recovering the tau a decoding method is *actually* operating at. No torch.

Every method in the power family is affine in log space:

    adjusted = (1 - tau) * log p_pri + tau * log p_ctx + c

so tau is recoverable from a method's output by regressing `adjusted - log p_pri`
on `log p_ctx - log p_pri`.

**Why this module exists.** A method's name and its own `get_tau()` can both be
wrong about the thing the paper is entirely about — whether it interpolates
(tau < 1) or extrapolates (tau > 1). The vendored repo's `CoCoADecoding` reports
`global_alpha` (0.5) while operating at `alpha + gamma` (1.5). So regime claims are
settled here, by measurement, rather than read off a label.

**Why the centring matters.** `c` is never zero in practice: normalising leaves a
constant, and CAD and AdaCAD combine *raw* logits, so their output differs from the
log-space form by the two partition functions. An uncentred projection is biased by
`c * sum(basis) / ||basis||^2`. Empirically that turns a true tau of 2.5 into an
estimate of -0.30 — a confident, plausible, wrong number on the wrong side of the
only boundary that matters. Mean-centring both vectors fits the intercept
implicitly and makes the estimate exact.
"""

from __future__ import annotations

import numpy as np


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))


def fit_tau(adjusted: np.ndarray, logp_ctx: np.ndarray,
            logp_pri: np.ndarray) -> float:
    """Least-squares tau, with the additive constant fitted implicitly.

    `adjusted` may be unnormalised logits; it is log-softmaxed first, which is
    harmless because the power family's argmax is invariant to normalisation.
    Returns NaN when the two distributions coincide — then no tau is
    distinguishable from any other, which is unidentifiable, not zero.
    """
    adj = log_softmax(adjusted)
    lp_ctx = np.asarray(logp_ctx, dtype=np.float64)
    lp_pri = np.asarray(logp_pri, dtype=np.float64)
    basis = lp_ctx - lp_pri
    target = adj - lp_pri
    basis = basis - basis.mean()
    target = target - target.mean()
    denom = float((basis * basis).sum())
    if denom < 1e-12:
        return float("nan")
    return float((target * basis).sum() / denom)


def fit_residual(adjusted: np.ndarray, logp_ctx: np.ndarray,
                 logp_pri: np.ndarray, tau: float) -> float:
    """Max absolute error of the fitted affine form, after normalising both sides.

    Large residual means the method is not affine in log space and cannot be
    summarised by a single tau at all — which is the useful answer, not a failure.
    """
    if tau != tau:
        return float("nan")
    adj = log_softmax(adjusted)
    pred = log_softmax((1 - tau) * np.asarray(logp_pri, dtype=np.float64)
                       + tau * np.asarray(logp_ctx, dtype=np.float64))
    return float(np.abs(adj - pred).max())


def classify(tau: float, atol: float = 1e-6) -> str:
    if tau != tau:
        return "unidentifiable"
    if abs(tau - 1.0) <= atol:
        return "pure context"
    return "extrapolation" if tau > 1.0 else "interpolation"


def describe(adjusted: np.ndarray, logp_ctx: np.ndarray, logp_pri: np.ndarray,
             reported_tau: float | None = None,
             affine_tol: float = 1e-3) -> dict:
    """Fitted tau, residual, regime, and whether a self-report agrees."""
    tau = fit_tau(adjusted, logp_ctx, logp_pri)
    residual = fit_residual(adjusted, logp_ctx, logp_pri, tau)
    return {
        "effective_tau": tau,
        "reported_tau": reported_tau,
        "affine_residual": residual,
        "affine_in_log_space": bool(residual == residual and residual < affine_tol),
        "regime": classify(tau),
        "self_report_matches": bool(reported_tau is not None and tau == tau
                                    and abs(reported_tau - tau) < 1e-3),
    }
