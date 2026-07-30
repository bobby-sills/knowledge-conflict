"""Test 3a: analytic reachability.

For a token pair (a, b), the power family q(y) ∝ p_pri(y)^(1-τ) p_ctx(y)^τ gives

    log q(a) - log q(b) = (1-τ)·ℓ_pri(a,b) + τ·ℓ_ctx(a,b)

which is zero at

    τ*(a,b) = − ℓ_pri(a,b) / (ℓ_ctx(a,b) − ℓ_pri(a,b))

so τ* is where the model's preference between a and b flips. The theory predicts
τ* ∈ (0,1) for resistance cases: the prior favours the gold, the context favours
the corruption, and somewhere between "pure context" and "pure prior" the
preference crosses over. If that holds empirically, plain interpolation is
sufficient for resistance and the extrapolation regime every existing method
lives in (τ > 1) is not just unnecessary but pointed the wrong way.

Reading the sign conventions: with a = gold and b = the corruption, resistance
means ℓ_pri > 0 (prior prefers gold) and ℓ_ctx < 0 (context prefers the
corruption), giving τ* = ℓ_pri / (ℓ_pri − ℓ_ctx) ∈ (0, 1). Correction is the
mirror image — ℓ_pri < 0, ℓ_ctx > 0 — which also lands in (0, 1). A τ* outside
[0, 1] means both distributions agree about the pair and no crossover exists in
the interpolation regime; those cases are reported separately rather than clipped,
because clipping them into the histogram would be the difference between a
finding and an artefact.
"""

from __future__ import annotations

import numpy as np


def log_ratio(logp, a: int, b: int) -> float:
    """ℓ(a,b) = log[p(a)/p(b)], from a log-prob vector."""
    lp = np.asarray(logp, dtype=np.float64)
    return float(lp[a] - lp[b])


def tau_star(l_pri: float, l_ctx: float, eps: float = 1e-9) -> float:
    """Pairwise reversal threshold. NaN when the two log-ratios coincide.

    A zero denominator means the prior and the context prefer a over b by exactly
    the same margin: no value of τ changes the ordering, so there is no crossover
    to report. NaN, not a large number.
    """
    denom = l_ctx - l_pri
    if abs(denom) < eps:
        return float("nan")
    return float(-l_pri / denom)


def competitor(logp_ctx, gold_token: int) -> int:
    """The strongest alternative to the gold token under the context distribution.

    The context's own top choice (excluding gold) is the token the document is
    actually pushing, which is what the gold has to beat.
    """
    lp = np.asarray(logp_ctx, dtype=np.float64).copy()
    lp[gold_token] = -np.inf
    return int(np.argmax(lp))


def case_reachability(logp_pri, logp_ctx, gold_token: int,
                      stated_token: int | None = None) -> dict:
    """τ* for one conflict case, against two choices of competitor.

    `b_ctxtop` — the context's top non-gold token. The general case.
    `b_stated` — the first token of the object the document asserts. For a
                 corrupted document this is the specific falsehood we care about,
                 and it need not be the context's argmax (the model may prefer
                 some third token entirely, which is itself worth seeing).
    """
    b_top = competitor(logp_ctx, gold_token)
    out = {
        "gold_token": int(gold_token),
        "competitor_ctxtop": int(b_top),
        "l_pri_ctxtop": log_ratio(logp_pri, gold_token, b_top),
        "l_ctx_ctxtop": log_ratio(logp_ctx, gold_token, b_top),
    }
    out["tau_star_ctxtop"] = tau_star(out["l_pri_ctxtop"], out["l_ctx_ctxtop"])

    if stated_token is not None and stated_token != gold_token:
        l_pri = log_ratio(logp_pri, gold_token, stated_token)
        l_ctx = log_ratio(logp_ctx, gold_token, stated_token)
        out.update({
            "competitor_stated": int(stated_token),
            "l_pri_stated": l_pri,
            "l_ctx_stated": l_ctx,
            "tau_star_stated": tau_star(l_pri, l_ctx),
        })
    return out


def summarise(taus, in_range=(0.0, 1.0)) -> dict:
    """Distribution of τ* plus the fraction inside the interpolation regime."""
    t = np.asarray([x for x in taus if x is not None], dtype=np.float64)
    finite = t[np.isfinite(t)]
    if finite.size == 0:
        return {"n": 0, "n_finite": 0, "frac_in_interpolation": float("nan")}
    lo, hi = in_range
    inside = (finite > lo) & (finite < hi)
    return {
        "n": int(t.size),
        "n_finite": int(finite.size),
        "frac_undefined": float(1 - finite.size / t.size),
        "frac_in_interpolation": float(inside.mean()),
        "frac_below_0": float((finite <= lo).mean()),
        "frac_above_1": float((finite >= hi).mean()),
        "median": float(np.median(finite)),
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
    }


def summarise_by_state(records, tau_field: str = "tau_star_ctxtop") -> dict:
    """`summarise` split by conflict state — the Test 3a histogram, as numbers."""
    by_state: dict[str, list] = {}
    for r in records:
        by_state.setdefault(r["state"], []).append(r.get(tau_field))
    return {state: summarise(vals) for state, vals in sorted(by_state.items())}
