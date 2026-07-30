"""Test 2's predictors: the resist-or-correct decision signals.

Ours (internal), computed from the context-free forward pass:
    internal_knowledge     Inside-Out score at the layer chosen on the layer split
    internal_margin        top-1 minus top-2 candidate log-prob at that layer
    trajectory_stability   fraction of layers where the eventual answer is top-1

Theirs (external), computed from the two output distributions:
    prior_entropy          H(p_pri)
    prior_max              max p_pri
    entropy_gap            H(p_pri) - H(p_ctx)          CoCoA
    jsd                    JSD(p_pri || p_ctx)          AdaCAD
    renyi                  D_2(p_ctx || p_pri)          CoCoA
    log_popularity         log10 subject pageviews      the frequency control
    self_consistency       agreement across the 8 closed-book samples

The popularity control earns its place: entity frequency correlates with
everything here, and a signal that merely detects "this is a famous entity" would
produce a respectable AUC while carrying no information about the model's
internal state. If log-popularity matches the internal signal, the internal
signal is a frequency detector.

All distribution signals are computed over the full vocabulary at the first
answer-token position, in nats, at capture time — Tests 2 and 4 are re-analyses
of these numbers and never touch the model again.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config

INTERNAL_SIGNALS = ("internal_knowledge", "internal_margin", "trajectory_stability")
EXTERNAL_SIGNALS = ("prior_entropy", "prior_max", "entropy_gap", "jsd", "renyi",
                    "log_popularity", "self_consistency")
ALL_SIGNALS = INTERNAL_SIGNALS + EXTERNAL_SIGNALS

# Sign convention: every signal is oriented so that **larger means "resist"**
# (trust the prior). Without a fixed convention, half the AUCs would come out
# below 0.5 and the "best signal" comparison would be meaningless. A signal whose
# natural direction points the other way is negated here, once, explicitly.
RESIST_ORIENTATION = {
    "internal_knowledge": +1,    # model knows the answer  -> resist
    "internal_margin": +1,       # confident internal belief -> resist
    "trajectory_stability": +1,  # stable belief -> resist
    "prior_entropy": -1,         # uncertain prior -> correct
    "prior_max": +1,             # confident prior -> resist
    "entropy_gap": -1,           # prior sharper than ctx (gap<0) -> resist
    "jsd": 0,                    # symmetric: no a-priori direction (see below)
    "renyi": 0,
    "log_popularity": +1,        # famous entity -> prior likely right -> resist
    "self_consistency": +1,      # consistent samples -> resist
}


# --------------------------------------------------------------------------- #
# Distribution functionals (numpy; inputs are log-probs over the full vocab)
# --------------------------------------------------------------------------- #

def _as_logprobs(x: Sequence[float] | np.ndarray) -> np.ndarray:
    lp = np.asarray(x, dtype=np.float64)
    if lp.ndim != 1:
        raise ValueError(f"expected a 1-D log-prob vector, got {lp.shape}")
    return lp


def entropy(logprobs) -> float:
    """H(p) in nats, from log-probs, with 0*log0 = 0.

    Masked rather than written as `np.where(p > 0, p * lp, 0.0)`: `where`
    evaluates both branches, so a token with zero mass computes 0 * -inf = NaN and
    then discards it. The answer is the same but the run is littered with invalid-op
    warnings, and over a 128k vocabulary most tokens have zero mass — so the real
    NaN that matters would be invisible among them.
    """
    lp = _as_logprobs(logprobs)
    p = np.exp(lp)
    m = p > 0
    return float(-(p[m] * lp[m]).sum())


def max_prob(logprobs) -> float:
    return float(np.exp(_as_logprobs(logprobs).max()))


def kl(logp, logq) -> float:
    """KL(p || q) in nats. Masked for the same reason as `entropy`."""
    lp, lq = _as_logprobs(logp), _as_logprobs(logq)
    p = np.exp(lp)
    m = p > 0
    return float((p[m] * (lp[m] - lq[m])).sum())


def jsd(logp, logq) -> float:
    """Jensen-Shannon divergence, AdaCAD's signal. In nats, so in [0, ln 2]."""
    lp, lq = _as_logprobs(logp), _as_logprobs(logq)
    m = 0.5 * (np.exp(lp) + np.exp(lq))
    logm = np.full_like(m, -np.inf)
    nz = m > 0
    logm[nz] = np.log(m[nz])
    return 0.5 * (kl(lp, logm) + kl(lq, logm))


def renyi(logp, logq, alpha: float = config.RENYI_ALPHA) -> float:
    """Renyi divergence D_alpha(p || q), CoCoA's signal.

    D_alpha = 1/(alpha-1) * log sum_y p(y)^alpha q(y)^(1-alpha), computed in log
    space via logsumexp. Doing it in probability space overflows for alpha=2 the
    moment q has a token with negligible mass, which over a 128k vocabulary is
    every single time.
    """
    if abs(alpha - 1.0) < 1e-12:
        return kl(logp, logq)
    lp, lq = _as_logprobs(logp), _as_logprobs(logq)
    terms = alpha * lp + (1.0 - alpha) * lq
    m = terms.max()
    total = m + np.log(np.exp(terms - m).sum())
    return float(total / (alpha - 1.0))


def distribution_signals(logp_pri, logp_ctx,
                         alpha: float = config.RENYI_ALPHA) -> dict:
    """All five output-distribution signals from one pair of distributions."""
    h_pri, h_ctx = entropy(logp_pri), entropy(logp_ctx)
    return {
        "prior_entropy": h_pri,
        "ctx_entropy": h_ctx,
        "prior_max": max_prob(logp_pri),
        "ctx_max": max_prob(logp_ctx),
        "entropy_gap": h_pri - h_ctx,
        "jsd": jsd(logp_pri, logp_ctx),
        "renyi": renyi(logp_ctx, logp_pri, alpha=alpha),
        "kl_ctx_pri": kl(logp_ctx, logp_pri),
    }


# --------------------------------------------------------------------------- #
# Self-consistency, from Test 0's closed-book samples
# --------------------------------------------------------------------------- #

def self_consistency(samples: Sequence[str]) -> float:
    """Share of the 8 closed-book samples that give the modal answer.

    Not "share that are correct" — that would leak the label straight into a
    predictor of the label. This is agreement, computable without knowing the
    gold answer, which is what makes it a usable decoding-time signal and a fair
    baseline.
    """
    from .vendor import normalize_answer
    if not samples:
        return float("nan")
    norm = [normalize_answer(s) for s in samples]
    counts: dict[str, int] = {}
    for s in norm:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.values()) / len(norm)


# --------------------------------------------------------------------------- #
# Orientation
# --------------------------------------------------------------------------- #

def orient(name: str, values: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    """Flip a signal so that larger means "resist".

    For the symmetric divergences (JSD, Renyi) there is no principled direction —
    they measure *how much* the two distributions differ, not which to trust. Both
    orientations are equally defensible a priori, so the direction is fitted on
    `labels`, and the fitting must therefore happen on the train split only. The
    fitted direction is one free parameter granted to a baseline, which is the
    right way to lose an argument with a reviewer rather than win one cheaply.
    """
    values = np.asarray(values, dtype=np.float64)
    sign = RESIST_ORIENTATION.get(name, 0)
    if sign != 0:
        return sign * values
    if labels is None:
        return values
    from .stats import auc
    a = auc(values, labels)
    return values if (np.isnan(a) or a >= 0.5) else -values


def fit_orientations(table: dict[str, np.ndarray], labels: np.ndarray) -> dict[str, int]:
    """Directions for the sign-free signals, fitted on the train split.

    Returns name -> +1/-1 for every signal, so that the analysis stage can apply
    exactly the same directions to the layer split without refitting.
    """
    from .stats import auc
    out = {}
    for name, values in table.items():
        sign = RESIST_ORIENTATION.get(name, 0)
        if sign != 0:
            out[name] = sign
            continue
        a = auc(np.asarray(values, dtype=np.float64), labels)
        out[name] = 1 if (np.isnan(a) or a >= 0.5) else -1
    return out


def apply_orientations(table: dict[str, np.ndarray],
                       signs: dict[str, int]) -> dict[str, np.ndarray]:
    return {name: signs.get(name, 1) * np.asarray(v, dtype=np.float64)
            for name, v in table.items()}
