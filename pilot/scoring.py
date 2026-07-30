"""Test 1: the Inside-Out knowledge score, computed from external and internal
candidate scores.

The definition (arXiv 2503.15299): knowledge for a question is the fraction of
(correct, incorrect) candidate pairs in which the correct candidate is ranked
higher. Score the candidates with output token probabilities and that fraction is
*external* knowledge; score them with intermediate computations and it is
*internal* knowledge. With one correct answer it is exactly the AUC of the score
against the correctness labels over the candidate set.

Two details that decide whether the comparison is fair:

**Ties count as half.** A scorer that assigns identical scores to a correct and
an incorrect candidate has not ranked the correct one higher, and it has not
ranked it lower either. Counting ties as 1.0 inflates any coarse or saturated
scorer — and the logit lens at early layers produces plenty of near-ties. This is
the standard AUC tie convention and it is the one Inside-Out's pair-counting
definition implies.

**Same position, same granularity.** The logit lens reads one position: the final
question token. So the primary external score is also read there — the
distribution over the *first* answer token. Scoring the external side over a full
multi-token span and the internal side over one token would compare a sentence
scorer to a word scorer and call the difference "hidden knowledge". The
full-span score is still logged, as a secondary.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# --------------------------------------------------------------------------- #
# Knowledge score
# --------------------------------------------------------------------------- #

def knowledge_score(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Fraction of (correct, incorrect) pairs where the correct one ranks higher.

    Ties contribute 0.5. Returns NaN when there is no such pair, or when any
    score is missing (NaN) — a partially scored candidate set would otherwise
    quietly change the denominator between facts.
    """
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(correct, dtype=bool)
    if s.shape != c.shape or s.ndim != 1:
        raise ValueError("scores and correct must be 1-D and the same length")
    if not np.isfinite(s).all():
        return float("nan")
    pos, neg = s[c], s[~c]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    wins = (diff > 0).sum() + 0.5 * (diff == 0).sum()
    return float(wins / (pos.size * neg.size))


def knowledge_score_matrix(scores: np.ndarray, correct: Sequence[bool]) -> np.ndarray:
    """`knowledge_score` for each row of a (n_rows, n_candidates) matrix.

    Used to get the score at all 33 layers in one call.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"expected 2-D (rows, candidates), got {scores.shape}")
    return np.array([knowledge_score(row, correct) for row in scores])


def top1_index(scores: Sequence[float]) -> int:
    """Index of the highest-scoring candidate. First index wins ties, which keeps
    the gold-first candidate ordering from mattering: gold is index 0, so a tie
    between gold and a distractor would otherwise resolve in gold's favour and
    flatter every scorer. See `top1_is_correct`."""
    s = np.asarray(scores, dtype=np.float64)
    return int(np.argmax(s))


def top1_is_correct(scores: Sequence[float], correct: Sequence[bool]) -> bool:
    """Whether the argmax candidate is correct, counting a tie at the top as wrong.

    Deliberately strict. Candidate index 0 is always the gold answer, so
    `argmax`'s first-index tie-break would credit every tie to gold.
    """
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(correct, dtype=bool)
    if not np.isfinite(s).all():
        return False
    best = s.max()
    tied = np.flatnonzero(s == best)
    if tied.size > 1 and not c[tied].all():
        return False
    return bool(c[tied[0]])


# --------------------------------------------------------------------------- #
# Layer selection (uses the layer split only)
# --------------------------------------------------------------------------- #

def best_layer(per_fact_layer_scores: np.ndarray) -> tuple[int, np.ndarray]:
    """Layer maximising mean internal knowledge score.

    Input is (n_facts, n_layers); rows with NaN are ignored per layer. Returns
    (layer_index, per_layer_means). **Must be called on the layer-selection split
    only** — this is the one choice in the pilot fitted to data, and fitting it on
    the same facts it is reported on is the classic way to manufacture a positive
    Test 1.
    """
    arr = np.asarray(per_fact_layer_scores, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (n_facts, n_layers), got {arr.shape}")
    with np.errstate(invalid="ignore"):
        means = np.nanmean(arr, axis=0)
    if np.isnan(means).all():
        raise ValueError("no finite internal scores at any layer")
    return int(np.nanargmax(means)), means


# --------------------------------------------------------------------------- #
# Internal predictors derived from the per-layer candidate scores
# --------------------------------------------------------------------------- #

def internal_margin(layer_scores: Sequence[float]) -> float:
    """Gap between the best and second-best candidate at one layer.

    A confident internal belief should show a wide margin; a model that is merely
    picking the least-bad option should not.
    """
    s = np.sort(np.asarray(layer_scores, dtype=np.float64))[::-1]
    if s.size < 2 or not np.isfinite(s[:2]).all():
        return float("nan")
    return float(s[0] - s[1])


def trajectory_stability(per_layer_scores: np.ndarray,
                         answer_index: int | None = None) -> float:
    """Fraction of layers at which the eventual answer is already top-1.

    "Eventual answer" is the final layer's top-1 unless `answer_index` is given.
    An answer the model settles on early and never revisits is a different kind of
    belief from one it arrives at in the last two layers, and the spec asks for
    that as a Test 2 predictor.
    """
    arr = np.asarray(per_layer_scores, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected (n_layers, n_candidates), got {arr.shape}")
    finite = np.isfinite(arr).all(axis=1)
    if not finite.any():
        return float("nan")
    if answer_index is None:
        answer_index = int(np.argmax(arr[np.flatnonzero(finite)[-1]]))
    hits = np.argmax(arr[finite], axis=1) == answer_index
    return float(hits.mean())


def first_layer_locked(per_layer_scores: np.ndarray,
                       answer_index: int | None = None) -> int:
    """Earliest layer after which the eventual answer stays top-1 forever.

    Reported alongside `trajectory_stability`: they separate "top-1 at scattered
    layers" from "top-1 from layer 14 onward", which the fraction alone conflates.
    """
    arr = np.asarray(per_layer_scores, dtype=np.float64)
    finite_rows = np.flatnonzero(np.isfinite(arr).all(axis=1))
    if finite_rows.size == 0:
        return -1
    if answer_index is None:
        answer_index = int(np.argmax(arr[finite_rows[-1]]))
    tops = np.argmax(arr[finite_rows], axis=1)
    locked = finite_rows[-1]
    for pos in range(len(tops) - 1, -1, -1):
        if tops[pos] != answer_index:
            break
        locked = finite_rows[pos]
    return int(locked)


# --------------------------------------------------------------------------- #
# Divergence between internal and external
# --------------------------------------------------------------------------- #

def divergence_report(records: Sequence[dict]) -> dict:
    """Test 1's headline numbers.

    Each record needs: `internal_scores` and `external_scores` (candidate-aligned
    lists) and `correct_mask`.

    Reports the divergence rate — how often internal top-1 differs from external
    top-1 — and, on that divergent subset, how often each side is the correct one.
    That second number is the one the kill criterion turns on: a large divergence
    where internal is no better means the internals hold nothing the output
    distribution lacks.
    """
    n = 0
    diverged = 0
    int_right = ext_right = both_wrong = 0
    int_scores, ext_scores, states = [], [], []
    for r in records:
        cm = r["correct_mask"]
        i_s, e_s = r["internal_scores"], r["external_scores"]
        if not (np.isfinite(np.asarray(i_s, float)).all()
                and np.isfinite(np.asarray(e_s, float)).all()):
            continue
        n += 1
        int_scores.append(knowledge_score(i_s, cm))
        ext_scores.append(knowledge_score(e_s, cm))
        states.append(r.get("state"))
        if top1_index(i_s) != top1_index(e_s):
            diverged += 1
            i_ok = top1_is_correct(i_s, cm)
            e_ok = top1_is_correct(e_s, cm)
            int_right += int(i_ok and not e_ok)
            ext_right += int(e_ok and not i_ok)
            both_wrong += int(not i_ok and not e_ok)
    if n == 0:
        raise ValueError("no fully scored records")
    return {
        "n_facts": n,
        "internal_knowledge_mean": float(np.nanmean(int_scores)),
        "external_knowledge_mean": float(np.nanmean(ext_scores)),
        "knowledge_gap": float(np.nanmean(int_scores) - np.nanmean(ext_scores)),
        "divergence_rate": diverged / n,
        "n_diverged": diverged,
        "on_divergent_internal_correct": int_right / diverged if diverged else float("nan"),
        "on_divergent_external_correct": ext_right / diverged if diverged else float("nan"),
        "on_divergent_both_wrong": both_wrong / diverged if diverged else float("nan"),
        "internal_scores": int_scores,
        "external_scores": ext_scores,
        "states": states,
    }


def divergence_result_or_none(records: Sequence[dict]) -> dict | None:
    """`divergence_report` without the raw per-fact vectors, or None if the subset
    is empty. Used for the collision-restricted secondary read, where the subset
    can legitimately be too small to say anything."""
    if len(records) < 20:
        return None
    rep = divergence_report(records)
    return {k: v for k, v in rep.items()
            if k not in ("internal_scores", "external_scores", "states")}


def check_test1_kill(report: dict, kill) -> dict:
    """Evaluate Test 1's kill criterion against the thresholds fixed in config."""
    low_divergence = report["divergence_rate"] < kill.min_divergence_rate
    internal_loses = not (report["on_divergent_internal_correct"]
                          > report["on_divergent_external_correct"])
    fired = bool(low_divergence or (kill.divergent_subset_internal_must_win
                                    and internal_loses))
    reasons = []
    if low_divergence:
        reasons.append(
            f"divergence rate {report['divergence_rate']:.3f} < "
            f"{kill.min_divergence_rate}")
    if internal_loses:
        reasons.append(
            f"on the divergent subset internal is correct "
            f"{report['on_divergent_internal_correct']:.3f} vs external "
            f"{report['on_divergent_external_correct']:.3f}")
    return {"test": "test1", "fired": fired, "reasons": reasons}
