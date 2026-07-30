"""Reading captured rows back into arrays. No torch, no GPU.

Kept separate from `capture.py` on purpose. Tests 1, 2 and 4 are re-analyses of
saved data, and the spec is explicit that they must not be reruns of the model —
so the code that reads capture output has to be importable on a machine with no
torch and no GPU. If these helpers lived in `capture.py`, importing them would
pull in torch and the "re-analysis" would quietly require the same environment as
the capture.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def internal_matrix(prior_row: Mapping) -> np.ndarray:
    """(L+1, n_candidates) internal log-probs, with JSON nulls restored to NaN."""
    return np.array([[np.nan if v is None else v for v in layer]
                     for layer in prior_row["internal_logp_by_layer"]],
                    dtype=np.float64)


def external_vector(prior_row: Mapping) -> np.ndarray:
    """Primary external score: log p_pri of each candidate's first answer token."""
    return np.array([np.nan if v is None else v
                     for v in prior_row["external_first_token_logp"]],
                    dtype=np.float64)


def external_full_span_vector(prior_row: Mapping) -> np.ndarray:
    """Secondary external score: length-normalised full-span log-prob.

    Not comparable to the internal score — the lens sees one position — so it is a
    supporting number, never the headline. Returns an empty array when the capture
    ran with --skip-full-span.
    """
    spans = prior_row.get("external_full_span") or []
    return np.array([np.nan if s.get("mean_logp") is None else s["mean_logp"]
                     for s in spans], dtype=np.float64)
