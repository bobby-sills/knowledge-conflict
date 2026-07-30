"""Assembling the Test 2 / Test 4 predictor table from saved capture rows.

Pure re-analysis: no model, no GPU. Test 2's binary is defined only on cases
where the document disagrees with the model — correction and resistance — and the
label is "should we resist", which is true exactly on resistance cases. Agreement
cases are excluded by construction, not by filtering on an outcome.

The internal predictors are read at a single layer, chosen on the layer split and
passed in. Choosing it here from the same rows being scored would be the standard
way to manufacture a positive Test 2, so the layer is an argument, never a
default.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from . import signals as sig
from .records import external_vector, internal_matrix
from .scoring import (internal_margin, knowledge_score, trajectory_stability,
                      first_layer_locked)


CONFLICT_STATES = ("correction", "resistance")


def build_predictor_table(ctx_rows: Sequence[Mapping],
                          prior_by_id: Mapping[str, Mapping],
                          layer: int,
                          consistency_by_id: Mapping[str, float] | None = None,
                          states: Sequence[str] = CONFLICT_STATES) -> dict:
    """One row per conflict case; one column per predictor.

    Returns
        {"case_ids", "labels", "states", "table": {name: np.ndarray}}
    where `labels[i]` is True when the right action is to resist.
    """
    consistency_by_id = consistency_by_id or {}
    case_ids, labels, state_list = [], [], []
    cols: dict[str, list[float]] = {name: [] for name in sig.ALL_SIGNALS}

    for row in ctx_rows:
        if row["state"] not in states:
            continue
        prior_row = prior_by_id.get(row["fact_id"])
        if prior_row is None:
            continue

        internal = internal_matrix(prior_row)          # (L+1, n_cand)
        if layer >= internal.shape[0]:
            raise IndexError(
                f"layer {layer} requested but only {internal.shape[0]} captured")
        cm = prior_row["correct_mask"]
        at_layer = internal[layer]

        case_ids.append(row["case_id"])
        state_list.append(row["state"])
        labels.append(row["state"] == "resistance")

        cols["internal_knowledge"].append(knowledge_score(at_layer, cm))
        cols["internal_margin"].append(internal_margin(at_layer))
        cols["trajectory_stability"].append(trajectory_stability(internal))
        cols["prior_entropy"].append(row["prior_entropy"])
        cols["prior_max"].append(row["prior_max"])
        cols["entropy_gap"].append(row["entropy_gap"])
        cols["jsd"].append(row["jsd"])
        cols["renyi"].append(row["renyi"])
        lp = row.get("log_popularity")
        cols["log_popularity"].append(np.nan if lp is None else float(lp))
        consistency = consistency_by_id.get(row["fact_id"])
        cols["self_consistency"].append(
            np.nan if consistency is None else float(consistency))

    table = {k: np.asarray(v, dtype=np.float64) for k, v in cols.items()}
    return {"case_ids": case_ids, "labels": np.asarray(labels, dtype=bool),
            "states": state_list, "table": table}


def external_knowledge_table(prior_rows: Sequence[Mapping], layer: int) -> dict:
    """Test 1's per-fact numbers: knowledge scores and the divergence inputs."""
    records = []
    for row in prior_rows:
        internal = internal_matrix(row)
        if layer >= internal.shape[0]:
            continue
        records.append({
            "fact_id": row["fact_id"],
            "split": row["split"],
            "correct_mask": row["correct_mask"],
            "internal_scores": internal[layer],
            "external_scores": external_vector(row),
            "usable_first_token": row.get("usable_first_token", True),
            "locked_layer": first_layer_locked(internal),
        })
    return {"records": records}


def per_layer_internal_scores(prior_rows: Sequence[Mapping]) -> np.ndarray:
    """(n_facts, n_layers) matrix of internal knowledge scores. Feeds layer choice."""
    out = []
    for row in prior_rows:
        internal = internal_matrix(row)
        cm = row["correct_mask"]
        out.append([knowledge_score(internal[l], cm) for l in range(internal.shape[0])])
    return np.asarray(out, dtype=np.float64)


def filter_split(rows: Sequence[Mapping], splits: Sequence[str]) -> list[Mapping]:
    keep = set(splits)
    return [r for r in rows if r.get("split") in keep]


def consistency_from_prior_samples(screen_rows: Sequence[Mapping]) -> dict[str, float]:
    """fact_id -> self-consistency over the 8 closed-book samples."""
    return {r["fact_id"]: sig.self_consistency(r.get("samples") or [])
            for r in screen_rows}


def signal_for_routing(table: dict, case_ids: Sequence[str],
                       name: str, signs: Mapping[str, int]) -> dict[str, float]:
    """case_id -> oriented signal value, for `powerfamily.SignalRouter`."""
    values = signs.get(name, 1) * np.asarray(table[name], dtype=np.float64)
    return {cid: float(v) for cid, v in zip(case_ids, values)}
