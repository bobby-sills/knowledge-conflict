"""End-to-end analysis on synthetic capture rows. No model, no GPU.

This is the test that catches wiring errors: whether the layer index actually
selects the layer, whether the labels line up with the cases, whether agreement
cases are excluded from Test 2's binary. Individually-correct functions wired
together wrongly produce a plausible AUC table, which is the worst kind of bug
this pilot can have.
"""

import numpy as np
import pytest

from pilot.analysis import (CONFLICT_STATES, build_predictor_table,
                           consistency_from_prior_samples,
                           external_knowledge_table, filter_split,
                           per_layer_internal_scores, signal_for_routing)
from pilot.records import (external_full_span_vector, external_vector,
                          internal_matrix)


N_LAYERS = 5
N_CAND = 3


def _prior_row(fact_id, split="train", best_layer=3, gold_wins=True):
    """A capture row where the gold candidate wins only at `best_layer`."""
    internal = []
    for layer in range(N_LAYERS):
        if layer == best_layer and gold_wins:
            internal.append([-0.1, -2.0, -3.0])       # gold (index 0) on top
        else:
            internal.append([-3.0, -0.1, -2.0])       # a distractor on top
    return {
        "fact_id": fact_id,
        "split": split,
        "correct_mask": [True, False, False],
        "internal_logp_by_layer": internal,
        "external_first_token_logp": [-2.5, -0.2, -1.0],   # external prefers d1
        "external_full_span": [{"mean_logp": -1.0, "sum_logp": -2.0, "n_tokens": 2},
                               {"mean_logp": -0.5, "sum_logp": -1.0, "n_tokens": 2},
                               {"mean_logp": -3.0, "sum_logp": -6.0, "n_tokens": 2}],
        "candidate_token_ids": [10, 11, 12],
        "usable_first_token": True,
        "prior_entropy": 1.0,
        "prior_max": 0.8,
        "log_popularity": 3.0,
    }


def _ctx_row(case_id, fact_id, state, split="train", **kw):
    row = {
        "case_id": case_id, "fact_id": fact_id, "state": state, "split": split,
        "doc_variant": "corrupted" if state == "resistance" else "faithful",
        "prior_entropy": 1.0, "prior_max": 0.8, "entropy_gap": -0.5,
        "jsd": 0.3, "renyi": 0.7, "log_popularity": 3.0,
        "gold_token": 10, "stated_token": 11,
        "tau_star_ctxtop": 0.4,
    }
    row.update(kw)
    return row


class TestRecords:
    def test_internal_matrix_shape_and_nan(self):
        row = _prior_row("f1")
        row["internal_logp_by_layer"][0][1] = None
        mat = internal_matrix(row)
        assert mat.shape == (N_LAYERS, N_CAND)
        assert np.isnan(mat[0, 1])

    def test_external_vector_nan(self):
        row = _prior_row("f1")
        row["external_first_token_logp"][2] = None
        vec = external_vector(row)
        assert np.isnan(vec[2])
        assert vec[0] == -2.5

    def test_full_span_vector(self):
        assert external_full_span_vector(_prior_row("f1"))[1] == -0.5

    def test_full_span_missing_is_empty(self):
        row = _prior_row("f1")
        del row["external_full_span"]
        assert external_full_span_vector(row).size == 0


class TestPerLayerScores:
    def test_shape_matches_facts_by_layers(self):
        rows = [_prior_row(f"f{i}") for i in range(4)]
        mat = per_layer_internal_scores(rows)
        assert mat.shape == (4, N_LAYERS)

    def test_the_planted_layer_wins(self):
        from pilot.scoring import best_layer
        rows = [_prior_row(f"f{i}", best_layer=3) for i in range(6)]
        layer, means = best_layer(per_layer_internal_scores(rows))
        assert layer == 3
        assert means[3] == 1.0

    def test_layers_where_a_distractor_wins_score_low(self):
        rows = [_prior_row(f"f{i}", best_layer=3) for i in range(6)]
        mat = per_layer_internal_scores(rows)
        assert mat[:, 0].mean() < 0.5


class TestExternalKnowledgeTable:
    def test_records_carry_both_scorings(self):
        rows = [_prior_row("f1")]
        recs = external_knowledge_table(rows, layer=3)["records"]
        assert len(recs) == 1
        assert recs[0]["internal_scores"][0] == -0.1
        assert recs[0]["external_scores"][0] == -2.5

    def test_out_of_range_layer_is_skipped(self):
        assert external_knowledge_table([_prior_row("f1")], layer=99)["records"] == []

    def test_divergence_is_detected_on_the_planted_setup(self):
        # internal picks the gold at layer 3, external picks a distractor.
        from pilot.scoring import divergence_report
        rows = [_prior_row(f"f{i}", best_layer=3) for i in range(10)]
        recs = external_knowledge_table(rows, layer=3)["records"]
        rep = divergence_report(recs)
        assert rep["divergence_rate"] == 1.0
        assert rep["on_divergent_internal_correct"] == 1.0
        assert rep["on_divergent_external_correct"] == 0.0


class TestPredictorTable:
    def _fixture(self):
        prior_by_id = {f"f{i}": _prior_row(f"f{i}") for i in range(4)}
        ctx_rows = [
            _ctx_row("c0", "f0", "correction"),
            _ctx_row("c1", "f1", "resistance"),
            _ctx_row("c2", "f2", "agreement"),
            _ctx_row("c3", "f3", "resistance", split="layer"),
        ]
        return ctx_rows, prior_by_id

    def test_agreement_cases_are_excluded(self):
        ctx_rows, prior_by_id = self._fixture()
        out = build_predictor_table(ctx_rows, prior_by_id, layer=3)
        assert "c2" not in out["case_ids"]
        assert set(out["states"]) <= set(CONFLICT_STATES)

    def test_labels_mark_resistance(self):
        ctx_rows, prior_by_id = self._fixture()
        out = build_predictor_table(ctx_rows, prior_by_id, layer=3)
        pairs = dict(zip(out["case_ids"], out["labels"]))
        assert pairs["c0"] is np.False_ or pairs["c0"] == False   # noqa: E712
        assert pairs["c1"]
        assert pairs["c3"]

    def test_every_declared_signal_is_present(self):
        from pilot.signals import ALL_SIGNALS
        ctx_rows, prior_by_id = self._fixture()
        out = build_predictor_table(ctx_rows, prior_by_id, layer=3)
        assert set(out["table"]) == set(ALL_SIGNALS)
        for name, col in out["table"].items():
            assert col.shape == out["labels"].shape, name

    def test_layer_argument_changes_the_internal_columns(self):
        # The planted setup makes layer 3 the good one; layer 0 must differ.
        ctx_rows, prior_by_id = self._fixture()
        good = build_predictor_table(ctx_rows, prior_by_id, layer=3)
        bad = build_predictor_table(ctx_rows, prior_by_id, layer=0)
        assert not np.allclose(good["table"]["internal_knowledge"],
                               bad["table"]["internal_knowledge"])
        assert good["table"]["internal_knowledge"][0] == 1.0
        assert bad["table"]["internal_knowledge"][0] == 0.0

    def test_out_of_range_layer_raises(self):
        ctx_rows, prior_by_id = self._fixture()
        with pytest.raises(IndexError):
            build_predictor_table(ctx_rows, prior_by_id, layer=99)

    def test_missing_prior_row_is_skipped(self):
        ctx_rows, _ = self._fixture()
        out = build_predictor_table(ctx_rows, {}, layer=3)
        assert out["case_ids"] == []

    def test_self_consistency_is_joined_by_fact(self):
        ctx_rows, prior_by_id = self._fixture()
        out = build_predictor_table(ctx_rows, prior_by_id, layer=3,
                                    consistency_by_id={"f1": 0.75})
        pairs = dict(zip(out["case_ids"], out["table"]["self_consistency"]))
        assert pairs["c1"] == 0.75
        assert np.isnan(pairs["c0"])

    def test_trajectory_stability_reflects_the_planted_trajectory(self):
        # The gold is top-1 at exactly one of five layers, but the *eventual*
        # answer (final layer's top-1) is a distractor, top-1 at four of five.
        ctx_rows, prior_by_id = self._fixture()
        out = build_predictor_table(ctx_rows, prior_by_id, layer=3)
        assert out["table"]["trajectory_stability"][0] == pytest.approx(4 / 5)


class TestFilterAndRouting:
    def test_filter_split(self):
        rows = [{"split": "train"}, {"split": "layer"}, {"split": "report"}]
        assert len(filter_split(rows, ("train", "layer"))) == 2

    def test_filter_split_excludes_report_by_default_usage(self):
        rows = [{"split": "report"}]
        assert filter_split(rows, ("train", "layer")) == []

    def test_consistency_from_samples(self):
        rows = [{"fact_id": "f1", "samples": ["Paris", "Paris", "Lyon"]}]
        out = consistency_from_prior_samples(rows)
        assert out["f1"] == pytest.approx(2 / 3)

    def test_zero_consistency_is_kept_not_turned_into_nan(self):
        # A truthiness check here would map a genuine 0.0 to NaN and drop the case
        # from the AUC.
        from pilot.analysis import build_predictor_table
        prior_by_id = {"f0": _prior_row("f0")}
        ctx = [_ctx_row("c0", "f0", "resistance")]
        out = build_predictor_table(ctx, prior_by_id, layer=3,
                                    consistency_by_id={"f0": 0.0})
        assert out["table"]["self_consistency"][0] == 0.0

    def test_signal_for_routing_applies_the_sign(self):
        table = {"prior_entropy": np.array([1.0, 2.0])}
        out = signal_for_routing(table, ["a", "b"], "prior_entropy",
                                 {"prior_entropy": -1})
        assert out == {"a": -1.0, "b": -2.0}
