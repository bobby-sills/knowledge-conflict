"""Tests for AUC, bootstrap CIs, and the error-correlation matrix.

Test 2's kill criterion is "+0.05 AUC with non-overlapping CIs", so these are the
functions the gate is made of.
"""

import numpy as np
import pytest

from pilot.config import KillCriteria
from pilot.stats import (accuracy_at_best_threshold, auc, bootstrap_auc_table,
                         check_test2_kill, error_correlation_matrix,
                         error_vectors, non_overlapping, paired_difference)


class TestAUC:
    def test_perfect_separation(self):
        assert auc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0

    def test_inverted(self):
        assert auc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0

    def test_chance_on_constant_scores(self):
        # All ties -> 0.5, which is the whole reason for the rank formulation.
        assert auc([0.5] * 6, [True, True, True, False, False, False]) == 0.5

    def test_partial_ties(self):
        assert auc([1.0, 1.0, 0.0], [True, False, False]) == pytest.approx(0.75)

    def test_matches_sklearn_including_ties(self):
        sk = pytest.importorskip("sklearn.metrics")
        rng = np.random.default_rng(7)
        for _ in range(30):
            n = int(rng.integers(10, 80))
            # Round to force ties, which is where a threshold-sweep AUC diverges.
            scores = np.round(rng.normal(size=n), 1)
            labels = rng.random(n) < 0.4
            if labels.all() or not labels.any():
                continue
            assert auc(scores, labels) == pytest.approx(
                sk.roc_auc_score(labels, scores))

    def test_nan_scores_dropped_pairwise(self):
        # NaN means "not computable for this question", not "average value".
        assert auc([np.nan, 0.1, 0.9], [True, False, True]) == 1.0

    def test_degenerate_labels_give_nan(self):
        assert np.isnan(auc([0.1, 0.9], [True, True]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            auc([0.1, 0.2], [True])


class TestThreshold:
    def test_finds_separating_threshold(self):
        acc, cut = accuracy_at_best_threshold([0.1, 0.2, 0.8, 0.9],
                                              [False, False, True, True])
        assert acc == 1.0
        assert 0.2 < cut <= 0.8

    def test_all_same_class_predictable(self):
        acc, _ = accuracy_at_best_threshold([0.1, 0.2], [True, True])
        assert acc == 1.0


class TestBootstrap:
    def _table(self, n=200, seed=3):
        rng = np.random.default_rng(seed)
        labels = rng.random(n) < 0.5
        strong = labels + rng.normal(0, 0.4, n)
        weak = labels + rng.normal(0, 2.0, n)
        noise = rng.normal(0, 1, n)
        return {"strong": strong, "weak": weak, "noise": noise}, labels

    def test_ci_brackets_the_point_estimate(self):
        table, labels = self._table()
        boot = bootstrap_auc_table(table, labels, n_boot=200)
        for name, s in boot["per_signal"].items():
            assert s["lo"] <= s["auc"] <= s["hi"], name

    def test_noise_ci_contains_half(self):
        table, labels = self._table()
        boot = bootstrap_auc_table(table, labels, n_boot=300)
        s = boot["per_signal"]["noise"]
        assert s["lo"] < 0.5 < s["hi"]

    def test_shared_draws_are_paired(self):
        # Identical predictors must have a zero-width paired difference. This is
        # what independent per-predictor resampling would get wrong, and it is the
        # comparison Test 2's gate rests on.
        table, labels = self._table()
        table["strong_copy"] = table["strong"].copy()
        boot = bootstrap_auc_table(table, labels, n_boot=100)
        d = paired_difference(boot, "strong", "strong_copy")
        assert d["diff"] == pytest.approx(0.0)
        assert d["lo"] == pytest.approx(0.0)
        assert d["hi"] == pytest.approx(0.0)

    def test_paired_difference_detects_a_real_gap(self):
        table, labels = self._table()
        boot = bootstrap_auc_table(table, labels, n_boot=300)
        d = paired_difference(boot, "strong", "noise")
        assert d["diff"] > 0.2
        assert d["lo"] > 0
        assert d["p_a_gt_b"] > 0.99

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            bootstrap_auc_table({"a": [0.1, 0.2, 0.3]}, [True, False], n_boot=10)


class TestNonOverlapping:
    def test_disjoint(self):
        assert non_overlapping({"lo": 0.7, "hi": 0.8}, {"lo": 0.5, "hi": 0.6})

    def test_overlapping(self):
        assert not non_overlapping({"lo": 0.6, "hi": 0.8}, {"lo": 0.5, "hi": 0.65})


class TestErrorCorrelation:
    def test_identical_predictors_correlate_perfectly(self):
        # Both wrong on exactly the same two questions.
        labels = np.array([True, True, False, False])
        table = {"a": np.array([1.0, 0.0, 1.0, 0.0]),
                 "b": np.array([1.0, 0.0, 1.0, 0.0])}
        errs = error_vectors(table, labels, {"a": 0.5, "b": 0.5})
        info = error_correlation_matrix(errs)
        assert errs["a"].tolist() == [0.0, 1.0, 1.0, 0.0]
        assert np.asarray(info["matrix"])[0, 1] == pytest.approx(1.0)
        assert info["degenerate"] == []

    def test_a_never_wrong_predictor_is_flagged_not_zeroed(self):
        # Zero variance in the errors means the correlation is 0/0. Reporting 0.0
        # would read as "uncorrelated errors" — the finding that keeps the project
        # alive as an ensemble — so it must stay NaN and be flagged.
        labels = np.array([True, True, False, False])
        table = {"perfect": np.array([1.0, 1.0, 0.0, 0.0]),
                 "flawed": np.array([1.0, 0.0, 1.0, 0.0])}
        errs = error_vectors(table, labels, {"perfect": 0.5, "flawed": 0.5})
        info = error_correlation_matrix(errs)
        mat = np.asarray(info["matrix"])
        assert info["degenerate"] == ["perfect"]
        assert np.isnan(mat[0, 1])
        assert mat[0, 0] == 1.0
        assert info["error_rates"]["perfect"] == 0.0

    def test_complementary_predictors_have_negative_correlation(self):
        # a is wrong exactly where b is right: the "complementary rather than
        # superior" case the spec asks us to check for.
        labels = np.array([True, True, False, False])
        table = {"a": np.array([1.0, 0.0, 0.0, 0.0]),
                 "b": np.array([0.0, 1.0, 1.0, 1.0])}
        errs = error_vectors(table, labels, {"a": 0.5, "b": 0.5})
        assert errs["a"].tolist() == [0.0, 1.0, 0.0, 0.0]
        assert errs["b"].tolist() == [1.0, 0.0, 1.0, 1.0]
        assert np.asarray(error_correlation_matrix(errs)["matrix"])[0, 1] < 0

    def test_missing_values_count_as_errors(self):
        labels = np.array([True, False])
        errs = error_vectors({"a": np.array([np.nan, 0.0])}, labels, {"a": 0.5})
        assert errs["a"].tolist() == [1.0, 0.0]


class TestTest2Gate:
    def _boot(self, internal_auc, external_auc, width=0.01):
        names = ["internal_knowledge", "prior_entropy"]
        per = {
            "internal_knowledge": {"auc": internal_auc, "lo": internal_auc - width,
                                   "hi": internal_auc + width},
            "prior_entropy": {"auc": external_auc, "lo": external_auc - width,
                              "hi": external_auc + width},
        }
        n_boot = 200
        rng = np.random.default_rng(0)
        draws = np.stack([rng.normal(internal_auc, width / 2, n_boot),
                          rng.normal(external_auc, width / 2, n_boot)], axis=1)
        return {"per_signal": per, "_draws": draws, "_names": names,
                "n_boot": n_boot, "ci": 0.95}

    def test_passes_on_a_clear_win(self):
        out = check_test2_kill(self._boot(0.80, 0.70),
                               ["internal_knowledge"], ["prior_entropy"])
        assert out["fired"] is False
        assert out["margin"] == pytest.approx(0.10)

    def test_fires_on_a_small_margin(self):
        out = check_test2_kill(self._boot(0.72, 0.70),
                               ["internal_knowledge"], ["prior_entropy"])
        assert out["fired"] is True
        assert out["narrow_failure"] is True

    def test_fires_on_overlapping_cis_despite_margin(self):
        out = check_test2_kill(self._boot(0.80, 0.70, width=0.08),
                               ["internal_knowledge"], ["prior_entropy"])
        assert out["fired"] is True
        assert any("CIs overlap" in r for r in out["reasons"])

    def test_a_loss_is_not_a_narrow_failure(self):
        out = check_test2_kill(self._boot(0.60, 0.70),
                               ["internal_knowledge"], ["prior_entropy"])
        assert out["fired"] is True
        assert out["narrow_failure"] is False

    def test_thresholds_come_from_config_not_the_call(self):
        # Guardrail: the numbers are fixed in config, in advance.
        k = KillCriteria()
        assert k.min_auc_margin == 0.05
        assert k.require_non_overlapping_ci is True
