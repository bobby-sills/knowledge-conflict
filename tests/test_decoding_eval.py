"""Tests for EM scoring, oracle tuning, and the Test 3 / Test 4 gates.

No torch: this is the half of Test 3 that must run after a dead GPU session.
"""

import numpy as np
import pytest

from pilot.config import KILL
from pilot.decoding_eval import (check_test3_kill, evaluate_predictions,
                                 oracle_from_sweep, tune_oracle)
from pilot.permutation import (check_test4_kill, permutation_auc,
                               permute_within_all, permute_within_state)


def _row(case_id, state, text, gold="George Washington", tau=None):
    return {"case_id": case_id, "state": state, "text": text,
            "gold_aliases": [gold], "tau": tau}


class TestEvaluatePredictions:
    def test_per_state_and_overall(self):
        rows = [_row("a", "correction", "George Washington"),
                _row("b", "correction", "John Adams"),
                _row("c", "resistance", "George Washington")]
        out = evaluate_predictions(rows)
        assert out["overall"] == pytest.approx(2 / 3)
        assert out["correction"] == 0.5
        assert out["resistance"] == 1.0
        assert out["n_correction"] == 2

    def test_normalisation_handles_punctuation_and_articles(self):
        out = evaluate_predictions([_row("a", "correction", "the George Washington.")])
        assert out["overall"] == 1.0

    def test_substring_is_reported_alongside_strict_em(self):
        # An instruct model wrapping the answer in a phrase fails strict EM but
        # passes substring EM; both are reported so neither is chosen quietly.
        rows = [_row("a", "correction", "It was George Washington who led them")]
        out = evaluate_predictions(rows, metric="em")
        assert out["overall"] == 0.0
        assert out["overall_substring"] == 1.0

    def test_empty_is_nan_not_zero(self):
        assert np.isnan(evaluate_predictions([])["overall"])


class TestTuneOracle:
    def test_picks_the_best_tau_per_state(self):
        rows_by_tau = {
            0.3: [_row("a", "resistance", "George Washington"),
                  _row("b", "correction", "wrong")],
            1.5: [_row("a", "resistance", "wrong"),
                  _row("b", "correction", "George Washington")],
        }
        out = tune_oracle(rows_by_tau)
        assert out["tau_by_state"]["resistance"] == 0.3
        assert out["tau_by_state"]["correction"] == 1.5

    def test_ties_break_toward_tau_one(self):
        # A flat curve must not produce a spuriously extreme constant.
        rows_by_tau = {
            0.0: [_row("a", "agreement", "George Washington")],
            1.0: [_row("a", "agreement", "George Washington")],
            2.5: [_row("a", "agreement", "George Washington")],
        }
        assert tune_oracle(rows_by_tau)["tau_by_state"]["agreement"] == 1.0

    def test_curves_are_returned_for_plotting(self):
        rows_by_tau = {0.5: [_row("a", "resistance", "George Washington")],
                       1.5: [_row("a", "resistance", "nope")]}
        curves = tune_oracle(rows_by_tau)["curves"]["resistance"]
        assert curves == {0.5: 1.0, 1.5: 0.0}


class TestOracleFromSweep:
    def test_selects_the_row_at_each_states_tau(self):
        rows_by_tau = {
            0.3: [_row("a", "resistance", "right"), _row("b", "correction", "no")],
            1.5: [_row("a", "resistance", "no"), _row("b", "correction", "right")],
        }
        picked = oracle_from_sweep(rows_by_tau,
                                   {"resistance": 0.3, "correction": 1.5})
        assert {(r["case_id"], r["text"]) for r in picked} == {("a", "right"),
                                                              ("b", "right")}
        assert all(r["method"] == "oracle3" for r in picked)

    def test_is_identical_to_regenerating(self):
        # Selecting the already-generated row is provably the same as generating
        # at the routed tau, which is what lets the oracle be free.
        rows_by_tau = {0.3: [_row("a", "resistance", "x")],
                       1.0: [_row("a", "resistance", "y")]}
        picked = oracle_from_sweep(rows_by_tau, {"resistance": 0.3})
        assert len(picked) == 1 and picked[0]["text"] == "x"

    def test_unknown_state_is_skipped(self):
        rows_by_tau = {0.3: [_row("a", "agreement", "x")]}
        assert oracle_from_sweep(rows_by_tau, {"resistance": 0.3}) == []


class TestTest3Gate:
    def test_passes_on_a_large_ceiling(self):
        out = check_test3_kill({"overall": 0.60, "resistance": 0.40},
                               {"grd": {"overall": 0.50, "resistance": 0.21}})
        assert out["fired"] is False
        assert out["gain"] == pytest.approx(0.10)
        assert out["resistance_gain"] == pytest.approx(0.19)

    def test_fires_when_the_oracle_barely_beats_the_baseline(self):
        out = check_test3_kill({"overall": 0.52, "resistance": 0.2},
                               {"grd": {"overall": 0.50, "resistance": 0.2}})
        assert out["fired"] is True

    def test_compares_against_the_strongest_baseline(self):
        out = check_test3_kill(
            {"overall": 0.60, "resistance": 0.3},
            {"cad": {"overall": 0.30, "resistance": 0.02},
             "grd": {"overall": 0.58, "resistance": 0.21}})
        assert out["best_baseline"] == "grd"
        assert out["fired"] is True

    def test_no_baselines_fires(self):
        assert check_test3_kill({"overall": 0.9}, {})["fired"] is True

    def test_threshold_comes_from_config(self):
        assert KILL.min_oracle_gain_over_best_baseline_em == 0.05


class TestPermutation:
    def test_shuffle_preserves_the_marginal(self):
        # Holding the marginal fixed is what makes the control meaningful: a
        # resample from a fitted distribution would change the routing mixture and
        # confound "adaptivity does nothing" with "a different global mixture".
        rng = np.random.default_rng(0)
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = permute_within_all(v, rng)
        assert sorted(out) == sorted(v)

    def test_within_state_shuffle_stays_inside_each_state(self):
        rng = np.random.default_rng(0)
        v = np.array([1.0, 2.0, 10.0, 20.0])
        states = ["a", "a", "b", "b"]
        out = permute_within_state(v, states, rng)
        assert sorted(out[:2]) == [1.0, 2.0]
        assert sorted(out[2:]) == [10.0, 20.0]

    def test_null_auc_sits_at_chance(self):
        rng = np.random.default_rng(2)
        labels = rng.random(300) < 0.5
        values = labels + rng.normal(0, 0.3, 300)
        out = permutation_auc(values, labels, n_shuffles=40)
        assert out["real_auc"] > 0.9
        assert out["null_mean"] == pytest.approx(0.5, abs=0.05)
        assert out["p_value"] < 0.05


class TestTest4Gate:
    def test_passes_when_shuffling_destroys_the_gain(self):
        out = check_test4_kill(real_gain=0.10, permuted_gains=[0.01, 0.0, -0.01])
        assert out["fired"] is False
        assert out["recovered_fraction"] == pytest.approx(0.0, abs=0.02)

    def test_fires_when_shuffling_keeps_the_gain(self):
        out = check_test4_kill(real_gain=0.10, permuted_gains=[0.09, 0.08, 0.10])
        assert out["fired"] is True
        assert "global rescale" in out["reasons"][0]

    def test_no_real_gain_fires(self):
        # Nothing for the control to explain, so this is not a pass.
        out = check_test4_kill(real_gain=-0.01, permuted_gains=[0.0])
        assert out["fired"] is True

    def test_no_valid_permutations_fires(self):
        assert check_test4_kill(0.1, [float("nan")])["fired"] is True

    def test_threshold_comes_from_config(self):
        assert KILL.max_permuted_fraction_of_real_gain == 0.50
