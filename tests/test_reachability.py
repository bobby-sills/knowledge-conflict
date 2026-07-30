"""Tests for the pairwise reversal threshold tau*.

The defining property: at tau = tau*, the power family is exactly indifferent
between a and b. Everything else in Test 3a is a histogram of this number, so the
algebra is checked directly against the family rather than against a remembered
formula.
"""

import numpy as np
import pytest

from pilot.reachability import (case_reachability, competitor, log_ratio,
                                summarise, summarise_by_state, tau_star)


def combined_log_ratio(l_pri: float, l_ctx: float, tau: float) -> float:
    """log q(a) - log q(b) under q ∝ p_pri^(1-tau) p_ctx^tau."""
    return (1 - tau) * l_pri + tau * l_ctx


class TestTauStar:
    def test_is_the_indifference_point(self):
        for l_pri, l_ctx in [(2.0, -3.0), (-1.5, 4.0), (0.5, -0.25), (-2.0, -8.0)]:
            t = tau_star(l_pri, l_ctx)
            assert combined_log_ratio(l_pri, l_ctx, t) == pytest.approx(0.0, abs=1e-9)

    def test_resistance_case_lands_in_interpolation(self):
        # Resistance: prior prefers the gold (l_pri > 0), context prefers the
        # corruption (l_ctx < 0). Theory says tau* in (0,1).
        t = tau_star(l_pri=3.0, l_ctx=-2.0)
        assert 0.0 < t < 1.0
        assert t == pytest.approx(3.0 / 5.0)

    def test_correction_case_also_lands_in_interpolation(self):
        # Mirror image: prior wrong, context right.
        t = tau_star(l_pri=-2.0, l_ctx=4.0)
        assert 0.0 < t < 1.0
        assert t == pytest.approx(2.0 / 6.0)

    def test_agreement_gives_no_crossover_in_range(self):
        # Both distributions prefer a; no tau in (0,1) flips the pair.
        t = tau_star(l_pri=3.0, l_ctx=5.0)
        assert not (0.0 < t < 1.0)

    def test_equal_log_ratios_are_undefined(self):
        # No value of tau changes the ordering, so there is no crossover: NaN,
        # not a large number.
        assert np.isnan(tau_star(2.0, 2.0))
        assert np.isnan(tau_star(0.0, 0.0))

    def test_preference_flips_across_tau_star(self):
        l_pri, l_ctx = 3.0, -2.0
        t = tau_star(l_pri, l_ctx)
        assert combined_log_ratio(l_pri, l_ctx, t - 0.1) > 0    # gold preferred
        assert combined_log_ratio(l_pri, l_ctx, t + 0.1) < 0    # corruption wins

    def test_matches_the_spec_formula(self):
        l_pri, l_ctx = 1.7, -0.9
        assert tau_star(l_pri, l_ctx) == pytest.approx(-l_pri / (l_ctx - l_pri))


class TestLogRatio:
    def test_difference_of_logprobs(self):
        lp = np.log(np.array([0.5, 0.25, 0.25]))
        assert log_ratio(lp, 0, 1) == pytest.approx(np.log(2.0))


class TestCompetitor:
    def test_excludes_the_gold(self):
        lp = np.log(np.array([0.6, 0.3, 0.1]))
        assert competitor(lp, gold_token=0) == 1

    def test_finds_the_top_alternative(self):
        lp = np.log(np.array([0.1, 0.2, 0.7]))
        assert competitor(lp, gold_token=2) == 1


class TestCaseReachability:
    def test_reports_both_competitors(self):
        # gold=0; ctx prefers token 2; the document states token 1.
        logp_pri = np.log(np.array([0.7, 0.2, 0.1]))
        logp_ctx = np.log(np.array([0.1, 0.2, 0.7]))
        out = case_reachability(logp_pri, logp_ctx, gold_token=0, stated_token=1)
        assert out["competitor_ctxtop"] == 2
        assert out["competitor_stated"] == 1
        assert 0.0 < out["tau_star_ctxtop"] < 1.0
        assert np.isfinite(out["tau_star_stated"])

    def test_stated_equal_to_gold_is_skipped(self):
        lp = np.log(np.array([0.5, 0.5]))
        out = case_reachability(lp, lp, gold_token=0, stated_token=0)
        assert "tau_star_stated" not in out


class TestSummarise:
    def test_fraction_in_interpolation(self):
        s = summarise([0.2, 0.5, 1.5, -0.3])
        assert s["n"] == 4
        assert s["frac_in_interpolation"] == pytest.approx(0.5)
        assert s["frac_above_1"] == pytest.approx(0.25)
        assert s["frac_below_0"] == pytest.approx(0.25)

    def test_undefined_values_are_counted_not_dropped(self):
        s = summarise([0.5, np.nan, np.nan, 0.5])
        assert s["n"] == 4
        assert s["n_finite"] == 2
        assert s["frac_undefined"] == pytest.approx(0.5)
        # The fraction in range is over the finite values, not over n.
        assert s["frac_in_interpolation"] == 1.0

    def test_empty_is_safe(self):
        assert summarise([])["n"] == 0

    def test_by_state(self):
        records = [{"state": "resistance", "tau_star_ctxtop": 0.4},
                   {"state": "resistance", "tau_star_ctxtop": 0.6},
                   {"state": "correction", "tau_star_ctxtop": 1.8}]
        out = summarise_by_state(records)
        assert out["resistance"]["frac_in_interpolation"] == 1.0
        assert out["correction"]["frac_in_interpolation"] == 0.0
