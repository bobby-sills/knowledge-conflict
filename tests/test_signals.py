"""Tests for the output-distribution signals and the orientation convention."""

import numpy as np
import pytest

from pilot.signals import (ALL_SIGNALS, EXTERNAL_SIGNALS, INTERNAL_SIGNALS,
                           RESIST_ORIENTATION, apply_orientations,
                           distribution_signals, entropy, fit_orientations, jsd,
                           kl, max_prob, orient, renyi, self_consistency)


def logp(probs):
    return np.log(np.asarray(probs, dtype=np.float64))


class TestEntropy:
    def test_uniform_is_log_n(self):
        for n in (2, 8, 1000):
            assert entropy(logp([1 / n] * n)) == pytest.approx(np.log(n))

    def test_deterministic_is_zero(self):
        assert entropy(logp([1.0, 0.0, 0.0])) == pytest.approx(0.0)

    def test_zero_mass_does_not_produce_nan(self):
        # 0*log0 must be 0; over a 128k vocabulary this is the common case.
        assert np.isfinite(entropy(logp([0.5, 0.5, 0.0, 0.0])))

    def test_rejects_2d(self):
        with pytest.raises(ValueError):
            entropy(np.zeros((2, 3)))


class TestMaxProb:
    def test_reads_the_top_mass(self):
        assert max_prob(logp([0.7, 0.2, 0.1])) == pytest.approx(0.7)


class TestKL:
    def test_identical_is_zero(self):
        p = logp([0.5, 0.3, 0.2])
        assert kl(p, p) == pytest.approx(0.0)

    def test_non_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            p = rng.dirichlet(np.ones(6))
            q = rng.dirichlet(np.ones(6))
            assert kl(logp(p), logp(q)) >= -1e-12

    def test_zero_p_mass_is_skipped(self):
        # p(y)=0 contributes nothing even where q(y)=0 too.
        assert np.isfinite(kl(logp([0.5, 0.5, 0.0]), logp([0.4, 0.6, 0.0])))


class TestJSD:
    def test_identical_is_zero(self):
        p = logp([0.5, 0.3, 0.2])
        assert jsd(p, p) == pytest.approx(0.0)

    def test_disjoint_support_is_ln2(self):
        # The maximum of JSD in nats.
        assert jsd(logp([1.0, 0.0]), logp([0.0, 1.0])) == pytest.approx(np.log(2))

    def test_symmetric(self):
        p, q = logp([0.7, 0.3]), logp([0.2, 0.8])
        assert jsd(p, q) == pytest.approx(jsd(q, p))

    def test_bounded(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            p, q = rng.dirichlet(np.ones(5)), rng.dirichlet(np.ones(5))
            assert -1e-12 <= jsd(logp(p), logp(q)) <= np.log(2) + 1e-12


class TestRenyi:
    def test_alpha_one_reduces_to_kl(self):
        p, q = logp([0.6, 0.3, 0.1]), logp([0.2, 0.3, 0.5])
        assert renyi(p, q, alpha=1.0) == pytest.approx(kl(p, q))

    def test_approaches_kl_as_alpha_approaches_one(self):
        p, q = logp([0.6, 0.3, 0.1]), logp([0.2, 0.3, 0.5])
        assert renyi(p, q, alpha=1.0001) == pytest.approx(kl(p, q), abs=1e-3)

    def test_identical_is_zero(self):
        p = logp([0.5, 0.3, 0.2])
        assert renyi(p, p, alpha=2.0) == pytest.approx(0.0)

    def test_closed_form_alpha_two(self):
        # D_2(p||q) = log sum p^2/q
        p = np.array([0.5, 0.5])
        q = np.array([0.25, 0.75])
        expected = np.log(np.sum(p ** 2 / q))
        assert renyi(logp(p), logp(q), alpha=2.0) == pytest.approx(expected)

    def test_survives_a_vanishing_q(self):
        # Computed in probability space this overflows; over a real vocabulary a
        # near-zero q is guaranteed, so the log-space path is not optional.
        p = np.array([0.5, 0.5])
        q = np.array([1 - 1e-300, 1e-300])
        assert np.isfinite(renyi(logp(p), logp(q), alpha=2.0))


class TestDistributionSignals:
    def test_all_six_present(self):
        p, q = logp([0.7, 0.2, 0.1]), logp([0.1, 0.2, 0.7])
        out = distribution_signals(p, q)
        for key in ("prior_entropy", "prior_max", "entropy_gap", "jsd", "renyi"):
            assert key in out and np.isfinite(out[key])

    def test_entropy_gap_sign(self):
        sharp = logp([0.98, 0.01, 0.01])
        flat = logp([1 / 3, 1 / 3, 1 / 3])
        # Sharp prior, flat context -> H(pri) - H(ctx) < 0
        assert distribution_signals(sharp, flat)["entropy_gap"] < 0
        assert distribution_signals(flat, sharp)["entropy_gap"] > 0


class TestSelfConsistency:
    def test_all_agree(self):
        assert self_consistency(["Paris", "Paris", "Paris"]) == 1.0

    def test_normalisation_applies(self):
        # "the paris." and "Paris" normalise to the same string.
        assert self_consistency(["Paris", "the paris.", "Paris"]) == 1.0

    def test_all_differ(self):
        assert self_consistency(["a", "b", "c"]) == pytest.approx(1 / 3)

    def test_modal_not_correct(self):
        # Agreement, not accuracy: it must be computable without the gold answer,
        # or it would leak the Test 2 label into a predictor of that label.
        assert self_consistency(["wrong", "wrong", "right"]) == pytest.approx(2 / 3)

    def test_empty_is_nan(self):
        assert np.isnan(self_consistency([]))


class TestOrientation:
    def test_every_signal_has_a_declared_direction(self):
        for name in ALL_SIGNALS:
            assert name in RESIST_ORIENTATION

    def test_internal_and_external_are_disjoint(self):
        assert not set(INTERNAL_SIGNALS) & set(EXTERNAL_SIGNALS)

    def test_negated_signals_flip(self):
        vals = np.array([1.0, 2.0, 3.0])
        assert np.allclose(orient("prior_entropy", vals), -vals)
        assert np.allclose(orient("prior_max", vals), vals)

    def test_symmetric_divergence_direction_is_fitted(self):
        # JSD has no a-priori direction: it measures how much two distributions
        # differ, not which to trust. Here larger JSD marks correction, so the
        # fitted direction must be negative.
        labels = np.array([True, True, False, False])
        vals = np.array([0.1, 0.2, 0.8, 0.9])
        oriented = orient("jsd", vals, labels)
        assert np.allclose(oriented, -vals)

    def test_fit_and_apply_are_consistent(self):
        labels = np.array([True, True, False, False])
        table = {"jsd": np.array([0.1, 0.2, 0.8, 0.9]),
                 "prior_max": np.array([0.9, 0.8, 0.2, 0.1])}
        signs = fit_orientations(table, labels)
        assert signs["jsd"] == -1
        assert signs["prior_max"] == 1
        applied = apply_orientations(table, signs)
        assert np.allclose(applied["jsd"], -table["jsd"])
        assert np.allclose(applied["prior_max"], table["prior_max"])

    def test_fitted_direction_is_reused_not_refitted(self):
        # Directions fitted on train must be applied unchanged to the reported
        # split, even when they are the wrong way round there.
        signs = {"jsd": -1}
        table = {"jsd": np.array([0.9, 0.8, 0.1, 0.2])}
        assert np.allclose(apply_orientations(table, signs)["jsd"], -table["jsd"])
