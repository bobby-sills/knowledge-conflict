"""Tests for recovering a method's effective tau.

This instrument exists because names and self-reports lie about the one thing the
paper is about — interpolation (tau<1) versus extrapolation (tau>1). It has to be
exact for every affine form the baselines actually use, including the ones that mix
raw logits rather than log-probabilities.
"""

import numpy as np
import pytest

from pilot.regime import classify, describe, fit_residual, fit_tau, log_softmax


V = 2000


@pytest.fixture
def dists():
    """Deliberately mismatched scales and offsets, as real logits are."""
    rng = np.random.default_rng(1)
    raw_ctx = rng.normal(0, 4, V)
    raw_pri = rng.normal(3, 6, V)
    return raw_ctx, raw_pri, log_softmax(raw_ctx), log_softmax(raw_pri)


class TestFitTau:
    @pytest.mark.parametrize("tau", [0.0, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5])
    def test_exact_on_the_power_family(self, dists, tau):
        _, _, lp_ctx, lp_pri = dists
        adjusted = (1 - tau) * lp_pri + tau * lp_ctx
        assert fit_tau(adjusted, lp_ctx, lp_pri) == pytest.approx(tau, abs=1e-9)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0])
    def test_exact_on_cad_which_mixes_raw_logits(self, dists, alpha):
        # CAD is (1+a)*logits_ctx - a*logits_prior on RAW logits, so its output
        # differs from the log-space form by the two partition functions. The fit
        # must absorb that; tau is still 1 + alpha.
        raw_ctx, raw_pri, lp_ctx, lp_pri = dists
        adjusted = (1 + alpha) * raw_ctx - alpha * raw_pri
        assert fit_tau(adjusted, lp_ctx, lp_pri) == pytest.approx(1 + alpha, abs=1e-9)

    def test_detects_cocoas_alpha_plus_gamma(self, dists):
        # The finding this module was built for: the vendored CoCoA runs at
        # alpha + gamma = 1.5 (extrapolation, Table 1's CoCoA*), while its own
        # get_tau() reports alpha = 0.5 (interpolation).
        _, _, lp_ctx, lp_pri = dists
        alpha, gamma = 0.5, 1.0
        adjusted = (alpha * lp_ctx + (1 - alpha) * lp_pri
                    + gamma * (lp_ctx - lp_pri))
        assert fit_tau(adjusted, lp_ctx, lp_pri) == pytest.approx(1.5, abs=1e-9)
        assert classify(1.5) == "extrapolation"
        assert classify(0.5) == "interpolation"

    def test_gamma_zero_recovers_equation_5s_pure_blend(self, dists):
        _, _, lp_ctx, lp_pri = dists
        alpha = 0.5
        adjusted = alpha * lp_ctx + (1 - alpha) * lp_pri
        assert fit_tau(adjusted, lp_ctx, lp_pri) == pytest.approx(alpha, abs=1e-9)

    def test_invariant_to_an_additive_constant(self, dists):
        # Normalisation, partition functions and any constant offset must not move
        # the estimate.
        _, _, lp_ctx, lp_pri = dists
        adjusted = (1 - 1.7) * lp_pri + 1.7 * lp_ctx
        for c in (-50.0, 0.0, 123.4):
            assert fit_tau(adjusted + c, lp_ctx, lp_pri) == pytest.approx(1.7, abs=1e-9)

    def test_uncentred_projection_would_be_badly_wrong(self, dists):
        # Guards the centring itself. Without it, a true tau of 2.5 estimates as
        # about -0.3: confident, plausible, and on the wrong side of the only
        # boundary that matters.
        _, _, lp_ctx, lp_pri = dists
        tau = 2.5
        adjusted = log_softmax((1 - tau) * lp_pri + tau * lp_ctx)
        basis, target = lp_ctx - lp_pri, adjusted - lp_pri
        uncentred = float((target * basis).sum() / (basis * basis).sum())
        assert fit_tau(adjusted, lp_ctx, lp_pri) == pytest.approx(tau, abs=1e-9)
        assert abs(uncentred - tau) > 1.0
        assert uncentred < 1.0 < tau      # misclassifies the regime

    def test_identical_distributions_are_unidentifiable(self, dists):
        _, _, lp_ctx, _ = dists
        assert np.isnan(fit_tau(lp_ctx, lp_ctx, lp_ctx))
        assert classify(float("nan")) == "unidentifiable"


class TestResidual:
    def test_near_zero_for_an_affine_method(self, dists):
        _, _, lp_ctx, lp_pri = dists
        adjusted = (1 - 1.5) * lp_pri + 1.5 * lp_ctx
        tau = fit_tau(adjusted, lp_ctx, lp_pri)
        assert fit_residual(adjusted, lp_ctx, lp_pri, tau) < 1e-9

    def test_large_for_a_non_affine_method(self, dists):
        # A method that is not affine in log space cannot be summarised by one tau,
        # and the residual says so rather than the fit quietly succeeding.
        raw_ctx, raw_pri, lp_ctx, lp_pri = dists
        adjusted = np.tanh(raw_ctx) * 6 - raw_pri ** 2
        tau = fit_tau(adjusted, lp_ctx, lp_pri)
        assert fit_residual(adjusted, lp_ctx, lp_pri, tau) > 1e-3

    def test_nan_tau_gives_nan_residual(self, dists):
        _, _, lp_ctx, lp_pri = dists
        assert np.isnan(fit_residual(lp_ctx, lp_ctx, lp_pri, float("nan")))


class TestDescribe:
    def test_flags_a_disagreeing_self_report(self, dists):
        _, _, lp_ctx, lp_pri = dists
        adjusted = (1 - 1.5) * lp_pri + 1.5 * lp_ctx
        out = describe(adjusted, lp_ctx, lp_pri, reported_tau=0.5)
        assert out["effective_tau"] == pytest.approx(1.5, abs=1e-9)
        assert out["reported_tau"] == 0.5
        assert out["self_report_matches"] is False
        assert out["regime"] == "extrapolation"
        assert out["affine_in_log_space"] is True

    def test_accepts_an_agreeing_self_report(self, dists):
        _, _, lp_ctx, lp_pri = dists
        adjusted = (1 - 0.3) * lp_pri + 0.3 * lp_ctx
        out = describe(adjusted, lp_ctx, lp_pri, reported_tau=0.3)
        assert out["self_report_matches"] is True
        assert out["regime"] == "interpolation"

    def test_no_self_report_is_not_a_match(self, dists):
        _, _, lp_ctx, lp_pri = dists
        out = describe(lp_ctx, lp_ctx, lp_pri)
        assert out["self_report_matches"] is False


class TestLogSoftmax:
    def test_normalises(self):
        lp = log_softmax(np.array([1.0, 2.0, 3.0]))
        assert np.exp(lp).sum() == pytest.approx(1.0)

    def test_stable_on_large_values(self):
        lp = log_softmax(np.array([1e4, 1e4 + 1, 1e4 - 1]))
        assert np.isfinite(lp).all()
        assert np.exp(lp).sum() == pytest.approx(1.0)
