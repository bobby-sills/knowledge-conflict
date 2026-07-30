"""Tests for the torch-dependent code. Skipped where torch is absent.

The `test_lens_*` and `test_generation_*` classes load a real (tiny) model, so
they are marked `slow` and skipped unless `--run-slow` is passed. Run them on
Colab before the expensive capture:

    pytest tests/test_torch_paths.py --run-slow -q

The lens self-check is the one the spec singles out: verify the final layer
reproduces the model's actual logits before trusting any Test 1 number.
"""

import pytest

torch = pytest.importorskip("torch")

from pilot.powerfamily import PowerFamily, _position_ids   # noqa: E402


class TestPowerFamilyMath:
    def _dists(self):
        # p_pri prefers token 0, p_ctx prefers token 1.
        logits_pri = torch.tensor([[2.0, 0.0, -1.0]])
        logits_ctx = torch.tensor([[0.0, 2.0, -1.0]])
        return logits_ctx, logits_pri

    def test_tau_zero_is_the_pure_prior(self):
        ctx, pri = self._dists()
        out = PowerFamily(0.0).get_next_token_logits(ctx, pri)
        assert int(out.argmax()) == 0
        # Up to an additive constant, tau=0 is exactly log p_pri.
        expected = torch.log_softmax(pri.float(), dim=-1)
        assert torch.allclose(out - out.max(), expected - expected.max(), atol=1e-6)

    def test_tau_one_is_the_pure_context(self):
        ctx, pri = self._dists()
        out = PowerFamily(1.0).get_next_token_logits(ctx, pri)
        assert int(out.argmax()) == 1
        expected = torch.log_softmax(ctx.float(), dim=-1)
        assert torch.allclose(out - out.max(), expected - expected.max(), atol=1e-6)

    def test_interpolation_sits_between(self):
        ctx, pri = self._dists()
        half = PowerFamily(0.5).get_next_token_logits(ctx, pri)
        lp_pri = torch.log_softmax(pri.float(), dim=-1)
        lp_ctx = torch.log_softmax(ctx.float(), dim=-1)
        assert torch.allclose(half, 0.5 * lp_pri + 0.5 * lp_ctx, atol=1e-6)

    def test_extrapolation_suppresses_prior_favoured_tokens(self):
        ctx, pri = self._dists()
        lp_pri = torch.log_softmax(pri.float(), dim=-1)
        at_one = PowerFamily(1.0).get_next_token_logits(ctx, pri)
        at_two = PowerFamily(2.0).get_next_token_logits(ctx, pri)
        # tau>1 puts negative weight on the prior, so the prior's favourite loses
        # ground relative to tau=1.
        gap_one = at_one[0, 0] - at_one[0, 1]
        gap_two = at_two[0, 0] - at_two[0, 1]
        assert gap_two < gap_one
        assert lp_pri[0, 0] > lp_pri[0, 1]

    def test_reversal_happens_at_tau_star(self):
        # Ties the power family back to reachability.tau_star: the same crossover.
        from pilot.reachability import tau_star
        ctx, pri = self._dists()
        lp_pri = torch.log_softmax(pri.float(), dim=-1)[0]
        lp_ctx = torch.log_softmax(ctx.float(), dim=-1)[0]
        l_pri = float(lp_pri[0] - lp_pri[1])
        l_ctx = float(lp_ctx[0] - lp_ctx[1])
        t = tau_star(l_pri, l_ctx)
        assert 0 < t < 1
        below = PowerFamily(t - 0.05).get_next_token_logits(ctx, pri)
        above = PowerFamily(t + 0.05).get_next_token_logits(ctx, pri)
        assert int(below.argmax()) == 0
        assert int(above.argmax()) == 1

    def test_get_tau_reports_the_constant(self):
        assert PowerFamily(1.3).get_tau() == pytest.approx(1.3)

    def test_accepts_unbatched_and_batched(self):
        ctx, pri = self._dists()
        out = PowerFamily(0.5).get_next_token_logits(ctx, pri)
        assert out.shape == (1, 3)


class TestPositionIds:
    def test_ignores_left_padding(self):
        # Left padding puts the first real token at index k, not 0. Feeding the
        # default 0..T-1 positions would rotate every RoPE embedding by k.
        mask = torch.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
        pos = _position_ids(mask)
        assert pos[0].tolist() == [0, 0, 0, 1, 2]
        assert pos[1].tolist() == [0, 1, 2, 3, 4]

    def test_never_negative(self):
        assert int(_position_ids(torch.zeros((1, 3), dtype=torch.long)).min()) == 0

    def test_last_position_is_the_real_length_minus_one(self):
        mask = torch.tensor([[0, 1, 1]])
        assert int(_position_ids(mask)[0, -1]) == 1


class TestRegimeProbeOnTensors:
    """The torch adapter around `pilot.regime`.

    The estimator's arithmetic is covered exactly, on CPU, in test_regime.py. What
    is left to check here is the tensor plumbing: that the adapter feeds it the
    right three vectors and passes a self-report through.
    """

    def _dists(self):
        logits_pri = torch.tensor([[2.0, 0.0, -1.0, 0.5]])
        logits_ctx = torch.tensor([[0.0, 2.0, -1.0, -0.5]])
        return logits_ctx, logits_pri

    @pytest.mark.parametrize("tau", [0.0, 0.3, 1.0, 1.5, 2.5])
    def test_recovers_the_power_family_tau(self, tau):
        from pilot.vendor import effective_tau
        ctx, pri = self._dists()
        assert effective_tau(PowerFamily(tau), ctx, pri) == pytest.approx(tau,
                                                                         abs=1e-4)

    def test_flags_a_disagreeing_self_report(self):
        from pilot.vendor import regime_report

        class Liar:
            name = "liar"

            def get_next_token_logits(self, lc, lp):
                return PowerFamily(1.6).get_next_token_logits(lc, lp)

            def get_tau(self, lc, lp):
                return 0.4

        ctx, pri = self._dists()
        r = regime_report(Liar(), ctx, pri)
        assert r["effective_tau"] == pytest.approx(1.6, abs=1e-4)
        assert r["reported_tau"] == 0.4
        assert r["self_report_matches"] is False
        assert r["regime"] == "extrapolation"
        assert r["method"] == "liar"

    def test_handles_a_no_argument_get_tau(self):
        # PowerFamily.get_tau takes *args; the adapter must cope with either shape.
        from pilot.vendor import regime_report
        ctx, pri = self._dists()
        assert regime_report(PowerFamily(0.3), ctx, pri)["reported_tau"] == \
            pytest.approx(0.3)

    def test_non_affine_method_shows_a_large_residual(self):
        from pilot.vendor import regime_report

        class Nonlinear:
            name = "nonlinear"

            def get_next_token_logits(self, lc, lp):
                return torch.tanh(lc.float()) * 6.0 - lp.float() ** 2

        ctx, pri = self._dists()
        r = regime_report(Nonlinear(), ctx, pri)
        assert r["affine_in_log_space"] is False
        assert r["affine_residual"] > 1e-3


class TestRouters:
    def _case(self, case_id="c1", state="resistance"):
        return {"case_id": case_id, "state": state}

    def test_oracle_routes_by_state(self):
        from pilot.powerfamily import OracleRouter
        r = OracleRouter({"correction": 1.5, "resistance": 0.3, "agreement": 1.0})
        assert r.for_case(self._case(state="resistance")).tau == 0.3
        assert r.for_case(self._case(state="correction")).tau == 1.5

    def test_signal_router_thresholds(self):
        from pilot.powerfamily import SignalRouter
        r = SignalRouter({"hi": 0.9, "lo": 0.1}, threshold=0.5,
                         tau_resist=0.3, tau_correct=1.5)
        assert r.for_case(self._case("hi")).tau == 0.3
        assert r.for_case(self._case("lo")).tau == 1.5

    def test_signal_router_falls_back_on_nan(self):
        from pilot.powerfamily import SignalRouter
        r = SignalRouter({"x": float("nan")}, threshold=0.5, tau_resist=0.3,
                         tau_correct=1.5)
        assert r.for_case(self._case("x")).tau == 1.5
        assert r.for_case(self._case("missing")).tau == 1.5


# --------------------------------------------------------------------------- #
# Real-model tests
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def tiny_bundle(request):
    """A real but tiny causal LM. Needs network access on first run."""
    if not request.config.getoption("--run-slow"):
        pytest.skip("needs --run-slow (downloads a model)")
    from pilot.model import load_model
    return load_model("sshleifer/tiny-gpt2", dtype="float32", device_map=None)


class TestUlp:
    """The dtype grid spacing the lens tolerance is derived from.

    This exists because the first Colab run died on a tolerance of 2e-2 against a
    reconstruction error of 0.0625 — which was *half a ULP*, i.e. arithmetically
    optimal. An absolute tolerance below the dtype's own floor cannot be met.
    """

    def test_bfloat16_spacing_at_llama_logit_magnitudes(self):
        from pilot.lens import _ulp
        # 23.875 is in [16, 32), bf16 has 8 bits of significand -> 2**(4-7).
        assert _ulp(23.875, torch.bfloat16) == pytest.approx(0.125)
        # The observed err_direct was exactly half of this.
        assert 0.0625 == pytest.approx(0.5 * _ulp(23.875, torch.bfloat16))

    def test_spacing_doubles_across_a_power_of_two(self):
        from pilot.lens import _ulp
        assert _ulp(15.9, torch.bfloat16) == pytest.approx(0.0625)
        assert _ulp(16.1, torch.bfloat16) == pytest.approx(0.125)

    def test_float32_is_far_finer_than_bfloat16(self):
        from pilot.lens import _ulp
        assert _ulp(24.0, torch.float32) < _ulp(24.0, torch.bfloat16) / 1000

    def test_degenerate_magnitudes_do_not_explode(self):
        from pilot.lens import _ulp
        assert _ulp(0.0, torch.bfloat16) == 0.0
        assert _ulp(float("nan"), torch.bfloat16) == 0.0
        assert _ulp(float("inf"), torch.bfloat16) == 0.0

    def test_a_bf16_tolerance_is_looser_than_the_old_hardcoded_one(self):
        # Pins the actual bug: the old tol=2e-2 was below the bf16 floor at
        # Llama-3 logit magnitudes, so the check could never pass on this model.
        from pilot.lens import _ulp
        assert 4.0 * _ulp(23.875, torch.bfloat16) > 2e-2


@pytest.mark.slow
class TestLensAgainstRealModel:
    def test_calibrate_reproduces_the_models_logits(self, tiny_bundle):
        # The check the spec insists on: if the lens cannot reproduce the final
        # layer, no Test 1 number is trustworthy.
        from pilot.lens import calibrate
        report = calibrate(tiny_bundle)
        assert report["passed"] is True
        assert report["lens_mode"] in ("post_norm", "pre_norm")
        assert min(report["err_direct"], report["err_after_norm"]) <= report["tol"]

    def test_the_conventions_are_separated_or_both_correct(self, tiny_bundle):
        # The wrong convention should be off by orders of magnitude, not a hair —
        # otherwise the detection is a coin flip. Asserting only `bad > good`, as
        # this test originally did, would pass on a coin flip.
        #
        # The exception this fixture may exercise: tiny-gpt2 is randomly
        # initialised, so its ln_f is still weight=1/bias=0 and therefore close to
        # idempotent. Then both conventions reproduce the logits and the choice is
        # immaterial. That is a pass, not a coin flip.
        from pilot.lens import calibrate
        r = calibrate(tiny_bundle)
        assert r["discrimination"] >= 20.0 or r["both_within_tol"]

    def test_reconstruction_ranks_tokens_like_the_model(self, tiny_bundle):
        from pilot.lens import calibrate
        r = calibrate(tiny_bundle)
        assert r["argmax_agrees"] is True
        assert r["topk_agrees"] is True

    def test_tolerance_is_derived_from_the_dtype_not_hardcoded(self, tiny_bundle):
        from pilot.lens import _ulp, calibrate
        r = calibrate(tiny_bundle)
        assert r["tol_overridden"] is False
        expected = max(4.0 * _ulp(r["max_abs_logit"],
                                  getattr(torch, r["compute_dtype"].split(".")[-1])),
                       1e-4)
        assert r["tol"] == pytest.approx(expected)

    def test_an_impossible_tolerance_still_fails_loudly(self, tiny_bundle):
        # The magnitude check must remain capable of failing when overridden.
        from pilot.lens import calibrate
        with pytest.raises(RuntimeError, match="magnitude"):
            calibrate(tiny_bundle, tol=0.0)

    def test_demanding_absurd_discrimination_fails(self, tiny_bundle):
        # The discrimination check must remain capable of failing.
        from pilot.lens import calibrate
        if calibrate(tiny_bundle)["both_within_tol"]:
            pytest.skip("this fixture's final norm is near-idempotent, so the "
                        "discrimination gate is correctly bypassed — see the real "
                        "model, where the two conventions differ by ~90x")
        with pytest.raises(RuntimeError, match="discrimination"):
            calibrate(tiny_bundle, min_discrimination=1e12)

    def test_hidden_states_count_is_layers_plus_one(self, tiny_bundle):
        from pilot.model import forward_last
        out = forward_last(tiny_bundle, ["hello world"], want_hidden=True)
        assert out["hidden"].shape[1] == tiny_bundle.n_layers + 1

    def test_lens_token_logprobs_shape_and_normalisation(self, tiny_bundle):
        from pilot.lens import calibrate, lens_token_logprobs
        from pilot.model import forward_last
        calibrate(tiny_bundle)
        out = forward_last(tiny_bundle, ["hello", "world"], want_hidden=True)
        ids = [10, 20, 30]
        lp = lens_token_logprobs(tiny_bundle, out["hidden"], ids)
        assert lp.shape == (2, tiny_bundle.n_layers + 1, 3)
        assert bool((lp <= 0).all())        # log-probs

    def test_final_layer_lens_matches_the_external_scores(self, tiny_bundle):
        # The final-layer lens *is* the output distribution, so the internal score
        # at the last layer must equal the external score. If this fails, the
        # internal-vs-external comparison is measuring an artefact.
        from pilot.lens import calibrate, lens_token_logprobs
        from pilot.model import forward_last
        calibrate(tiny_bundle)
        out = forward_last(tiny_bundle, ["the capital of France is"], want_hidden=True)
        ids = [10, 20, 30]
        external = torch.log_softmax(out["logits"], dim=-1)[0, ids]
        internal_last = lens_token_logprobs(tiny_bundle, out["hidden"], ids)[0, -1]
        assert torch.allclose(external, internal_last, atol=1e-3)


@pytest.mark.slow
class TestGenerationLoop:
    def test_two_row_batch_produces_text(self, tiny_bundle):
        from pilot.lens import calibrate
        from pilot.powerfamily import generate
        calibrate(tiny_bundle)
        out = generate(tiny_bundle, "Context: Paris.\nQ: capital of France?\nA:",
                       "Q: capital of France?\nA:", 1.0, max_new_tokens=4)
        assert isinstance(out["text"], str)
        assert out["n_tokens"] <= 4

    def test_tau_zero_equals_prior_only_greedy(self, tiny_bundle):
        # tau=0 must ignore the document entirely. A generation loop that feeds
        # each row its own argmax, or mixes up the rows, fails this.
        from pilot.powerfamily import generate
        prior = "Q: capital of France?\nA:"
        a = generate(tiny_bundle, "Context: Rome is the capital.\n" + prior,
                     prior, 0.0, max_new_tokens=5)
        b = generate(tiny_bundle, "Context: totally different text.\n" + prior,
                     prior, 0.0, max_new_tokens=5)
        assert a["text"] == b["text"]

    def test_float_method_is_coerced_to_power_family(self, tiny_bundle):
        from pilot.powerfamily import generate
        out = generate(tiny_bundle, "a", "a", 0.5, max_new_tokens=2)
        assert out["taus"] == [0.5] * len(out["taus"])


@pytest.mark.slow
class TestTokenisation:
    def test_first_answer_token_is_context_sensitive(self, tiny_bundle):
        from pilot.model import first_answer_token_id
        tid = first_answer_token_id(tiny_bundle, "The capital is", " Paris")
        assert isinstance(tid, int)

    def test_answer_token_ids_are_the_suffix(self, tiny_bundle):
        from pilot.model import answer_token_ids
        prompt, answer = "The capital is", " Paris today"
        ids = answer_token_ids(tiny_bundle, prompt, answer)
        assert ids is not None and len(ids) >= 2
        first = ids[0]
        from pilot.model import first_answer_token_id
        assert first_answer_token_id(tiny_bundle, prompt, answer) == first
