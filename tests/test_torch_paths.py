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

    def test_the_two_conventions_are_clearly_separated(self, tiny_bundle):
        # The wrong convention should be off by orders of magnitude, not a hair —
        # otherwise the detection is a coin flip.
        from pilot.lens import calibrate
        r = calibrate(tiny_bundle)
        good = min(r["err_direct"], r["err_after_norm"])
        bad = max(r["err_direct"], r["err_after_norm"])
        assert bad > good

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
