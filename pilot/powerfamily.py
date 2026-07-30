"""Test 3b: the power family, the τ sweep, and three-way oracle routing.

    q(y) ∝ p_pri(y)^(1-τ) · p_ctx(y)^τ
    log q(y) = (1-τ)·log p_pri(y) + τ·log p_ctx(y) + const

τ=0 is the pure prior, τ=1 the pure context, τ∈(0,1) interpolation and τ>1
extrapolation. CAD, AdaCAD, CoCoA and COIECD are all fixed or adaptively-chosen
points in this family; the sweep and the oracle are ours.

**The oracle must be three-way.** A one-sided gate that only ever raises the
correction strength measures the ceiling of a method that cannot resist, which is
the wrong ceiling: it is bounded above by "always follow the context" and would
make resistance look unreachable no matter how good the signal is. So the oracle
routes to one of three τ values chosen per state, including a τ < 1 for
resistance.

Generation runs both prompts as a **two-row batch sharing one KV cache**, so a
step is one forward call rather than two. That is also what makes the spec's
open question answerable: whether the "2× cost" framing in the literature
survives batching. `timing.py` measures it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from . import config
from .model import ModelBundle, stop_token_ids


# --------------------------------------------------------------------------- #
# Combiners
# --------------------------------------------------------------------------- #

class PowerFamily:
    """Fixed-τ member of the family. Combines in log space, unnormalised.

    Normalising would cost a second softmax per step and cannot change the argmax,
    so the returned tensor is a valid set of logits but not a log-probability
    vector. Anything downstream that needs probabilities must softmax it.
    """

    def __init__(self, tau: float):
        self.tau = float(tau)
        self.name = f"power_tau{self.tau:g}"

    def get_next_token_logits(self, logits_ctx: torch.Tensor,
                              logits_prior: torch.Tensor) -> torch.Tensor:
        lp_ctx = F.log_softmax(logits_ctx.float(), dim=-1)
        lp_pri = F.log_softmax(logits_prior.float(), dim=-1)
        return (1.0 - self.tau) * lp_pri + self.tau * lp_ctx

    def get_tau(self, *_args) -> float:
        return self.tau


@dataclass
class OracleRouter:
    """Per-question τ chosen from the ground-truth conflict state.

    Not a decoding method — an upper bound. It answers "if the resist-or-correct
    decision were free, how much would it buy?", which is the only question Test 3
    asks. The three constants are tuned on the train split (`tune_oracle`).
    """
    tau_by_state: Mapping[str, float]
    name: str = "oracle3"

    def for_case(self, case: Mapping[str, Any]) -> PowerFamily:
        return PowerFamily(self.tau_by_state[case["state"]])


@dataclass
class SignalRouter:
    """Per-question τ chosen by thresholding a signal. Used by Test 4.

    `tau_resist` when the signal (oriented so larger = resist) is at or above
    `threshold`, `tau_correct` otherwise. This is the smallest honest instantiation
    of the eventual method: one threshold, two τ values, both fitted on train.
    """
    signal: Mapping[str, float]          # case_id -> oriented signal value
    threshold: float
    tau_resist: float
    tau_correct: float
    name: str = "signal_routed"

    def for_case(self, case: Mapping[str, Any]) -> PowerFamily:
        val = self.signal.get(case["case_id"], float("nan"))
        if val != val:                   # NaN -> fall back to context-trusting
            return PowerFamily(self.tau_correct)
        return PowerFamily(self.tau_resist if val >= self.threshold
                           else self.tau_correct)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #

def _position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Positions that ignore left padding.

    With left padding the first real token sits at index k, not 0, and feeding the
    default 0..T-1 positions would rotate every RoPE embedding by k. The model
    still produces fluent text, so this is another failure that does not announce
    itself — it just makes the two rows of the batch incomparable, which is fatal
    when the whole method is a comparison between them.
    """
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.clamp(min=0)


@torch.no_grad()
def generate(bundle: ModelBundle,
             prompt_ctx: str,
             prompt_pri: str,
             method: Any,
             max_new_tokens: int = config.GEN_MAX_NEW_TOKENS,
             stop_ids: set[int] | None = None) -> dict:
    """Greedy generation under any combiner. Returns the decoded text and the τ path.

    `method` may be
      * a float — treated as a fixed τ,
      * an object with `get_next_token_logits(logits_ctx, logits_prior)` (the
        vendored CAD / AdaCAD / CoCoA / COIECD classes, and PowerFamily),
      * an object with `select_next_token(...)` and `reset()` (vendored GRD, which
        is stateful across the sequence).

    The two prompts go through as one left-padded batch of 2, row 0 = context,
    row 1 = prior, and both rows are fed the *same* chosen token each step. Feeding
    each row its own argmax instead would let the prior branch wander onto a
    different sequence, and p_pri would stop being "the prior for this
    continuation".
    """
    if isinstance(method, (int, float)):
        method = PowerFamily(float(method))
    stop_ids = stop_ids if stop_ids is not None else stop_token_ids(bundle)
    if hasattr(method, "reset"):
        method.reset()

    tok = bundle.tokenizer
    enc = tok([prompt_ctx, prompt_pri], return_tensors="pt", padding=True,
              add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.model.device)
    attn = enc["attention_mask"].to(bundle.model.device)

    out = bundle.model(input_ids=input_ids, attention_mask=attn,
                       position_ids=_position_ids(attn), use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()
    next_pos = _position_ids(attn)[:, -1:] + 1

    produced: list[int] = []
    taus: list[float | None] = []
    for _ in range(max_new_tokens):
        logits_ctx, logits_pri = logits[0:1], logits[1:2]

        if hasattr(method, "select_next_token"):
            token = int(method.select_next_token(logits_ctx[0], logits_pri[0]))
            tau = None
        else:
            adjusted = method.get_next_token_logits(logits_ctx, logits_pri)
            token = int(torch.argmax(adjusted, dim=-1).item())
            tau = method.get_tau(logits_ctx, logits_pri) if hasattr(method, "get_tau") else None
        taus.append(tau)

        if token in stop_ids:
            break
        produced.append(token)

        step_ids = torch.full((2, 1), token, dtype=torch.long,
                              device=bundle.model.device)
        attn = torch.cat([attn, torch.ones((2, 1), dtype=attn.dtype,
                                           device=attn.device)], dim=1)
        out = bundle.model(input_ids=step_ids, attention_mask=attn,
                           position_ids=next_pos, past_key_values=past,
                           use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()
        next_pos = next_pos + 1

    text = tok.decode(produced, skip_special_tokens=True).strip()
    return {"text": text, "n_tokens": len(produced),
            "taus": [t for t in taus if t is not None]}


# --------------------------------------------------------------------------- #
# Evaluation, oracle tuning and the Test 3 gate live in `decoding_eval.py`, which
# imports no torch — the analysis half of Test 3 must run on a CPU-only machine.
# Re-exported here so callers have one import site.
# --------------------------------------------------------------------------- #

from .decoding_eval import (check_test3_kill, evaluate_predictions,  # noqa: E402
                            oracle_from_sweep, tune_oracle)

__all__ = ["PowerFamily", "OracleRouter", "SignalRouter", "generate",
           "evaluate_predictions", "tune_oracle", "oracle_from_sweep",
           "check_test3_kill"]
