"""Wall-clock cost of the two-pass scheme, batched and unbatched.

The literature describes contrastive decoding as "2x cost". Whether that is true
of a real implementation is an open question the spec flags as nearly free to
answer while we are already holding the model, so we answer it: a two-row batch
shares the same kernel launches and the same weight reads, and on a memory-bound
8B decode step the second row is close to free. If that is what the numbers say,
the framing in the literature is about FLOPs rather than latency.

Four configurations, same prompts, same token budget:
    single            one prompt, plain greedy
    two_pass_batched  ctx and prior as one batch of 2, our generation loop
    two_pass_serial   ctx and prior as two separate forward calls per step
    single_batched2   one prompt duplicated into a batch of 2 — the control that
                      separates "the second row costs nothing" from "our loop has
                      overhead the plain path does not"
"""

from __future__ import annotations

import time
from typing import Sequence

import torch

from . import config
from .model import ModelBundle, stop_token_ids
from .powerfamily import PowerFamily, _position_ids, generate


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def _generate_single(bundle: ModelBundle, prompt: str, max_new_tokens: int,
                     n_rows: int = 1) -> int:
    """Plain greedy decode of `n_rows` copies of one prompt. Returns tokens made."""
    tok = bundle.tokenizer
    enc = tok([prompt] * n_rows, return_tensors="pt", padding=True,
              add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.model.device)
    attn = enc["attention_mask"].to(bundle.model.device)
    out = bundle.model(input_ids=input_ids, attention_mask=attn,
                       position_ids=_position_ids(attn), use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1, :]
    next_pos = _position_ids(attn)[:, -1:] + 1
    stops = stop_token_ids(bundle)
    made = 0
    for _ in range(max_new_tokens):
        token = int(torch.argmax(logits[0]).item())
        if token in stops:
            break
        made += 1
        step = torch.full((n_rows, 1), token, dtype=torch.long,
                          device=bundle.model.device)
        attn = torch.cat([attn, torch.ones((n_rows, 1), dtype=attn.dtype,
                                           device=attn.device)], dim=1)
        out = bundle.model(input_ids=step, attention_mask=attn,
                           position_ids=next_pos, past_key_values=past,
                           use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :]
        next_pos = next_pos + 1
    return made


@torch.no_grad()
def _generate_two_pass_serial(bundle: ModelBundle, prompt_ctx: str, prompt_pri: str,
                              tau: float, max_new_tokens: int) -> int:
    """The naive implementation: two independent forward calls per step."""
    tok = bundle.tokenizer
    method = PowerFamily(tau)
    stops = stop_token_ids(bundle)
    states = []
    for prompt in (prompt_ctx, prompt_pri):
        enc = tok([prompt], return_tensors="pt", add_special_tokens=False)
        ids = enc["input_ids"].to(bundle.model.device)
        attn = enc["attention_mask"].to(bundle.model.device)
        out = bundle.model(input_ids=ids, attention_mask=attn, use_cache=True)
        states.append({"past": out.past_key_values, "attn": attn,
                       "logits": out.logits[:, -1, :],
                       "pos": torch.tensor([[ids.shape[1]]],
                                           device=bundle.model.device)})
    made = 0
    for _ in range(max_new_tokens):
        adjusted = method.get_next_token_logits(states[0]["logits"],
                                                states[1]["logits"])
        token = int(torch.argmax(adjusted, dim=-1).item())
        if token in stops:
            break
        made += 1
        for st in states:
            step = torch.full((1, 1), token, dtype=torch.long,
                              device=bundle.model.device)
            st["attn"] = torch.cat(
                [st["attn"], torch.ones((1, 1), dtype=st["attn"].dtype,
                                        device=st["attn"].device)], dim=1)
            out = bundle.model(input_ids=step, attention_mask=st["attn"],
                               position_ids=st["pos"],
                               past_key_values=st["past"], use_cache=True)
            st["past"], st["logits"] = out.past_key_values, out.logits[:, -1, :]
            st["pos"] = st["pos"] + 1
    return made


def measure(bundle: ModelBundle,
            pairs: Sequence[tuple[str, str]],
            max_new_tokens: int = config.GEN_MAX_NEW_TOKENS,
            warmup: int = 1) -> dict:
    """Time the four configurations over the same prompt pairs.

    A warmup pass runs first and is discarded: the first decode step on a fresh
    CUDA context pays for kernel autotuning and allocator growth, which on short
    generations is larger than the effect being measured.
    """
    if len(pairs) <= warmup:
        raise ValueError("need more prompt pairs than warmup iterations")

    for ctx, pri in pairs[:warmup]:
        generate(bundle, ctx, pri, PowerFamily(1.0), max_new_tokens=max_new_tokens)
        _generate_single(bundle, ctx, max_new_tokens)

    work = pairs[warmup:]
    results: dict[str, object] = {}

    def run(name: str, fn):
        _sync()
        t0 = time.perf_counter()
        tokens = sum(fn(ctx, pri) for ctx, pri in work)
        _sync()
        dt = time.perf_counter() - t0
        results[name] = {
            "seconds": dt,
            "n_sequences": len(work),
            "n_tokens": tokens,
            "sec_per_sequence": dt / len(work),
            "ms_per_token": 1000 * dt / max(tokens, 1),
        }

    run("single", lambda ctx, _pri: _generate_single(bundle, ctx, max_new_tokens))
    run("single_batched2",
        lambda ctx, _pri: _generate_single(bundle, ctx, max_new_tokens, n_rows=2))
    run("two_pass_batched",
        lambda ctx, pri: generate(bundle, ctx, pri, PowerFamily(1.0),
                                  max_new_tokens=max_new_tokens)["n_tokens"])
    run("two_pass_serial",
        lambda ctx, pri: _generate_two_pass_serial(bundle, ctx, pri, 1.0,
                                                   max_new_tokens))

    base = results["single"]["sec_per_sequence"]      # type: ignore[index]
    for r in results.values():
        r["overhead_vs_single"] = r["sec_per_sequence"] / base   # type: ignore[index]

    # Token counts differ between configurations (different combiners stop at
    # different points), so per-sequence latency is the honest comparison and
    # per-token is reported for context.
    results["_note"] = (
        "overhead_vs_single is per-sequence latency relative to plain greedy. "
        "Token counts differ between configurations because the combiners stop "
        "at different points; ms_per_token normalises for that.")
    return results
