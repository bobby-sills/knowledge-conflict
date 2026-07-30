"""The logit lens, and the check that it is actually a lens.

Read the residual stream at layer L, push it through the model's own final layer
norm and unembedding, and you get a distribution over the vocabulary as if the
model had stopped computing at L. No training, no labels — which is exactly why
the pilot starts here rather than with trained probes.

**The trap this module exists to avoid.** In HuggingFace Llama, the last element
of `output_hidden_states` has *already* been through `model.model.norm` — the
model applies the final norm and then appends the result. Applying `norm` again
to that element double-normalises it. RMSNorm is not idempotent (it re-applies
its learned weight), so the result is wrong but plausible: still a valid
distribution, still peaked on sensible tokens, just quietly not the model's
distribution. Every Test 1 number would be off with nothing to show it.

So we do not assume. `calibrate()` reconstructs the final-layer logits both ways
and compares them against the logits the model actually returned, then records
which convention holds. If neither reconstruction matches, it raises — a
mis-detected lens must stop the pipeline, not degrade it.

**How close is "matches".** Not an absolute number of logits. In bf16 a logit of
magnitude ~24 lives on a grid spaced 0.125 apart, so a *correctly rounded*
reconstruction still differs from the model's own by 0.0625 — half a ULP, the
floor. An absolute tolerance below that fails no matter how right the lens is,
which is a bug in the check rather than a finding about the lens. So the
magnitude tolerance is derived from the compute dtype (see `_ulp`), and the
load-bearing criteria are dtype-independent: the two conventions must be
separated by a wide margin, and the reconstruction must rank tokens the way the
model does. See DECISIONS.md §10.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

from .model import ModelBundle, forward_last


# Bits of significand precision, including the implicit leading bit.
_SIGNIFICAND_BITS = {
    torch.bfloat16: 8, torch.float16: 11, torch.float32: 24, torch.float64: 53,
}


def _ulp(magnitude: float, dtype: torch.dtype) -> float:
    """Spacing of `dtype`'s representable grid at `magnitude`.

    The unit in the last place. Two numbers closer together than this are the same
    number in `dtype`, so it is the floor on any reconstruction error — no amount
    of correct arithmetic gets below half of it.
    """
    bits = _SIGNIFICAND_BITS.get(dtype, 24)
    if not (magnitude > 0.0) or not math.isfinite(magnitude):
        return 0.0
    return 2.0 ** (math.floor(math.log2(magnitude)) - (bits - 1))


def _final_norm(bundle: ModelBundle):
    m = bundle.model
    for attr in ("model", "transformer"):
        inner = getattr(m, attr, None)
        if inner is not None:
            for nattr in ("norm", "final_layernorm", "ln_f"):
                norm = getattr(inner, nattr, None)
                if norm is not None:
                    return norm
    if hasattr(m, "get_decoder"):
        dec = m.get_decoder()
        for nattr in ("norm", "final_layernorm", "ln_f"):
            norm = getattr(dec, nattr, None)
            if norm is not None:
                return norm
    raise AttributeError("could not locate the model's final layer norm")


def _unembed(bundle: ModelBundle):
    head = getattr(bundle.model, "lm_head", None)
    if head is None:
        raise AttributeError("could not locate lm_head")
    return head


@torch.no_grad()
def calibrate(bundle: ModelBundle,
              probe_prompts: Sequence[str] | None = None,
              tol: float | None = None,
              ulp_budget: float = 4.0,
              min_discrimination: float = 20.0,
              abs_floor: float = 1e-4,
              rank_k: int = 8) -> dict:
    """Decide whether the last hidden state is pre- or post-final-norm.

    Sets `bundle.lens_mode` to "post_norm" (last state already normed) or
    "pre_norm" (norm must be applied), and returns the reconstruction errors so
    they can go in the manifest. Raises RuntimeError if the lens cannot be shown
    to reproduce the model's own logits.

    Three things must hold, and each can fail independently:

    1. **Magnitude.** The winning convention's max absolute logit error is within
       `tol`. `tol` is *derived from the model's dtype*, not fixed: it is
       `ulp_budget` times the ULP of the compute dtype at the observed logit
       magnitude. In bf16 a logit of ~24 sits on a grid spaced 0.125 apart, so
       correctly-rounded arithmetic still lands 0.0625 away and an absolute
       tolerance below that is unsatisfiable however right the lens is. Pass an
       explicit `tol` to override.
    2. **Discrimination.** The winner must beat the loser by `min_discrimination`
       *unless both conventions are within `tol`*, in which case the choice is
       immaterial — that happens when the final norm is near-idempotent (a
       LayerNorm still at weight=1, bias=0), and both reconstructions are the
       model's own logits. What this rules out is the dangerous middle: two
       conventions that are similarly *wrong*, where the detection is a coin flip.
    3. **Rank agreement.** The reconstruction must put the same token first and the
       same `rank_k` tokens in the top `rank_k`. This is what the lens is actually
       used for — ranking candidate answers — and unlike (1) it does not move when
       the dtype does.
    """
    probe_prompts = list(probe_prompts or [
        "Question: What is the capital of France?\nAnswer:",
        "The first president of the United States was",
    ])
    out = forward_last(bundle, probe_prompts, want_hidden=True)
    logits, hidden = out["logits"], out["hidden"]
    h_last = hidden[:, -1, :]                      # (B, H)

    norm, head = _final_norm(bundle), _unembed(bundle)
    param_dtype = next(head.parameters()).dtype

    direct = head(h_last.to(param_dtype)).float()             # assume post-norm
    normed = head(norm(h_last.to(param_dtype))).float()       # assume pre-norm

    err_direct = float((direct - logits).abs().max())
    err_normed = float((normed - logits).abs().max())

    # The error floor the model's own arithmetic imposes, so a failure can be read
    # as "the lens is wrong" rather than "the dtype cannot do better than this".
    max_abs_logit = float(logits.abs().max())
    ulp = _ulp(max_abs_logit, param_dtype)
    derived_tol = max(ulp_budget * ulp, abs_floor)
    tol_used = derived_tol if tol is None else float(tol)

    if err_direct <= err_normed:
        mode, recon, err_best, err_worst = "post_norm", direct, err_direct, err_normed
    else:
        mode, recon, err_best, err_worst = "pre_norm", normed, err_normed, err_direct

    discrimination = err_worst / max(err_best, 1e-12)
    both_within_tol = err_worst <= tol_used

    k = min(rank_k, int(logits.shape[-1]))
    argmax_ok = bool((recon.argmax(-1) == logits.argmax(-1)).all())
    a, b = recon.topk(k, -1).indices, logits.topk(k, -1).indices
    topk_ok = all(set(a[i].tolist()) == set(b[i].tolist()) for i in range(a.shape[0]))

    failures = []
    if err_best > tol_used:
        failures.append(f"magnitude: best error {err_best:.4g} > tol {tol_used:.4g} "
                        f"({err_best / ulp:.2f} ULP, budget {ulp_budget})")
    if discrimination < min_discrimination and not both_within_tol:
        failures.append(f"discrimination: {discrimination:.2f}x < "
                        f"{min_discrimination}x while the losing convention "
                        f"({err_worst:.4g}) is outside tol — the two are similarly "
                        f"wrong, so the detection is a coin flip")
    if not argmax_ok:
        failures.append("rank: reconstruction and model disagree on the top-1 token")
    if not topk_ok:
        failures.append(f"rank: top-{k} token sets differ")

    if failures:
        raise RuntimeError(
            "logit lens self-check FAILED. Do not trust any Test 1 number until "
            "this is resolved — the lens is not reading what it thinks it is "
            "reading.\n  " + "\n  ".join(failures) +
            f"\n  err_direct={err_direct:.4g}  err_after_norm={err_normed:.4g}"
            f"\n  max|logit|={max_abs_logit:.4g}  {param_dtype} ULP there={ulp:.4g}"
            f"  -> best error is {err_best / ulp if ulp else float('nan'):.2f} ULP")

    bundle.lens_mode = mode
    report = {
        "lens_mode": mode,
        "err_direct": err_direct,
        "err_after_norm": err_normed,
        "tol": tol_used,
        "tol_derived": derived_tol,
        "tol_overridden": tol is not None,
        "max_abs_logit": max_abs_logit,
        "compute_dtype": str(param_dtype),
        "ulp_at_max_logit": ulp,
        "err_best_in_ulp": (err_best / ulp) if ulp else None,
        "discrimination": discrimination,
        "both_within_tol": both_within_tol,
        "argmax_agrees": argmax_ok,
        "topk_agrees": topk_ok,
        "rank_k": k,
        "n_hidden_states": int(hidden.shape[1]),
        "n_layers": bundle.n_layers,
        "probe_prompts": len(probe_prompts),
        "passed": True,
    }
    print(f"[lens] mode={mode}  err={err_best:.3g} "
          f"({report['err_best_in_ulp']:.2f} ULP of {param_dtype}, tol {tol_used:.3g})"
          f"  discrimination={discrimination:.0f}x  top-{k} rank agrees={topk_ok}")
    return report


@torch.no_grad()
def lens_logits(bundle: ModelBundle, hidden: torch.Tensor,
                layer: int | None = None) -> torch.Tensor:
    """Project residual streams to vocabulary logits.

    `hidden` is (..., H) for a single layer, or (B, L+1, H) with `layer=None` to
    project every layer at once. The final layer is projected without re-applying
    the norm when `bundle.lens_mode == "post_norm"`.
    """
    if bundle.lens_mode == "unknown":
        raise RuntimeError("call lens.calibrate() before using the lens")
    norm, head = _final_norm(bundle), _unembed(bundle)
    param_dtype = next(head.parameters()).dtype

    def project(h: torch.Tensor, already_normed: bool) -> torch.Tensor:
        h = h.to(param_dtype)
        return head(h if already_normed else norm(h)).float()

    if layer is not None:
        last = bundle.n_layers
        return project(hidden, already_normed=(layer == last and
                                               bundle.lens_mode == "post_norm"))

    if hidden.ndim != 3:
        raise ValueError(f"expected (B, L+1, H) when layer is None, got {tuple(hidden.shape)}")
    n_states = hidden.shape[1]
    outs = []
    for idx in range(n_states):
        is_last = idx == n_states - 1
        outs.append(project(hidden[:, idx, :],
                            already_normed=is_last and bundle.lens_mode == "post_norm"))
    return torch.stack(outs, dim=1)      # (B, L+1, vocab)


@torch.no_grad()
def lens_token_logprobs(bundle: ModelBundle, hidden: torch.Tensor,
                        token_ids: Sequence[int],
                        chunk_rows: int = 64) -> torch.Tensor:
    """Log-probs of specific tokens at every layer: (B, L+1, len(token_ids)).

    All (row, layer) pairs are flattened into one matrix and unembedded as a single
    GEMM per chunk, rather than one small matmul per layer. Per fact that is 33
    projections of a 4096-vector through a 4096x128256 matrix; as 33 separate
    calls it is launch-bound and takes far longer than the forward pass that
    produced the hidden states.

    Only the candidate columns are kept, but the *full* vocabulary is still
    computed: a log-prob needs the log-sum-exp over every token, so there is no
    correct shortcut that touches only the candidates. `chunk_rows` caps how many
    (row, layer) pairs are in flight — each one costs `vocab` floats, so 64 rows of
    a 128k vocabulary is ~33 MB in fp32.
    """
    if hidden.ndim != 3:
        raise ValueError(f"expected (B, L+1, H), got {tuple(hidden.shape)}")
    if bundle.lens_mode == "unknown":
        raise RuntimeError("call lens.calibrate() before using the lens")

    b, n_states, h_dim = hidden.shape
    ids = torch.as_tensor(list(token_ids), dtype=torch.long, device=hidden.device)
    norm, head = _final_norm(bundle), _unembed(bundle)
    param_dtype = next(head.parameters()).dtype

    flat = hidden.reshape(b * n_states, h_dim).to(param_dtype)
    # Which flattened rows are a final layer that has already been normed.
    is_last = torch.zeros(b * n_states, dtype=torch.bool, device=hidden.device)
    if bundle.lens_mode == "post_norm":
        is_last[torch.arange(b, device=hidden.device) * n_states + (n_states - 1)] = True

    out = torch.empty((b * n_states, ids.numel()), dtype=torch.float32,
                      device=hidden.device)
    for start in range(0, flat.shape[0], chunk_rows):
        stop = min(start + chunk_rows, flat.shape[0])
        block = flat[start:stop]
        mask = is_last[start:stop]
        if mask.any():
            normed = norm(block)
            block = torch.where(mask.unsqueeze(-1), block, normed)
        else:
            block = norm(block)
        logits = head(block).float()
        out[start:stop] = torch.log_softmax(logits, dim=-1).index_select(1, ids)
    return out.reshape(b, n_states, ids.numel())
