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
"""

from __future__ import annotations

from typing import Sequence

import torch

from .model import ModelBundle, forward_last


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
              tol: float = 2e-2) -> dict:
    """Decide whether the last hidden state is pre- or post-final-norm.

    Sets `bundle.lens_mode` to "post_norm" (last state already normed) or
    "pre_norm" (norm must be applied), and returns the reconstruction errors so
    they can go in the manifest. Raises RuntimeError if neither convention
    reproduces the model's logits.

    `tol` is on max absolute logit error. bf16 logits of magnitude ~20 carry
    ~0.06 of representation error on their own, so the two hypotheses are
    separated by orders of magnitude, not by a hair: the wrong one is typically
    off by whole logits.
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

    if err_direct <= tol and err_direct < err_normed:
        mode = "post_norm"
    elif err_normed <= tol and err_normed < err_direct:
        mode = "pre_norm"
    else:
        raise RuntimeError(
            "logit lens self-check FAILED: neither convention reproduces the "
            f"model's logits (direct max-abs-err={err_direct:.4g}, "
            f"after-norm={err_normed:.4g}, tol={tol}). Do not trust any Test 1 "
            "number until this is resolved — the lens is not reading what it "
            "thinks it is reading.")

    bundle.lens_mode = mode
    report = {
        "lens_mode": mode,
        "err_direct": err_direct,
        "err_after_norm": err_normed,
        "tol": tol,
        "n_hidden_states": int(hidden.shape[1]),
        "n_layers": bundle.n_layers,
        "probe_prompts": len(probe_prompts),
        "passed": True,
    }
    print(f"[lens] mode={mode}  err_direct={err_direct:.3g}  "
          f"err_after_norm={err_normed:.3g}  (tol {tol})")
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
