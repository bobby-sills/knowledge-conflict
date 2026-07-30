"""Model loading and the batched forward passes everything else is built on.

Left padding, everywhere. With left padding the last position of every row in a
batch is that row's final prompt token, so "the residual stream at the final
question token" is just `hidden_states[layer][:, -1, :]` — no per-row index
arithmetic to get subtly wrong. Right padding would silently read the residual
stream above a pad token for every sequence except the longest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from . import config


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    n_layers: int          # transformer blocks; hidden_states has n_layers + 1
    hidden: int
    vocab: int
    device: str
    lens_mode: str = "unknown"   # set by lens.calibrate()

    @property
    def n_hidden_states(self) -> int:
        return self.n_layers + 1


def hf_token() -> str | None:
    """Llama-3 is a gated repo. Token from Colab Secrets or the environment —
    never from a cell, and never written to the manifest."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        from google.colab import userdata   # type: ignore
        for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN"):
            try:
                tok = userdata.get(key)
                if tok:
                    os.environ["HF_TOKEN"] = tok
                    return tok
            except Exception:
                continue
    except Exception:
        pass
    return None


def load_model(model_id: str = config.MODEL_ID,
               dtype: str = config.DTYPE,
               device_map: str = "auto") -> ModelBundle:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = hf_token()
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        # Llama-3 ships no pad token. EOS is the conventional stand-in; it is
        # only ever used in masked-out positions.
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device_map, token=token)
    model.eval()

    cfg = model.config
    device = str(next(model.parameters()).device)
    return ModelBundle(
        model=model, tokenizer=tokenizer,
        n_layers=int(cfg.num_hidden_layers),
        hidden=int(cfg.hidden_size),
        vocab=int(getattr(cfg, "vocab_size", len(tokenizer))),
        device=device,
    )


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #

def encode(bundle: ModelBundle, prompts: Sequence[str]) -> dict:
    """Left-padded batch. `add_special_tokens=False` because our prompts already
    carry whatever BOS/header the chat template inserted; letting the tokenizer
    add another BOS shifts every position and quietly changes the distributions."""
    tok = bundle.tokenizer
    enc = tok(list(prompts), return_tensors="pt", padding=True,
              add_special_tokens=False)
    return {k: v.to(bundle.model.device) for k, v in enc.items()}


def first_answer_token_id(bundle: ModelBundle, prompt: str, answer: str) -> int | None:
    """Token id the model would have to emit first to start writing `answer`.

    Done by tokenising prompt and prompt+answer and taking the first token past
    the prompt's length, rather than tokenising the answer alone. BPE is
    context-sensitive: " politician" tokenises differently depending on what
    precedes it, and the standalone form is often not what the model would
    actually emit. Returns None when the prompt's tokenisation is not a prefix of
    the joint tokenisation (a real if rare BPE re-merge), so callers can count
    and exclude those candidates instead of scoring the wrong token.
    """
    tok = bundle.tokenizer
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    j_ids = tok(prompt + answer, add_special_tokens=False)["input_ids"]
    if len(j_ids) <= len(p_ids) or list(j_ids[:len(p_ids)]) != list(p_ids):
        return None
    return int(j_ids[len(p_ids)])


def answer_token_ids(bundle: ModelBundle, prompt: str, answer: str) -> list[int] | None:
    """All tokens of `answer` in the context of `prompt` (for full-span scoring)."""
    tok = bundle.tokenizer
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    j_ids = tok(prompt + answer, add_special_tokens=False)["input_ids"]
    if len(j_ids) <= len(p_ids) or list(j_ids[:len(p_ids)]) != list(p_ids):
        return None
    return [int(t) for t in j_ids[len(p_ids):]]


# --------------------------------------------------------------------------- #
# Forward passes
# --------------------------------------------------------------------------- #

@torch.no_grad()
def forward_last(bundle: ModelBundle, prompts: Sequence[str],
                 want_hidden: bool = True) -> dict:
    """One forward pass per prompt; returns final-position logits and hidden states.

    Returns
        logits : (B, vocab)     float32, at the final prompt token
        hidden : (B, L+1, H)    float32, same position, layer 0 = embeddings
    """
    enc = encode(bundle, prompts)
    out = bundle.model(**enc, output_hidden_states=want_hidden, use_cache=False)
    logits = out.logits[:, -1, :].float()
    hidden = None
    if want_hidden:
        hidden = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=1).float()
    return {"logits": logits, "hidden": hidden,
            "n_tokens": int(enc["attention_mask"].sum().item())}


@torch.no_grad()
def teacher_forced_logprob(bundle: ModelBundle, prompt: str,
                           answers: Sequence[str]) -> list[dict]:
    """Length-normalised and summed log-prob of each answer span after `prompt`.

    The secondary external score. Kept separate from the first-token score
    because it is *not* comparable to the logit lens: the lens only ever sees one
    position, so a multi-token external score would be measuring something the
    internal score structurally cannot.
    """
    results = []
    for answer in answers:
        ids = answer_token_ids(bundle, prompt, answer)
        if ids is None:
            results.append({"answer": answer, "sum_logp": None,
                            "mean_logp": None, "n_tokens": 0})
            continue
        p_ids = bundle.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = torch.tensor([list(p_ids) + ids], device=bundle.model.device)
        out = bundle.model(input_ids=full, use_cache=False)
        logprobs = torch.log_softmax(out.logits[0].float(), dim=-1)
        # Position len(p_ids)-1 predicts the first answer token, and so on.
        total = 0.0
        for k, tid in enumerate(ids):
            total += float(logprobs[len(p_ids) - 1 + k, tid])
        results.append({"answer": answer, "sum_logp": total,
                        "mean_logp": total / len(ids), "n_tokens": len(ids)})
    return results


@torch.no_grad()
def sample_answers(bundle: ModelBundle, prompt: str, n: int,
                   temperature: float = config.PRIOR_TEMPERATURE,
                   top_p: float = config.PRIOR_TOP_P,
                   max_new_tokens: int = config.PRIOR_MAX_NEW_TOKENS,
                   seed: int | None = None) -> list[str]:
    """`n` sampled continuations, as one batch. Used by Test 0's prior screening."""
    if seed is not None:
        torch.manual_seed(seed)
    enc = encode(bundle, [prompt] * n)
    out = bundle.model.generate(
        **enc, do_sample=True, temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens,
        pad_token_id=bundle.tokenizer.pad_token_id,
    )
    gen = out[:, enc["input_ids"].shape[1]:]
    return [bundle.tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen]


def stop_token_ids(bundle: ModelBundle) -> set[int]:
    """EOS plus Llama-3's `<|eot_id|>`, which ends an assistant turn."""
    tok = bundle.tokenizer
    ids = {tok.eos_token_id}
    for name in ("<|eot_id|>", "<|end_of_text|>"):
        tid = tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    return {i for i in ids if isinstance(i, int)}
