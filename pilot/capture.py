"""The one expensive pass. Everything Tests 1-4 need is logged here, once.

The rule from the spec: Tests 2 and 4 must be *re-analyses of saved data*, not
reruns of the model. So this stage is deliberately greedy about what it writes —
per-layer internal scores, external scores, all output-distribution signals, the
top-k of both distributions, the residual stream at every layer, the gold and
stated token ids. Anything not written here is a second GPU session later.

Split in two, because the prior pass is per *fact* and the context pass is per
*case*:

    capture_prior.jsonl   one row per fact   (question-only pass; the lens lives here)
    capture_ctx.jsonl     one row per case   (fact x document variant; joint signals)

The faithful and corrupted variants of a fact share one question, so the prior
pass would otherwise run twice and — because sampling is off and the prompt is
identical — produce two identical copies of the same 33-layer residual stream.
"""

from __future__ import annotations

import numpy as np
import torch

from . import config, signals as sig
from .documents import context_prompt, prior_prompt
from .factset import candidates, correct_mask
from .lens import lens_token_logprobs
from .model import (ModelBundle, first_answer_token_id, forward_last,
                    teacher_forced_logprob)


# --------------------------------------------------------------------------- #
# Candidate token resolution
# --------------------------------------------------------------------------- #

def resolve_candidate_tokens(bundle: ModelBundle, fact: dict,
                             prompt: str) -> dict:
    """First-token id for every candidate, plus the collision diagnostics.

    Two candidates that share a first token are indistinguishable to a
    first-token scorer — "Paris" and "Parma" do not collide, but "politician" and
    "political leader" very much do. Those collisions cap the achievable knowledge
    score through no fault of the model, so they are counted and reported rather
    than silently absorbed into the numbers.
    """
    cands = candidates(fact)
    ids: list[int | None] = [first_answer_token_id(bundle, prompt, c) for c in cands]
    resolved = [i for i in ids if i is not None]
    n_unresolved = sum(1 for i in ids if i is None)
    n_unique = len(set(resolved))
    gold_id = ids[0]
    # A gold that collides with a distractor's first token makes this fact
    # unusable for first-token scoring; flagged, not dropped, so the rate is
    # visible in the report.
    gold_collides = bool(gold_id is not None and resolved.count(gold_id) > 1)
    return {
        "candidates": cands,
        "candidate_token_ids": ids,
        "n_unresolved_candidates": n_unresolved,
        "n_unique_first_tokens": n_unique,
        "first_token_collision": n_unique < len(resolved),
        "gold_first_token_collides": gold_collides,
        "usable_first_token": bool(gold_id is not None and not gold_collides
                                   and n_unresolved == 0),
    }


def _topk(logprobs: np.ndarray, k: int = config.TOPK_SAVE) -> dict:
    idx = np.argsort(logprobs)[::-1][:k]
    return {"ids": [int(i) for i in idx],
            "logprobs": [round(float(logprobs[i]), 6) for i in idx]}


# --------------------------------------------------------------------------- #
# Prior pass (per fact)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def capture_prior_batch(bundle: ModelBundle, facts: list[dict],
                        want_residuals: bool = config.SAVE_RESIDUALS,
                        full_span_scores: bool = True) -> tuple[list[dict], np.ndarray | None]:
    """Question-only pass for a batch of facts.

    Returns (rows, residuals) where residuals is (B, L+1, H) fp16 for the
    ResidualStore, or None.
    """
    prompts = [prior_prompt(f, bundle.tokenizer) for f in facts]
    out = forward_last(bundle, prompts, want_hidden=True)
    logits, hidden = out["logits"], out["hidden"]
    logp_pri_full = torch.log_softmax(logits, dim=-1)

    rows = []
    for i, fact in enumerate(facts):
        tokinfo = resolve_candidate_tokens(bundle, fact, prompts[i])
        ids = tokinfo["candidate_token_ids"]
        cm = correct_mask(fact)

        # Unresolved candidates get -inf, which `knowledge_score` turns into NaN
        # for the whole fact rather than a partial ranking.
        safe_ids = [i2 if i2 is not None else 0 for i2 in ids]
        ext = logp_pri_full[i, safe_ids].float().cpu().numpy()
        ext = np.where([i2 is None for i2 in ids], np.nan, ext)

        lens_lp = lens_token_logprobs(bundle, hidden[i:i + 1], safe_ids)[0]
        internal = lens_lp.float().cpu().numpy()          # (L+1, n_cand)
        internal = np.where(np.asarray([i2 is None for i2 in ids])[None, :],
                            np.nan, internal)

        lp = logp_pri_full[i].float().cpu().numpy()
        row = {
            "fact_id": fact["fact_id"],
            "split": fact["split"],
            "prompt_pri": prompts[i],
            "correct_mask": cm,
            "external_first_token_logp": [None if np.isnan(v) else float(v) for v in ext],
            "internal_logp_by_layer": [[None if np.isnan(v) else round(float(v), 6)
                                        for v in layer] for layer in internal],
            "prior_entropy": sig.entropy(lp),
            "prior_max": sig.max_prob(lp),
            "prior_topk": _topk(lp),
            "log_popularity": fact.get("log_s_pop"),
            "n_layers_captured": int(internal.shape[0]),
            **{k: v for k, v in tokinfo.items() if k != "candidates"},
        }
        if full_span_scores:
            spans = teacher_forced_logprob(bundle, prompts[i], tokinfo["candidates"])
            row["external_full_span"] = [
                {"mean_logp": s["mean_logp"], "sum_logp": s["sum_logp"],
                 "n_tokens": s["n_tokens"]} for s in spans]
        rows.append(row)

    resid = None
    if want_residuals:
        resid = hidden.to(torch.float16).cpu().numpy()
    return rows, resid


# --------------------------------------------------------------------------- #
# Context pass (per case)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def capture_ctx_batch(bundle: ModelBundle, cases: list[dict],
                      facts_by_id: dict[str, dict],
                      prior_by_id: dict[str, dict]) -> list[dict]:
    """Document+question pass for a batch of cases, plus the joint signals.

    The joint signals (entropy gap, JSD, Renyi) need both distributions at the
    same position. p_ctx comes from this pass; p_pri is reconstructed from the
    prior pass's saved top-k **only for the candidate tokens**, while the
    divergences need the full vocabulary — so the prior pass is repeated here for
    the same batch. That is one extra forward per case, and it is the price of
    having the divergences be exactly right rather than approximated from a
    truncated distribution.
    """
    facts = [facts_by_id[c["fact_id"]] for c in cases]
    ctx_prompts = [context_prompt(f, c["document"], bundle.tokenizer)
                   for f, c in zip(facts, cases)]
    pri_prompts = [prior_prompt(f, bundle.tokenizer) for f in facts]

    out_ctx = forward_last(bundle, ctx_prompts, want_hidden=False)
    out_pri = forward_last(bundle, pri_prompts, want_hidden=False)
    logp_ctx = torch.log_softmax(out_ctx["logits"], dim=-1).float().cpu().numpy()
    logp_pri = torch.log_softmax(out_pri["logits"], dim=-1).float().cpu().numpy()

    rows = []
    for i, (case, fact) in enumerate(zip(cases, facts)):
        prior_row = prior_by_id.get(fact["fact_id"], {})
        gold_tok = None
        cand_ids = prior_row.get("candidate_token_ids")
        if cand_ids:
            gold_tok = cand_ids[0]
        stated_tok = first_answer_token_id(bundle, ctx_prompts[i],
                                          case["stated_object"])

        dist = sig.distribution_signals(logp_pri[i], logp_ctx[i])
        row = {
            "case_id": case["case_id"],
            "fact_id": fact["fact_id"],
            "split": case["split"],
            "state": case["state"],
            "doc_variant": case["doc_variant"],
            "stated_object": case["stated_object"],
            "gold_token": gold_tok,
            "stated_token": stated_tok,
            "ctx_topk": _topk(logp_ctx[i]),
            "pri_topk": _topk(logp_pri[i]),
            "ctx_argmax": int(np.argmax(logp_ctx[i])),
            "pri_argmax": int(np.argmax(logp_pri[i])),
            "log_popularity": fact.get("log_s_pop"),
            **dist,
        }
        # Reachability inputs, computed here while both full distributions exist.
        if gold_tok is not None:
            from .reachability import case_reachability
            row.update(case_reachability(logp_pri[i], logp_ctx[i], gold_tok,
                                         stated_tok))
        rows.append(row)
    return rows


# Reading capture output back lives in `records.py`, which imports no torch — see
# the note there about why re-analysis must not require a GPU environment.
